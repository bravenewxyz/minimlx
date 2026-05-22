"""Local tool-use loop for `minimlx code`.

One turn = one call to `engine.stream(…, tools=…)` followed by execution of
any tool_use blocks the model produced. We keep looping turns until the
model responds with text only (no tool call) or a safety cap is hit.

The streaming renderer shows live tokens/sec and strips reasoning blocks. We
buffer the raw text of each turn and post-parse with
`tooluse.parse_content_blocks` so we can separate text replies from tool
calls reliably — mid-stream text/tool interleave is easy to get wrong, and a
single post-parse after stop is accurate.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from minimlx import defaults as _D

from minimlx.codetools import CodeTools, ToolResult
from minimlx.engine import Engine
from minimlx.prompts import system_prompt
from minimlx.render import _split_channels
from minimlx.tooluse import (
    GEMMA4_CLOSE,
    GEMMA4_OPEN,
    GENERIC_CLOSE,
    GENERIC_OPEN,
    anthropic_tools_to_transformers,
    build_tool_system_prompt,
    flatten_history,
    parse_content_blocks,
    template_supports_tools,
)


def _stabilize_for_markdown(s: str) -> str:
    """Close unclosed triple-backtick fences so mid-stream partial code blocks
    don't make the entire tail render as code until the closing fence arrives."""
    fence_count = 0
    for line in s.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fence_count += 1
    if fence_count % 2 == 1:
        return s + "\n```"
    return s


# Best-effort name extractors for in-flight tool calls. Both formats put the
# name very early in the body, so we can pull it out before the close tag
# arrives. See `tooluse._parse_*_body` for the authoritative parsers — these
# regexes are strictly for the live placeholder.
_GEMMA4_NAME_RE = re.compile(r"call:\s*([A-Za-z_][\w\-.]*)")
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _preview_tool_body(body: str) -> str:
    """Best-effort one-liner of a tool call's name + args for live display.
    Args are shown in full (no truncation). Returns an empty string when the
    body isn't recognizably formed yet — caller falls back to the bare
    placeholder."""
    body = body.strip()
    m = _GEMMA4_NAME_RE.search(body)
    if m:
        name = m.group(1)
        brace = body.find("{", m.end())
        if brace == -1:
            return name
        return f"{name}{body[brace:].rstrip()}"
    m = _JSON_NAME_RE.search(body)
    if m:
        # Show the full JSON body when available; the raw `{…}` after the
        # name field is more informative than just the name alone.
        brace = body.find("{")
        if brace != -1:
            return body[brace:].rstrip()
        return m.group(1)
    return ""


def _mask_tool_markers(s: str) -> str:
    """Replace complete and partial tool-call blocks with a short placeholder
    for display only. Includes the tool name (+ truncated args) when we can
    extract them; falls back to a bare `[tool call…]` if not. The parser
    operates on the raw buffer, not this masked view."""
    pairs = ((GEMMA4_OPEN, GEMMA4_CLOSE), (GENERIC_OPEN, GENERIC_CLOSE))
    out = s
    for open_m, close_m in pairs:
        while True:
            i = out.find(open_m)
            if i == -1:
                break
            j = out.find(close_m, i + len(open_m))
            if j == -1:
                preview = _preview_tool_body(out[i + len(open_m):])
                placeholder = f"[tool call: {preview}…]" if preview else "[tool call…]"
                out = out[:i] + placeholder
                break
            preview = _preview_tool_body(out[i + len(open_m):j])
            placeholder = f"[tool call: {preview}]" if preview else "[tool call]"
            out = out[:i] + placeholder + out[j + len(close_m):]
    return out


MAX_TURNS_PER_REQUEST = 12


@dataclass
class TurnResult:
    text_blocks: list[str]
    tool_calls: list[dict]
    stop_reason: str


def _drain_and_render(
    chunks: Iterable,
    console: Console,
    label: str,
    refresh_per_second: int = 12,
) -> str:
    """Drain the chunk stream while rendering text (reasoning stripped) to the
    terminal. Returns the raw unfiltered buffer for downstream parsing.

    Three streams land in different places:
      - **Thinking content** → `console.print` (dim italic) → scrollback.
      - **Tool-call bodies** → `console.print` (dim cyan) → scrollback. Live
        widget shows a compact `[calling write_file…]` marker in their place.
      - **Answer text** → `Live` widget with markdown rendering.

    Routing the long-form bodies (thinking, tool calls) to scrollback rather
    than holding them in `Live` keeps the live frame bounded — once a frame
    grows past terminal height, Rich can no longer redraw the off-screen
    portion and the UI looks stuck.
    """
    raw = ""
    n_tok = 0
    t0 = time.perf_counter()

    md_cache: dict[str, Any] = {"t": 0.0, "body": None}
    MD_INTERVAL = 0.25  # seconds

    thinking_header_printed = False
    thinking_finalized = False  # True after we've printed the post-thinking newline
    printed_thinking = 0
    # Per tool-call: keys are the body-start byte offsets in `raw`.
    streamed_tool_bodies: dict[int, dict[str, Any]] = {}
    live: Live | None = None

    def _flush_thinking() -> None:
        """Append any new thinking text to scrollback. No-op once finalized."""
        nonlocal printed_thinking, thinking_header_printed
        if thinking_finalized:
            return
        thinking_now, _, _ = _split_channels(raw)
        if thinking_now and not thinking_header_printed:
            console.print("[dim italic]thinking…[/]")
            thinking_header_printed = True
        new_text = thinking_now[printed_thinking:]
        if new_text:
            console.print(new_text, end="", style="italic dim", highlight=False)
            printed_thinking = len(thinking_now)

    def _finalize_thinking() -> None:
        """Flush any remaining thinking content and print the separator. The
        scrollback is then frozen — the live widget owns the answer phase."""
        nonlocal thinking_finalized
        if thinking_finalized:
            return
        _flush_thinking()
        if thinking_header_printed:
            console.print()
        thinking_finalized = True

    def _stream_tool_bodies(answer: str) -> None:
        """Stream new tool-call body bytes to scrollback. Idempotent —
        tracks per-body progress so each chunk only prints what's new."""
        pairs = ((GEMMA4_OPEN, GEMMA4_CLOSE), (GENERIC_OPEN, GENERIC_CLOSE))
        for open_m, close_m in pairs:
            cursor = 0
            while True:
                i = answer.find(open_m, cursor)
                if i == -1:
                    break
                body_start = i + len(open_m)
                j = answer.find(close_m, body_start)
                body_end = j if j != -1 else len(answer)

                state = streamed_tool_bodies.get(body_start)
                if state is None:
                    # First time seeing this open marker — extract the
                    # tool name (if visible yet) and print a header.
                    preview = _preview_tool_body(answer[body_start:body_end])
                    name = preview.split("{", 1)[0] if preview else "tool"
                    console.print()
                    console.print(f"[dim cyan]→ {name} body:[/]")
                    state = {"printed": 0, "closed": False}
                    streamed_tool_bodies[body_start] = state

                new_text = answer[body_start + state["printed"]:body_end]
                if new_text:
                    console.print(new_text, end="", style="dim cyan", highlight=False)
                    state["printed"] = body_end - body_start
                if j != -1 and not state["closed"]:
                    console.print()  # newline closes the body block
                    state["closed"] = True

                cursor = (j + len(close_m)) if j != -1 else len(answer)

    def _compact_mask(answer: str) -> str:
        """Replace each tool-call block with a tiny `[calling NAME…]` marker
        so the Live widget body stays bounded. The full body is already in
        scrollback via `_stream_tool_bodies`."""
        pairs = ((GEMMA4_OPEN, GEMMA4_CLOSE), (GENERIC_OPEN, GENERIC_CLOSE))
        out = answer
        for open_m, close_m in pairs:
            while True:
                i = out.find(open_m)
                if i == -1:
                    break
                j = out.find(close_m, i + len(open_m))
                if j == -1:
                    preview = _preview_tool_body(out[i + len(open_m):])
                    name = preview.split("{", 1)[0] if preview else "tool"
                    out = out[:i] + f"[calling {name}…]"
                    break
                preview = _preview_tool_body(out[i + len(open_m):j])
                name = preview.split("{", 1)[0] if preview else "tool"
                out = out[:i] + f"[called {name}]" + out[j + len(close_m):]
        return out

    def _render_answer(final: bool) -> Group:
        thinking_v, answer_v, _ = _split_channels(raw)
        dt = time.perf_counter() - t0
        tps = n_tok / dt if dt > 0 else 0.0
        masked = _compact_mask(answer_v)
        if masked.strip():
            now = time.perf_counter()
            need_reparse = (
                final
                or md_cache["body"] is None
                or (now - md_cache["t"]) >= MD_INTERVAL
            )
            if need_reparse:
                try:
                    stable = masked if final else _stabilize_for_markdown(masked)
                    md_cache["body"] = Markdown(stable, code_theme="monokai")
                except Exception:
                    md_cache["body"] = Text(masked)
                md_cache["t"] = now
            b: Any = md_cache["body"]
        else:
            b = Text(masked)
        extras = f" · thought {len(thinking_v)} chars" if thinking_v else ""
        s = Text(
            f"  {n_tok} tok · {tps:5.1f} tok/s{extras}",
            style="dim green" if final else "dim cyan",
        )
        return Group(b, s)

    try:
        for chunk in chunks:
            raw += chunk.text
            n_tok += chunk.n_tokens
            thinking, answer, still_thinking = _split_channels(raw)

            if still_thinking:
                _flush_thinking()
                continue

            if not (answer.strip() or thinking or thinking_header_printed):
                continue

            # One-shot finalize: idempotent flush + separator. Handles the
            # edge case where open + close arrive in the same chunk before
            # phase 1 ever fires, and the normal case where phase 1 had been
            # streaming gradually. Returns immediately on subsequent calls.
            _finalize_thinking()
            # Tool-call bodies stream straight to scrollback in dim cyan;
            # the Live widget below only sees the compact `[calling X…]`
            # placeholder via `_compact_mask`. Rich routes `console.print`
            # calls inside an active Live context above the live area.
            _stream_tool_bodies(answer)
            if live is None:
                live = Live(
                    Group(Text(""), Text("", style="dim")),
                    console=console,
                    refresh_per_second=refresh_per_second,
                    transient=False,
                    vertical_overflow="visible",
                )
                live.__enter__()
            live.update(_render_answer(final=False))
    except KeyboardInterrupt:
        pass
    finally:
        if live is not None:
            try:
                live.update(_render_answer(final=True))
            except Exception:
                pass
            live.__exit__(None, None, None)
        elif thinking_header_printed and not thinking_finalized:
            # Pure-thinking output (no answer phase reached) — close the
            # streamed thought block with a newline so the prompt doesn't
            # dangle off the last italic line.
            _flush_thinking()
            console.print()
    return raw


def _render_tool_call(console: Console, name: str, inp: dict) -> None:
    body = Text()
    body.append(f"{name}", style="bold cyan")
    body.append("(")
    try:
        body.append(json.dumps(inp, ensure_ascii=False), style="dim")
    except Exception:
        body.append(str(inp), style="dim")
    body.append(")")
    console.print(body)


def _render_tool_result(console: Console, r: ToolResult) -> None:
    preview = r.output if len(r.output) < 400 else r.output[:400] + f"\n[…+{len(r.output) - 400} chars]"
    style = "red" if r.is_error else "dim"
    console.print(Panel(Text(preview, style=style), border_style=style, expand=False))


def _one_turn(
    engine: Engine,
    messages: list[dict],
    tools: list[dict],
    console: Console,
    *,
    temp: float,
    top_p: float,
    max_tokens: int,
) -> TurnResult:
    """Run one engine turn, return text blocks + tool_use blocks found."""
    transformers_tools = anthropic_tools_to_transformers(tools)
    flattened = flatten_history(messages)

    # For models whose template doesn't render tools natively, the format
    # lesson goes into a *front-loaded* system message and the engine will
    # quietly drop the `tools=` arg when apply_chat_template rejects it.
    supports = template_supports_tools(engine.tokenizer) if engine._loaded else True

    kwargs: dict[str, Any] = dict(max_tokens=max_tokens, temp=temp, top_p=top_p)
    if supports:
        kwargs["tools"] = transformers_tools

    raw = _drain_and_render(
        engine.stream(flattened, **kwargs),
        console,
        label=engine.model_id.split("/")[-1],
    )

    # Strip reasoning before tool-block parsing so a `<tool_call>` the model
    # *reasoned about* isn't mistaken for a real call.
    _, answer, _ = _split_channels(raw)
    blocks, stop_reason = parse_content_blocks(answer)
    text_blocks = [b["text"] for b in blocks if b.get("type") == "text"]
    tool_calls = [b for b in blocks if b.get("type") == "tool_use"]
    return TurnResult(text_blocks=text_blocks, tool_calls=tool_calls, stop_reason=stop_reason)


def run_code_session(
    engine: Engine,
    console: Console,
    *,
    initial_prompt: str,
    cwd: Path,
    allow_bash: bool = False,
    unrestricted: bool = False,
    temp: float = _D.CODE_TEMP,
    top_p: float = _D.TOP_P,
    max_tokens: int = _D.CODE_MAX_TOKENS,
) -> list[dict]:
    """Single-request agentic loop. Returns the full message history."""
    engine.load()
    supports = template_supports_tools(engine.tokenizer)

    tools_impl = CodeTools(cwd=cwd, allow_bash=allow_bash, unrestricted=unrestricted)
    tool_schemas = tools_impl.schemas()

    sys_parts = [system_prompt(engine.model_id, template_supports_tools=supports)]
    if not supports:
        sys_parts.append(build_tool_system_prompt(tool_schemas))
    sys_prompt = "\n\n".join(p for p in sys_parts if p)

    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [{"type": "text", "text": initial_prompt}]},
    ]

    console.print(Panel.fit(
        Text.assemble(
            ("minimlx code", "bold cyan"),
            ("  ·  ", "dim"),
            (engine.model_id, "dim"),
            ("  ·  cwd=", "dim"),
            (str(cwd), "dim"),
            (f"  ·  tools: {len(tool_schemas)}", "dim"),
            (f"  ·  bash: {'on' if allow_bash else 'off'}", "dim"),
        ),
        border_style="cyan",
    ))

    for turn_i in range(MAX_TURNS_PER_REQUEST):
        turn = _one_turn(
            engine, messages, tool_schemas, console,
            temp=temp, top_p=top_p, max_tokens=max_tokens,
        )

        assistant_content: list[dict] = []
        for t in turn.text_blocks:
            if t.strip():
                assistant_content.append({"type": "text", "text": t})
        for c in turn.tool_calls:
            assistant_content.append(c)
        if not assistant_content:
            assistant_content = [{"type": "text", "text": ""}]
        messages.append({"role": "assistant", "content": assistant_content})

        if not turn.tool_calls:
            return messages

        tool_results_blocks: list[dict] = []
        for call in turn.tool_calls:
            _render_tool_call(console, call["name"], call.get("input") or {})
            result = tools_impl.run(call["name"], call.get("input") or {})
            _render_tool_result(console, result)
            tool_results_blocks.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": [{"type": "text", "text": result.output}],
                "is_error": result.is_error,
            })
        messages.append({"role": "user", "content": tool_results_blocks})

    console.print(Text(f"[stopped: reached {MAX_TURNS_PER_REQUEST}-turn cap]",
                       style="yellow"))
    return messages
