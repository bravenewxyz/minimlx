from __future__ import annotations
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from minimlx.aliases import load_aliases, load_preset, resolve_alias
from minimlx.codeloop import _drain_and_render, _mask_tool_markers
from minimlx.codetools import CodeTools
from minimlx.engine import Engine
from minimlx.prompts import system_prompt
from minimlx.render import _split_channels, stream_response
from minimlx.speak import Speaker
from minimlx.store import Store
from minimlx.toolrender import render_call, render_result
from minimlx.tooluse import (
    anthropic_tools_to_transformers,
    build_tool_system_prompt,
    flatten_history,
    parse_content_blocks,
    template_supports_tools,
)

from minimlx import defaults as D
HISTORY_FILE = D.CONFIG_DIR / "history"

MAX_TOOL_TURNS_PER_MESSAGE = 8


def _keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    return kb


def _print_help(console: Console) -> None:
    console.print(
        "[dim]/clear /new /reset · clear conversation (keep current model)[/]\n"
        "[dim]/save F        · save transcript to file F[/]\n"
        "[dim]/model         · show current model[/]\n"
        "[dim]/models [A]    · list aliases, or switch to alias A[/]\n"
        "[dim]/stats         · show last-turn stats[/]\n"
        "[dim]/speak         · toggle TTS on/off[/]\n"
        "[dim]/voice NAME    · set TTS voice (e.g. Samantha)[/]\n"
        "[dim]/rate WPM      · set TTS words per minute[/]\n"
        "[dim]/stop          · stop current TTS playback[/]\n"
        "[dim]/help          · this help[/]\n"
        "[dim]/exit          · quit (Ctrl-D also exits)[/]"
    )


def _render_alias_table(console: Console, current_id: str) -> None:
    from rich.table import Table
    aliases = load_aliases()
    table = Table(title="aliases", show_lines=False)
    table.add_column("alias", style="cyan")
    table.add_column("resolves to")
    table.add_column("preset", style="dim")
    table.add_column("", style="green")
    for k in sorted(aliases):
        preset = load_preset(k)
        bits: list[str] = []
        if preset.get("draft"):
            bits.append(f"draft={preset['draft']}×{preset.get('num_draft_tokens', 4)}")
        if "temp" in preset:
            bits.append(f"temp={preset['temp']}")
        active = "●" if resolve_alias(k, aliases) == current_id else ""
        table.add_row(k, aliases[k], " · ".join(bits), active)
    console.print(table)


def _prompt_alias(console: Console, aliases: dict[str, str]) -> str | None:
    """Prompt for an alias name with fuzzy completion. Empty input cancels."""
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.completion import FuzzyWordCompleter
    completer = FuzzyWordCompleter(sorted(aliases.keys()))
    try:
        choice = pt_prompt("alias › ", completer=completer).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice:
        return None
    if choice not in aliases:
        console.print(f"[red]unknown alias:[/] {choice}")
        return None
    return choice


def _mlx_reclaim() -> None:
    """Best-effort: run GC and clear MLX's Metal allocator cache.

    Call between model swaps and on conversation resets to avoid Metal
    command-buffer timeouts caused by fragmented residue from prior runs.
    """
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _swap_engine(current: Engine, new_alias: str) -> Engine:
    """Build a fresh Engine for `new_alias`, preserving kv/quant settings."""
    aliases = load_aliases()
    preset = load_preset(new_alias)
    draft = preset.get("draft")
    num_draft = int(preset.get("num_draft_tokens", 4))
    return Engine(
        model_id=resolve_alias(new_alias, aliases),
        draft_model_id=resolve_alias(draft, aliases) if draft else None,
        num_draft_tokens=num_draft,
        max_kv_size=current.max_kv_size,
        kv_bits=current.kv_bits,
        kv_group_size=current.kv_group_size,
        quantized_kv_start=current.quantized_kv_start,
    )


def _assistant_visible_text(msg: dict) -> str:
    """Concatenate the text blocks of an assistant message, ignoring tool_use.

    Works for both string-content (plain chat) and list-content (tool-use)
    assistant messages.
    """
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)
    return str(c)


def _save_transcript(messages: list[dict], path: str) -> None:
    chunks: list[str] = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            # Skip tool_use / tool_result blocks — save only visible text.
            if role == "user" and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in c
            ):
                continue
            parts = [
                b.get("text", "")
                for b in c
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = "".join(parts)
        else:
            text = str(c)
        if not text.strip():
            continue
        chunks.append(f"## {role}\n{text}")
    Path(path).write_text("\n\n".join(chunks))


def _run_tool_loop(
    engine: Engine,
    console: Console,
    messages: list[dict],
    tools_impl: CodeTools,
    tool_schemas: list[dict],
    *,
    max_tokens: int,
    temp: float,
    top_p: float,
) -> str:
    """Run the tool-use loop for a single user turn.

    Messages already include the latest user message at entry. Appends
    assistant (and any synthesized user-tool_result) messages in place.
    Returns the final visible assistant text for the turn.
    """
    supports = template_supports_tools(engine.tokenizer) if engine._loaded else True
    transformers_tools = anthropic_tools_to_transformers(tool_schemas)

    final_text = ""
    for turn_i in range(MAX_TOOL_TURNS_PER_MESSAGE):
        flattened = flatten_history(messages)
        kwargs: dict[str, Any] = dict(max_tokens=max_tokens, temp=temp, top_p=top_p)
        if supports:
            kwargs["tools"] = transformers_tools

        raw = _drain_and_render(
            engine.stream(flattened, **kwargs),
            console,
            label=engine.model_id.split("/")[-1],
        )

        _, answer, _ = _split_channels(raw)
        blocks, _stop = parse_content_blocks(answer)
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        tool_calls = [b for b in blocks if b.get("type") == "tool_use"]

        assistant_content: list[dict] = []
        for tb in text_blocks:
            if tb.get("text", "").strip():
                assistant_content.append({"type": "text", "text": tb["text"]})
        for c in tool_calls:
            assistant_content.append(c)
        if not assistant_content:
            assistant_content = [{"type": "text", "text": ""}]
        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_calls:
            final_text = "".join(
                b["text"] for b in assistant_content if b.get("type") == "text"
            )
            return final_text

        tool_results_blocks: list[dict] = []
        for call in tool_calls:
            name = call.get("name", "")
            inp = call.get("input") or {}
            render_call(console, name, inp)
            result = tools_impl.run(name, inp)
            render_result(console, name, inp, result)
            tool_results_blocks.append({
                "type": "tool_result",
                "tool_use_id": call.get("id", ""),
                "content": [{"type": "text", "text": result.output}],
                "is_error": result.is_error,
            })
        messages.append({"role": "user", "content": tool_results_blocks})

    console.print(Text("[stopped: tool loop cap reached]", style="dim yellow"))
    # final_text will be whatever the last assistant visible text was, if any.
    if messages and messages[-1].get("role") == "assistant":
        final_text = _assistant_visible_text(messages[-1])
    return final_text


def run_chat(
    engine: Engine,
    store: Store,
    console: Console,
    *,
    system: str | None = None,
    conversation_id: int | None = None,
    max_tokens: int = 2048,
    temp: float = D.TEMP,
    top_p: float = D.TOP_P,
    speaker: Speaker | None = None,
    tools_on: bool = True,
    allow_bash: bool = True,
    cwd: Path | None = None,
    unrestricted: bool = False,
) -> None:
    D.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if speaker is None:
        speaker = Speaker(enabled=False)

    tts_hint = "  [dim]· TTS on[/]" if speaker.enabled else ""

    # Tool-use plumbing.
    tools_impl: CodeTools | None = None
    tool_schemas: list[dict] = []
    tools_hint = ""
    if tools_on:
        engine.load()
        effective_cwd = (cwd or Path.cwd()).resolve()
        tools_impl = CodeTools(cwd=effective_cwd, allow_bash=allow_bash, unrestricted=unrestricted)
        tool_schemas = tools_impl.schemas()
        scope_hint = " · unrestricted" if unrestricted else ""
        tools_hint = (
            f"  [dim]· tools: {len(tool_schemas)}"
            f" · bash: {'on' if allow_bash else 'off'}{scope_hint}[/]"
        )

    console.print(Panel.fit(
        f"[bold cyan]minimlx chat[/]  [dim]· {engine.model_id}[/]{tts_hint}{tools_hint}\n"
        "[dim]Enter submits · Alt+Enter newline · /help · Ctrl-C stops gen · Ctrl-D exits[/]",
        border_style="cyan",
    ))

    messages: list[dict] = []
    if conversation_id is not None:
        messages = store.load_messages(conversation_id)
        console.print(f"[dim]-- continuing conversation {conversation_id} ({len(messages)} msgs) --[/]")

    # Build the system prompt. When tools are on the tool-use system prompt
    # comes first, then the user-supplied `system` string.
    if not messages:
        sys_parts: list[str] = []
        if tools_on:
            supports = template_supports_tools(engine.tokenizer)
            sys_parts.append(system_prompt(engine.model_id, template_supports_tools=supports))
            if not supports:
                sys_parts.append(build_tool_system_prompt(tool_schemas))
        if system:
            sys_parts.append(system)
        sys_joined = "\n\n".join(p for p in sys_parts if p)
        if sys_joined:
            messages.append({"role": "system", "content": sys_joined})

    if conversation_id is None:
        conversation_id = store.new_conversation(engine.model_id, system)

    def _turns() -> int:
        return sum(1 for m in messages if m["role"] == "user" and (
            isinstance(m.get("content"), str) or
            (isinstance(m.get("content"), list) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in m.get("content", [])
            ))
        ))

    def _toolbar() -> HTML:
        tts = "  spk:on" if speaker.enabled else ""
        tl = "  tools:on" if tools_on else ""
        return HTML(
            f" <b>{engine.model_id.split('/')[-1]}</b> · turns: {_turns()}{tts}{tl} · /help "
        )

    session: PromptSession = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        multiline=True,
        key_bindings=_keybindings(),
        bottom_toolbar=_toolbar,
        prompt_continuation=lambda w, *_: "." * (w - 1) + " ",
    )

    while True:
        try:
            line = session.prompt("› ").strip()
        except EOFError:
            speaker.stop()
            console.print("[dim]bye[/]")
            break
        except KeyboardInterrupt:
            speaker.stop()
            console.print("[dim]bye[/]")
            break
        speaker.stop()
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""
            if cmd in ("/exit", "/quit", "/q"):
                break
            if cmd in ("/reset", "/clear", "/new"):
                messages = [m for m in messages if m["role"] == "system"]
                conversation_id = store.new_conversation(engine.model_id, system)
                _mlx_reclaim()
                console.print("[dim]-- reset --[/]")
                continue
            if cmd == "/help":
                _print_help(console)
                continue
            if cmd == "/model":
                console.print(f"[dim]{engine.model_id}[/]")
                continue
            if cmd == "/models":
                aliases_map = load_aliases()
                target: str | None = None
                if arg.strip():
                    target = arg.strip()
                    if target not in aliases_map:
                        console.print(f"[red]unknown alias:[/] {target}")
                        continue
                else:
                    _render_alias_table(console, engine.model_id)
                    target = _prompt_alias(console, aliases_map)
                    if target is None:
                        console.print("[dim]-- cancelled --[/]")
                        continue
                new_model_id = resolve_alias(target, aliases_map)
                if new_model_id == engine.model_id:
                    console.print(f"[dim]already using {target}[/]")
                    continue
                console.print(f"[dim]-- switching to {target} ({new_model_id}) --[/]")
                # Capture kv settings + build the new engine shell (no weights yet).
                kv_cfg = dict(
                    max_kv_size=engine.max_kv_size,
                    kv_bits=engine.kv_bits,
                    kv_group_size=engine.kv_group_size,
                    quantized_kv_start=engine.quantized_kv_start,
                )
                preset_ = load_preset(target)
                draft_ = preset_.get("draft")
                num_draft_ = int(preset_.get("num_draft_tokens", 4))
                new_engine = Engine(
                    model_id=new_model_id,
                    draft_model_id=resolve_alias(draft_, aliases_map) if draft_ else None,
                    num_draft_tokens=num_draft_,
                    **kv_cfg,
                )
                # Free the old engine BEFORE loading new weights so peak unified
                # memory doesn't double. If the new load fails we're modelless
                # and must break the chat loop — safer than silently continuing.
                old_model_id = engine.model_id
                del engine
                _mlx_reclaim()
                try:
                    new_engine.load()
                except Exception as e:
                    console.print(f"[red]failed to load {target}:[/] {e}")
                    console.print(f"[red]old engine ({old_model_id}) was freed; exit and restart.[/]")
                    break
                engine = new_engine
                _mlx_reclaim()
                # Re-evaluate tool-template support for the new tokenizer.
                if tools_on and tools_impl is not None:
                    supports = template_supports_tools(engine.tokenizer)
                    console.print(
                        f"[dim]-- {target} loaded · tools: "
                        f"{'native' if supports else 'generic (system-prompt)'} --[/]"
                    )
                else:
                    console.print(f"[dim]-- {target} loaded --[/]")
                continue
            if cmd == "/stats":
                s = engine.last_stats
                if s:
                    console.print(
                        f"[dim]prompt: {s.prompt_tokens} tok @ {s.prompt_tps:.1f} tok/s  · "
                        f"gen: {s.generated_tokens} tok @ {s.generation_tps:.1f} tok/s  · "
                        f"peak: {s.peak_memory_gb:.1f} GB[/]"
                    )
                else:
                    console.print("[dim]no stats yet[/]")
                continue
            if cmd == "/speak":
                speaker.set_enabled(not speaker.enabled)
                console.print(f"[dim]TTS {'on' if speaker.enabled else 'off'}[/]")
                continue
            if cmd == "/voice":
                if not arg:
                    console.print(f"[dim]voice: {speaker.voice or 'default'}[/]")
                else:
                    speaker.set_voice(arg.strip())
                    console.print(f"[dim]voice set to {speaker.voice}[/]")
                continue
            if cmd == "/rate":
                if not arg:
                    console.print(f"[dim]rate: {speaker.rate or 'default'}[/]")
                else:
                    try:
                        speaker.set_rate(int(arg.strip()))
                        console.print(f"[dim]rate set to {speaker.rate} wpm[/]")
                    except ValueError:
                        console.print("[red]rate must be an integer (words per minute)[/]")
                continue
            if cmd == "/stop":
                speaker.stop()
                console.print("[dim]-- tts stopped --[/]")
                continue
            if cmd == "/save":
                if not arg:
                    console.print("[red]usage: /save <file>[/]")
                    continue
                _save_transcript(messages, arg)
                console.print(f"[dim]saved to {arg}[/]")
                continue
            console.print(f"[red]unknown command: {cmd}[/]  [dim](/help)[/]")
            continue

        # Persist the user message up-front (plain-text view for the store).
        store.append_message(conversation_id, "user", line)

        if tools_on and tools_impl is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": line}]})
            try:
                final_text = _run_tool_loop(
                    engine, console, messages, tools_impl, tool_schemas,
                    max_tokens=max_tokens, temp=temp, top_p=top_p,
                )
            except KeyboardInterrupt:
                console.print("[dim]-- interrupted --[/]")
                final_text = ""
            store.append_message(conversation_id, "assistant", final_text)
            console.print()
            if speaker.enabled and final_text.strip():
                speaker.speak(_mask_tool_markers(final_text))
        else:
            messages.append({"role": "user", "content": line})
            reply = stream_response(
                engine.stream(messages, max_tokens=max_tokens, temp=temp, top_p=top_p),
                console,
            )
            messages.append({"role": "assistant", "content": reply})
            store.append_message(conversation_id, "assistant", reply)
            console.print()
            if speaker.enabled and reply.strip():
                speaker.speak(reply)
