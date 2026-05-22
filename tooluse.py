"""Tool-use plumbing for the Anthropic-compatible server.

Four jobs:

1. **Tool schema normalisation** — map Anthropic tool defs into the
   OpenAI-style shape that `tokenizer.apply_chat_template(tools=…)` expects.

2. **History flattening** — convert inbound Anthropic `tool_use` /
   `tool_result` content blocks into `tool_calls` / `tool_responses` fields
   on message dicts so the model's own chat template renders them in its
   native format (Gemma 4's `<|tool_call>call:name{…}<tool_call|>`, Qwen's
   `<tool_call>{…}</tool_call>`, etc.).

3. **Output parsing** — scan model output for complete tool calls in either
   format:
     - generic `<tool_call>{json}</tool_call>` (Qwen, Llama-style)
     - Gemma 4  `<|tool_call>call:name{args}<tool_call|>`  (with `<|"|>`
       string delimiters and unquoted keys — parsed via mlx-lm's helper)
   and produce Anthropic-shaped `tool_use` content blocks with fresh IDs.

4. **Streaming state machine** — emit Anthropic SSE events
   (`content_block_start`/`content_block_delta`/`content_block_stop` with
   `text_delta` or `input_json_delta`) as tokens arrive, and signal
   `should_stop()` once a complete tool_use is emitted so the server can
   break out of generation before the model hallucinates a fake tool result.

**Fallback system prompt** `build_tool_system_prompt` is kept as an optional
escape hatch for models whose chat template has no tool support at all — it
is *not* injected when the template handles tools natively, to avoid
teaching a format that conflicts with the model's training.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

# Generic / Qwen / Llama style
GENERIC_OPEN = "<tool_call>"
GENERIC_CLOSE = "</tool_call>"
# Gemma 4 native
GEMMA4_OPEN = "<|tool_call>"
GEMMA4_CLOSE = "<tool_call|>"

# When the tool-call scanner sees ANY of these format pairs in the output,
# it treats the contents as a complete tool call and attempts to parse them.
_TOOL_FORMATS: tuple[tuple[str, str, str], ...] = (
    (GEMMA4_OPEN, GEMMA4_CLOSE, "gemma4"),
    (GENERIC_OPEN, GENERIC_CLOSE, "generic"),
)

_GENERIC_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# XML-style function call used by Qwen Claude-Opus-distilled variants:
#   <function=name> <parameter=p1>v1</parameter> … </function>
_XML_FUNC_RE = re.compile(
    r"<function=(?P<name>[^>\s]+)\s*>(?P<body>.*?)</function>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=(?P<pname>[^>\s]+)\s*>(?P<pval>.*?)</parameter>",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def anthropic_tools_to_transformers(tools: list[dict] | None) -> list[dict] | None:
    """Normalise Anthropic tool defs for `tokenizer.apply_chat_template(tools=…)`.

    Anthropic uses `name` / `description` / `input_schema`. The standard
    shape that chat templates iterate over is OpenAI-style
    `{type: function, function: {name, description, parameters}}`.
    """
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        if not name:
            continue
        schema = t.get("input_schema") or t.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", "") or "",
                "parameters": schema,
            },
        })
    return out or None


def template_supports_tools(tokenizer: Any) -> bool:
    """Heuristic: does the tokenizer's chat_template actually render `tools`?

    We look for `tools` / `tool_call` / `tool_calls` substrings in the
    template source. When present, `apply_chat_template(tools=…)` will do the
    right thing and we should NOT layer a system-prompt instruction on top
    that teaches a different format.
    """
    tmpl = getattr(tokenizer, "chat_template", None)
    if not isinstance(tmpl, str):
        return False
    return ("tool_call" in tmpl) or ("tool_calls" in tmpl) or ("tool_response" in tmpl)


def build_tool_system_prompt(tools: list[dict] | None) -> str:
    """Fallback tool instructions for models whose chat_template ignores `tools`.

    Teach a simple generic format (`<tool_call>{"name":…,"input":…}</tool_call>`)
    that our parser reads. Only inject this when `template_supports_tools` is
    False — otherwise it conflicts with the model's native training.
    """
    if not tools:
        return ""
    lines: list[str] = [
        "You have access to the tools listed below. When you need to use a tool, emit exactly one",
        "<tool_call> block per call. Nothing else on those lines. The block must contain a single",
        "JSON object with `name` (the tool name) and `input` (an object of arguments).",
        "",
        "Example:",
        "<tool_call>",
        '{"name": "read_file", "input": {"path": "/tmp/foo.txt"}}',
        "</tool_call>",
        "",
        "After you emit a <tool_call> block you must STOP. Do not write anything after it.",
        "The tool's result will be provided in the next message. Never invent results.",
        "",
        "Available tools:",
    ]
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "?")
        desc = (t.get("description") or "").strip().split("\n")[0]
        lines.append(f"- {name}: {desc}")
        schema = t.get("input_schema") or t.get("parameters") or {}
        props = (schema or {}).get("properties") or {}
        required = set((schema or {}).get("required") or [])
        for pname, pinfo in props.items():
            ptype = (pinfo or {}).get("type", "any")
            pdesc = (pinfo or {}).get("description", "")
            mark = "*" if pname in required else ""
            lines.append(f"    - {mark}{pname} ({ptype}): {pdesc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# History flattening (structure-preserving)
# ---------------------------------------------------------------------------


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)
    return str(content)


def flatten_history(messages: list[dict]) -> list[dict]:
    """Convert Anthropic content-block messages into dicts the model's chat
    template can render natively.

    Produces messages with optional fields the major chat templates look for:
      - `content`  — text parts joined (always a string)
      - `tool_calls` — OpenAI-style on assistant messages carrying tool_use
      - `tool_responses` — Gemma-style on user messages carrying tool_result
      - a synthetic `role="tool"` message with `tool_call_id` + `content` is
        emitted for OpenAI-style templates that look for it

    Different templates consume different subsets of these fields. The
    unused ones are silently ignored, so one call produces input that works
    across Gemma 4, Qwen, Llama 3, etc.
    """
    id_to_name: dict[str, str] = {}
    out: list[dict] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tid = block.get("id", "") or _new_tool_id()
                name = block.get("name", "")
                inp = block.get("input", {}) or {}
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except json.JSONDecodeError:
                        inp = {"_raw": inp}
                if not isinstance(inp, dict):
                    inp = {}
                id_to_name[tid] = name
                tool_calls.append({
                    "id": tid,
                    "type": "function",
                    "function": {"name": name, "arguments": inp},
                })
            elif btype == "tool_result":
                tid = block.get("tool_use_id", "")
                tc = block.get("content", "")
                if isinstance(tc, list):
                    tc = "\n".join(
                        b.get("text", "") for b in tc
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                tool_results.append({
                    "tool_call_id": tid,
                    "name": id_to_name.get(tid, "unknown"),
                    "response": tc,
                    "is_error": bool(block.get("is_error")),
                })
            elif btype == "image":
                text_parts.append("[image omitted — server is text-only]")

        msg: dict = {"role": role, "content": "\n".join(p for p in text_parts if p)}

        if tool_calls and role == "assistant":
            msg["tool_calls"] = tool_calls

        if tool_results:
            # Gemma-style: tool_responses field on the same (user) message.
            msg["tool_responses"] = [
                {"name": r["name"], "response": r["response"]}
                for r in tool_results
            ]
            out.append(msg)
            # OpenAI-style: emit a separate role="tool" message per result,
            # which Qwen / Llama templates look for. Templates that don't
            # recognise role=tool just render an empty turn.
            for r in tool_results:
                out.append({
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "name": r["name"],
                    "content": r["response"],
                })
            continue

        out.append(msg)

    return out


# ---------------------------------------------------------------------------
# Tool-call body parsers
# ---------------------------------------------------------------------------


def _new_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _parse_generic_body(body: str) -> dict | None:
    """Parse a `<tool_call>…</tool_call>` body into `{name, input}`.

    Supports two body dialects:
      1. Hermes/Qwen JSON: `{"name": "...", "arguments": {...}}`
      2. XML function form used by Claude-distilled Qwen variants:
         `<function=NAME><parameter=P1>v1</parameter>…</function>`
    """
    body = body.strip()
    # Try JSON first (Hermes/Qwen native format).
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        stripped = body.strip("`\n ")
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict) and obj.get("name"):
        return {"name": str(obj["name"]), "input": _extract_input(obj)}
    # Fall back to XML function form.
    return _parse_xml_function_body(body)


def _parse_xml_function_body(body: str) -> dict | None:
    """Parse XML-style function call body (Claude-distilled Qwen format)."""
    fn = _XML_FUNC_RE.search(body)
    if fn is None:
        return None
    name = fn.group("name").strip()
    if not name:
        return None
    inner = fn.group("body")
    params: dict[str, Any] = {}
    for m in _XML_PARAM_RE.finditer(inner):
        pname = m.group("pname").strip()
        if not pname:
            continue
        raw = m.group("pval").strip("\n").rstrip()
        # Best-effort coercion: try JSON (numbers, bools, null, arrays, objects).
        try:
            params[pname] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            params[pname] = raw
    return {"name": name, "input": params}


def _parse_gemma4_body(body: str) -> dict | None:
    """Parse a `<|tool_call>call:name{args}<tool_call|>` body.

    Delegates to mlx-lm's gemma4 tool parser, which handles Gemma's unusual
    argument format (unquoted keys, `<|"|>` string delimiters, balanced
    recursive braces).
    """
    try:
        from mlx_lm.tool_parsers.gemma4 import parse_tool_call
    except Exception:
        return None
    try:
        parsed = parse_tool_call(body.strip())
    except Exception:
        return None
    if isinstance(parsed, list):
        if not parsed:
            return None
        parsed = parsed[0]
    if not isinstance(parsed, dict) or not parsed.get("name"):
        return None
    args = parsed.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return {"name": str(parsed["name"]), "input": args}


_BODY_PARSERS: dict[str, Callable[[str], dict | None]] = {
    "generic": _parse_generic_body,
    "gemma4": _parse_gemma4_body,
}


def _extract_input(obj: dict) -> dict:
    """Accept both Anthropic `input` and OpenAI/Qwen `arguments` keys."""
    inp = obj.get("input")
    if inp is None:
        inp = obj.get("arguments", {})
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except json.JSONDecodeError:
            inp = {"_raw": inp}
    return inp if isinstance(inp, dict) else {}


def _find_earliest_tool_call(text: str, start_at: int) -> tuple[int, int, str] | None:
    """Locate the earliest complete tool-call block at or after `start_at`.

    Returns (start_index, end_index_exclusive, format_name) or None.
    `end_index_exclusive` points past the closing marker.
    """
    best: tuple[int, int, str] | None = None
    for open_m, close_m, name in _TOOL_FORMATS:
        s = text.find(open_m, start_at)
        if s == -1:
            continue
        e = text.find(close_m, s + len(open_m))
        if e == -1:
            continue
        end_excl = e + len(close_m)
        if best is None or s < best[0]:
            best = (s, end_excl, name)
    return best


def _find_earliest_open(text: str, start_at: int) -> tuple[int, str, str] | None:
    """Earliest open marker in `text[start_at:]`, ignoring whether it closes."""
    best: tuple[int, str, str] | None = None
    for open_m, close_m, name in _TOOL_FORMATS:
        s = text.find(open_m, start_at)
        if s == -1:
            continue
        if best is None or s < best[0]:
            best = (s, open_m, close_m)
    return best


def _body_for_format(text: str, start: int, end_excl: int, fmt: str) -> str:
    open_m, close_m, _ = next(f for f in _TOOL_FORMATS if f[2] == fmt)
    return text[start + len(open_m):end_excl - len(close_m)]


# ---------------------------------------------------------------------------
# Non-streaming output parser
# ---------------------------------------------------------------------------


def parse_content_blocks(text: str) -> tuple[list[dict], str]:
    """Split model output into Anthropic content blocks.

    Returns `(blocks, stop_reason)` where `blocks` is a list of
    `{type: "text", text: str}` or `{type: "tool_use", id, name, input}`
    dicts. `stop_reason` is `"tool_use"` if any tool_use was produced, else
    `"end_turn"`.
    """
    blocks: list[dict] = []
    cursor = 0
    had_tool = False

    while cursor < len(text):
        found = _find_earliest_tool_call(text, cursor)
        if found is None:
            tail = text[cursor:]
            if tail.strip():
                blocks.append({"type": "text", "text": tail})
            break

        start, end_excl, fmt = found

        if start > cursor:
            pre = text[cursor:start]
            if pre.strip():
                blocks.append({"type": "text", "text": pre})

        body = _body_for_format(text, start, end_excl, fmt)
        parsed = _BODY_PARSERS[fmt](body)
        if parsed is not None:
            blocks.append({
                "type": "tool_use",
                "id": _new_tool_id(),
                "name": parsed["name"],
                "input": parsed["input"],
            })
            had_tool = True
        else:
            blocks.append({"type": "text", "text": text[start:end_excl]})

        cursor = end_excl

    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    return blocks, ("tool_use" if had_tool else "end_turn")


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------


EventSink = Callable[[str, dict], None]


class StreamingToolParser:
    """Scans a streaming text buffer and emits Anthropic content_block events
    for mixed text / tool_use output.

    Supports both `<tool_call>…</tool_call>` (generic) and
    `<|tool_call>call:NAME{…}<tool_call|>` (Gemma 4 native) formats.

    Usage:
        parser = StreamingToolParser(send_event, clean_text_fn)
        for chunk in engine.stream(...):
            parser.push(chunk.text)
            if parser.should_stop():
                break
        parser.flush()
        stop_reason = parser.stop_reason()

    `clean_text_fn(raw_buf) -> (thinking_str, answer_str, still_thinking_bool)`
    is the `minimlx.render._split_channels`-style filter used to hide
    `<think>` / `<|channel>thought` reasoning while streaming.
    """

    def __init__(
        self,
        send: EventSink,
        clean_text_fn: Callable[[str], tuple[str, str, bool]] | None = None,
    ):
        self._send = send
        self._clean = clean_text_fn
        self._raw_buf = ""
        self._decided_upto = 0
        self._block_index = -1
        self._current_type: str | None = None
        self._had_tool = False
        self._output_chars = 0

    # ---- block lifecycle ----

    def _open_text(self) -> None:
        self._block_index += 1
        self._current_type = "text"
        self._send("content_block_start", {
            "type": "content_block_start",
            "index": self._block_index,
            "content_block": {"type": "text", "text": ""},
        })

    def _close_current(self) -> None:
        if self._current_type is not None:
            self._send("content_block_stop", {
                "type": "content_block_stop",
                "index": self._block_index,
            })
            self._current_type = None

    def _emit_text_delta(self, delta: str) -> None:
        if not delta:
            return
        if self._current_type != "text":
            self._close_current()
            self._open_text()
        self._send("content_block_delta", {
            "type": "content_block_delta",
            "index": self._block_index,
            "delta": {"type": "text_delta", "text": delta},
        })
        self._output_chars += len(delta)

    def _emit_tool_use(self, name: str, inp: dict) -> None:
        self._close_current()
        self._block_index += 1
        self._current_type = "tool_use"
        tid = _new_tool_id()
        self._send("content_block_start", {
            "type": "content_block_start",
            "index": self._block_index,
            "content_block": {
                "type": "tool_use",
                "id": tid,
                "name": name,
                "input": {},
            },
        })
        json_str = json.dumps(inp, ensure_ascii=False)
        self._send("content_block_delta", {
            "type": "content_block_delta",
            "index": self._block_index,
            "delta": {"type": "input_json_delta", "partial_json": json_str},
        })
        self._close_current()
        self._had_tool = True
        self._output_chars += len(json_str)

    # ---- scanning ----

    def push(self, chunk: str) -> None:
        self._raw_buf += chunk
        self._process(final=False)

    def flush(self) -> None:
        self._process(final=True)
        self._close_current()

    def _process(self, final: bool) -> None:
        if self._clean is not None:
            _, clean, still_thinking = self._clean(self._raw_buf)
            if still_thinking and not final:
                return
        else:
            clean = self._raw_buf

        while self._decided_upto < len(clean):
            # Look for a COMPLETE tool-call block starting at or after the cursor.
            found = _find_earliest_tool_call(clean, self._decided_upto)

            if found is None:
                # No complete tool-call ahead. We may still have a partial
                # open marker tail — hold it back so we don't stream a
                # half-emitted marker as text.
                next_open = _find_earliest_open(clean, self._decided_upto)

                stable_end = len(clean)
                if not final:
                    # Guard against partial open markers at the tail.
                    for open_m, _close, _name in _TOOL_FORMATS:
                        for n in range(1, len(open_m)):
                            if clean.endswith(open_m[:n]):
                                candidate = len(clean) - n
                                if candidate < stable_end:
                                    stable_end = candidate
                                break
                    # Also hold text after a known-open-but-unclosed marker.
                    if next_open is not None and next_open[0] < stable_end:
                        stable_end = next_open[0]

                if stable_end > self._decided_upto:
                    self._emit_text_delta(clean[self._decided_upto:stable_end])
                    self._decided_upto = stable_end
                break

            start, end_excl, fmt = found

            if start > self._decided_upto:
                self._emit_text_delta(clean[self._decided_upto:start])
                self._decided_upto = start

            body = _body_for_format(clean, start, end_excl, fmt)
            parsed = _BODY_PARSERS[fmt](body)
            if parsed is not None:
                self._emit_tool_use(parsed["name"], parsed["input"])
            else:
                self._emit_text_delta(clean[start:end_excl])

            self._decided_upto = end_excl

    # ---- outcome ----

    def should_stop(self) -> bool:
        """Signal to the server that generation should end immediately.

        Fires after the first complete tool_use block is emitted. This stops
        the model from hallucinating fake tool results when its native stop
        conditions (EOS / turn markers) don't fire reliably.
        """
        return self._had_tool

    def stop_reason(self) -> str:
        return "tool_use" if self._had_tool else "end_turn"

    @property
    def output_chars(self) -> int:
        return self._output_chars
