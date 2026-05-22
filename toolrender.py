"""Rich-based rendering of tool calls and their results."""
from __future__ import annotations

import json
import os
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from minimlx.codetools import ToolResult


_MAX_OUTPUT_BYTES = 20_000

_EXT_TO_LEXER = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx",
    ".json": "json", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".rst": "rst",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rs": "rust", ".go": "go", ".rb": "ruby", ".java": "java",
    ".kt": "kotlin", ".swift": "swift", ".lua": "lua", ".php": "php",
    ".html": "html", ".htm": "html", ".xml": "xml",
    ".css": "css", ".scss": "scss", ".sql": "sql",
    ".ini": "ini", ".cfg": "ini",
}


def _lexer_for(path: str) -> str:
    try:
        _, ext = os.path.splitext(path or "")
        return _EXT_TO_LEXER.get(ext.lower(), "text")
    except Exception:
        return "text"


def _truncate_output(text: str) -> tuple[str, bool]:
    if len(text) > _MAX_OUTPUT_BYTES:
        return text[:_MAX_OUTPUT_BYTES], True
    return text, False


def _short_hint(name: str, inp: dict) -> str:
    if not isinstance(inp, dict):
        return ""
    if name in ("read_file", "write_file", "edit_file", "ls"):
        v = inp.get("path")
        return str(v) if v else ""
    if name == "bash":
        v = str(inp.get("command", "")).splitlines()
        first = v[0] if v else ""
        return first[:60] + ("…" if len(first) > 60 else "")
    if name == "grep":
        pat = inp.get("pattern", "")
        path = inp.get("path")
        return f"'{pat}' in {path}" if path else f"'{pat}'"
    if name == "glob":
        v = inp.get("pattern")
        return f"'{v}'" if v else ""
    return ""


def render_call(console: Console, name: str, inp: dict) -> None:
    """Render the 'about to run' banner for a tool call."""
    try:
        hint = _short_hint(name, inp or {})
        t = Text("· ", style="dim")
        t.append(f"running {name}", style="dim")
        if hint:
            t.append(f" {hint}", style="dim italic")
        t.append("…", style="dim")
        console.print(t)
    except Exception:
        try:
            console.print(f"· running {name}…", style="dim")
        except Exception:
            pass


def render_result(console: Console, name: str, inp: dict, result: ToolResult) -> None:
    """Render the tool's output with tool-specific styling."""
    try:
        fn = _RESULT_RENDERERS.get(name, _render_fallback)
        fn(console, inp or {}, result)
    except Exception:
        try:
            _render_fallback(console, inp or {}, result)
        except Exception:
            pass


def _error_panel(console: Console, title: str, text: str) -> None:
    out, _ = _truncate_output(text or "")
    console.print(Panel(Text(out, style="red"), title=title, border_style="red", expand=False))


def _render_read_file(console: Console, inp: dict, result: ToolResult) -> None:
    path = str(inp.get("path", ""))
    offset = inp.get("offset")
    text, truncated = _truncate_output(result.output)
    if result.is_error:
        _error_panel(console, f"📖 read {path}", text)
        return
    lines = text.splitlines()
    n_total = len(lines)
    preview = lines[:30]
    extra = n_total - len(preview)
    start_line = int(offset) if isinstance(offset, int) and offset else 1
    end_line = start_line + n_total - 1 if n_total else start_line
    range_str = f"{start_line}-{end_line}" if n_total else "0"
    try:
        syntax: Any = Syntax(
            "\n".join(preview), _lexer_for(path),
            line_numbers=True, start_line=start_line,
            word_wrap=False, theme="ansi_dark",
        )
    except Exception:
        syntax = Text("\n".join(preview))
    parts: list[Any] = [syntax]
    if extra > 0:
        parts.append(Text(f"[+{extra} more lines]", style="dim"))
    if truncated:
        parts.append(Text("[truncated at 20 KB]", style="dim"))
    console.print(Panel(Group(*parts), title=f"📖 read {path}:{range_str}", border_style="green", expand=False))


def _render_write_file(console: Console, inp: dict, result: ToolResult) -> None:
    path = str(inp.get("path", ""))
    content = inp.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    if result.is_error:
        _error_panel(console, f"✏ write {path}", result.output)
        return
    all_lines = content.splitlines()
    n_lines = len(all_lines)
    n_bytes = len(content.encode("utf-8", errors="replace"))
    preview = all_lines[:15]
    try:
        syntax: Any = Syntax(
            "\n".join(preview), _lexer_for(path),
            line_numbers=True, word_wrap=False, theme="ansi_dark",
        )
    except Exception:
        syntax = Text("\n".join(preview))
    parts: list[Any] = [Text(f"✏ wrote {path} ({n_lines} lines, {n_bytes} bytes)", style="green"), syntax]
    if n_lines > 15:
        parts.append(Text(f"[+{n_lines - 15} more lines]", style="dim"))
    console.print(Panel(Group(*parts), border_style="green", expand=False))


def _render_edit_file(console: Console, inp: dict, result: ToolResult) -> None:
    path = str(inp.get("path", ""))
    old = inp.get("old_string", "") or ""
    new = inp.get("new_string", "") or ""
    if not isinstance(old, str):
        old = str(old)
    if not isinstance(new, str):
        new = str(new)
    if result.is_error:
        _error_panel(console, f"✦ edit {path}", result.output)
        return
    try:
        old_all = old.splitlines()
        new_all = new.splitlines()
        old_lines = old_all[:20]
        new_lines = new_all[:20]
        diff = Text()
        for i, ln in enumerate(old_lines):
            if i:
                diff.append("\n")
            diff.append(f"- {ln}", style="red")
        if old_lines and new_lines:
            diff.append("\n")
        for i, ln in enumerate(new_lines):
            if i:
                diff.append("\n")
            diff.append(f"+ {ln}", style="green")
        parts: list[Any] = [diff]
        old_extra = max(0, len(old_all) - len(old_lines))
        new_extra = max(0, len(new_all) - len(new_lines))
        if old_extra or new_extra:
            parts.append(Text(f"[trimmed: -{old_extra} / +{new_extra} more lines]", style="dim"))
        console.print(Panel(Group(*parts), title=f"✦ edit {path}", border_style="green", expand=False))
    except Exception:
        _render_fallback(console, inp, result)


def _render_bash(console: Console, inp: dict, result: ToolResult) -> None:
    cmd = str(inp.get("command", ""))
    out, truncated = _truncate_output(result.output)
    lines = out.splitlines()
    preview = lines[:25]
    extra = len(lines) - len(preview)
    parts: list[Any] = [
        Text(cmd or "[no command]", style="cyan"),
        Text("\n".join(preview) if preview else "[no output]", style="dim"),
    ]
    if extra > 0:
        parts.append(Text(f"[+{extra} more lines]", style="dim"))
    if truncated:
        parts.append(Text("[truncated at 20 KB]", style="dim"))
    title = "$ bash"
    border = "green"
    if result.is_error:
        border = "red"
        first_line = lines[0] if lines else ""
        if first_line.startswith("exit "):
            title = f"$ bash [{first_line}]"
    console.print(Panel(Group(*parts), title=title, border_style=border, expand=False))


def _render_grep(console: Console, inp: dict, result: ToolResult) -> None:
    pattern = str(inp.get("pattern", ""))
    path = str(inp.get("path", "") or ".")
    out, truncated = _truncate_output(result.output)
    title = f"🔎 grep '{pattern}' in {path}"
    if result.is_error:
        _error_panel(console, title, out)
        return
    if out.strip() in ("", "[no matches]"):
        console.print(Panel(Text("[no matches]", style="dim"), title=title, border_style="cyan", expand=False))
        return
    lines = out.splitlines()
    preview = lines[:20]
    extra = len(lines) - len(preview)
    body = Text()
    for i, line in enumerate(preview):
        if i:
            body.append("\n")
        parts = line.split(":", 2)
        if len(parts) == 3:
            file_s, lineno, rest = parts
            body.append(file_s, style="cyan")
            body.append(":", style="dim")
            body.append(lineno, style="dim")
            body.append(":", style="dim")
            try:
                idx = rest.lower().find(pattern.lower()) if pattern else -1
                if idx >= 0:
                    body.append(rest[:idx])
                    body.append(rest[idx:idx + len(pattern)], style="yellow bold")
                    body.append(rest[idx + len(pattern):])
                else:
                    body.append(rest)
            except Exception:
                body.append(rest)
        else:
            body.append(line)
    children: list[Any] = [body]
    if extra > 0:
        children.append(Text(f"[+{extra} more]", style="dim"))
    if truncated:
        children.append(Text("[truncated at 20 KB]", style="dim"))
    console.print(Panel(Group(*children), title=title, border_style="green", expand=False))


def _render_list_like(
    console: Console, title: str, out: str, truncated: bool,
    is_error: bool, cap: int, empty_marker: str, style_fn,
) -> None:
    if is_error:
        _error_panel(console, title, out)
        return
    if out.strip() in ("", empty_marker):
        console.print(Panel(Text(empty_marker, style="dim"), title=title, border_style="cyan", expand=False))
        return
    lines = out.splitlines()
    preview = lines[:cap]
    extra = len(lines) - len(preview)
    body = Text()
    for i, line in enumerate(preview):
        if i:
            body.append("\n")
        body.append(line, style=style_fn(line))
    children: list[Any] = [body]
    if extra > 0:
        children.append(Text(f"[+{extra} more]", style="dim"))
    if truncated:
        children.append(Text("[truncated at 20 KB]", style="dim"))
    console.print(Panel(Group(*children), title=title, border_style="green", expand=False))


def _render_glob(console: Console, inp: dict, result: ToolResult) -> None:
    pattern = str(inp.get("pattern", ""))
    out, truncated = _truncate_output(result.output)
    _render_list_like(
        console, f"⋯ glob '{pattern}'", out, truncated,
        result.is_error, 25, "[no matches]", lambda _l: "cyan",
    )


def _render_ls(console: Console, inp: dict, result: ToolResult) -> None:
    path = str(inp.get("path", "") or ".")
    out, truncated = _truncate_output(result.output)
    _render_list_like(
        console, f"📂 ls {path}", out, truncated,
        result.is_error, 40, "[empty]",
        lambda l: "cyan" if l.endswith("/") else "dim",
    )


def _render_fallback(console: Console, inp: dict, result: ToolResult) -> None:
    try:
        args_json = json.dumps(inp, indent=2, default=str)
    except Exception:
        args_json = str(inp)
    out, truncated = _truncate_output(result.output or "")
    style = "red" if result.is_error else None
    body = Text()
    body.append("args:\n", style="dim")
    body.append(args_json)
    body.append("\n\nresult:\n", style="dim")
    body.append(out, style=style)
    if truncated:
        body.append("\n[truncated at 20 KB]", style="dim")
    border = "red" if result.is_error else "cyan"
    console.print(Panel(body, title="tool", border_style=border, expand=False))


_RESULT_RENDERERS = {
    "read_file": _render_read_file,
    "write_file": _render_write_file,
    "edit_file": _render_edit_file,
    "bash": _render_bash,
    "grep": _render_grep,
    "glob": _render_glob,
    "ls": _render_ls,
}
