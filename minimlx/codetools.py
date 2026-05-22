"""Cursor-like local tool set for `minimlx code`.

Seven tools, all scoped to a working directory passed in at construction time:

    read_file(path, offset?, limit?) — file contents, optional line slice
    write_file(path, content)       — create or overwrite
    edit_file(path, old, new)       — single exact-string replacement
    ls(path=".")                    — directory listing
    glob(pattern)                   — recursive glob
    grep(pattern, path=".")         — regex search (line-oriented)
    bash(command, timeout=30)       — shell, only enabled when allow_bash=True

Each `.run(name, input)` call returns a plain string that we feed back into
the model as a `tool_result`. Errors are reported as readable strings, not
exceptions — the model can read them and decide what to do next.

Path handling: every incoming path is resolved relative to `cwd` and then
checked to be within `cwd`. Escaping via `..` or absolute paths outside the
sandbox returns a refusal string instead of reading the host filesystem.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_READ_BYTES = 256_000
MAX_GREP_MATCHES = 200
MAX_LS_ENTRIES = 500
MAX_GLOB_MATCHES = 500


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read the contents of a text file. Optionally return only a line range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "offset": {"type": "integer", "description": "1-indexed first line to return. Optional."},
                "limit": {"type": "integer", "description": "Max lines to return. Optional."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create a new file or overwrite an existing file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace one exact occurrence of old_string with new_string in a file. Fails if old_string is missing or appears more than once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "ls",
        "description": "List entries in a directory (non-recursive). Dirs have a trailing /.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Defaults to '.'."},
            },
        },
    },
    {
        "name": "glob",
        "description": "Recursively find files matching a glob pattern (e.g. '**/*.py').",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex pattern in files. Returns matching lines with file:line prefix.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "File or directory to search. Defaults to '.'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command in the working directory and return combined stdout+stderr. Disabled by default for safety.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds. Default 30, max 120."},
            },
            "required": ["command"],
        },
    },
]


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class CodeTools:
    def __init__(self, cwd: Path, allow_bash: bool = False, unrestricted: bool = False):
        self.cwd = cwd.resolve()
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.allow_bash = allow_bash
        self.unrestricted = unrestricted

    def schemas(self) -> list[dict]:
        if self.allow_bash:
            return TOOL_SCHEMAS
        return [t for t in TOOL_SCHEMAS if t["name"] != "bash"]

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            fn = getattr(self, f"_tool_{name}", None)
            if fn is None:
                return ToolResult(f"unknown tool: {name}", is_error=True)
            return fn(args or {})
        except Exception as e:
            return ToolResult(f"tool {name} raised {type(e).__name__}: {e}", is_error=True)

    # ---- path helpers ----

    def _safe_path(self, p: str) -> Path | None:
        if not isinstance(p, str) or not p:
            return None
        candidate = (self.cwd / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
        if self.unrestricted:
            return candidate
        try:
            candidate.relative_to(self.cwd)
        except ValueError:
            return None
        return candidate

    def _scope_error(self, requested: str) -> ToolResult:
        """Build a directive error that names the attempted path and tells
        the model how to recover, instead of just `refused: path outside…`."""
        try:
            attempted = (
                str(Path(requested).resolve()) if os.path.isabs(requested)
                else str((self.cwd / requested).resolve())
            )
        except Exception:
            attempted = requested
        return ToolResult(
            f"refused: {attempted!r} is outside the working directory "
            f"{str(self.cwd)!r}. Use a path under the working directory "
            f"(e.g. a relative path like 'index.html'), or ask the user to "
            f"re-run with `-C <dir>` to widen the scope or `--unrestricted` "
            f"to disable the scope check.",
            is_error=True,
        )

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.cwd)) or "."
        except ValueError:
            return str(p)

    # ---- tools ----

    def _tool_read_file(self, args: dict) -> ToolResult:
        requested = args.get("path", "")
        path = self._safe_path(requested)
        if path is None:
            return self._scope_error(requested)
        if not path.exists():
            return ToolResult(f"not found: {self._rel(path)}", is_error=True)
        if path.is_dir():
            return ToolResult(f"is a directory: {self._rel(path)}", is_error=True)
        raw = path.read_bytes()
        if len(raw) > MAX_READ_BYTES:
            raw = raw[:MAX_READ_BYTES]
            truncated_note = f"\n\n[truncated at {MAX_READ_BYTES} bytes]"
        else:
            truncated_note = ""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(f"binary file ({len(raw)} bytes, not UTF-8): {self._rel(path)}", is_error=True)
        offset = args.get("offset")
        limit = args.get("limit")
        if offset is not None or limit is not None:
            lines = text.splitlines(keepends=True)
            start = max(1, int(offset or 1)) - 1
            end = len(lines) if limit is None else start + max(1, int(limit))
            text = "".join(lines[start:end])
        return ToolResult(text + truncated_note)

    def _tool_write_file(self, args: dict) -> ToolResult:
        requested = args.get("path", "")
        path = self._safe_path(requested)
        if path is None:
            return self._scope_error(requested)
        content = args.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return ToolResult(f"wrote {len(content)} chars to {self._rel(path)}")

    def _tool_edit_file(self, args: dict) -> ToolResult:
        requested = args.get("path", "")
        path = self._safe_path(requested)
        if path is None:
            return self._scope_error(requested)
        if not path.exists():
            return ToolResult(f"not found: {self._rel(path)}", is_error=True)
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        if not isinstance(old, str) or not old:
            return ToolResult("old_string must be a non-empty string", is_error=True)
        if not isinstance(new, str):
            new = str(new)
        text = path.read_text()
        count = text.count(old)
        if count == 0:
            return ToolResult("old_string not found", is_error=True)
        if count > 1:
            return ToolResult(f"old_string matches {count} times — provide more context to make it unique", is_error=True)
        path.write_text(text.replace(old, new, 1))
        return ToolResult(f"edited {self._rel(path)} (1 replacement)")

    def _tool_ls(self, args: dict) -> ToolResult:
        requested = args.get("path") or "."
        p = self._safe_path(requested)
        if p is None:
            return self._scope_error(requested)
        if not p.exists():
            return ToolResult(f"not found: {self._rel(p)}", is_error=True)
        if not p.is_dir():
            return ToolResult(f"not a directory: {self._rel(p)}", is_error=True)
        entries = sorted(p.iterdir())[:MAX_LS_ENTRIES]
        lines = [e.name + ("/" if e.is_dir() else "") for e in entries]
        return ToolResult("\n".join(lines) if lines else "[empty]")

    def _tool_glob(self, args: dict) -> ToolResult:
        pattern = args.get("pattern", "")
        if not pattern:
            return ToolResult("pattern required", is_error=True)
        matches: list[str] = []
        for root, dirs, files in os.walk(self.cwd):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), self.cwd)
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f, pattern):
                    matches.append(rel)
                    if len(matches) >= MAX_GLOB_MATCHES:
                        break
            if len(matches) >= MAX_GLOB_MATCHES:
                break
        matches.sort()
        return ToolResult("\n".join(matches) if matches else "[no matches]")

    def _tool_grep(self, args: dict) -> ToolResult:
        pattern = args.get("pattern", "")
        if not pattern:
            return ToolResult("pattern required", is_error=True)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(f"bad regex: {e}", is_error=True)
        requested = args.get("path") or "."
        base = self._safe_path(requested)
        if base is None:
            return self._scope_error(requested)
        matches: list[str] = []
        targets: list[Path] = [base] if base.is_file() else []
        if base.is_dir():
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    targets.append(Path(root) / f)
        for t in targets:
            try:
                with t.open("r", errors="replace") as fh:
                    for i, line in enumerate(fh, start=1):
                        if rx.search(line):
                            matches.append(f"{self._rel(t)}:{i}:{line.rstrip()}")
                            if len(matches) >= MAX_GREP_MATCHES:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if len(matches) >= MAX_GREP_MATCHES:
                break
        return ToolResult("\n".join(matches) if matches else "[no matches]")

    def _tool_bash(self, args: dict) -> ToolResult:
        if not self.allow_bash:
            return ToolResult("bash is disabled. Rerun with --allow-bash to enable.", is_error=True)
        cmd = args.get("command", "")
        if not cmd:
            return ToolResult("command required", is_error=True)
        timeout = min(120, max(1, int(args.get("timeout") or 30)))
        try:
            proc = subprocess.run(
                ["/bin/sh", "-c", cmd],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"timed out after {timeout}s", is_error=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_READ_BYTES:
            out = out[:MAX_READ_BYTES] + f"\n\n[truncated at {MAX_READ_BYTES} bytes]"
        if proc.returncode != 0:
            return ToolResult(f"exit {proc.returncode}\n{out}", is_error=True)
        return ToolResult(out or "[no output]")
