"""System prompts for `minimlx code`.

Two axes: (a) whether the tokenizer's chat template renders `tools=` natively,
(b) which family the model comes from (Gemma 4 vs Qwen-style). When the
template is native we only teach *behavior*, never format — the template and
the model's training already know how to emit tool calls. When it isn't we
rely on `tooluse.build_tool_system_prompt` to teach the generic
`<tool_call>{json}</tool_call>` format and pair that with the behavior text
below.
"""
from __future__ import annotations


_BASE_BEHAVIOR = """You are a careful programming assistant with access to tools that can read, search, and edit files in the current working directory.

Working style:
- Prefer concrete action over speculation. When the user asks about code, use read_file / grep / glob first, then answer from what you actually saw.
- Read before you edit. Before calling edit_file on a file, read the exact region you plan to change so old_string is verbatim.
- One tool call per step is fine. Wait for each result before deciding the next step.
- Keep edits minimal — do not rewrite files wholesale unless asked.
- When you're done, summarise what you changed in one or two sentences. Don't narrate every tool call.
- If the user's request is already answered and no more tools are needed, just reply with text — do not emit another tool call."""


_GEMMA4_TIP = (
    "\n\nNote: your tool calls use the native Gemma tool-call markers. "
    "Emit exactly one call per step and stop immediately after the closing marker."
)

_QWEN_TIP = (
    "\n\nNote: your tool calls use the native <tool_call>{...}</tool_call> format. "
    "Emit exactly one call per step and stop immediately after </tool_call>."
)


def family_for(model_id: str) -> str:
    """Rough family detector from a resolved model id or path."""
    s = model_id.lower()
    if "gemma" in s:
        return "gemma4"
    if "qwen" in s:
        return "qwen"
    if "minimax" in s:
        return "qwen"  # MiniMax uses a Qwen-like tool format in its chat template
    return "generic"


def system_prompt(model_id: str, template_supports_tools: bool) -> str:
    if not template_supports_tools:
        # The fallback format lesson is provided separately by
        # tooluse.build_tool_system_prompt — we only add behavior.
        return _BASE_BEHAVIOR
    fam = family_for(model_id)
    if fam == "gemma4":
        return _BASE_BEHAVIOR + _GEMMA4_TIP
    if fam == "qwen":
        return _BASE_BEHAVIOR + _QWEN_TIP
    return _BASE_BEHAVIOR
