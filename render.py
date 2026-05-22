from __future__ import annotations
import time
from typing import Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from minimlx.engine import Chunk


# (open_marker, close_marker) pairs for each model's thinking channel.
_THINK_PAIRS: tuple[tuple[str, str], ...] = (
    ("<|channel>thought", "<channel|>"),  # Gemma 4
    ("<think>",           "</think>"),     # Qwen / Qwopus / DeepSeek-R1 / etc.
)
_END_MARKERS: tuple[str, ...] = ("<turn|>", "<|im_end|>", "<|eot_id|>")


def _split_channels(buf: str) -> tuple[str, str, bool]:
    """Split a raw stream buffer into (thinking, answer, still_thinking).

    Handles multiple model-specific reasoning markers and strips trailing EOS
    sequences so partial markers don't flicker on screen.
    """
    # Strip trailing EOS markers (complete or partial) from the buffer first.
    for end in _END_MARKERS:
        if end in buf:
            buf = buf.split(end, 1)[0]
    for end in _END_MARKERS:
        buf = _strip_partial_end(buf, end)

    # Locate the earliest opening marker and its pair close.
    best: tuple[int, str, str] | None = None
    for open_m, close_m in _THINK_PAIRS:
        idx = buf.find(open_m)
        if idx == -1:
            continue
        if best is None or idx < best[0]:
            best = (idx, open_m, close_m)

    if best is None:
        # No opening marker found. If a lone close marker exists, the chat
        # template prepended the open tag to the prompt and the model's stream
        # starts already inside a thinking block — split around the close tag.
        for _open_m, close_m in _THINK_PAIRS:
            idx = buf.find(close_m)
            if idx != -1:
                thinking = buf[:idx].lstrip()
                answer = buf[idx + len(close_m):].lstrip()
                for end in _END_MARKERS:
                    answer = _strip_partial_end(answer, end)
                return thinking, answer, False
        # No complete marker either way — but the tail may still be a partial
        # thinking-open marker that will become complete on the next token.
        # Hold back any suffix that matches a prefix of any known open marker
        # so downstream streaming can't emit a half-formed `<|ch` as text.
        hold = 0
        for open_m, _close in _THINK_PAIRS:
            for n in range(len(open_m) - 1, 0, -1):
                if buf.endswith(open_m[:n]):
                    if n > hold:
                        hold = n
                    break
        if hold > 0:
            return "", buf[:len(buf) - hold], False
        return "", buf, False

    t_start, open_m, close_m = best
    prefix = buf[:t_start]
    after_open = buf[t_start + len(open_m):]
    if after_open.startswith("\n"):
        after_open = after_open[1:]

    t_end = after_open.find(close_m)
    if t_end == -1:
        thinking = _strip_partial_end(after_open, close_m).lstrip()
        return thinking, prefix, True

    thinking = after_open[:t_end].lstrip()
    answer = prefix + after_open[t_end + len(close_m):]
    for end in _END_MARKERS:
        answer = _strip_partial_end(answer, end)
    return thinking, answer.lstrip(), False


def _strip_partial_end(s: str, marker: str) -> str:
    """Strip trailing `marker` (complete or partial) so half-emitted markers
    don't flicker on screen."""
    if marker in s:
        s = s.split(marker, 1)[0]
    for n in range(len(marker) - 1, 0, -1):
        if s.endswith(marker[:n]):
            return s[:-n]
    return s


def stream_response(
    chunks: Iterable[Chunk],
    console: Console,
    refresh_per_second: int = 12,
) -> str:
    """Render a stream of tokens with a live tokens/sec footer.

    Two-phase rendering:
      1. While the model is in its thinking channel, write each new thought
         chunk straight to `console.print` (dim italic) so it lands in the
         terminal's scrollback. Long chains-of-thought stay scrollable after
         the run.
      2. Once the answer channel opens, switch to a `Live` widget that
         markdown-renders the answer with a tok/s footer. The frame that
         persists when Live exits becomes the final scrollback line.

    Returns the complete post-filter answer. A KeyboardInterrupt during the
    stream finalizes the render cleanly and returns whatever was produced.
    """
    buf = ""
    n_tok = 0
    t0 = time.perf_counter()
    interrupted = False

    md_cache: dict = {"t": 0.0, "body": None}
    MD_INTERVAL = 0.25

    thinking_header_printed = False
    thinking_finalized = False
    printed_thinking = 0  # chars of thinking already written to scrollback
    live: Live | None = None

    def _flush_thinking() -> None:
        """Append any new thinking text to scrollback. No-op once finalized."""
        nonlocal printed_thinking, thinking_header_printed
        if thinking_finalized:
            return
        thinking_now, _, _ = _split_channels(buf)
        if thinking_now and not thinking_header_printed:
            console.print("[dim italic]thinking…[/]")
            thinking_header_printed = True
        new_text = thinking_now[printed_thinking:]
        if new_text:
            console.print(new_text, end="", style="italic dim", highlight=False)
            printed_thinking = len(thinking_now)

    def _finalize_thinking() -> None:
        """Flush any remaining thinking + print separator. Idempotent."""
        nonlocal thinking_finalized
        if thinking_finalized:
            return
        _flush_thinking()
        if thinking_header_printed:
            console.print()
        thinking_finalized = True

    def _render_answer(is_final: bool) -> Group:
        thinking_v, answer_v, _ = _split_channels(buf)
        dt = time.perf_counter() - t0
        tps = n_tok / dt if dt > 0 else 0.0
        if answer_v.strip() and not interrupted:
            from minimlx.codeloop import _stabilize_for_markdown
            now = time.perf_counter()
            need_reparse = (
                is_final
                or md_cache["body"] is None
                or (now - md_cache["t"]) >= MD_INTERVAL
            )
            if need_reparse:
                try:
                    stable = answer_v if is_final else _stabilize_for_markdown(answer_v)
                    md_cache["body"] = Markdown(stable, code_theme="monokai")
                except Exception:
                    md_cache["body"] = Text(answer_v)
                md_cache["t"] = now
            body_text = md_cache["body"]
        else:
            body_text = Text(answer_v)
        extras = f" · thought {len(thinking_v)} chars" if thinking_v else ""
        status_text = Text(
            f"  {n_tok} tok · {tps:5.1f} tok/s{extras}",
            style="dim green" if is_final and not interrupted else "dim cyan",
        )
        if is_final and interrupted:
            status_text = Text(status_text.plain + " · interrupted", style="dim yellow")
        return Group(body_text, status_text)

    try:
        for chunk in chunks:
            buf += chunk.text
            n_tok += chunk.n_tokens
            thinking, answer, still_thinking = _split_channels(buf)

            if still_thinking:
                _flush_thinking()
                continue

            # Past thinking — but skip empty-buffer / partial-marker frames
            # so we don't flash an empty Live region.
            if not (answer.strip() or thinking or thinking_header_printed):
                continue

            # One-shot finalize: idempotent flush + separator. Handles the
            # edge case where open + close arrive in the same chunk before
            # phase 1 ever fires.
            _finalize_thinking()
            if live is None:
                live = Live(
                    Group(Text(""), Text("", style="dim")),
                    console=console,
                    refresh_per_second=refresh_per_second,
                    transient=False,
                    # Long tool-call payloads or long answers will exceed
                    # terminal height — `visible` lets the body scroll past
                    # the screen instead of getting Rich-ellipsised at the
                    # bottom of the live frame.
                    vertical_overflow="visible",
                )
                live.__enter__()
            live.update(_render_answer(is_final=False))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if live is not None:
            try:
                live.update(_render_answer(is_final=True))
            except Exception:
                pass
            live.__exit__(None, None, None)
        elif thinking_header_printed and not thinking_finalized:
            _flush_thinking()
            console.print()

    _, answer, _ = _split_channels(buf)
    return answer
