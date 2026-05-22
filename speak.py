from __future__ import annotations
import re
import shutil
import subprocess
from typing import Optional


_CODE_BLOCK = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_ITAL = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_UBOLD = re.compile(r"__([^_\n]+)__")
_UITAL = re.compile(r"(?<!_)_([^_\n]+)_(?!_)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BLANKS = re.compile(r"\n{3,}")


def strip_markdown_for_tts(text: str) -> str:
    """Strip the markdown constructs that are painful to listen to."""
    text = _CODE_BLOCK.sub(" (code block omitted) ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADER.sub("", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITAL.sub(r"\1", text)
    text = _UBOLD.sub(r"\1", text)
    text = _UITAL.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


class Speaker:
    """Non-blocking text-to-speech via macOS `say`.

    Only one utterance plays at a time — calling `speak()` while a previous
    utterance is still playing terminates the previous one first. `wait()`
    blocks until the current utterance finishes (for one-shot CLI use);
    `stop()` is idempotent and safe to call at program exit.
    """

    def __init__(
        self,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        enabled: bool = True,
    ):
        self.voice = voice
        self.rate = rate
        self.enabled = enabled and shutil.which("say") is not None
        self._proc: Optional[subprocess.Popen] = None

    def available(self) -> bool:
        return self.enabled

    def speak(self, text: str) -> None:
        if not self.enabled:
            return
        clean = strip_markdown_for_tts(text)
        if not clean:
            return
        self.stop()
        cmd: list[str] = ["say"]
        if self.voice:
            cmd += ["-v", self.voice]
        if self.rate:
            cmd += ["-r", str(self.rate)]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.enabled = False
            return
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(clean.encode("utf-8"))
        except BrokenPipeError:
            pass
        finally:
            try:
                self._proc.stdin.close()
            except Exception:
                pass

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except ProcessLookupError:
                pass
        self._proc = None

    def wait(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.wait()
        except KeyboardInterrupt:
            self.stop()
            return
        self._proc = None

    def set_voice(self, voice: Optional[str]) -> None:
        self.voice = voice

    def set_rate(self, rate: Optional[int]) -> None:
        self.rate = rate

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled and shutil.which("say") is not None

    @staticmethod
    def list_voices(english_only: bool = True) -> list[tuple[str, str]]:
        """Return [(name, locale)] from `say -v ?` output."""
        if shutil.which("say") is None:
            return []
        try:
            out = subprocess.check_output(["say", "-v", "?"], text=True)
        except subprocess.CalledProcessError:
            return []
        voices: list[tuple[str, str]] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "Samantha         en_US    # ..."
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            locale = parts[1]
            if english_only and not locale.startswith("en"):
                continue
            voices.append((name, locale))
        return voices
