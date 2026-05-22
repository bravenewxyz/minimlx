"""Voice conversation mode: listen → transcribe → generate → speak → loop."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from minimlx import defaults as _D
from minimlx.engine import Engine
from minimlx.listen import Transcriber, record_audio
from minimlx.render import stream_response
from minimlx.speak import Speaker
from minimlx.store import Store


def run_conversation(
    engine: Engine,
    store: Store,
    console: Console,
    *,
    transcriber: Transcriber,
    system: str | None = None,
    conversation_id: int | None = None,
    max_tokens: int = 2048,
    temp: float = _D.TEMP,
    top_p: float = _D.TOP_P,
    speaker: Speaker | None = None,
    silence_threshold: float = _D.SILENCE_THRESHOLD,
    silence_duration: float = _D.SILENCE_DURATION,
) -> None:
    if speaker is None:
        speaker = Speaker(enabled=True)

    console.print(Panel.fit(
        f"[bold cyan]minimlx conversation[/]  [dim]· {engine.model_id}[/]\n"
        f"[dim]STT: {transcriber.model_id} · lang: {transcriber.language}[/]\n"
        "[dim]Speak after the prompt · silence auto-stops recording · Ctrl-C to skip/exit[/]",
        border_style="cyan",
    ))

    messages: list[dict] = []
    if conversation_id is not None:
        messages = store.load_messages(conversation_id)
        console.print(f"[dim]-- continuing conversation {conversation_id} ({len(messages)} msgs) --[/]")
    if not messages and system:
        messages.append({"role": "system", "content": system})
    if conversation_id is None:
        conversation_id = store.new_conversation(engine.model_id, system)

    def _turns() -> int:
        return sum(1 for m in messages if m["role"] == "user")

    # Pre-load the STT model so first turn isn't slow.
    console.print("[dim]loading speech-to-text model…[/]")
    transcriber._load()
    console.print("[dim]ready.[/]\n")

    while True:
        # --- Record ---
        console.print(f"[bold green]listening[/] [dim](turn {_turns() + 1})[/]")
        try:
            audio = record_audio(
                silence_threshold=silence_threshold,
                silence_duration=silence_duration,
            )
        except KeyboardInterrupt:
            console.print("\n[dim]bye[/]")
            speaker.stop()
            break

        if len(audio) < 16_000 * 0.3:
            console.print("[dim]-- too short, skipped --[/]")
            continue

        # --- Transcribe ---
        console.print("[dim]transcribing…[/]", end="")
        try:
            text = transcriber.transcribe(audio)
        except KeyboardInterrupt:
            console.print(" [dim]skipped[/]")
            continue

        if not text:
            console.print(" [dim]-- no speech detected --[/]")
            continue
        console.print()
        console.print(Text(f"  you: {text}", style="bold"))
        console.print()

        messages.append({"role": "user", "content": text})
        store.append_message(conversation_id, "user", text)

        # --- Generate ---
        try:
            reply = stream_response(
                engine.stream(messages, max_tokens=max_tokens, temp=temp, top_p=top_p),
                console,
            )
        except KeyboardInterrupt:
            reply = ""
        except Exception as e:
            console.print(f"[red]error:[/] {e}")
            continue

        if reply:
            messages.append({"role": "assistant", "content": reply})
            store.append_message(conversation_id, "assistant", reply)
        console.print()

        # --- Speak ---
        if speaker.enabled and reply.strip():
            try:
                speaker.speak(reply)
                speaker.wait()
            except KeyboardInterrupt:
                speaker.stop()
