from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from minimlx import __version__, defaults as D
from minimlx.aliases import load_aliases, load_preset, resolve_alias
from minimlx.engine import Engine
from minimlx.models import list_cached, pull as pull_model, remove as remove_model
from minimlx.render import stream_response
from minimlx.speak import Speaker
from minimlx.store import Store

app = typer.Typer(
    name="minimlx",
    help="Very fast Google Gemma 4 on Apple Silicon via MLX.",
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
)

DB_PATH = D.DB_PATH

KNOWN_SUBCOMMANDS = {
    "ask", "chat", "code", "conversation", "pull", "ls", "rm", "logs", "models", "voices", "serve",
    "version", "--help", "-h", "--version", "--install-completion", "--show-completion",
    "--conversation",
}


def _build_engine(
    model: str,
    draft: Optional[str],
    num_draft_tokens: int,
    max_kv_size: Optional[int],
    kv_bits: Optional[int],
) -> Engine:
    aliases = load_aliases()
    preset = load_preset(model)
    # CLI flags override preset. Draft is the big one: if the user didn't
    # pass --draft, fall back to whatever the alias prefers.
    effective_draft = draft if draft is not None else preset.get("draft")
    effective_num_draft = (
        num_draft_tokens if num_draft_tokens != 4 else preset.get("num_draft_tokens", 4)
    )
    return Engine(
        model_id=resolve_alias(model, aliases),
        draft_model_id=resolve_alias(effective_draft, aliases) if effective_draft else None,
        num_draft_tokens=effective_num_draft,
        max_kv_size=max_kv_size,
        kv_bits=kv_bits,
    )


def _apply_temp_preset(model: str, temp: float, default: float = D.TEMP) -> float:
    """Return the preset temp if the user didn't explicitly override it."""
    if temp != default:
        return temp
    preset = load_preset(model)
    return preset.get("temp", temp)


@app.command(help="Run a one-shot prompt (also the default if you just pass text).")
def ask(
    prompt: list[str] = typer.Argument(..., help="The prompt text."),
    model: str = typer.Option(D.MODEL, "-m", "--model", help="Model id or alias."),
    draft: Optional[str] = typer.Option(None, "--draft", help="Draft model for speculative decoding."),
    num_draft_tokens: int = typer.Option(D.NUM_DRAFT_TOKENS, "--num-draft-tokens"),
    max_tokens: int = typer.Option(D.MAX_TOKENS, "-n", "--max-tokens", help="Max tokens to generate (-1 = unlimited)."),
    temp: float = typer.Option(D.TEMP, "-t", "--temp"),
    top_p: float = typer.Option(D.TOP_P, "--top-p"),
    system: Optional[str] = typer.Option(None, "-s", "--system"),
    max_kv_size: Optional[int] = typer.Option(None, "--max-kv-size"),
    kv_bits: Optional[int] = typer.Option(None, "--kv-bits"),
    continue_: bool = typer.Option(False, "-c", "--continue", help="Continue last conversation."),
    speak: bool = typer.Option(False, "-S", "--speak", help="Speak the answer aloud via macOS `say`."),
    voice: Optional[str] = typer.Option(None, "--voice", help="TTS voice (e.g. Samantha, Alex)."),
    rate: Optional[int] = typer.Option(None, "--rate", help="TTS words per minute."),
) -> None:
    console = Console()
    store = Store(DB_PATH)
    temp = _apply_temp_preset(model, temp)
    engine = _build_engine(model, draft, num_draft_tokens, max_kv_size, kv_bits)
    speak = speak or voice is not None or rate is not None
    speaker = Speaker(voice=voice, rate=rate, enabled=speak)

    text = " ".join(prompt)
    if text == "-":
        text = sys.stdin.read()

    messages: list[dict] = []
    conv_id: Optional[int] = None
    if continue_:
        conv_id = store.last_conversation_id()
        if conv_id is not None:
            messages = store.load_messages(conv_id)
    if not messages and system:
        messages.append({"role": "system", "content": system})
    if conv_id is None:
        conv_id = store.new_conversation(engine.model_id, system)
    messages.append({"role": "user", "content": text})
    store.append_message(conv_id, "user", text)

    try:
        reply = stream_response(
            engine.stream(messages, max_tokens=max_tokens, temp=temp, top_p=top_p),
            console,
        )
    except Exception as e:
        console.print(f"[red]error:[/] {e}")
        raise typer.Exit(1)
    store.append_message(conv_id, "assistant", reply)
    console.print()

    if speaker.available() and reply.strip():
        try:
            speaker.speak(reply)
            speaker.wait()
        except KeyboardInterrupt:
            speaker.stop()


@app.command(help="Interactive chat REPL with streaming output and slash commands.")
def chat(
    model: str = typer.Option(D.MODEL, "-m", "--model"),
    draft: Optional[str] = typer.Option(None, "--draft"),
    num_draft_tokens: int = typer.Option(D.NUM_DRAFT_TOKENS, "--num-draft-tokens"),
    max_tokens: int = typer.Option(D.MAX_TOKENS, "-n", "--max-tokens", help="Max tokens to generate (-1 = unlimited)."),
    temp: float = typer.Option(D.TEMP, "-t", "--temp"),
    top_p: float = typer.Option(D.TOP_P, "--top-p"),
    system: Optional[str] = typer.Option(None, "-s", "--system"),
    max_kv_size: Optional[int] = typer.Option(None, "--max-kv-size"),
    kv_bits: Optional[int] = typer.Option(None, "--kv-bits"),
    continue_: bool = typer.Option(False, "-c", "--continue"),
    speak: bool = typer.Option(False, "-S", "--speak", help="Speak each reply aloud via macOS `say`."),
    voice: Optional[str] = typer.Option(None, "--voice", help="TTS voice (e.g. Samantha, Alex)."),
    rate: Optional[int] = typer.Option(None, "--rate", help="TTS words per minute."),
    tools: bool = typer.Option(True, "--tools/--no-tools", help="Enable the local tool-use loop (read/edit/grep/bash/…)."),
    allow_bash: bool = typer.Option(True, "--allow-bash/--no-bash", help="Enable the bash tool (on by default)."),
    cwd: Path = typer.Option(Path.cwd(), "-C", "--cwd", help="Working directory the tools are scoped to."),
    unrestricted: bool = typer.Option(False, "--unrestricted", help="Allow tools to read/write paths outside --cwd. Off by default."),
) -> None:
    from minimlx.chat import run_chat
    console = Console()
    store = Store(DB_PATH)
    temp = _apply_temp_preset(model, temp)
    engine = _build_engine(model, draft, num_draft_tokens, max_kv_size, kv_bits)
    speak = speak or voice is not None or rate is not None
    speaker = Speaker(voice=voice, rate=rate, enabled=speak)
    conv_id = store.last_conversation_id() if continue_ else None
    run_chat(
        engine, store, console,
        system=system, conversation_id=conv_id,
        max_tokens=max_tokens, temp=temp, top_p=top_p,
        speaker=speaker,
        tools_on=tools, allow_bash=allow_bash, cwd=cwd.resolve(),
        unrestricted=unrestricted,
    )


@app.command(help="Run a one-shot programming task with local tool use (read/edit/grep/…).")
def code(
    prompt: list[str] = typer.Argument(..., help="What you want done."),
    model: str = typer.Option(D.MODEL, "-m", "--model"),
    draft: Optional[str] = typer.Option(None, "--draft"),
    num_draft_tokens: int = typer.Option(D.NUM_DRAFT_TOKENS, "--num-draft-tokens"),
    max_tokens: int = typer.Option(D.CODE_MAX_TOKENS, "-n", "--max-tokens"),
    temp: float = typer.Option(D.CODE_TEMP, "-t", "--temp"),
    top_p: float = typer.Option(D.TOP_P, "--top-p"),
    cwd: Path = typer.Option(Path.cwd(), "-C", "--cwd", help="Working directory the tools are scoped to."),
    allow_bash: bool = typer.Option(True, "--allow-bash/--no-bash", help="Enable the bash tool (on by default)."),
    unrestricted: bool = typer.Option(False, "--unrestricted", help="Allow tools to read/write paths outside --cwd. Off by default."),
    max_kv_size: Optional[int] = typer.Option(None, "--max-kv-size"),
    kv_bits: Optional[int] = typer.Option(None, "--kv-bits"),
) -> None:
    from minimlx.codeloop import run_code_session
    console = Console()
    temp = _apply_temp_preset(model, temp)
    engine = _build_engine(model, draft, num_draft_tokens, max_kv_size, kv_bits)
    text = " ".join(prompt)
    if text == "-":
        text = sys.stdin.read()
    run_code_session(
        engine, console,
        initial_prompt=text,
        cwd=cwd.resolve(),
        allow_bash=allow_bash,
        unrestricted=unrestricted,
        temp=temp, top_p=top_p, max_tokens=max_tokens,
    )


@app.command(help="Voice conversation: speak → transcribe → generate → TTS → loop.")
def conversation(
    model: str = typer.Option(D.MODEL, "-m", "--model"),
    draft: Optional[str] = typer.Option(None, "--draft"),
    num_draft_tokens: int = typer.Option(D.NUM_DRAFT_TOKENS, "--num-draft-tokens"),
    max_tokens: int = typer.Option(D.MAX_TOKENS, "-n", "--max-tokens", help="Max tokens to generate (-1 = unlimited)."),
    temp: float = typer.Option(D.TEMP, "-t", "--temp"),
    top_p: float = typer.Option(D.TOP_P, "--top-p"),
    system: Optional[str] = typer.Option(None, "-s", "--system"),
    max_kv_size: Optional[int] = typer.Option(None, "--max-kv-size"),
    kv_bits: Optional[int] = typer.Option(None, "--kv-bits"),
    continue_: bool = typer.Option(False, "-c", "--continue"),
    voice: Optional[str] = typer.Option(None, "--voice", help="TTS voice (e.g. Samantha, Alex)."),
    rate: Optional[int] = typer.Option(None, "--rate", help="TTS words per minute."),
    language: str = typer.Option("en", "-l", "--language", help="STT language code (e.g. en, fr, de, ja)."),
    stt_model: Optional[str] = typer.Option(None, "--stt-model", help=f"STT model id (default: {D.STT_MODEL} alias)."),
    silence_threshold: float = typer.Option(D.SILENCE_THRESHOLD, "--silence-threshold", help="RMS amplitude for silence detection."),
    silence_duration: float = typer.Option(D.SILENCE_DURATION, "--silence-duration", help="Seconds of silence to auto-stop recording."),
) -> None:
    from minimlx.conversation import run_conversation
    from minimlx.listen import Transcriber
    console = Console()
    store = Store(DB_PATH)
    engine = _build_engine(model, draft, num_draft_tokens, max_kv_size, kv_bits)
    speaker = Speaker(voice=voice, rate=rate, enabled=True)
    stt_id = resolve_alias(stt_model or D.STT_MODEL)
    transcriber = Transcriber(model_id=stt_id, language=language)
    conv_id = store.last_conversation_id() if continue_ else None
    run_conversation(
        engine, store, console,
        transcriber=transcriber,
        system=system, conversation_id=conv_id,
        max_tokens=max_tokens, temp=temp, top_p=top_p,
        speaker=speaker,
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
    )


@app.command(help="Download a model to the local HuggingFace cache.")
def pull(repo: str = typer.Argument(..., help="HF repo id or alias (e.g. gemma4).")) -> None:
    console = Console()
    resolved = resolve_alias(repo)
    pull_model(resolved, console)


@app.command("ls", help="List cached models.")
def ls_cmd(
    filter: str = typer.Option("gemma", "-f", "--filter", help="Substring filter on repo id."),
) -> None:
    console = Console()
    entries = list_cached(name_filter=filter)
    if not entries:
        console.print("[dim]no models cached matching filter[/]")
        return
    table = Table(title="cached models", show_lines=False)
    table.add_column("repo", style="cyan")
    table.add_column("size", justify="right")
    table.add_column("files", justify="right", style="dim")
    table.add_column("last access", style="dim")
    for e in entries:
        ts = datetime.fromtimestamp(e["last_accessed"]).strftime("%Y-%m-%d %H:%M")
        table.add_row(e["repo_id"], f"{e['size_gb']:.2f} GB", str(e["nb_files"]), ts)
    console.print(table)


@app.command(help="Remove a cached model.")
def rm(repo: str = typer.Argument(..., help="Repo id to remove.")) -> None:
    console = Console()
    n = remove_model(resolve_alias(repo))
    if n:
        console.print(f"[green]✓[/] removed {repo}")
    else:
        console.print(f"[red]not found:[/] {repo}")


@app.command(help="List built-in and user aliases, or show details for one alias.")
def models(
    alias: str = typer.Argument(None, help="Optional alias to show details for."),
) -> None:
    console = Console()
    aliases = load_aliases()
    if alias is None:
        table = Table(title="aliases")
        table.add_column("alias", style="cyan")
        table.add_column("resolves to")
        table.add_column("preset", style="dim")
        for k in sorted(aliases):
            preset = load_preset(k)
            bits = []
            if preset.get("draft"):
                bits.append(f"draft={preset['draft']}×{preset.get('num_draft_tokens', 4)}")
            if "temp" in preset:
                bits.append(f"temp={preset['temp']}")
            table.add_row(k, aliases[k], " · ".join(bits))
        console.print(table)
        return

    if alias not in aliases:
        console.print(f"[red]unknown alias:[/] {alias}")
        raise typer.Exit(1)

    chain = [alias]
    seen = {alias}
    cur = alias
    while aliases.get(cur) in aliases and aliases[cur] not in seen:
        cur = aliases[cur]
        chain.append(cur)
        seen.add(cur)
    target = aliases[cur]

    table = Table(title=f"alias: {alias}", show_header=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("alias", alias)
    if len(chain) > 1:
        table.add_row("chain", " → ".join(chain))
    table.add_row("resolves to", target)
    table.add_row("kind", "local path" if target.startswith("/") else "HF repo")

    preset = load_preset(alias)
    if not preset and len(chain) > 1:
        preset = load_preset(cur)
    if preset:
        for k, v in preset.items():
            table.add_row(f"preset.{k}", str(v))
    else:
        table.add_row("preset", "[dim]—[/]")
    console.print(table)


@app.command(help="Show recent conversation logs.")
def logs(n: int = typer.Option(10, "-n", "--limit")) -> None:
    console = Console()
    store = Store(DB_PATH)
    rows = store.list_conversations(limit=n)
    if not rows:
        console.print("[dim]no conversations logged[/]")
        return
    table = Table(title="conversations")
    table.add_column("id", justify="right", style="cyan")
    table.add_column("model")
    table.add_column("msgs", justify="right")
    table.add_column("when", style="dim")
    for (cid, model, created, count) in rows:
        ts = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(cid), model, str(count), ts)
    console.print(table)


@app.command(help="Start an Anthropic-API-compatible HTTP server for Claude Code.")
def serve(
    host: str = typer.Option(D.SERVER_HOST, "--host", help="Interface to bind."),
    port: int = typer.Option(D.SERVER_PORT, "--port", "-p", help="Port to listen on."),
    default_model: str = typer.Option(D.MODEL, "-m", "--default-model", help="Alias used for all requests when pinned (default), or as fallback otherwise."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    pin: bool = typer.Option(False, "--pin/--no-pin", help="Force every request to use the default model. Off by default — the server loads whatever model the client asks for."),
) -> None:
    from minimlx.server import run_server
    run_server(host=host, port=port, default_model=default_model, verbose=verbose, pin=pin)


@app.command(help="List TTS voices available to macOS `say`.")
def voices(all: bool = typer.Option(False, "--all", "-a", help="Show non-English voices too.")) -> None:
    console = Console()
    vs = Speaker.list_voices(english_only=not all)
    if not vs:
        console.print("[red]macOS `say` not available[/]")
        raise typer.Exit(1)
    table = Table(title="say voices" + ("" if all else " (english only — --all for more)"))
    table.add_column("name", style="cyan")
    table.add_column("locale", style="dim")
    for name, locale in vs:
        table.add_row(name, locale)
    console.print(table)


@app.command(help="Print version.")
def version() -> None:
    typer.echo(f"minimlx {__version__}")


def _route_bare_prompt() -> None:
    """If argv[1] isn't a known subcommand and doesn't start with -, inject 'ask'.

    This lets users write `minimlx "hello"` as a shortcut for `minimlx ask "hello"`.
    ``--conversation`` is promoted to the ``conversation`` subcommand so
    ``minimlx --conversation`` works as a convenient shorthand.
    """
    if len(sys.argv) < 2:
        return
    # Promote --conversation anywhere in argv to the subcommand form.
    if "--conversation" in sys.argv:
        sys.argv.remove("--conversation")
        sys.argv.insert(1, "conversation")
        return
    first = sys.argv[1]
    if first in KNOWN_SUBCOMMANDS or first.startswith("-"):
        return
    sys.argv.insert(1, "ask")


def main() -> None:
    _route_bare_prompt()
    app()


if __name__ == "__main__":
    main()
