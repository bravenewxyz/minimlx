"""Anthropic Messages API-compatible HTTP server wrapping minimlx engines.

Exposes POST /v1/messages (streaming + non-streaming) so Claude Code and other
Anthropic SDK clients can talk to local MLX models as if they were claude.ai.

**Not implemented**:
- Image / document content blocks. Text-only.
- Prompt caching headers. Ignored silently.

**Threading**: a single global `_engine_lock` serializes both model swaps and
generation, because mlx-lm's model/KV cache is not concurrency-safe. This is
fine for a single-user local workstation. For multi-user, swap in mlx-lm's
own batching server.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Iterator

# Keep the banner clean. All three env vars have to be set BEFORE importing
# mlx_lm / transformers / huggingface_hub so the libraries pick them up.
#
# - HF_HUB_DISABLE_PROGRESS_BARS: `snapshot_download` spams a "Fetching N
#   files" tqdm bar on every load() call, even when the files are cached.
# - TOKENIZERS_PARALLELISM: the tokenizers Rust backend forks workers that
#   leak macOS semaphore handles when the parent process exits, causing the
#   "resource_tracker: … leaked semaphore objects" warning on shutdown.
# - TRANSFORMERS_VERBOSITY: silences transformers' UserWarning about the
#   "mistral regex pattern" in Qwen3.5 tokenizers. The warning suggests
#   passing `fix_mistral_regex=True`, but that kwarg only applies to real
#   Mistral tokenizers — forwarding it to a Qwen tokenizer crashes inside
#   transformers. The Qwen tokenization is already correct for our use; the
#   warning is a false positive.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# tqdm creates a daemon monitor thread (_monitor.py TMonitor) that polls
# progress bars. After Metal inference, a late GPU callback can fire on
# that thread without the GIL held → Fatal Python error. Disabling the
# monitor prevents the crash (the progress bars still work, just without
# the background polling).
os.environ.setdefault("TQDM_MONITOR_INTERVAL", "0")
# Suppress the `multiprocessing.resource_tracker` "leaked semaphore" warning
# that fires in the resource-tracker subprocess (not the parent) when the
# server is killed by SIGTERM. Passing this through PYTHONWARNINGS lets the
# child pick it up via env.
_prev_warn = os.environ.get("PYTHONWARNINGS", "")
_mp_filter = "ignore::UserWarning:multiprocessing.resource_tracker"
if _mp_filter not in _prev_warn:
    os.environ["PYTHONWARNINGS"] = (
        f"{_prev_warn},{_mp_filter}" if _prev_warn else _mp_filter
    )

import warnings  # noqa: E402
warnings.filterwarnings(
    "ignore",
    message=r".*incorrect regex pattern.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*fix_mistral_regex.*",
)

from minimlx.aliases import load_aliases, load_preset, resolve_alias
from minimlx.engine import Engine
from minimlx.promptcache import PromptCacheStore
from minimlx.render import _split_channels
from minimlx.tooluse import (
    StreamingToolParser,
    anthropic_tools_to_transformers,
    build_tool_system_prompt,
    flatten_history,
    parse_content_blocks,
    template_supports_tools,
)

# Settable by run_server().
from minimlx import defaults as _D
_DEFAULT_MODEL = _D.MODEL
_VERBOSE = False
_PINNED = False  # If True, every request uses _DEFAULT_MODEL regardless of
                 # what the client's `model` field asks for. Prevents model
                 # swapping, which crashes under the PrismML mlx fork when
                 # freeing a 31B target triggers late Metal callbacks.


def _short_model(name: str) -> str:
    """Trim a repo path/id to a short label for log lines."""
    return name.rsplit("/", 1)[-1]


def _log_stats(
    model_name: str,
    resolved_target: str,
    path: str,
    stats: Any,
    stop_reason: str,
    n_output_fallback: int,
    prefix_hit: int = 0,
    cache_stats: dict | None = None,
) -> None:
    """Always-on single-line summary after each request — shows tok/s etc."""
    if stats is None:
        in_tok = 0
        out_tok = n_output_fallback
        prompt_tps = 0.0
        gen_tps = 0.0
        peak_gb = 0.0
    else:
        in_tok = int(getattr(stats, "prompt_tokens", 0))
        out_tok = int(getattr(stats, "generated_tokens", n_output_fallback))
        prompt_tps = float(getattr(stats, "prompt_tps", 0.0))
        gen_tps = float(getattr(stats, "generation_tps", 0.0))
        peak_gb = float(getattr(stats, "peak_memory_gb", 0.0))
    asked = _short_model(model_name)
    used = _short_model(resolved_target)
    model_label = f"{asked}" if asked == used else f"{asked} → {used}"
    cache_part = ""
    if prefix_hit:
        cache_part = f"  ·  cache {prefix_hit} tok hit"
    elif cache_stats and cache_stats.get("entries"):
        cache_part = f"  ·  cache miss ({cache_stats['entries']} entries)"
    print(
        f"  {path}  {model_label}  "
        f"prompt {in_tok} tok @ {prompt_tps:6.1f} t/s  ·  "
        f"gen {out_tok} tok @ {gen_tps:6.1f} t/s  ·  "
        f"peak {peak_gb:5.1f} GB  ·  stop={stop_reason}{cache_part}",
        flush=True,
    )


def _prepare_cache(
    engine: Engine,
    store: PromptCacheStore,
    messages: list[dict],
    tools: list[dict] | None,
) -> tuple[Any, int]:
    """Tokenize the prompt and look up the best cached prefix.

    Returns (prompt_cache, prefix_len). When the engine has a draft model
    the returned list is a *combined* cache: target layers followed by
    draft layers, which is the layout mlx-lm's speculative_generate_step
    expects (it slices at `prompt_cache[:len(model.layers)]`).

    On a cold miss the first request fills both target and draft caches
    during prefill + generation. On store we save the combined cache so
    future lookups return both portions already warm.
    """
    from mlx_lm.models.cache import make_prompt_cache
    try:
        _, tokens = engine.tokenize_prompt(messages, tools=tools)
    except Exception:
        return None, 0

    cache, prefix_len = store.lookup(tokens)

    if cache is not None:
        # Cache hit — may be target-only or combined depending on what was
        # stored. If the engine now has a draft model but the stored cache
        # is target-only (from a prior non-draft session), extend it with
        # fresh draft layers. The draft will prefill from scratch on this
        # first warm request, but subsequent requests will hit the full
        # combined cache.
        if engine.draft_model is not None:
            n_target = len(engine.model.layers)
            if len(cache) <= n_target:
                draft_cache = make_prompt_cache(engine.draft_model)
                cache = list(cache) + list(draft_cache)
    else:
        # Cold miss — create fresh caches.
        target_cache = list(make_prompt_cache(engine.model))
        if engine.draft_model is not None:
            draft_cache = list(make_prompt_cache(engine.draft_model))
            cache = target_cache + draft_cache
        else:
            cache = target_cache
        prefix_len = 0

    _cache_tokens_by_id[id(cache)] = tokens
    return cache, prefix_len


def _finalize_cache(
    engine: Engine,
    store: PromptCacheStore,
    messages: list[dict],
    tools: list[dict] | None,
    prompt_cache: Any,
    generated_tokens: int,
) -> None:
    """Save the post-generation cache back to the store, trimmed to cover
    only the prompt tokens (not the generated output).

    For combined caches (target + draft layers), trim_prompt_cache iterates
    all layers and trims each by `generated_tokens`. Both target and draft
    caches stay synchronised because speculative decoding keeps the draft's
    accepted token count aligned with the target's.
    """
    if prompt_cache is None:
        return
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
    try:
        tokens = _cache_tokens_by_id.pop(id(prompt_cache), None)
        if tokens is None:
            _, tokens = engine.tokenize_prompt(messages, tools=tools)
        if not can_trim_prompt_cache(prompt_cache):
            return
        if generated_tokens > 0:
            trim_prompt_cache(prompt_cache, generated_tokens)
        store.store(tokens, prompt_cache)
    except Exception as e:
        _log(f"cache finalize failed: {type(e).__name__}: {e}")


# Thread-local scratch dict for passing token lists between _prepare and
# _finalize without mutating the cache object. id() is unique as long as the
# cache object lives past the lookup → store cycle, which it does here.
_cache_tokens_by_id: dict[int, list[int]] = {}

# Global engine slot — one model loaded at a time. Two locks:
#   _swap_lock — serializes model loads/unloads
# The server is single-threaded (no ThreadingMixIn) to avoid Metal GIL
# crashes, so _swap_lock is the only concurrency guard needed — and it only
# matters if we later add async request handling.
_swap_lock = threading.Lock()
_current_engine: Engine | None = None
_current_target: str | None = None
_current_cache_store: PromptCacheStore | None = None


def _sync_metal() -> None:
    """Drain pending Metal ops before handing the GIL back to the idle
    server loop. Without this, a late Metal callback on a C thread can
    re-enter Python after the request handler returned, tripping
    `PyThreadState_Get: the GIL is released`.
    """
    try:
        import mlx.core as mx
        mx.synchronize()
    except Exception:
        pass


def _log(msg: str) -> None:
    if _VERBOSE:
        print(f"[minimlx-server] {msg}", flush=True)


def _resolve(model_name: str | None) -> tuple[str, str]:
    """Map an inbound model name to (target, alias_name).

    Returns both the resolved path and the original alias name, so the
    caller can look up per-alias presets like draft-model pairing.

    Follows alias chains: if the resolved value is itself an alias key,
    resolve again (max 5 hops to avoid loops). This lets users write
    `my-fav = "qwen36-35b-claude-opus-abliterated"` in aliases.toml
    and have it resolve through to the local path.
    """
    aliases = load_aliases()
    if _PINNED or not model_name:
        name = _DEFAULT_MODEL
        target = resolve_alias(name, aliases)
        return _chase(target, aliases), name

    # Direct key match
    if model_name in aliases:
        return _chase(aliases[model_name], aliases), model_name
    # Strip provider prefix (e.g. "google/gemma-4-e4b" → "gemma-4-e4b")
    tail = model_name.split("/", 1)[-1]
    if tail in aliases:
        return _chase(aliases[tail], aliases), tail
    # Substring match (e.g. "claude-haiku-4-5-20251001" matches alias key "claude-haiku")
    for key in aliases:
        if key in model_name:
            return _chase(aliases[key], aliases), key
    # Raw HF repo id
    if "/" in model_name:
        return model_name, model_name
    # Fall back to default
    return _chase(resolve_alias(_DEFAULT_MODEL, aliases), aliases), _DEFAULT_MODEL


def _chase(target: str, aliases: dict[str, str], max_hops: int = 5) -> str:
    """Follow alias chains: if `target` is itself an alias key, resolve it."""
    for _ in range(max_hops):
        if target in aliases and aliases[target] != target:
            target = aliases[target]
        else:
            break
    return target


@contextmanager
def _engine_for(model_name: str | None) -> Iterator[tuple[Engine, PromptCacheStore]]:
    """Ensure the requested model is loaded, then yield (engine, cache_store).

    Honors per-alias presets (e.g. gemma4 → draft=gemma4-draft,
    num_draft_tokens=2). Swaps happen under `_swap_lock`; generation happens
    Server is single-threaded to avoid Metal GIL crashes.
    """
    global _current_engine, _current_target, _current_cache_store
    target, alias_name = _resolve(model_name)
    with _swap_lock:
        if _current_target != target:
            if _current_engine is not None:
                _log(f"unloading {_current_target}")
                # Drain ALL pending Metal ops before freeing any model
                # buffers. Without this, a late Metal callback can fire on
                # a background C thread after gc.collect() has released the
                # model arrays, tripping PyThreadState_Get with GIL-released.
                _sync_metal()
                _current_engine = None
                _current_cache_store = None
                gc.collect()
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass
            # Apply any preset (draft pairing, num_draft_tokens, etc.) from
            # the alias before instantiating. Missing preset → plain Engine.
            preset = load_preset(alias_name)
            draft_alias = preset.get("draft")
            draft_target = None
            if draft_alias:
                aliases = load_aliases()
                draft_target = aliases.get(draft_alias, draft_alias)
            num_draft = int(preset.get("num_draft_tokens", 4))
            _log(f"loading {target}"
                 + (f"  draft={draft_alias}×{num_draft}" if draft_target else ""))
            eng = Engine(
                model_id=target,
                draft_model_id=draft_target,
                num_draft_tokens=num_draft,
            )
            eng.load()
            _current_engine = eng
            _current_target = target
            _current_cache_store = PromptCacheStore(model_id=target)
            _log(f"loaded {target}")
        engine = _current_engine
        cache_store = _current_cache_store
    yield engine, cache_store  # type: ignore[misc]


def _convert_messages(
    system: Any,
    messages: list[dict],
    tools: list[dict] | None,
) -> list[dict]:
    """Flatten Anthropic-shaped input into minimlx's {role, content} messages.

    The `tools` parameter is *not* injected here — it's passed separately to
    `engine.stream(tools=…)` so the tokenizer's own `apply_chat_template` can
    render it in the model's native format. If the template rejects tools,
    the engine falls back and `build_tool_system_prompt` is merged into the
    system message by this function as a last resort.
    """
    out: list[dict] = []
    system_text = ""

    if isinstance(system, str):
        system_text = system
    elif isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        system_text = "\n".join(p for p in parts if p)

    if system_text:
        out.append({"role": "system", "content": system_text})

    out.extend(flatten_history(messages))
    return out


def _anthropic_stop_reason(finish: str) -> str:
    if finish == "length":
        return "max_tokens"
    return "end_turn"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress the default BaseHTTPRequestHandler per-request HTTP line.
        # The per-request stats line from `_log_stats` is more informative.
        return

    # ---- HTTP helpers -----------------------------------------------------

    def _send_json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._send_json(status, {
            "type": "error",
            "error": {"type": err_type, "message": message},
        })

    # ---- routing ----------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._send_json(200, {"status": "ok", "default_model": _DEFAULT_MODEL})
            return
        if self.path.startswith("/v1/models"):
            aliases = load_aliases()
            data = [
                {
                    "id": name,
                    "type": "model",
                    "display_name": name,
                    "created_at": "2026-04-11T00:00:00Z",
                    "target": target,
                }
                for name, target in sorted(aliases.items())
            ]
            self._send_json(200, {"data": data, "has_more": False, "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None})
            return
        self._send_error_json(404, f"unknown path {self.path}", "not_found_error")

    def do_POST(self) -> None:  # noqa: N802
        if not (self.path == "/v1/messages" or self.path.startswith("/v1/messages?")):
            self._send_error_json(404, f"unknown path {self.path}", "not_found_error")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            if not raw:
                self._send_error_json(400, "empty body")
                return
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as e:
                self._send_error_json(400, f"invalid json: {e}")
                return
            self._handle_messages(body)
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._send_error_json(500, f"{type(e).__name__}: {e}", "api_error")
            except Exception:
                pass

    # ---- messages handler -------------------------------------------------

    def _handle_messages(self, body: dict) -> None:
        model_in = body.get("model")
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens", 1024))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.95))
        system = body.get("system")
        messages_in = body.get("messages", []) or []
        tools_in = body.get("tools") or []
        tools_norm = anthropic_tools_to_transformers(tools_in)

        msgs = _convert_messages(system, messages_in, tools_in)

        if not any(m["role"] in ("user", "assistant", "tool") for m in msgs):
            self._send_error_json(400, "no user/assistant messages")
            return

        _log(f"POST /v1/messages  model={model_in}  stream={stream}  msgs={len(msgs)}  tools={len(tools_in)}  max_tokens={max_tokens}")

        try:
            with _engine_for(model_in) as (engine, cache_store):
                # Only inject our generic tool system prompt when the
                # tokenizer's own chat template has no tool support — it
                # teaches a different format than the model's native one and
                # would otherwise confuse models like Gemma 4.
                if tools_in and not template_supports_tools(engine.tokenizer):
                    instructions = build_tool_system_prompt(tools_in)
                    if msgs and msgs[0].get("role") == "system":
                        msgs[0]["content"] = (msgs[0]["content"] + "\n\n" + instructions).strip()
                    else:
                        msgs.insert(0, {"role": "system", "content": instructions})

                client_model = model_in or _DEFAULT_MODEL
                resolved = engine.model_id  # what we actually loaded
                if stream:
                    self._stream_response(engine, cache_store, msgs, tools_norm, client_model, resolved, max_tokens, temperature, top_p)
                else:
                    self._sync_response(engine, cache_store, msgs, tools_norm, client_model, resolved, max_tokens, temperature, top_p)
        except Exception as e:
            traceback.print_exc()
            try:
                self._send_error_json(500, f"{type(e).__name__}: {e}", "api_error")
            except Exception:
                pass

    def _sync_response(
        self,
        engine: Engine,
        cache_store: PromptCacheStore,
        messages: list[dict],
        tools: list[dict] | None,
        model_name: str,
        resolved_target: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        prompt_cache, prefix_len = _prepare_cache(engine, cache_store, messages, tools)
        # Always pass the cache (even on cold) so finalize can store the
        # filled cache afterward. For combined caches (target+draft), the
        # empty combined list tells stream_generate to use it as the store.
        cache_for_gen = prompt_cache

        buf = ""
        n_out = 0
        for chunk in engine.stream(
            messages, max_tokens=max_tokens, temp=temperature, top_p=top_p,
            tools=tools, prompt_cache=cache_for_gen,
        ):
            buf += chunk.text
            n_out += 1
        _sync_metal()
        _, answer, _ = _split_channels(buf)

        content_blocks, parsed_stop = parse_content_blocks(answer)

        stats = engine.last_stats
        in_tok = stats.prompt_tokens if stats else 0
        out_tok = stats.generated_tokens if stats else n_out
        model_stop = _anthropic_stop_reason(stats.finish_reason if stats else "stop")
        stop_reason = parsed_stop if parsed_stop == "tool_use" else model_stop

        _finalize_cache(engine, cache_store, messages, tools, prompt_cache, out_tok)

        _log_stats(model_name, resolved_target, "POST /v1/messages  (sync) ", stats, stop_reason, n_out,
                   prefix_hit=prefix_len, cache_stats=cache_store.stats())

        self._send_json(200, {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        })

    def _stream_response(
        self,
        engine: Engine,
        cache_store: PromptCacheStore,
        messages: list[dict],
        tools: list[dict] | None,
        model_name: str,
        resolved_target: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        prompt_cache, prefix_len = _prepare_cache(engine, cache_store, messages, tools)
        cache_for_gen = prompt_cache

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(event: str, data: dict) -> None:
            line = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        parser = StreamingToolParser(send, clean_text_fn=_split_channels)

        try:
            send("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })
            send("ping", {"type": "ping"})

            n_out = 0
            last_ping = time.time()

            for chunk in engine.stream(
                messages, max_tokens=max_tokens, temp=temperature, top_p=top_p,
                tools=tools, prompt_cache=cache_for_gen,
            ):
                n_out += 1
                parser.push(chunk.text)
                if parser.should_stop():
                    break
                if time.time() - last_ping > 10:
                    send("ping", {"type": "ping"})
                    last_ping = time.time()
            _sync_metal()

            parser.flush()

            stats = engine.last_stats
            in_tok = stats.prompt_tokens if stats else 0
            out_tok = stats.generated_tokens if stats else n_out
            model_stop = _anthropic_stop_reason(stats.finish_reason if stats else "stop")
            stop_reason = parser.stop_reason()
            if stop_reason != "tool_use":
                stop_reason = model_stop

            _finalize_cache(engine, cache_store, messages, tools, prompt_cache, out_tok)

            _log_stats(model_name, resolved_target, "POST /v1/messages (stream)", stats, stop_reason, n_out,
                       prefix_hit=prefix_len, cache_stats=cache_store.stats())

            send("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            })
            send("message_stop", {"type": "message_stop"})
        except (BrokenPipeError, ConnectionResetError):
            _log("client disconnected mid-stream")


class _Server(HTTPServer):
    # No ThreadingMixIn — all requests handled on the main thread.
    # Metal GPU callbacks fire on Apple dispatch threads that have no Python
    # ThreadState. With threading, a callback during select() or handler
    # thread teardown crashes CPython. Running synchronously keeps the GIL
    # held during generation (when callbacks fire) and only releases it
    # during the idle select() — at which point all Metal work has been
    # drained by mx.synchronize().
    allow_reuse_address = True
    timeout = 0.5


def _kill_tqdm_monitor() -> None:
    """Belt-and-suspenders: if tqdm's TMonitor thread started before our env
    var took effect, kill it now. The thread polls progress bars and can
    receive late Metal callbacks without the GIL → fatal crash."""
    try:
        import tqdm
        mon = getattr(tqdm.tqdm, "monitor", None)
        if mon is not None and hasattr(mon, "kill"):
            mon.kill()
            tqdm.tqdm.monitor = None
    except Exception:
        pass


def run_server(
    host: str = _D.SERVER_HOST,
    port: int = _D.SERVER_PORT,
    default_model: str = _D.MODEL,
    verbose: bool = False,
    pin: bool = False,
) -> None:
    global _DEFAULT_MODEL, _VERBOSE, _PINNED
    _DEFAULT_MODEL = default_model
    _VERBOSE = verbose
    _PINNED = pin
    _kill_tqdm_monitor()

    # Resolve the default alias early so we fail fast on bad config.
    target = _resolve(default_model)
    aliases = load_aliases()
    resolved = aliases.get(default_model, target)

    srv = _Server((host, port), _Handler)
    url = f"http://{host}:{port}"
    pin_line = "  mode:           pinned (all requests → default model)\n" if pin else ""
    banner = (
        f"\n  minimlx server  listening on {url}\n"
        f"  default model:  {default_model}  ({resolved})\n"
        f"{pin_line}\n"
        f"  wire Claude Code to this server:\n\n"
        f"    export ANTHROPIC_BASE_URL={url}\n"
        f"    export ANTHROPIC_AUTH_TOKEN=minimlx\n"
        f"    claude --model {default_model}\n\n"
        f"  ctrl-c to stop\n"
    )
    print(banner, flush=True)

    # Route SIGTERM through the same clean-shutdown path as Ctrl-C so that
    # `kill` / `pkill` / supervisor-sent TERM signals let atexit handlers run
    # and release multiprocessing resources instead of leaking them.
    import signal

    def _term(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _term)
    except (OSError, ValueError):
        pass

    try:
        # Single-threaded request loop with Metal synchronisation between
        # requests. Without threading, generation runs on the main thread
        # which holds the GIL → Metal callbacks land safely. Between
        # requests we synchronise to drain any residual GPU work before
        # entering the select() idle wait (which releases the GIL).
        import mlx.core as mx
        while True:
            srv.handle_request()
            try:
                mx.synchronize()
            except Exception:
                pass
    except KeyboardInterrupt:
        print("\n  stopping…", flush=True)
    finally:
        srv.server_close()
