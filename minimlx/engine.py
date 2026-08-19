from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from minimlx import defaults as _D
from minimlx.defaults import WIRED_LIMIT_BYTES as _WIRED_LIMIT_BYTES
from minimlx.models import resolve as _resolve_repo


def _is_dflash_repo(repo_id: str | None) -> bool:
    """Heuristic: is `repo_id` a z-lab DFlash block-diffusion draft?

    DFlash drafts ship as `z-lab/<base>-DFlash` (or `-DFlash-bN`). We match
    on the trailing path component so local-path overrides (e.g. a converted
    copy under ~/.cache/minimlx/models/foo-DFlash-4bit) still route correctly.
    """
    if not repo_id:
        return False
    tail = repo_id.rstrip("/").rsplit("/", 1)[-1].lower()
    return "dflash" in tail


def _is_mtp_repo(repo_id: str | None) -> bool:
    """Heuristic: is `repo_id` a Google Multi-Token-Prediction "assistant" drafter?

    Google's MTP drafters ship as `mlx-community/<base>-it-assistant-bf16`.
    They route through mlx-vlm's `draft_kind="mtp"` path, not mlx-lm's nor
    DFlash's. We match on `assistant` or a trailing `-mtp` suffix so local
    aliases (e.g. `gemma4-moe-mtp`) resolve correctly.
    """
    if not repo_id:
        return False
    tail = repo_id.rstrip("/").rsplit("/", 1)[-1].lower()
    return "assistant" in tail or tail.endswith("-mtp")


def _is_mtplx_repo(repo_id: str | None) -> bool:
    """Heuristic: is `repo_id` an MTPLX build with a native MTP head?

    MTPLX artifacts carry the multi-token-prediction head in the model repo
    itself (`mtp.safetensors` alongside the trunk) instead of pairing with a
    separate draft model, and they ship as `<base>-MTPLX-<profile>`. They
    need the `mtplx` runtime to use that head; plain mlx-lm loads the same
    weights but silently drops the MTP tensors and decodes autoregressively.
    """
    if not repo_id:
        return False
    tail = repo_id.rstrip("/").rsplit("/", 1)[-1].lower()
    return "mtplx" in tail


class _DropMtplxChatter:
    """Stdout proxy that swallows mtplx's unsolicited `[mtplx] ...` lines.

    mtplx prints a compiled-verify prewarm report straight to stdout from
    inside the verify path — no env gate, no logger. That fires on the
    generation thread mid-stream, so it lands in the middle of a rendered
    answer ("The[mtplx] compiled-verify prewarm {...} user is asking..."). It
    cannot be redirected per-thread, since `sys.stdout` is process-global and
    the consumer is rendering concurrently. Filtering the one prefix is the
    surgical option: `print()` resolves `sys.stdout` when it is called, while
    rich's Console captured the real stream when it was built, so this
    intercepts mtplx's chatter and nothing else.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._dropped = False

    def write(self, text: str) -> int:
        if text.startswith("[mtplx] "):
            self._dropped = True
            return len(text)
        # `print()` emits its terminator as a separate write; drop the
        # newline belonging to a line we just swallowed.
        if self._dropped:
            self._dropped = False
            if text == "\n":
                return 1
        return self._wrapped.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _mtplx_drop_caches() -> None:
    """Let go of the compiled verify graphs that pin an MTPLX model's weights.

    mtplx memoizes compiled verify steps and Metal kernels in module globals,
    and those closures hold the model. Dropping the runtime and running a full
    gc frees exactly nothing — measured on the 27B build, active unified
    memory sat at 20.37 GB after `rt = None; gc.collect(); mx.clear_cache()`
    and fell to 0.00 GB the moment these were cleared.

    There is no public teardown API, so this reaches for private names and
    shrugs off anything that has moved. The cost of missing them is a model's
    worth of memory held until the process exits; the cost of clearing them
    needlessly is recompiling a handful of kernels.
    """
    import gc
    import sys

    for module, attr in (
        ("mtplx.graphbank", "_SHARED_VERIFY_STEPS"),
        ("mtplx.graphbank", "_PREWARMED_BUCKETS"),
        ("mtplx.verify_kernels", "_KERNEL_CACHE"),
    ):
        try:
            getattr(sys.modules[module], attr).clear()
        except Exception:
            pass
    gc.collect()


def _quiet_mtplx_stdout() -> None:
    """Install the stdout filter once per process."""
    import sys

    if not isinstance(sys.stdout, _DropMtplxChatter):
        sys.stdout = _DropMtplxChatter(sys.stdout)


def _mtplx_meta(path: str) -> dict:
    """Load an MTPLX build's `mtplx_runtime.json`, or `{}` if it has none."""
    import json
    from pathlib import Path as _Path

    try:
        return json.loads((_Path(path) / "mtplx_runtime.json").read_text())
    except Exception:
        return {}


def _mtplx_depth_for(meta: dict, default: int = 3) -> int:
    """Pick the speculative depth for an MTPLX build.

    `mtplx_runtime.json` records both the depth the publisher measured as
    fastest (`speed_evidence.depth`) and the ceiling the MTP head was built
    for (`mtp_depth_max`). Prefer the measured depth, clamped to the ceiling
    — drafting deeper than the head supports just burns forward passes on
    tokens it cannot propose. Optimal depth is machine-specific, so
    `MINIMLX_MTPLX_DEPTH` overrides both.
    """
    import os

    override = os.environ.get("MINIMLX_MTPLX_DEPTH", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    ceiling = int(meta.get("mtp_depth_max") or default)
    depth = default
    evidence = meta.get("speed_evidence")
    if isinstance(evidence, dict) and evidence.get("depth"):
        depth = int(evidence["depth"])
    return max(1, min(depth, ceiling))


@dataclass
class Chunk:
    text: str
    token: int
    tps: float
    # Number of decoded tokens this chunk's `text` covers. Always 1 for the
    # vanilla mlx-lm path; can be >1 for DFlash/MTP block-decoding where
    # each iteration yields multiple accepted tokens at once. Live UI
    # renderers should add this (not 1) to their running counter.
    n_tokens: int = 1


@dataclass
class Stats:
    prompt_tokens: int
    generated_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    finish_reason: str


class Engine:
    """Thin MLX-LM wrapper. Load once, stream forever.

    Lazy-imports mlx_lm at first use so that `minimlx --help` and cache
    management commands don't pay the import cost.
    """

    def __init__(
        self,
        model_id: str,
        draft_model_id: Optional[str] = None,
        num_draft_tokens: int = 4,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ):
        self.model_id = model_id
        self.draft_model_id = draft_model_id
        self.num_draft_tokens = num_draft_tokens
        self.max_kv_size = max_kv_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self._loaded = False
        self._last_stats: Optional[Stats] = None
        self._is_dflash = _is_dflash_repo(draft_model_id)
        self._is_mtp = _is_mtp_repo(draft_model_id)
        # MTPLX is a property of the *target*, not the draft: the MTP head
        # rides along inside the model repo.
        self._is_mtplx = _is_mtplx_repo(model_id)
        self._mvlm_processor: Any = None  # mlx-vlm processor when MTP is active
        self._mtplx_req: Any = None       # job queue into the mtplx worker thread
        self._mtplx_depth = 0             # tuned speculative depth for that runtime
        self._mtplx_opts: dict = {}       # verify config the profile env requires

    def load(self) -> None:
        if self._loaded:
            return
        self._configure_download_progress()

        if self._is_mtplx:
            self._load_mtplx()
            return

        if self._is_mtp:
            # MTP needs mlx-vlm's wrapper because the drafter consumes the
            # target's last-layer hidden state via shared_kv hooks that are
            # only exposed on `mlx_vlm`'s wrapped models. Both target and
            # drafter come from mlx-vlm in this mode.
            try:
                from mlx_vlm.utils import load as mvlm_load
                from mlx_vlm.speculative import load_drafter as mvlm_load_drafter
            except ImportError as e:
                raise ImportError(
                    "MTP draft requested but mlx-vlm 0.5.0+ is unavailable. "
                    "Install with:\n"
                    "  pip install \"mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm\""
                ) from e
            try:
                import mlx.core as _mx
                _mx.set_wired_limit(_WIRED_LIMIT_BYTES)
            except Exception:
                pass
            self.model, self._mvlm_processor = mvlm_load(_resolve_repo(self.model_id))
            self.tokenizer = (
                self._mvlm_processor.tokenizer
                if hasattr(self._mvlm_processor, "tokenizer")
                else self._mvlm_processor
            )
            self.draft_model = mvlm_load_drafter(self.draft_model_id, kind="mtp")
            self._loaded = True
            return

        from mlx_lm import load as mlx_load
        # On Apple Silicon, weights paged out by Metal cause spurious slowdowns
        # mid-generation. Wire a generous chunk so 30B-class targets stay
        # resident. Best-effort — the call is unsupported on older mlx builds.
        try:
            import mlx.core as _mx
            _mx.set_wired_limit(_WIRED_LIMIT_BYTES)
        except Exception:
            pass
        self.model, self.tokenizer = mlx_load(_resolve_repo(self.model_id))
        self.draft_model = None
        if self.draft_model_id:
            if self._is_dflash:
                try:
                    from dflash.model_mlx import load_draft as dflash_load_draft
                except ImportError as e:
                    raise ImportError(
                        "DFlash draft requested but the `dflash` package is not "
                        "installed. Install with:\n"
                        "  pip install \"dflash[mlx] @ git+https://github.com/z-lab/dflash\"\n"
                        f"(original error: {e})"
                    ) from e
                self.draft_model = dflash_load_draft(_resolve_repo(self.draft_model_id))
            else:
                self.draft_model, _ = mlx_load(_resolve_repo(self.draft_model_id))
        self._loaded = True

    def _load_mtplx(self) -> None:
        """Load an MTPLX artifact onto a thread that then owns it for good.

        MTPLX builds carry their multi-token-prediction head in the model repo
        (`mtp.safetensors` beside the trunk), so the model drafts ahead of
        itself with no second model resident and verifies each block by
        rejection sampling — the output distribution is unchanged, only the
        wall-clock moves. mlx-lm reads the same trunk happily but drops the
        MTP tensors in `sanitize()`, losing the entire speedup; hence a
        dedicated backend rather than a draft-model pairing.

        Everything that touches MLX happens on one long-lived worker thread,
        because the runtime profile installs verify-specialized Metal kernels
        that are thread-affine. Loading on one thread and generating on
        another, or letting a per-request thread exit underneath them, aborts
        the process (SIGTRAP, "PyThreadState_Get ... the GIL is released") —
        and does it *after* printing a perfectly good answer, which is a
        miserable way to find out. A single thread that loads, generates, and
        never exits keeps the profile's throughput and live streaming both.
        """
        import queue
        import threading

        try:
            import mtplx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MTPLX model requested but the `mtplx` package is not installed. "
                "Install with:\n"
                "  pip install mtplx\n"
                "It pulls mlx>=0.32, above dflash's declared mlx==0.31.2 pin; "
                "the `*-dflash` drafts run fine there in practice.\n"
                f"(original error: {e})"
            ) from e

        _quiet_mtplx_stdout()

        # `mtplx.load` takes a directory, not a Hub id, so materialize first.
        path = _resolve_repo(self.model_id, local_dir=True)
        meta = _mtplx_meta(path)

        req: Any = queue.Queue()
        self._mtplx_req = req
        ready: Any = queue.Queue()
        # The worker takes the queue as an argument and holds it for life.
        # It must not reach back through `self`: `release()` clears
        # `self._mtplx_req`, and a worker reading the attribute each loop
        # would raise, exit, and take the thread-affine verify kernels down
        # with it — the SIGTRAP this whole design exists to avoid.
        threading.Thread(
            target=self._mtplx_worker,
            args=(path, meta, ready, req),
            name="mtplx-runtime",
            daemon=True,
        ).start()

        kind, payload = ready.get()
        if kind == "error":
            self._mtplx_req = None
            raise payload
        self.tokenizer, self._mtplx_depth, self._mtplx_opts = payload
        self.model = None       # the runtime lives on the worker, not here
        self.draft_model = None
        self._loaded = True

    def _mtplx_worker(self, path: str, meta: dict, ready: Any, req: Any) -> None:
        """Own an MTPLX runtime for this Engine's lifetime. Never returns."""
        import os

        opts: dict = {}
        try:
            # A build names the runtime profile it was tuned against, and a
            # profile is nothing but env vars: verify-specialized matmul
            # kernels, compiled verify, packed-GQA attention. They are read
            # while the model loads and while kernels compile, so they have to
            # be set before the load.
            #
            # The profile and the generate options are one unit, not two
            # knobs. `turbo` sets MTPLX_SKIP_VERIFY_SNAPSHOT=1, which is only
            # correct under the verify strategy mtplx's own CLI pairs it with;
            # applying the profile and then calling `generate_mtpk` with its
            # conservative library defaults raises "capture commit failed" on
            # the first rejected block. Applied together they are worth ~1.9x
            # on this model (measured M5 Max, 27B Optimized-Speed, depth 3:
            # 26 -> 50 tok/s), which is most of the point of an MTPLX build.
            #
            # `MINIMLX_MTPLX_SAFE=1` falls back to mtplx's library defaults,
            # for an artifact whose architecture these settings don't suit.
            profile = meta.get("recommended_profile")
            if profile and not os.environ.get("MINIMLX_MTPLX_SAFE", "").strip():
                try:
                    from mtplx.profiles import apply_profile_env

                    apply_profile_env(str(profile))
                    opts = {
                        "verify_strategy": "capture_commit",
                        "verify_core": "linear-gdn-from-conv-tape",
                        "mtp_history_policy": "committed",
                    }
                except Exception:
                    opts = {}

            import mlx.core as mx

            try:
                mx.set_wired_limit(_WIRED_LIMIT_BYTES)
            except Exception:
                pass

            import mtplx

            rt = mtplx.load(path, mtp=True)
            ready.put(("ready", (rt.tokenizer, _mtplx_depth_for(meta), opts)))
        except BaseException as exc:
            ready.put(("error", exc))
            return

        from mtplx.generation import generate_mtpk
        from mtplx.sampling import SamplerConfig

        while True:
            job = req.get()
            ack = job.get("release")
            if ack is not None:
                # Released. Drop the weights but stay parked — exiting this
                # thread would tear the verify kernels down with it.
                rt = None
                _mtplx_drop_caches()
                try:
                    mx.clear_cache()
                except Exception:
                    pass
                ack.put(True)
                continue

            out_q = job["out"]
            if rt is None:
                out_q.put(("error", RuntimeError("this MTPLX runtime was released")))
                continue
            try:
                mx.reset_peak_memory()
            except Exception:
                pass

            # Detokenize here rather than in the consumer: MLX drops the GIL
            # across kernel launches, and keeping every tokenizer touch on the
            # thread that owns the model leaves the consumer nothing but
            # finished strings to handle.
            detok = rt.tokenizer.detokenizer
            detok.reset()

            def _on_tokens(tokens: list[int], _q: Any = out_q) -> None:
                for tok in tokens:
                    detok.add_token(int(tok))
                _q.put(("chunk", (detok.last_segment, len(tokens),
                                  int(tokens[-1]) if tokens else 0)))

            try:
                out = generate_mtpk(
                    rt,
                    job["prompt_ids"],
                    max_tokens=job["max_tokens"],
                    sampler=SamplerConfig(temperature=job["temp"], top_p=job["top_p"]),
                    speculative_depth=self._mtplx_depth,
                    seed=job["seed"],
                    token_callback=_on_tokens,
                    abort_check=job["stop"].is_set,
                    **opts,
                )
                detok.finalize()
                out_q.put(("done", (out, detok.last_segment)))
            except BaseException as exc:
                out_q.put(("error", exc))

    def release(self) -> None:
        """Drop model weights held by a background runtime, if any.

        Callers that swap models mid-process (chat's `/models`, the server's
        model switch) drop the old Engine first so peak unified memory doesn't
        carry two models at once. That is not enough for the MTPLX backend:
        its weights belong to a worker thread, which cannot simply be retired
        because exiting it tears down thread-affine Metal kernels and aborts
        the process. So ask it to let go instead. Safe to call on any Engine,
        any number of times.
        """
        import queue

        req, self._mtplx_req = self._mtplx_req, None
        if req is None:
            return
        # Wait for the worker to actually let go: the caller's next move is
        # loading another model, and the whole point is not to hold two at
        # once.
        ack: Any = queue.Queue()
        req.put({"release": ack})
        try:
            ack.get(timeout=60)
        except Exception:
            pass

    def _configure_download_progress(self) -> None:
        """Show Hugging Face download bars only when something actually needs
        downloading. When every model file is already cached the bars just
        flash degenerate `0.00B` lines and mangle later output, so suppress
        them in that case."""
        try:
            from huggingface_hub import try_to_load_from_cache
            from huggingface_hub.utils import (
                disable_progress_bars,
                enable_progress_bars,
            )
        except Exception:
            return

        def _ready(repo: str) -> bool:
            # Local paths and bare names have nothing to fetch from the Hub.
            if "/" not in repo or repo.startswith(("/", "~", ".")):
                return True
            from minimlx.models import split_subfolder

            repo_id, subfolder = split_subfolder(repo)
            probe = "config.json" if subfolder is None else f"{subfolder}/config.json"
            try:
                return isinstance(try_to_load_from_cache(repo_id, probe), str)
            except Exception:
                return True

        repos = [r for r in (self.model_id, self.draft_model_id) if r]
        try:
            if all(_ready(r) for r in repos):
                disable_progress_bars()
            else:
                enable_progress_bars()
        except Exception:
            pass

    def _build_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            # Retry without tools — some templates don't accept them.
            if tools:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
            flat: list[dict] = []
            sys_prefix = ""
            for m in messages:
                if m["role"] == "system":
                    sys_prefix += m["content"] + "\n\n"
                elif m["role"] == "user" and sys_prefix:
                    flat.append({"role": "user", "content": sys_prefix + m["content"]})
                    sys_prefix = ""
                else:
                    flat.append(m)
            if not flat and sys_prefix:
                flat.append({"role": "user", "content": sys_prefix.strip()})
            return self.tokenizer.apply_chat_template(
                flat, tokenize=False, add_generation_prompt=True
            )

    def tokenize_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, list[int]]:
        """Render messages through the chat template and tokenize the result.

        Returns (prompt_text, token_ids). The tokenized list is what the server's
        prompt-cache store uses for prefix matching.
        """
        self.load()
        prompt = self._build_prompt(messages, tools=tools)
        ids = self.tokenizer.encode(prompt)
        return prompt, list(ids)

    def stream(
        self,
        messages: list[dict],
        max_tokens: int = _D.MAX_TOKENS,
        temp: float = _D.TEMP,
        top_p: float = _D.TOP_P,
        tools: list[dict] | None = None,
        seed: int | None = None,
        prompt_cache: Any = None,
    ) -> Iterator[Chunk]:
        self.load()
        if self._is_mtplx:
            yield from self._stream_mtplx(
                messages,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                tools=tools,
                seed=seed,
            )
            return
        if self._is_mtp:
            yield from self._stream_mtp(
                messages,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                tools=tools,
                seed=seed,
            )
            return
        if self._is_dflash:
            yield from self._stream_dflash(
                messages,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                tools=tools,
                seed=seed,
            )
            return
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        # Reset the peak-memory counter so `last_stats.peak_memory_gb`
        # reflects *this* request, not the all-time process high water mark.
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

        prompt = self._build_prompt(messages, tools=tools)
        sampler = make_sampler(temp=temp, top_p=top_p)

        # Some chat templates (Qwen3.5 with reasoning) prepend `<think>\n` to
        # the assistant turn so the model starts already inside a thinking
        # block and the open marker never appears in the stream. Replay the
        # marker as a synthetic first chunk so downstream filters / tool
        # parsers see a complete `<think>...</think>` pair.
        prompt_tail = prompt.rstrip()
        synthetic_prefix: str | None = None
        if prompt_tail.endswith("<think>"):
            synthetic_prefix = "<think>\n"
        elif prompt_tail.endswith("<|channel>thought"):
            synthetic_prefix = "<|channel>thought\n"

        # If the caller supplied a pre-populated prompt cache (from a
        # PromptCacheStore hit), the cache's `offset` tells us how many
        # leading tokens of this prompt are already KV-filled. Slice them off
        # so `stream_generate` only prefills the *new* suffix — otherwise the
        # model would re-process the entire prompt on top of the cache.
        prompt_input: Any = prompt
        cache_offset = 0
        if prompt_cache is not None:
            try:
                cache_offset = int(getattr(prompt_cache[0], "offset", 0))
            except Exception:
                cache_offset = 0
        if cache_offset > 0:
            full_tokens = list(self.tokenizer.encode(prompt))
            if cache_offset >= len(full_tokens):
                # Cache already covers the whole prompt (edge case): send a
                # 1-token tail so stream_generate has something to decode.
                cache_offset = max(0, len(full_tokens) - 1)
            suffix = full_tokens[cache_offset:]
            prompt_input = mx.array(suffix)

        # mlx_lm.speculative_generate_step computes
        #   num_draft = min(max_tokens - ntoks, num_draft_tokens)
        # which goes negative when max_tokens == -1 (unlimited), producing an
        # empty draft batch and crashing on `mx.concatenate([])`. Clamp to a
        # large finite cap whenever a draft model is in play.
        effective_max_tokens = max_tokens
        if self.draft_model is not None and max_tokens < 0:
            effective_max_tokens = 1_000_000

        kwargs: dict = dict(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_input,
            max_tokens=effective_max_tokens,
            sampler=sampler,
            kv_group_size=self.kv_group_size,
        )
        if prompt_cache is not None:
            kwargs["prompt_cache"] = prompt_cache
        if self.draft_model is not None:
            kwargs["draft_model"] = self.draft_model
            kwargs["num_draft_tokens"] = self.num_draft_tokens
        if self.max_kv_size is not None:
            kwargs["max_kv_size"] = self.max_kv_size
        if self.kv_bits is not None:
            kwargs["kv_bits"] = self.kv_bits
            kwargs["quantized_kv_start"] = self.quantized_kv_start

        if synthetic_prefix is not None:
            yield Chunk(text=synthetic_prefix, token=-1, tps=0.0, n_tokens=0)

        last = None
        for r in stream_generate(**kwargs):
            last = r
            yield Chunk(
                text=getattr(r, "text", ""),
                token=getattr(r, "token", 0),
                tps=getattr(r, "generation_tps", 0.0),
            )
        if last is not None:
            # Prefer the live MLX counter (active unified-memory high water
            # mark) since `GenerationResponse.peak_memory` is zeroed in recent
            # mlx-lm. Fall back to the response attribute if the MLX call is
            # missing in some build.
            try:
                peak_bytes = int(mx.get_peak_memory())
            except Exception:
                peak_bytes = int(getattr(last, "peak_memory", 0))
            self._last_stats = Stats(
                prompt_tokens=int(getattr(last, "prompt_tokens", 0)),
                generated_tokens=int(getattr(last, "generation_tokens", 0)),
                prompt_tps=float(getattr(last, "prompt_tps", 0.0)),
                generation_tps=float(getattr(last, "generation_tps", 0.0)),
                peak_memory_gb=peak_bytes / 1e9,
                finish_reason=str(getattr(last, "finish_reason", "unknown")),
            )

    def _stream_mtplx(
        self,
        messages: list[dict],
        max_tokens: int,
        temp: float,
        top_p: float,
        tools: list[dict] | None,
        seed: int | None,
    ) -> Iterator[Chunk]:
        """Stream through the MTPLX runtime's native multi-token prediction.

        The runtime lives on the worker thread started by `_load_mtplx`; this
        hands it a job and drains the results. A result payload is a *delta* —
        the tokens accepted that round, stop tokens already stripped — which
        is why `n_tokens` is the block length: at draft depth k one round can
        commit up to k+1 tokens, and a live counter adding 1 per chunk would
        undercount by the exact factor MTPLX exists to deliver.
        """
        import queue
        import secrets
        import threading
        import time

        prompt = self._build_prompt(messages, tools=tools)
        prompt_ids = list(self.tokenizer.encode(prompt))

        prompt_tail = prompt.rstrip()
        synthetic_prefix: str | None = None
        if prompt_tail.endswith("<think>"):
            synthetic_prefix = "<think>\n"
        elif prompt_tail.endswith("<|channel>thought"):
            synthetic_prefix = "<|channel>thought\n"

        if self._mtplx_req is None:
            raise RuntimeError("this MTPLX runtime was released")

        out_q: Any = queue.Queue()
        # A consumer that stops early (Ctrl-C, chat `/stop`, a client
        # disconnect) would otherwise leave the worker generating against a
        # 27B model with nobody reading. `abort_check` is polled per round.
        stop = threading.Event()
        self._mtplx_req.put({
            "prompt_ids": prompt_ids,
            "max_tokens": max_tokens if max_tokens > 0 else 1_000_000,
            "temp": temp,
            "top_p": top_p,
            # mtplx seeds its own RNG and defaults to 0, so leaving it unset
            # would make every unseeded generation identical. Draw one.
            "seed": secrets.randbelow(2 ** 31) if seed is None else int(seed),
            "out": out_q,
            "stop": stop,
        })

        if synthetic_prefix is not None:
            yield Chunk(text=synthetic_prefix, token=-1, tps=0.0, n_tokens=0)

        started = time.perf_counter()
        produced = 0
        out = None
        tail = ""
        try:
            while True:
                kind, payload = out_q.get()
                if kind == "error":
                    raise payload
                if kind == "done":
                    out, tail = payload
                    break
                text, n_tokens, last_token = payload
                produced += n_tokens
                elapsed = time.perf_counter() - started
                yield Chunk(
                    text=text,
                    token=last_token,
                    tps=produced / elapsed if elapsed > 0 else 0.0,
                    n_tokens=n_tokens,
                )
        finally:
            stop.set()

        if tail:
            yield Chunk(text=tail, token=-1, tps=0.0, n_tokens=0)

        import mlx.core as mx

        try:
            peak_bytes = int(mx.get_peak_memory())
        except Exception:
            peak_bytes = 0
        stats = getattr(out, "stats", None)
        # `decode_tok_s` is the steady-state decode rate; `tok_s` folds in
        # prefill, which would understate the gain on short generations.
        gen_tps = float(
            getattr(stats, "decode_tok_s", 0.0) or getattr(stats, "tok_s", 0.0) or 0.0
        )
        self._last_stats = Stats(
            prompt_tokens=len(prompt_ids),
            generated_tokens=int(getattr(stats, "generated_tokens", produced) or produced),
            prompt_tps=float(getattr(stats, "prompt_tps", 0.0) or 0.0),
            generation_tps=gen_tps,
            peak_memory_gb=peak_bytes / 1e9,
            finish_reason=str(getattr(out, "finish_reason", None) or "stop"),
        )

    def _stream_mtp(
        self,
        messages: list[dict],
        max_tokens: int,
        temp: float,
        top_p: float,
        tools: list[dict] | None,
        seed: int | None,
    ) -> Iterator[Chunk]:
        """Stream through Google's Multi-Token-Prediction drafter via mlx-vlm.

        Requires mlx-vlm 0.5.0+ (Blaizzy/mlx-vlm main; not yet in v0.4.4).
        The drafter consumes the target's last-layer hidden state via
        `shared_kv` hooks in mlx-vlm's speculative path — that's why the
        target itself must be loaded through `mlx_vlm.utils.load`, not
        `mlx_lm.load`.
        """
        import mlx_vlm
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

        prompt = self._build_prompt(messages, tools=tools)

        prompt_tail = prompt.rstrip()
        synthetic_prefix: str | None = None
        if prompt_tail.endswith("<think>"):
            synthetic_prefix = "<think>\n"
        elif prompt_tail.endswith("<|channel>thought"):
            synthetic_prefix = "<|channel>thought\n"

        effective_max_tokens = max_tokens if max_tokens > 0 else 1_000_000

        if synthetic_prefix is not None:
            yield Chunk(text=synthetic_prefix, token=-1, tps=0.0, n_tokens=0)

        last = None
        for r in mlx_vlm.stream_generate(
            self.model,
            self._mvlm_processor,
            prompt,
            max_tokens=effective_max_tokens,
            temperature=temp,
            top_p=top_p,
            draft_model=self.draft_model,
            draft_kind="mtp",
        ):
            # mlx-vlm reports a running total in `generation_tokens`. The
            # delta per yield is the right `n_tokens` for live UI counting,
            # since MTP can flush multiple accepted tokens through the
            # detokenizer in a single iteration.
            cur_total = int(getattr(r, "generation_tokens", 0) or 0)
            prev_total = int(getattr(last, "generation_tokens", 0) or 0) if last is not None else 0
            n_tokens = max(1, cur_total - prev_total)
            last = r
            yield Chunk(
                text=getattr(r, "text", ""),
                token=int(getattr(r, "token", 0) or 0),
                tps=float(getattr(r, "generation_tps", 0.0)),
                n_tokens=n_tokens,
            )

        if last is not None:
            try:
                peak_bytes = int(mx.get_peak_memory())
            except Exception:
                peak_bytes = int(getattr(last, "peak_memory", 0) * 1e9)
            self._last_stats = Stats(
                prompt_tokens=int(getattr(last, "prompt_tokens", 0)),
                generated_tokens=int(getattr(last, "generation_tokens", 0)),
                prompt_tps=float(getattr(last, "prompt_tps", 0.0)),
                generation_tps=float(getattr(last, "generation_tps", 0.0)),
                peak_memory_gb=peak_bytes / 1e9,
                finish_reason=str(getattr(last, "finish_reason", None) or "stop"),
            )

    def _stream_dflash(
        self,
        messages: list[dict],
        max_tokens: int,
        temp: float,
        top_p: float,
        tools: list[dict] | None,
        seed: int | None,
    ) -> Iterator[Chunk]:
        """Stream through z-lab/dflash's block-diffusion speculative decoder.

        DFlash builds its own KV caches (target + draft) inside `stream_generate`
        and does not accept an external `prompt_cache`, so the server's
        cross-request prompt-cache reuse is bypassed for DFlash drafts. This is
        a deliberate tradeoff — DFlash's draft acceptance gain dominates the
        cache-reuse savings on the workloads it targets.
        """
        from dflash.model_mlx import stream_generate as dflash_stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        try:
            mx.reset_peak_memory()
        except Exception:
            pass

        prompt = self._build_prompt(messages, tools=tools)
        # At temp=0 a top_p sampler still pays for an `mx.sort` per call —
        # pass sampler=None and let DFlash use its cheap internal greedy
        # (`make_sampler(temp=0)` returns argmax). At temp>0 we keep the
        # caller's top_p in play.
        sampler = None if temp <= 0.0 else make_sampler(temp=temp, top_p=top_p)

        prompt_tail = prompt.rstrip()
        synthetic_prefix: str | None = None
        if prompt_tail.endswith("<think>"):
            synthetic_prefix = "<think>\n"
        elif prompt_tail.endswith("<|channel>thought"):
            synthetic_prefix = "<|channel>thought\n"

        # DFlash's stream_generate requires a positive max_tokens (it loops
        # `while n < max_tokens`). Map -1 (unlimited) to a large finite cap.
        effective_max_tokens = max_tokens if max_tokens > 0 else 1_000_000

        # `block_size=None` lets DFlash use the value baked into the draft
        # config (16 for the z-lab Gemma4 / Qwen3.6 drafts), which is also the
        # value tuned in the upstream README. We piggyback on `num_draft_tokens`
        # only if the caller explicitly raised it past mlx_lm's default of 4 —
        # otherwise the small default would silently clip DFlash's block.
        block_size: int | None = None
        if self.num_draft_tokens and self.num_draft_tokens > 4:
            block_size = int(self.num_draft_tokens)

        if synthetic_prefix is not None:
            yield Chunk(text=synthetic_prefix, token=-1, tps=0.0, n_tokens=0)

        last = None
        for r in dflash_stream_generate(
            self.model,
            self.draft_model,
            self.tokenizer,
            prompt,
            block_size=block_size,
            max_tokens=effective_max_tokens,
            temperature=temp,
            sampler=sampler,
        ):
            last = r
            tok_list = getattr(r, "tokens", []) or [0]
            yield Chunk(
                text=getattr(r, "text", ""),
                token=int(tok_list[-1]) if tok_list else 0,
                tps=float(getattr(r, "generation_tps", 0.0)),
                n_tokens=max(1, len(tok_list)),
            )

        if last is not None:
            try:
                peak_bytes = int(mx.get_peak_memory())
            except Exception:
                peak_bytes = int(getattr(last, "peak_memory", 0) * 1e9)
            self._last_stats = Stats(
                prompt_tokens=int(getattr(last, "prompt_tokens", 0)),
                generated_tokens=int(getattr(last, "generation_tokens", 0)),
                prompt_tps=float(getattr(last, "prompt_tps", 0.0)),
                generation_tps=float(getattr(last, "generation_tps", 0.0)),
                peak_memory_gb=peak_bytes / 1e9,
                finish_reason=str(getattr(last, "finish_reason", None) or "stop"),
            )

    @property
    def last_stats(self) -> Optional[Stats]:
        return self._last_stats
