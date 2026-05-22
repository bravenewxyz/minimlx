from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from minimlx import defaults as _D
from minimlx.defaults import WIRED_LIMIT_BYTES as _WIRED_LIMIT_BYTES


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
        self._mvlm_processor: Any = None  # mlx-vlm processor when MTP is active

    def load(self) -> None:
        if self._loaded:
            return
        if self._try_jang_load():
            self._loaded = True
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
            self.model, self._mvlm_processor = mvlm_load(self.model_id)
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
        self.model, self.tokenizer = mlx_load(self.model_id)
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
                self.draft_model = dflash_load_draft(self.draft_model_id)
            else:
                self.draft_model, _ = mlx_load(self.draft_model_id)
        self._loaded = True

    def _try_jang_load(self) -> bool:
        """Detect and load JANG-quantized models (JANGTQ / MXQ / JANG v2).

        Returns True if the model was loaded via jang-tools, False otherwise.
        """
        from pathlib import Path
        import json

        # Resolve HF repo to a local snapshot path
        try:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(
                self.model_id,
                allow_patterns=[
                    "*.json", "*.safetensors", "*.model", "*.txt",
                    "tokenizer*", "*.py", "*.md", "*.jinja",
                ],
            ))
        except Exception:
            return False

        # Check for jang_config.json
        jang_cfg_path = path / "jang_config.json"
        if not jang_cfg_path.exists():
            return False

        jang_cfg = json.loads(jang_cfg_path.read_text())

        # Use jang-tools v2 loader (handles JANG, MXQ, and MXTQ formats)
        try:
            from jang_tools.loader import _load_jang_v2
        except ImportError:
            raise ImportError(
                "This model requires jang-tools. Install from: "
                "pip install /path/to/jang-tools  (see github.com/jangq-ai/jangq)"
            )

        self.model, self.tokenizer = _load_jang_v2(path, jang_cfg)
        self.draft_model = None
        return True

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
