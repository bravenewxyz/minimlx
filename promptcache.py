"""Prompt KV-cache store for the server.

Implements a per-engine trie-style LRU cache of `mlx_lm` prompt caches so
that repeated prompt prefixes skip the prefill step.

Covers three related optimisations from the server's perspective:

- **System-prompt cache** (item 6): when the user asks N questions with the
  same system prompt, the tokens for that system prompt are KV-computed once
  and reused for every follow-up. First-token latency drops from hundreds of
  ms to sub-millisecond.

- **Conversation KV cache** (item 12): across turns of the same conversation,
  the full prefix (system + prior user/assistant turns) is cached. Each new
  user turn only prefills the new tokens.

- **Prefix trie** (item 13): keys are token-id sequences. Lookup walks every
  entry and picks the longest common prefix, so multiple overlapping
  conversations share KV for anything they have in common.

**How it works**

Each cache entry stores:
- `tokens`: the exact token ids the KV cache corresponds to
- `cache`: the `mlx_lm` prompt cache (list of per-layer caches) after having
  been filled with those tokens
- `size`: rough byte size for LRU accounting
- `ts`: last-use timestamp

`lookup(tokens)` finds the entry with the longest common prefix, deep-copies
its cache (to decouple from future in-place generation), trims the copy back
to the common prefix length, and returns `(cache, prefix_len)`.

`store(tokens, cache)` saves a deep-copied cache after eviction to keep the
store under `max_bytes`.

**Caveats**

- Cloning uses `copy.deepcopy` which actually duplicates the underlying
  mx.arrays. A ~500 MB cache copies in ~1 ms on M5 Max, so the overhead is
  small but not free.
- Not all cache types are trimmable — `RotatingKVCache` once full is not.
  Those entries are rejected on lookup when trimming is needed.
- Entries are keyed per `model_id`; different models never share entries.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Entry:
    tokens: list[int]
    cache: Any
    size: int
    ts: float = field(default_factory=time.time)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _estimate_cache_bytes(cache: Any) -> int:
    """Rough byte count of a prompt cache. Walks per-layer .state attrs and
    sums nbytes. Returns 0 if the structure doesn't expose sizes."""
    total = 0
    try:
        for layer in cache:
            # Layer caches expose the stored keys/values. Different impls use
            # different attribute names; grab any mx.array we can see.
            for attr in ("keys", "values", "state", "_state"):
                obj = getattr(layer, attr, None)
                if obj is None:
                    continue
                if hasattr(obj, "nbytes"):
                    total += int(obj.nbytes)
                elif isinstance(obj, (tuple, list)):
                    for el in obj:
                        if hasattr(el, "nbytes"):
                            total += int(el.nbytes)
    except Exception:
        return total
    return total


class PromptCacheStore:
    """Per-engine LRU store of prompt caches keyed by token-id sequences."""

    def __init__(self, model_id: str, max_bytes: int = 8 * 1024 ** 3):
        self.model_id = model_id
        self.max_bytes = max_bytes
        self._entries: list[_Entry] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.saved_prefix_tokens = 0  # running total of tokens we skipped prefilling

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": sum(e.size for e in self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "saved_prefix_tokens": self.saved_prefix_tokens,
            }

    def lookup(self, tokens: list[int]) -> tuple[Any | None, int]:
        """Return a (cache, prefix_len) for the longest cached prefix of
        `tokens`, or (None, 0) on miss. The returned cache is independent of
        the stored entry and may be mutated by the caller."""
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        with self._lock:
            best: _Entry | None = None
            best_plen = 0
            for e in self._entries:
                plen = _common_prefix_len(e.tokens, tokens)
                if plen > best_plen:
                    best = e
                    best_plen = plen
            if best is None or best_plen == 0:
                self.misses += 1
                return None, 0

            # Need to trim if the stored entry has more tokens than the common
            # prefix. Check trimmability before we commit to the deep copy.
            trim_amount = len(best.tokens) - best_plen
            if trim_amount > 0 and not can_trim_prompt_cache(best.cache):
                self.misses += 1
                return None, 0

            # Deep-copy the cache so generation's in-place mutation doesn't
            # corrupt the stored entry.
            cloned = copy.deepcopy(best.cache)
            if trim_amount > 0:
                trim_prompt_cache(cloned, trim_amount)

            best.ts = time.time()
            self.hits += 1
            self.saved_prefix_tokens += best_plen
            return cloned, best_plen

    def store(self, tokens: list[int], cache: Any) -> None:
        """Store a cache after generation. We deep-copy on store so the server
        can keep mutating the original cache without affecting us.

        Refuses to store non-trimmable caches (e.g., Qwen's chunked cache)
        because later `lookup` calls can't adjust them to a shorter prefix,
        which would cause cache/prompt desynchronisation.
        """
        from mlx_lm.models.cache import can_trim_prompt_cache
        if not can_trim_prompt_cache(cache):
            return
        cloned = copy.deepcopy(cache)
        size = _estimate_cache_bytes(cloned)
        with self._lock:
            # If an exact-prefix entry already exists with shorter tokens,
            # replace it (the new one is strictly better).
            prune_idxs = []
            for i, e in enumerate(self._entries):
                if len(e.tokens) <= len(tokens) and _common_prefix_len(e.tokens, tokens) == len(e.tokens):
                    prune_idxs.append(i)
            for i in reversed(prune_idxs):
                self._entries.pop(i)

            self._entries.append(_Entry(tokens=list(tokens), cache=cloned, size=size))
            self._evict_locked()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0
            self.saved_prefix_tokens = 0

    def _evict_locked(self) -> None:
        """Keep total bytes under `max_bytes`, evicting oldest entries first."""
        total = sum(e.size for e in self._entries)
        if total <= self.max_bytes:
            return
        # Oldest first
        self._entries.sort(key=lambda e: e.ts)
        while self._entries and total > self.max_bytes:
            dropped = self._entries.pop(0)
            total -= dropped.size
        # Keep insertion-ordered (newest at end) from here on.
        self._entries.sort(key=lambda e: e.ts)
