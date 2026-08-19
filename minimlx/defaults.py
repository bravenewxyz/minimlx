"""Centralized default values referenced across the codebase.

Edit constants here; consumers (cli.py, server.py, engine.py, aliases.py)
import them. The model registry — alias name → HF repo id — still lives
in `aliases.py`; this module is only for defaults and filesystem layout.
"""
from __future__ import annotations
from pathlib import Path

# --- Filesystem layout ----------------------------------------------------

CONFIG_DIR = Path.home() / ".minimlx"
DB_PATH = CONFIG_DIR / "logs.db"
ALIASES_FILE = CONFIG_DIR / "aliases.toml"
LOCAL_MODELS_DIR = Path.home() / ".cache" / "minimlx" / "models"

# --- Default model aliases ------------------------------------------------

# Default text model. Routed through the alias map in aliases.py.
#
# Qwen 3.8 27B, abliterated, in MTPLX's mixed 4/8-bit build: it carries its own
# multi-token-prediction head, so it speculates without a second model resident
# and without the `--draft` path Qwen 3.8 cannot use anyway (its linear-
# attention layers have a non-trimmable KV cache, which mlx-lm's speculative
# decoder rejects outright). Measured M5 Max, cold, 300 tok: ~50 tok/s decode
# at depth 3 under the build's `turbo` profile, ~21 GB resident.
#
# Requires the `mtplx` extra. `gemma4-moe` remains a `-m` away and is the
# fallback worth reaching for if mlx ever has to go back below 0.32.
MODEL = "qwen38-27b-mtplx"
STT_MODEL = "granite-stt"

# --- Sampling ------------------------------------------------------------

TEMP = 0.7
TOP_P = 0.95
# Lower temp for code-task one-shots — output should be more deterministic.
CODE_TEMP = 0.3

# --- Generation budget ---------------------------------------------------

# -1 = unlimited; engine clamps internally if a draft is in play.
MAX_TOKENS = -1
CODE_MAX_TOKENS = 4096

# --- Speculative decoding -------------------------------------------------

# Per-iteration draft tokens for the legacy mlx-lm spec path. DFlash
# ignores values ≤4 (uses the draft's config block size); MTP ignores
# this entirely (block size is baked into the assistant model).
NUM_DRAFT_TOKENS = 4

# --- Server ---------------------------------------------------------------

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1234

# --- Voice conversation ---------------------------------------------------

SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 1.5

# --- Apple Silicon residency ---------------------------------------------

# `mx.set_wired_limit(bytes)` pins this much unified memory so the OS
# doesn't page hot weights out mid-generation. 48 GB covers the bf16
# 31B target on a 128 GB box; harmless on smaller machines (call is
# best-effort and silently no-ops if it would exceed system memory).
WIRED_LIMIT_BYTES = int(48 * 1024 ** 3)
