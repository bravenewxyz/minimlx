from __future__ import annotations

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

# Re-export filesystem-layout constants so callers can import them from
# either `minimlx.aliases` or `minimlx.defaults`. Single source of truth
# is `defaults`.
from minimlx.defaults import CONFIG_DIR, ALIASES_FILE, LOCAL_MODELS_DIR  # noqa: F401

DEFAULTS: dict[str, str] = {
    # Gemma 4 family (dense + MoE)
    "gemma4":             "mlx-community/gemma-4-31b-it-8bit",
    "gemma4-max":         "mlx-community/gemma-4-31b-it-bf16",
    "gemma4-moe":         "mlx-community/gemma-4-26b-a4b-it-8bit",
    "gemma4-draft":       "mlx-community/gemma-4-e4b-it-4bit",

    # Gemma 4 MoE Claude-Opus distillation (TeichAI)
    "gemma4-moe-claude-opus":     str(LOCAL_MODELS_DIR / "gemma4-26b-claude-opus-distill-4bit"),
    "gemma4-moe-claude-opus-max": str(LOCAL_MODELS_DIR / "gemma4-26b-claude-opus-distill-bf16"),

    # Gemma 4 MoE abliterated (no mlx-community build yet;
    # locally converted from DuoNeural/Gemma-4-26B-A4B-Abliterated bf16)
    "gemma4-moe-abliterated": str(LOCAL_MODELS_DIR / "gemma4-26b-a4b-abliterated-4bit"),

    # Qwen 3.6 27B
    "qwen36-27b": "sabeshbesh/qwen3.6-27b-mlx-8bit",

    # Qwen 3.6 35B-A3B (MoE, 3B active)
    "qwen36-35b": "mlx-community/Qwen3.6-35B-A3B-8bit",

    # Qwen 3.6 35B-A3B Claude-Opus abliterated (no mlx-community build yet;
    # locally converted from huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.6-Opus-abliterated)
    "qwen36-35b-claude-opus-abliterated":     str(LOCAL_MODELS_DIR / "qwen36-35b-claude-opus-abliterated-4bit"),
    "qwen36-35b-claude-opus-abliterated-max": str(LOCAL_MODELS_DIR / "qwen36-35b-claude-opus-abliterated-bf16"),

    # Speech-to-text (IBM Granite 4.0 1B Speech — Apache 2.0, #2 on HF Open
    # ASR Leaderboard at 5.52 WER, en/fr/de/es/pt/ja/…). Auto language detect.
    "granite-stt": "ibm-granite/granite-4.0-1b-speech",

    # DFlash block-diffusion draft models (z-lab/dflash). The engine
    # auto-detects these by repo name and routes through the dflash MLX
    # backend instead of mlx_lm's vanilla speculative_generate_step.
    "gemma4-dflash":     "z-lab/gemma-4-31B-it-DFlash",
    "gemma4-moe-dflash": "z-lab/gemma-4-26B-A4B-it-DFlash",
    "qwen36-27b-dflash": "z-lab/Qwen3.6-27B-DFlash",
    "qwen36-35b-dflash": "z-lab/Qwen3.6-35B-A3B-DFlash",

    # 4-bit Gemma 4 targets — same family, ~half the bandwidth cost per
    # token vs 8-bit, so roughly 2× decode throughput on memory-bandwidth-
    # bound Apple Silicon. Pair with -dflash drafts for a stacked win.
    "gemma4-4bit":     "mlx-community/gemma-4-31b-it-4bit",
    "gemma4-moe-4bit": "mlx-community/gemma-4-26b-a4b-it-4bit",

    # Google's official Multi-Token-Prediction "assistant" drafters
    # (released 2026-05-06). Standard mlx-lm spec-decoding drafts — NOT
    # DFlash — so the engine routes through mlx_lm.stream_generate.
    # Smaller and lower-overhead than DFlash; Google reports ~2.2× on
    # 26B-A4B and ~3× on 31B Dense.
    "gemma4-mtp":     "mlx-community/gemma-4-31B-it-assistant-bf16",
    "gemma4-moe-mtp": "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
}


def load_aliases() -> dict[str, str]:
    aliases = dict(DEFAULTS)
    if ALIASES_FILE.exists():
        data = tomllib.loads(ALIASES_FILE.read_text())
        section = data.get("aliases", data)
        if isinstance(section, dict):
            aliases.update({k: str(v) for k, v in section.items()})
    return aliases


def resolve_alias(name: str, aliases: dict[str, str] | None = None) -> str:
    if aliases is None:
        aliases = load_aliases()
    return aliases.get(name, name)


# Per-alias inference defaults. CLI flags always win; a preset only fills in
# values the user didn't pass.
#
# History: presets used to pair `gemma4` with `gemma4-draft` (E4B) at
# `num_draft_tokens=2` for ~+13% over standalone. That preset broke when
# mlx-lm 0.31.3 tightened weight-loading checks against Gemma-4's layer-wise
# KV-reuse, and DFlash + MTP now dominate it by 5-9× anyway. Current
# defaults (measured M5 Max, gemma4-31B 8bit baseline = 14.5 tok/s):
#
#   gemma4 + gemma4-dflash          → 30.1 tok/s (2.08×)
#   gemma4-moe + gemma4-moe-dflash  → 112.9 tok/s (7.81×)
#   gemma4-moe-4bit + gemma4-moe-mtp→ 137.9 tok/s (9.55×)  (single-turn only)
#
# Why DFlash, not MTP, on the MoE chat default: MTP at temp>0 with the
# minimlx tool-use system prompt + `tools=` in the chat template diverges
# under sampling (drafter-target logit drift accumulates and the
# probability-ratio verifier accepts close-but-wrong tokens, leading to
# repetition loops in long generations). DFlash's strict token-equality
# verify is robust to this. MTP wins by ~4-15% on benchmarks but only on
# clean single-turn prompts at temp=0 — keep it as an opt-in via
# `--draft gemma4-moe-mtp` for those workloads.
#
# `gemma4-dflash` requires `pip install -e .[dflash]`. `gemma4-moe-mtp`
# requires `mlx-vlm 0.5.0+` from Blaizzy/mlx-vlm main. Each preset's
# Engine.load() raises a clear, actionable ImportError if the optional
# dependency is missing.
PRESETS: dict[str, dict] = {
    "gemma4":          {"draft": "gemma4-dflash"},
    "gemma4-max":      {"draft": "gemma4-dflash"},
    "gemma4-4bit":     {"draft": "gemma4-dflash"},
    "gemma4-moe":      {"draft": "gemma4-moe-dflash"},
    "gemma4-moe-4bit": {"draft": "gemma4-moe-dflash"},
    # Qwen-family models don't share a tokenizer-compatible small draft in
    # our library yet, so no preset — users can still pass --draft manually.
}


def load_preset(alias: str) -> dict:
    """Return a shallow copy of the preset for an alias, or an empty dict."""
    return dict(PRESETS.get(alias, {}))
