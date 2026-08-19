# minimlx

**Very fast macOS CLI for Google Gemma 4 on Apple Silicon via MLX.**

A local LLM command line for Apple Silicon Macs: an interactive chat REPL,
one-shot prompts, an agentic coding loop, and voice conversation — all running
on-device through [MLX](https://github.com/ml-explore/mlx). `minimlx serve` also
exposes an Anthropic-API-compatible endpoint, so you can point Claude Code at a
local model.

![minimlx chat](assets/preview.png)

## Features

- **Fast on-device inference** — Gemma 4, Qwen 3.6 and Qwen 3.8 on Apple
  Silicon via MLX, with speculative decoding (DFlash block-diffusion drafts,
  Google Multi-Token-Prediction drafters, and MTPLX builds that carry their
  own MTP head).
- **Interactive chat** — streaming output, reasoning display, and a local
  tool-use loop: read, write, edit, grep, glob, ls, and bash.
- **Agentic `code` command** — one-shot programming tasks with the same tools.
- **Voice conversation** — speak → transcribe → generate → speak, fully local.
- **Claude Code backend** — `minimlx serve` speaks the Anthropic API, so Claude
  Code can run against a local model.
- **Prompt and KV caching**, model aliases, and conversation logging.

## Requirements

- macOS on Apple Silicon (M-series)
- Python 3.11+

## Install

```sh
git clone https://github.com/bravenewxyz/minimlx
cd minimlx
pip install -e .
```

Optional extras:

```sh
pip install -e ".[conversation]"   # voice conversation (STT/TTS)
pip install -e ".[dflash]"         # DFlash speculative-decoding drafts
pip install -e ".[mtp]"            # Google Multi-Token-Prediction drafters
pip install -e ".[mtplx]"          # MTPLX runtime — required by the default model
```

`[mtplx]` wants mlx >= 0.32 while `[dflash]`'s metadata pins mlx == 0.31.2. pip
resolves to 0.32; dflash runs fine there in practice (verified against `gemma4`
and `gemma4-moe` on mlx 0.32.1), the pin is just conservative.

## Usage

```sh
minimlx chat                  # interactive chat REPL with tools
minimlx ask "explain crc32"   # one-shot prompt (bare text works too)
minimlx code "add a --json flag"   # one-shot agentic coding task
minimlx conversation          # voice conversation
minimlx serve                 # Anthropic-API server for Claude Code
```

Model and cache management:

```sh
minimlx pull gemma4           # download a model
minimlx models                # list model aliases
minimlx ls                    # list cached models
```

Run `minimlx --help` or `minimlx <command> --help` for all options.

## Qwen 3.8 (abliterated)

`qwen38-27b-mtplx` is the **default model** — `minimlx chat` with no `-m` loads
it. Three PocketAiHub builds are wired up, measured on an M5 Max / 128 GB with
300-token generations at `--temp 0.7`, decode rate from a cold model:

| alias | build | on disk | decode | prefill | peak RAM |
|---|---|---|---|---|---|
| **`qwen38-27b-mtplx`** (default) | 27B, mixed 4/8-bit, built-in MTP head | 20 GB | **57 tok/s** | 352 tok/s | 21.0 GB |
| `qwen38-9b-4bit` | 9B, 4-bit | 5.6 GB | 98 tok/s | 358 tok/s | 5.3 GB |
| `qwen38-9b` | 9B, 8-bit | 9.7 GB | 58 tok/s | 276 tok/s | 9.7 GB |
| `qwen38-27b-4bit` | 27B, 4-bit | 15 GB | 32 tok/s | 234 tok/s | 15.5 GB |
| `qwen38-27b` | 27B, 8-bit | 27 GB | 16 tok/s | 159 tok/s | 28.9 GB |

`qwen38-27b-6bit`, `qwen38-27b-2bit` and the `-max` (bf16) aliases resolve to
the same repos and download on first use. Sustained load roughly halves every
number above — an M-series laptop throttles well before the model does.

**Multi-quant repos.** The 9B and 27B MLX repos publish every precision in a
single repo, one subfolder per quant, so those aliases carry the subfolder
(`…-Abliterated-MLX/4bit`). `minimlx pull` and the engine fetch only the quant
you name.

**MTPLX.** The default is the same 27B trunk rebuilt with a
multi-token-prediction head in the model repo: it drafts ahead of itself with
no second model resident and verifies each block by rejection sampling, so the
output distribution is unchanged and only the wall-clock moves. It needs
`pip install -e ".[mtplx]"` — without that it still loads on plain mlx-lm, just
autoregressively at ~20 tok/s, since mlx-lm drops the MTP tensors.

minimlx applies the runtime profile the build ships with (`turbo`:
verify-specialised Metal kernels), worth ~1.8x, and the verify configuration
that profile requires — the two are one unit, and the profile alone fails on
the first rejected block. Those kernels are thread-affine, so the whole runtime
lives on one worker thread that loads, generates, and never exits; a thread
that exits under them aborts the process. `MINIMLX_MTPLX_SAFE=1` falls back to
mtplx's library defaults, and `MINIMLX_MTPLX_DEPTH` overrides the speculative
depth (default: the depth the build ships, which also measured fastest here).

**No `--draft` on this family.** Qwen 3.8's linear-attention layers have a
non-trimmable KV cache, so mlx-lm's speculative path refuses them outright
("Speculative decoding requires a trimmable prompt cache") — including a 9B
drafting for the 27B, which shares the tokenizer. The MTPLX build is the way to
speculate here.

These are abliterated checkpoints: refusal behaviour has been removed, so they
answer things the upstream instruct model declines.

## Using with Claude Code

Start the local server:

```sh
minimlx serve
```

Then point Claude Code at it:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_AUTH_TOKEN=minimlx
```
