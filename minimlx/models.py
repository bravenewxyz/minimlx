from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console

from minimlx.aliases import LOCAL_MODELS_DIR


# Files worth fetching for any model. `*.jinja` matters for the Qwen 3.x
# family, which ships its chat template as a standalone `chat_template.jinja`
# rather than inline in `tokenizer_config.json`.
_MODEL_FILES = [
    "*.json", "*.safetensors", "*.model", "*.txt",
    "tokenizer*", "*.jinja", "*.py", "*.md",
]


def _is_local_ref(model_id: str) -> bool:
    """Is this a filesystem path rather than a Hub repo id?"""
    return model_id.startswith(("/", "~", ".")) or Path(model_id).expanduser().exists()


def split_subfolder(model_id: str) -> tuple[str, str | None]:
    """Split an `org/repo/sub/dir` reference into `(repo_id, subfolder)`.

    Some publishers ship every precision of one model in a single repo, one
    subfolder per quant (`4bit/`, `8bit/`, `bf16/`, ...) — see the
    `qwen38-*` aliases. A bare `org/repo` has no subfolder and is returned
    unchanged; so is any local path.
    """
    if _is_local_ref(model_id):
        return model_id, None
    parts = [p for p in model_id.split("/") if p]
    if len(parts) < 3:
        return model_id, None
    return "/".join(parts[:2]), "/".join(parts[2:])


def resolve(model_id: str, *, local_dir: bool = False) -> str:
    """Return something `mlx_lm.load()` can open.

    Local paths and plain `org/repo` ids pass straight through. An
    `org/repo/subfolder` reference is materialized into the HF cache — that
    subfolder only — and handed back as a concrete directory, because
    mlx-lm's loader has no notion of a subfolder inside a repo.

    `local_dir=True` additionally materializes plain `org/repo` ids, for
    loaders that only accept a directory (the `mtplx` runtime is one).
    """
    repo, subfolder = split_subfolder(model_id)
    if subfolder is None and (not local_dir or _is_local_ref(model_id)):
        return model_id
    from huggingface_hub import snapshot_download

    prefix = "" if subfolder is None else f"{subfolder}/"
    patterns = [f"{prefix}{name}" for name in _MODEL_FILES]

    def _target(snapshot: str) -> Path:
        root = Path(snapshot)
        return root if subfolder is None else root / subfolder

    # Prefer the cache so a warm start costs no network round-trips. A repo
    # can be cached for one quant and not another, so confirm this subfolder
    # actually landed before trusting the hit.
    try:
        cached = _target(
            snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
        )
        if (cached / "config.json").exists() and any(cached.glob("*.safetensors")):
            return str(cached)
    except Exception:
        pass
    return str(_target(snapshot_download(repo, allow_patterns=patterns)))


def pull(repo_id: str, console: Console, revision: str | None = None) -> str:
    """Download a model to the HF cache."""
    from huggingface_hub import snapshot_download

    repo, subfolder = split_subfolder(repo_id)
    patterns = _MODEL_FILES if subfolder is None else [f"{subfolder}/{p}" for p in _MODEL_FILES]

    console.print(f"[cyan]pulling[/] [bold]{repo_id}[/]")
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=patterns,
    )
    if subfolder is not None:
        path = str(Path(path) / subfolder)
    console.print(f"[green]✓[/] [dim]{path}[/]")
    return path


def _walk_local_models(name_filter: str = "") -> list[dict[str, Any]]:
    if not LOCAL_MODELS_DIR.exists():
        return []
    needle = name_filter.lower()
    out: list[dict[str, Any]] = []
    for child in sorted(LOCAL_MODELS_DIR.iterdir()):
        if not child.is_dir():
            continue
        repo_id = f"local:{child.name}"
        if needle and needle not in repo_id.lower():
            continue
        size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
        files = sum(1 for p in child.rglob("*") if p.is_file())
        mtime = max((p.stat().st_mtime for p in child.rglob("*") if p.is_file()), default=child.stat().st_mtime)
        out.append({
            "repo_id": f"local:{child.name}",
            "size_gb": size / (1024 ** 3),
            "nb_files": files,
            "last_accessed": mtime,
            "path": str(child),
        })
    return out


def list_cached(name_filter: str = "") -> list[dict[str, Any]]:
    from huggingface_hub import scan_cache_dir

    needle = name_filter.lower()
    info = scan_cache_dir()
    out = []
    for repo in info.repos:
        if needle and needle not in repo.repo_id.lower():
            continue
        out.append({
            "repo_id": repo.repo_id,
            "size_gb": repo.size_on_disk / (1024 ** 3),
            "nb_files": repo.nb_files,
            "last_accessed": repo.last_accessed,
        })
    out.extend(_walk_local_models(name_filter))
    out.sort(key=lambda e: e["repo_id"])
    return out


def remove(repo_id: str) -> int:
    from huggingface_hub import scan_cache_dir

    if repo_id.startswith("local:"):
        name = repo_id[len("local:"):]
        path = LOCAL_MODELS_DIR / name
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return 1
        return 0

    info = scan_cache_dir()
    hashes: list[str] = []
    for repo in info.repos:
        if repo.repo_id == repo_id:
            hashes.extend(r.commit_hash for r in repo.revisions)
    if not hashes:
        return 0
    info.delete_revisions(*hashes).execute()
    return len(hashes)
