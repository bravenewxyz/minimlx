from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console

from minimlx.aliases import LOCAL_MODELS_DIR


def pull(repo_id: str, console: Console, revision: str | None = None) -> str:
    """Download a model to the HF cache."""
    from huggingface_hub import snapshot_download

    console.print(f"[cyan]pulling[/] [bold]{repo_id}[/]")
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[
            "*.json", "*.safetensors", "*.model", "*.txt",
            "tokenizer*", "*.py", "*.md",
        ],
    )
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
