"""Download / list / delete MLX models stored in LLM_MODELS_DIR (Epic 3)."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import (
    Settings,
    hf_auth_kwargs,
    hf_error_hint,
    hf_status_code,
    hf_token_value,
    load_settings,
)

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DOWNLOAD_USAGE = "Usage: make download MODEL=mlx-community/<repo>"
_DELETE_USAGE = "Usage: make delete MODEL=<name>"


class ModelError(RuntimeError):
    """User-facing model management error; the message is safe to print."""


@dataclass(frozen=True)
class ModelInfo:
    """A model folder sitting in LLM_MODELS_DIR."""

    name: str
    path: Path
    size_bytes: int

    @property
    def size(self) -> str:
        return format_size(self.size_bytes)


def models_root(settings: Settings) -> Path:
    """Directory that holds downloaded model folders."""
    return Path(settings.models_dir)


def validate_repo_id(repo_id: str | None) -> str:
    """Normalize a Hugging Face repo id; reject blank and path-like values."""
    if repo_id is None or not repo_id.strip():
        raise ModelError(f"MODEL is required. {_DOWNLOAD_USAGE}")
    repo_id = repo_id.strip()
    parts = repo_id.split("/")
    if len(parts) > 2 or any(not _SEGMENT.match(part) for part in parts):
        raise ModelError(
            f"Invalid MODEL {repo_id!r}: expected a Hugging Face repo id "
            "like mlx-community/<repo-name>."
        )
    return repo_id


def validate_model_name(name: str | None) -> str:
    """Normalize a model folder name; reject blank values and paths."""
    if name is None or not name.strip():
        raise ModelError(f"MODEL is required. {_DELETE_USAGE}")
    name = name.strip()
    parts = name.split("/")
    if len(parts) > 2 or any(not _SEGMENT.match(part) for part in parts):
        raise ModelError(
            f"Invalid MODEL {name!r}: expected a model name as shown by "
            "`make list` (a plain folder name), not a path."
        )
    return parts[-1]


def download_model(
    repo_id: str | None, *, settings: Settings | None = None
) -> Path:
    """Download a model snapshot into LLM_MODELS_DIR/<repo-name>.

    Re-downloading an already present model is idempotent (huggingface_hub
    reuses the files already on disk).
    """
    settings = settings or load_settings()
    repo_id = validate_repo_id(repo_id)
    target = models_root(settings) / repo_id.rsplit("/", 1)[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            **hf_auth_kwargs(settings),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a ModelError below
        raise ModelError(download_error_message(repo_id, exc, settings)) from None
    return target


def download_error_message(
    repo_id: str, exc: BaseException, settings: Settings
) -> str:
    """Readable, secret-free message for a failed download."""
    hint = hf_error_hint(exc)
    if hint is not None:
        return f"Failed to download {repo_id!r}: {hint}"
    if hf_status_code(exc) == 404:
        return (
            f"Repository {repo_id!r} was not found on Hugging Face "
            "(HTTP 404). Check the id, e.g. mlx-community/<repo-name>."
        )
    return f"Failed to download {repo_id!r}: {safe_error_text(exc, settings)}"


def safe_error_text(exc: BaseException, settings: Settings) -> str:
    """Exception text with the HF token redacted, if one is configured."""
    text = str(exc).strip() or exc.__class__.__name__
    token = hf_token_value(settings)
    if token:
        text = text.replace(token, "***")
    return text


def dir_size(path: Path) -> int:
    """Total size in bytes of all files under ``path``."""
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def format_size(size_bytes: int) -> str:
    """Human-readable size using GB / MB / KB units."""
    if size_bytes >= GB:
        return f"{size_bytes / GB:.2f} GB"
    if size_bytes >= MB:
        return f"{size_bytes / MB:.2f} MB"
    if size_bytes >= KB:
        return f"{size_bytes / KB:.2f} KB"
    return f"{size_bytes} B"


def list_models(settings: Settings | None = None) -> list[ModelInfo]:
    """Describe every model folder in LLM_MODELS_DIR, sorted by name."""
    settings = settings or load_settings()
    root = models_root(settings)
    if not root.is_dir():
        return []
    return [
        ModelInfo(name=entry.name, path=entry, size_bytes=dir_size(entry))
        for entry in sorted(root.iterdir())
        if entry.is_dir()
    ]


def delete_model(name: str | None, *, settings: Settings | None = None) -> Path:
    """Delete LLM_MODELS_DIR/<name>; refuse anything outside that directory."""
    settings = settings or load_settings()
    name = validate_model_name(name)
    root = models_root(settings).resolve()
    target = (root / name).resolve()
    if target == root or not target.is_relative_to(root):
        raise ModelError(
            f"Refusing to delete {name!r}: it does not resolve to a model "
            f"folder inside {root} (LLM_MODELS_DIR)."
        )
    if not target.exists():
        raise ModelError(
            f"Model {name!r} not found in {root}. Run `make list` to see "
            "the downloaded models."
        )
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise ModelError(
            f"Failed to delete {name!r}: {safe_error_text(exc, settings)}"
        ) from None
    return target
