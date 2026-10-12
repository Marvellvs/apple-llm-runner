"""Epic 3 - MLX model management.

US 3.1: `make download MODEL=mlx-community/<repo>` downloads a model into
        LLM_MODELS_DIR via huggingface_hub.snapshot_download; the local folder
        is named after the repo; re-download is idempotent; HF_TOKEN is passed
        when set; missing MODEL / unknown repo / denied access are readable.
US 3.2: `make list` shows the name, size and path of every downloaded model.
US 3.3: `make delete MODEL=<name>` removes a model directory and never
        touches anything outside LLM_MODELS_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from conftest import PROJECT_ROOT
from local_llm import model_manager
from local_llm.cli import main
from local_llm.config import Settings
from local_llm.model_manager import (
    ModelError,
    delete_model,
    download_model,
    format_size,
    list_models,
)

MAKEFILE = (PROJECT_ROOT / "Makefile").read_text()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Drop ambient LLM_*/HF_TOKEN vars and ignore any project .env file."""
    for name in list(os.environ):
        if name.startswith("LLM_") or name == "HF_TOKEN":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def settings(models_dir: Path) -> Settings:
    return Settings(models_dir=models_dir)


class FakeHTTPError(Exception):
    """Minimal stand-in for a huggingface_hub HTTP failure."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeHub:
    """Stand-in for huggingface_hub.snapshot_download."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.files = 1

    def __call__(
        self,
        repo_id: str,
        *,
        local_dir: str | None = None,
        token: str | None = None,
        **_: Any,
    ) -> str:
        self.calls.append(
            {"repo_id": repo_id, "local_dir": local_dir, "token": token}
        )
        if self.error is not None:
            raise self.error
        assert local_dir is not None
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        for index in range(self.files):
            (target / f"model-{index}.safetensors").write_bytes(b"x" * 8)
        return local_dir


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()
    monkeypatch.setattr(model_manager, "snapshot_download", fake)
    return fake


def make_model_dir(models_dir: Path, name: str, *sizes: int) -> Path:
    """Create a fake downloaded model with files of the given sizes."""
    path = models_dir / name
    path.mkdir(parents=True)
    for index, size in enumerate(sizes or (1,)):
        (path / f"weights-{index}.safetensors").write_bytes(b"x" * size)
    return path


# --------------------------------------------------------------------- US 3.1


def test_download_stores_the_model_under_its_repo_name(
    hub: FakeHub, settings: Settings, models_dir: Path
) -> None:
    path = download_model("mlx-community/Qwen3-0.6B-MLX-bf16", settings=settings)

    assert path == models_dir / "Qwen3-0.6B-MLX-bf16"
    assert (path / "model-0.safetensors").is_file()
    assert hub.calls == [
        {
            "repo_id": "mlx-community/Qwen3-0.6B-MLX-bf16",
            "local_dir": str(path),
            "token": None,
        }
    ]


def test_download_accepts_a_repo_id_without_namespace(
    hub: FakeHub, settings: Settings, models_dir: Path
) -> None:
    path = download_model("Qwen3-0.6B-MLX-bf16", settings=settings)
    assert path == models_dir / "Qwen3-0.6B-MLX-bf16"


def test_download_passes_the_token_to_hugging_face(
    hub: FakeHub, models_dir: Path
) -> None:
    settings = Settings(models_dir=models_dir, hf_token="hf_from_env")
    download_model("org/repo", settings=settings)
    assert hub.calls[0]["token"] == "hf_from_env"


def test_download_is_anonymous_without_a_token(
    hub: FakeHub, settings: Settings
) -> None:
    download_model("org/repo", settings=settings)
    assert hub.calls[0]["token"] is None


def test_download_is_idempotent(
    hub: FakeHub, settings: Settings
) -> None:
    first = download_model("org/repo", settings=settings)
    second = download_model("org/repo", settings=settings)

    assert first == second
    assert hub.calls[0] == hub.calls[1]
    assert len(hub.calls) == 2
    assert (first / "model-0.safetensors").is_file()


@pytest.mark.parametrize("repo_id", [None, "", "   "])
def test_download_requires_a_model(
    repo_id: str | None, hub: FakeHub, settings: Settings
) -> None:
    with pytest.raises(ModelError) as excinfo:
        download_model(repo_id, settings=settings)
    message = str(excinfo.value)
    assert "MODEL is required" in message
    assert "make download" in message
    assert hub.calls == []


@pytest.mark.parametrize(
    "repo_id",
    ["../evil", "/etc/passwd", "org/../../x", "a/b/c", "..", ".", "x\\y"],
)
def test_download_rejects_path_like_repo_ids(
    repo_id: str, hub: FakeHub, settings: Settings, models_dir: Path
) -> None:
    with pytest.raises(ModelError) as excinfo:
        download_model(repo_id, settings=settings)
    assert "Invalid MODEL" in str(excinfo.value)
    assert hub.calls == []
    assert not models_dir.exists()


def test_unknown_repository_is_reported_clearly(
    hub: FakeHub, settings: Settings
) -> None:
    hub.error = FakeHTTPError("Repository not found", status_code=404)
    with pytest.raises(ModelError) as excinfo:
        download_model("org/typo", settings=settings)
    message = str(excinfo.value)
    assert "org/typo" in message
    assert "404" in message
    assert "not found" in message


@pytest.mark.parametrize("status", [401, 403])
def test_access_denied_hints_at_hf_token(
    hub: FakeHub, settings: Settings, status: int
) -> None:
    hub.error = FakeHTTPError("gated repo", status_code=status)
    with pytest.raises(ModelError) as excinfo:
        download_model("org/private", settings=settings)
    assert "HF_TOKEN" in str(excinfo.value)


def test_download_error_never_prints_the_token(
    hub: FakeHub, models_dir: Path
) -> None:
    secret = "hf_super_secret_token"
    settings = Settings(models_dir=models_dir, hf_token=secret)
    hub.error = RuntimeError(f"boom while authenticated with {secret}")
    with pytest.raises(ModelError) as excinfo:
        download_model("org/repo", settings=settings)
    message = str(excinfo.value)
    assert secret not in message
    assert "***" in message
    assert "boom" in message


def test_unexpected_download_errors_become_model_errors(
    hub: FakeHub, settings: Settings
) -> None:
    hub.error = OSError("connection reset")
    with pytest.raises(ModelError) as excinfo:
        download_model("org/repo", settings=settings)
    assert "connection reset" in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_cli_download_prints_the_target_path(
    hub: FakeHub,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LLM_MODELS_DIR", str(tmp_path / "models"))
    assert main(["download", "org/repo"]) == 0
    assert str(tmp_path / "models" / "repo") in capsys.readouterr().out


def test_cli_download_without_a_model_fails_with_a_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["download"]) == 1
    err = capsys.readouterr().err
    assert "MODEL is required" in err
    assert "make download" in err


def test_cli_download_reports_an_unknown_repository(
    hub: FakeHub,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LLM_MODELS_DIR", str(tmp_path / "models"))
    hub.error = FakeHTTPError("Repository not found", status_code=404)
    assert main(["download", "org/nope"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_reports_invalid_configuration_as_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_PORT", "0")
    assert main(["list"]) == 1
    assert "Invalid configuration" in capsys.readouterr().err


def test_cli_exposes_the_epic_3_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("download", "list", "delete"):
        assert command in out


def test_makefile_download_target_requires_model() -> None:
    block = MAKEFILE.split("## download:", 1)[1].split("## ", 1)[0]
    assert 'test -n "$(MODEL)"' in block
    assert "-m local_llm download $(MODEL)" in block


def test_makefile_list_target_runs_the_cli() -> None:
    block = MAKEFILE.split("## list:", 1)[1].split("## ", 1)[0]
    assert "-m local_llm list" in block


def test_makefile_delete_target_requires_model() -> None:
    block = MAKEFILE.split("## delete:", 1)[1].split("## ", 1)[0]
    assert 'test -n "$(MODEL)"' in block
    assert "-m local_llm delete $(MODEL)" in block


# --------------------------------------------------------------------- US 3.2


def test_list_is_empty_when_the_directory_is_missing(settings: Settings) -> None:
    assert list_models(settings) == []


def test_list_is_empty_for_an_empty_directory(
    settings: Settings, models_dir: Path
) -> None:
    models_dir.mkdir(parents=True)
    assert list_models(settings) == []


def test_list_reports_name_size_and_path(
    settings: Settings, models_dir: Path
) -> None:
    make_model_dir(models_dir, "alpha", 300, 200)
    make_model_dir(models_dir, "beta", 2048)
    (models_dir / "stray-file.bin").write_bytes(b"x")

    models = list_models(settings)

    assert [model.name for model in models] == ["alpha", "beta"]
    assert models[0].path == models_dir / "alpha"
    assert models[0].size_bytes == 500
    assert models[0].size == "500 B"
    assert models[1].size_bytes == 2048
    assert models[1].size == "2.00 KB"


def test_list_counts_files_in_nested_directories(
    settings: Settings, models_dir: Path
) -> None:
    nested = models_dir / "alpha" / "chunks"
    nested.mkdir(parents=True)
    (nested / "part.bin").write_bytes(b"x" * 1024)
    (models_dir / "alpha" / "config.json").write_bytes(b"{}")

    model = list_models(settings)[0]
    assert model.size_bytes == 1024 + 2


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1024**2, "1.00 MB"),
        (1536 * 1024, "1.50 MB"),
        (1024**3, "1.00 GB"),
        (2304 * 1024**2, "2.25 GB"),
    ],
)
def test_format_size_uses_readable_units(size: int, expected: str) -> None:
    assert format_size(size) == expected


def test_cli_list_shows_models_with_size_and_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "models"
    make_model_dir(root, "alpha", 1024 * 1024)
    monkeypatch.setenv("LLM_MODELS_DIR", str(root))

    assert main(["list"]) == 0

    out = capsys.readouterr().out
    assert "alpha" in out
    assert "1.00 MB" in out
    assert str(root / "alpha") in out


def test_cli_list_on_an_empty_catalog_hints_at_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LLM_MODELS_DIR", str(tmp_path / "models"))

    assert main(["list"]) == 0

    out = capsys.readouterr().out
    assert "No models found" in out
    assert "make download" in out


# --------------------------------------------------------------------- US 3.3


def test_delete_removes_the_model_directory(
    settings: Settings, models_dir: Path
) -> None:
    make_model_dir(models_dir, "alpha", 10)
    make_model_dir(models_dir, "beta", 10)

    path = delete_model("alpha", settings=settings)

    assert path == (models_dir / "alpha").resolve()
    assert not (models_dir / "alpha").exists()
    assert (models_dir / "beta").is_dir()
    assert models_dir.is_dir()


def test_delete_accepts_a_full_repo_id(
    settings: Settings, models_dir: Path
) -> None:
    make_model_dir(models_dir, "Qwen3", 10)
    delete_model("mlx-community/Qwen3", settings=settings)
    assert not (models_dir / "Qwen3").exists()


def test_delete_of_a_missing_model_is_reported(
    settings: Settings, models_dir: Path
) -> None:
    models_dir.mkdir(parents=True)
    with pytest.raises(ModelError) as excinfo:
        delete_model("ghost", settings=settings)
    message = str(excinfo.value)
    assert "not found" in message
    assert "make list" in message


@pytest.mark.parametrize("name", [None, "", "   "])
def test_delete_requires_a_model_name(
    name: str | None, settings: Settings
) -> None:
    with pytest.raises(ModelError) as excinfo:
        delete_model(name, settings=settings)
    message = str(excinfo.value)
    assert "MODEL is required" in message
    assert "make delete" in message


@pytest.mark.parametrize(
    "name",
    ["..", ".", "../outside", "/etc", "a/b/c", "org/..", "x\\y"],
)
def test_delete_refuses_paths_outside_the_models_dir(
    name: str, settings: Settings, models_dir: Path, tmp_path: Path
) -> None:
    models_dir.mkdir(parents=True)
    victim = tmp_path / "precious.txt"
    victim.write_text("do not delete")

    with pytest.raises(ModelError) as excinfo:
        delete_model(name, settings=settings)

    assert "Invalid MODEL" in str(excinfo.value)
    assert victim.exists()
    assert list(models_dir.iterdir()) == []


def test_delete_refuses_a_symlink_pointing_outside(
    settings: Settings, models_dir: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-model"
    outside.mkdir()
    (outside / "weights.bin").write_bytes(b"x" * 32)
    models_dir.mkdir(parents=True)
    (models_dir / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ModelError) as excinfo:
        delete_model("escape", settings=settings)

    assert "Refusing to delete" in str(excinfo.value)
    assert "LLM_MODELS_DIR" in str(excinfo.value)
    assert outside.is_dir()
    assert (models_dir / "escape").is_symlink()


def test_delete_refuses_to_remove_the_models_dir_itself(
    settings: Settings, models_dir: Path
) -> None:
    models_dir.mkdir(parents=True)
    (models_dir / "self").symlink_to(models_dir, target_is_directory=True)

    with pytest.raises(ModelError) as excinfo:
        delete_model("self", settings=settings)

    assert "Refusing to delete" in str(excinfo.value)
    assert models_dir.is_dir()


def test_delete_wraps_filesystem_errors(
    settings: Settings, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_model_dir(models_dir, "alpha", 10)

    def boom(path: str) -> None:
        raise OSError("Permission denied")

    monkeypatch.setattr(model_manager.shutil, "rmtree", boom)

    with pytest.raises(ModelError) as excinfo:
        delete_model("alpha", settings=settings)
    assert "Permission denied" in str(excinfo.value)


def test_cli_delete_removes_the_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "models"
    make_model_dir(root, "alpha", 8)
    monkeypatch.setenv("LLM_MODELS_DIR", str(root))

    assert main(["delete", "alpha"]) == 0

    assert "Deleted" in capsys.readouterr().out
    assert not (root / "alpha").exists()


def test_cli_delete_of_a_missing_model_exits_with_code_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "models"
    root.mkdir(parents=True)
    monkeypatch.setenv("LLM_MODELS_DIR", str(root))

    assert main(["delete", "ghost"]) == 1

    err = capsys.readouterr().err
    assert "not found" in err
    assert "make list" in err


def test_cli_delete_without_a_model_name_exits_with_code_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["delete"]) == 1
    assert "MODEL is required" in capsys.readouterr().err
