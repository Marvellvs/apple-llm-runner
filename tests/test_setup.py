"""US 1.1: `make setup` creates .venv (Python >= 3.12), installs the package
editable, and is idempotent."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT

VENV_DIR = PROJECT_ROOT / ".venv"
MAKEFILE = PROJECT_ROOT / "Makefile"

KEY_PACKAGES = (
    "local-llm",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic-settings",
    "huggingface-hub",
    "mlx",
    "mlx-lm",
    "pytest",
    "pytest-cov",
)


def test_python_interpreter_is_at_least_3_12() -> None:
    assert sys.version_info >= (3, 12)


def test_venv_directory_created_by_setup() -> None:
    assert VENV_DIR.is_dir(), "run `make setup` first"
    assert (VENV_DIR / "pyvenv.cfg").is_file()


@pytest.mark.parametrize("package", KEY_PACKAGES)
def test_key_package_installed(package: str) -> None:
    version = metadata.version(package)
    assert version, f"{package} must have a non-empty version"


def test_package_installed_in_editable_mode() -> None:
    import local_llm

    package_path = Path(local_llm.__file__).resolve()
    assert package_path.is_relative_to(PROJECT_ROOT / "src" / "local_llm")


def test_metadata_version_matches_package_version() -> None:
    import local_llm

    assert metadata.version("local-llm") == local_llm.__version__


def test_setup_target_is_idempotent() -> None:
    makefile = MAKEFILE.read_text()
    setup_block = makefile.split("## setup:", 1)[1].split("## ", 1)[0]
    assert "test -d $(VENV) ||" in setup_block, (
        "setup must reuse an existing .venv instead of recreating it"
    )
    assert 'pip install --quiet -e ".[dev]"' in setup_block
