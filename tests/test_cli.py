"""US 1.2: `src/` layout, `pyproject.toml` and the CLI entry point."""

from __future__ import annotations

import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from local_llm.cli import main


def test_cli_help_exits_with_code_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_cli_no_args_prints_help_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


def test_cli_version_exits_with_code_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "local-llm" in capsys.readouterr().out


def test_python_m_entrypoint_runs_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["local_llm", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("local_llm", run_name="__main__")
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_cli_module_subprocess_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "local_llm", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=60,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_cli_console_script_is_installed() -> None:
    script = Path(sys.executable).parent / "local-llm"
    assert script.exists(), "console script `local-llm` must be installed"
    result = subprocess.run(
        [str(script), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_src_layout_exists() -> None:
    assert (PROJECT_ROOT / "src" / "local_llm" / "__init__.py").is_file()
    assert (PROJECT_ROOT / "src" / "local_llm" / "__main__.py").is_file()
    assert (PROJECT_ROOT / "src" / "local_llm" / "cli.py").is_file()


def test_pyproject_declares_project_metadata() -> None:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project = data["project"]
    assert project["name"] == "local-llm"
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"]["local-llm"] == "local_llm.cli:main"
    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_pyproject_declares_runtime_and_dev_dependencies() -> None:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    runtime = " ".join(data["project"]["dependencies"])
    for package in (
        "fastapi",
        "uvicorn",
        "pydantic",
        "huggingface_hub",
        "mlx",
        "mlx-lm",
    ):
        assert package in runtime
    dev = " ".join(data["project"]["optional-dependencies"]["dev"])
    assert "pytest" in dev
    assert "pytest-cov" in dev
