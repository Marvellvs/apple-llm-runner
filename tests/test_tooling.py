"""US 1.3: `make test` and `make coverage` exist; coverage gate is 80%."""

from __future__ import annotations

from conftest import PROJECT_ROOT

MAKEFILE = (PROJECT_ROOT / "Makefile").read_text()


def test_coverage_target_fails_under_80_percent() -> None:
    coverage_block = MAKEFILE.split("## coverage:", 1)[1].split("## ", 1)[0]
    assert "--cov=local_llm" in coverage_block
    assert "--cov-fail-under=80" in coverage_block


def test_test_target_runs_pytest() -> None:
    test_block = MAKEFILE.split("## test:", 1)[1].split("## ", 1)[0]
    assert "-m pytest" in test_block


def test_targets_fail_clearly_without_venv() -> None:
    guard_block = MAKEFILE.split("## check-venv:", 1)[1].split("## ", 1)[0]
    assert "make setup" in guard_block


def test_pytest_is_configured() -> None:
    import tomllib

    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
