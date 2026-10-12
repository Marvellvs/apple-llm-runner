PYTHON   ?= python3
VENV     := .venv
VENV_BIN := $(VENV)/bin
PY       := $(VENV_BIN)/python
# Build paths are anchored to the project root, not to the current directory.
DIST_DIR ?= $(CURDIR)/dist
WORK_DIR ?= $(CURDIR)/build/pyinstaller
PYI_CACHE := $(CURDIR)/build/pyinstaller-cache
PYI      := PYINSTALLER_CONFIG_DIR=$(PYI_CACHE) $(VENV_BIN)/pyinstaller --noconfirm --clean --distpath $(DIST_DIR) --workpath $(WORK_DIR)

.DEFAULT_GOAL := help

.PHONY: setup check-venv test coverage download list delete run binary binary-dir clean help

## setup: create .venv (Python >= 3.12) and install the package in editable mode with dev deps
setup:
	@$(PYTHON) -c 'import sys; print(f"Python {sys.version.split()[0]}"); sys.exit(0 if sys.version_info >= (3, 12) else f"ERROR: Python >= 3.12 required, got {sys.version.split()[0]} (set PYTHON=... and re-run)")'
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PY) -m pip install --quiet --upgrade pip
	@$(PY) -m pip install --quiet -e ".[dev]"
	@echo "setup OK: $(VENV) ready (Python $$($(PY) -V | cut -d' ' -f2))"

## check-venv: internal guard so other targets fail with a clear message
check-venv:
	@test -x $(PY) || (echo "ERROR: $(VENV) is missing. Run 'make setup' first." && exit 1)

## test: run the test suite
test: check-venv
	@$(PY) -m pytest

## coverage: run the test suite with coverage; fails if coverage < 80%
coverage: check-venv
	@$(PY) -m pytest --cov=local_llm --cov-report=term-missing --cov-fail-under=80

## download: download an MLX model; e.g. make download MODEL=mlx-community/<repo>
download: check-venv
	@test -n "$(MODEL)" || (echo "ERROR: MODEL is required. Usage: make download MODEL=mlx-community/<repo>" && exit 1)
	@$(PY) -m local_llm download $(MODEL)

## list: list downloaded models with their sizes and paths
list: check-venv
	@$(PY) -m local_llm list

## delete: delete a downloaded model; e.g. make delete MODEL=<name>
delete: check-venv
	@test -n "$(MODEL)" || (echo "ERROR: MODEL is required. Usage: make delete MODEL=<name>" && exit 1)
	@$(PY) -m local_llm delete $(MODEL)

## run: start the OpenAI-compatible server with the model from LLM_MODEL
run: check-venv
	@$(PY) -m local_llm serve

## binary: build dist/local-llm, one self-contained executable (Python + MLX + FastAPI inside)
binary: check-venv
	@$(PYI) packaging/local_llm.spec
	@echo "binary OK: $(DIST_DIR)/local-llm (run it with --help)"

## binary-dir: build dist/local-llm-dir/ as a directory instead (starts instantly, same files)
binary-dir: check-venv
	@LLM_ONEFILE=0 $(PYI) packaging/local_llm.spec
	@echo "binary OK: $(DIST_DIR)/local-llm-dir/local-llm"

## clean: remove venv, caches and build artifacts
clean:
	@rm -rf $(VENV) .pytest_cache .coverage .coverage.* htmlcov dist build *.egg-info src/*.egg-info
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__'))]"
	@echo "clean OK"

## help: list available targets
help:
	@grep -E '^## ' Makefile | sed 's/^## //'
