# Packaging `local-llm` as a single binary

`make binary` turns the whole project - CLI, FastAPI server, uvicorn, MLX, MLX
Metal kernels, transformers/tokenizers - into one self-contained macOS
executable. No Python, no virtualenv and no `pip install` are needed on the
machine that runs it.

Verified on macOS 26.6.2 / Apple Silicon (arm64) with Python 3.14.2,
PyInstaller 6.22.3, mlx 0.32.3, mlx-metal 0.32.3 and mlx-lm 0.32.0.

## Result

| Build | Artifact | Size | First launch | Later launches |
|---|---|---|---|---|
| `make binary` | `dist/local-llm` (one file) | ≈131 MiB | ~16 s | ~9.3 s |
| `make binary-dir` | `dist/local-llm-dir/local-llm` | 345 MB (directory) | ~9 s | ~0.27 s |

The first launch is slower because macOS validates a freshly built bundle; the
one-file executable additionally unpacks its ~345 MB payload into a temp
directory on *every* start. That is the whole trade-off: one file to copy vs.
instant start-up. Both artifacts are the same application.

## Building

```bash
make setup      # once: creates .venv and installs pyinstaller (dev extra)
make binary     # -> <project root>/dist/local-llm
```

Build paths are anchored to the project root, not to the current directory:
`DIST_DIR` defaults to `$(CURDIR)/dist` and `WORK_DIR` to
`$(CURDIR)/build/pyinstaller`, so `make -C /path/to/project binary` from
anywhere still writes to the project's own `dist/`. PyInstaller's cache is
redirected to `<project>/build/pyinstaller-cache` because its default location
(`~/Library/Application Support/pyinstaller`) is not always writable.

Invoking PyInstaller by hand works too, as long as the dist/work paths are given
explicitly (the spec prints a note when `distpath` is not `<project>/dist`):

```bash
PYINSTALLER_CONFIG_DIR=build/pyinstaller-cache .venv/bin/pyinstaller \
    --noconfirm --clean --distpath dist --workpath build/pyinstaller \
    packaging/local_llm.spec
```

### Switches

| Variable | Default | Meaning |
|---|---|---|
| `LLM_ONEFILE` | `1` | `0` builds a directory instead of a single file |
| `LLM_APP_NAME` | `local-llm` | Base name of the artifacts (`<name>` / `<name>-dir`) |
| `LLM_BUNDLE_MODEL_DIR` | unset | Embed a model directory, see below |
| `DIST_DIR` | `$(CURDIR)/dist` | Where the artifacts are written (Makefile only) |
| `WORK_DIR` | `$(CURDIR)/build/pyinstaller` | Intermediate build files (Makefile only) |
| `PYINSTALLER_CONFIG_DIR` | `build/pyinstaller-cache` | PyInstaller's own cache |

## What ends up in the binary

Inside: the `local_llm` package, CPython 3.14, FastAPI/Starlette/Pydantic,
uvicorn with its native speedups (uvloop, httptools, websockets, watchfiles),
`mlx.core` + `libmlx.dylib` + `libjaccl.dylib` + the 190 MB `mlx.metallib`,
transformers, tokenizers, safetensors, sentencepiece, numpy, huggingface_hub,
psutil.

Deliberately outside: **model weights** (1+ GB each, and a one-file build would
re-unpack them on every launch), the Hugging Face token and `.env`. Settings are
still read from the environment and from `./.env` relative to the *current*
directory, exactly as in the source install.

## Why the spec looks the way it does

Four non-obvious things have to be handled:

1. **`mlx` is a PEP 420 namespace package** - it has no `__init__.py`, so
   `mlx.__file__` is `None` and PyInstaller cannot see the pure-python helpers
   that `mlx.core` imports at initialisation (`mlx._reprlib_fix`). They are added
   with `collect_submodules("mlx")`; without this, `import mlx.core` fails with
   `ModuleNotFoundError: No module named 'mlx._reprlib_fix'`.
2. **`mlx-metal` installs its files inside the `mlx` package**, not under a
   package of its own: `mlx/lib/libmlx.dylib`, `mlx/lib/libjaccl.dylib` and
   `mlx/lib/mlx.metallib`. PyInstaller resolves `@rpath/libmlx.dylib` from
   `mlx/core*.so` and rewrites `LC_RPATH` while relocating the libraries, so all
   three are declared explicitly and placed in the bundle root. `libjaccl.dylib`
   is a plain run-time dependency of `libmlx.dylib` that the dependency analysis
   does not follow reliably; when it is missing, dyld fails with
   `Library not loaded: @rpath/libjaccl.dylib`.
3. **MLX finds `mlx.metallib` through `dladdr()`**, i.e. in the directory of the
   loaded `libmlx.dylib`, not through the import system. `packaging/entrypoint.py`
   therefore mirrors the metallib next to every `libmlx.dylib` it finds in the
   bundle before the first `import mlx`, so the build survives layout changes
   between PyInstaller versions.
4. **Dynamic imports by name.** `mlx-lm` loads model architectures and tool
   parsers via `importlib.import_module("mlx_lm.models.<arch>")` /
   `mlx_lm.tool_parsers.<name>`, and `transformers.AutoTokenizer` imports
   `transformers.models.<model_type>` on the fly. `collect_submodules("mlx_lm")`
   covers the first; the second is covered by a filtered
   `collect_submodules("transformers", ...)` that keeps only the model packages
   and their `tokenization_*` / `configuration_*` modules - the untrimmed tree is
   117 MB across 524 model packages, most of it torch/vision code this server
   never touches.

`tkinter`, `torch`, `datasets`, `matplotlib`, `PIL` and `pytest` are excluded to
keep the artifact small.

## Embedding a model (optional)

```bash
LLM_BUNDLE_MODEL_DIR=models/Qwen3-0.6B-MLX-bf16 make binary-dir
```

The directory is added as `models/<name>` inside the bundle. At start-up
`entrypoint.py` then defaults `LLM_MODELS_DIR` to it and, if it contains exactly
one model, sets `LLM_MODEL` too - so the resulting binary serves that model with
no configuration. Environment variables set by the user always win.

Use the **directory** build for this. Embedding a model in a one-file build
produces a >1 GB executable that re-extracts the whole model on every launch.

## Verified behaviour

The frozen artifacts were checked against a real model
(`Qwen3-0.6B-MLX-bf16`, MLX Metal path):

- `--version`, `--help`, `list`, and the `download` error path;
- `serve` with real generation through `POST /v1/chat/completions`, both
  non-streaming and streaming (`stream: true`);
- `POST /v1/responses`;
- started from an unrelated working directory with an absolute
  `LLM_MODELS_DIR`, i.e. portable outside the source tree.

`make test` (380 tests) still passes; the autotests exercise the package, not the
bundle.

## Distributing the binary

PyInstaller ad-hoc signs the bundle. To hand it to another Mac, sign it with a
Developer ID certificate and notarise it, otherwise Gatekeeper blocks the first
start and the user has to approve it in System Settings. `dist/` and `build/` are
git-ignored; `make clean` removes both.

The bundle freezes the exact versions installed in `.venv`. Re-run `make binary`
after updating dependencies so the binary and the source environment stay in
sync.
