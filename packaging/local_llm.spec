# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: `local-llm` as a self-contained macOS binary.

Build:
    .venv/bin/pyinstaller --noconfirm --clean packaging/local_llm.spec

Environment switches:
    LLM_ONEFILE=0                 build a directory instead of a single file
    LLM_BUNDLE_MODEL_DIR=<path>   embed a model directory (see packaging/README.md)
"""

from __future__ import annotations

import os
import pathlib
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

ROOT = pathlib.Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import mlx  # noqa: E402  (resolved from the build environment)

# `mlx` is a PEP 420 namespace package, so it has no __file__.
MLX_DIR = pathlib.Path(next(iter(mlx.__path__))).resolve()
MLX_LIB = MLX_DIR / "lib"

ONEFILE = os.environ.get("LLM_ONEFILE", "1") != "0"
BUNDLE_MODEL = os.environ.get("LLM_BUNDLE_MODEL_DIR", "").strip()
APP_NAME = os.environ.get("LLM_APP_NAME", "").strip() or "local-llm"

if pathlib.Path(DISTPATH).resolve() != ROOT / "dist":
    print(
        f"note: distpath is {DISTPATH}; `make binary` always builds into "
        f"{ROOT / 'dist'}"
    )


def keep_transformers_module(name: str) -> bool:
    """Filter the huge ``transformers`` tree down to what tokenizers need.

    ``AutoTokenizer`` resolves a tokenizer class by importing
    ``transformers.models.<model_type>`` at run time (and that package lazily
    imports its own ``tokenization_*``/``configuration_*`` modules), so those
    packages and files must be present; audio/vision/feature-extraction stacks
    are not needed by mlx-lm.
    """
    parts = name.split(".")
    if len(parts) < 3 or parts[:2] != ["transformers", "models"]:
        return True
    if len(parts) == 3:  # transformers.models.<model_type> package itself
        return True
    if parts[2] in {"auto", "auto_factory"}:
        return True
    return parts[-1].startswith(("tokenization", "configuration"))


# MLX ships its Metal kernels and shared libraries inside the `mlx` package
# (`mlx-metal` wheel); `mlx/core*.so` reaches them through LC_RPATH. They are
# declared explicitly - the kernel library is plain data (invisible to the
# dependency analysis) and libjaccl.dylib is a run-time dependency of
# libmlx.dylib that the analysis does not always follow - and all three are
# placed in the bundle root, where the rewritten rpaths look for them.
# entrypoint.py re-checks the metallib location at start-up.
binaries = [
    (str(MLX_LIB / "libmlx.dylib"), "."),
    (str(MLX_LIB / "libjaccl.dylib"), "."),
]
datas = [
    (str(MLX_LIB / "mlx.metallib"), "."),
    (str(ROOT / "README.md"), "."),
]

if BUNDLE_MODEL:
    model_dir = pathlib.Path(BUNDLE_MODEL).expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"LLM_BUNDLE_MODEL_DIR is not a directory: {model_dir}")
    datas.append((str(model_dir), f"models/{model_dir.name}"))

hiddenimports = [
    # The application package itself is imported through the src/ path hook.
    "local_llm",
    "local_llm.cli",
    "local_llm.config",
    "local_llm.inference",
    "local_llm.model_manager",
    "local_llm.monitoring",
    "local_llm.responses",
    "local_llm.schemas",
    "local_llm.server",
    "local_llm.store",
    # mlx-lm resolves model architectures and tool parsers by import string.
    *collect_submodules("mlx_lm"),
    # `mlx` is a PEP 420 namespace package whose extension module imports
    # sibling pure-python helpers (mlx._reprlib_fix, mlx.utils, ...), which
    # the dependency analysis cannot see.
    *collect_submodules("mlx"),
    # transformers: see keep_transformers_module().
    *collect_submodules("transformers", filter=keep_transformers_module),
    *collect_submodules("huggingface_hub"),
]

for dist in (
    "local-llm",
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "transformers",
    "tokenizers",
    "safetensors",
    "sentencepiece",
    "huggingface_hub",
    "numpy",
    "pydantic",
    "uvicorn",
):
    try:
        datas += copy_metadata(dist)
    except Exception:  # pragma: no cover - metadata is optional
        pass

a = Analysis(
    [str(ROOT / "packaging" / "entrypoint.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "torch",
        "datasets",
        "pytest",
        "PIL",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        # Distinct name so `make binary-dir` never overwrites (or is
        # overwritten by) the one-file build in dist/.
        name=f"{APP_NAME}-dir",
    )
