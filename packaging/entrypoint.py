"""Program entry point for the frozen (PyInstaller) build of `local-llm`.

PyInstaller executes this module instead of the console script declared in
``pyproject.toml``; it only prepares the bundle and then delegates to
:func:`local_llm.cli.main`, so the CLI behaves exactly like the source install.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _bundle_root() -> Path | None:
    """Directory that holds the unpacked application, or None when not frozen."""
    if not getattr(sys, "frozen", False):
        return None
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base)
    return Path(sys.executable).resolve().parent


def _colocate_mlx_metallib(root: Path) -> None:
    """Make sure ``mlx.metallib`` sits next to ``libmlx.dylib``.

    MLX locates its Metal kernel library by asking ``dladdr`` for the directory
    of the loaded ``libmlx.dylib``. The bundler is free to place the two files
    in different directories, so mirror the metallib before the first ``import
    mlx`` rather than relying on the bundle layout.
    """
    metallib = next(root.rglob("mlx.metallib"), None)
    if metallib is None:
        return
    for libmlx in root.rglob("libmlx.dylib"):
        neighbour = libmlx.parent / "mlx.metallib"
        if neighbour.exists():
            continue
        try:
            neighbour.symlink_to(metallib)
        except OSError:
            shutil.copy2(metallib, neighbour)


def _apply_bundled_model(root: Path) -> None:
    """Point the app at a model directory embedded by the bundler, if any.

    Only used by builds made with ``LLM_BUNDLE_MODEL_DIR``; environment
    variables set by the user always win.
    """
    import os

    bundled = root / "models"
    if not bundled.is_dir():
        return
    os.environ.setdefault("LLM_MODELS_DIR", str(bundled))
    if os.environ.get("LLM_MODEL"):
        return
    names = sorted(entry.name for entry in bundled.iterdir() if entry.is_dir())
    if len(names) == 1:
        os.environ["LLM_MODEL"] = names[0]


def main() -> int:
    """Run the CLI, preparing the frozen bundle first when necessary."""
    root = _bundle_root()
    if root is not None:
        _colocate_mlx_metallib(root)
        _apply_bundled_model(root)

    from local_llm.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
