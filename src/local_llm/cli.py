"""Command-line interface: serve / download / list / delete."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .config import SettingsError, load_settings
from .model_manager import (
    ModelError,
    delete_model,
    download_model,
    list_models,
    models_root,
)

PROG = "local-llm"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Local LLM server for macOS (Apple Silicon) with an "
            "OpenAI-compatible API, powered by MLX."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROG} {__version__}",
    )
    _add_commands(parser)
    return parser


def _add_commands(parser: argparse.ArgumentParser) -> None:
    """Register CLI subcommands (added incrementally with each epic)."""
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    commands.add_parser(
        "serve",
        help="start the OpenAI-compatible server with the model from LLM_MODEL",
    )

    download = commands.add_parser(
        "download",
        help="download an MLX model from Hugging Face into LLM_MODELS_DIR",
    )
    download.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="Hugging Face repo id, e.g. mlx-community/<repo>",
    )

    commands.add_parser(
        "list", help="list downloaded models with their sizes and paths"
    )

    delete = commands.add_parser("delete", help="delete a downloaded model")
    delete.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="model name as shown by `list` (or a repo id)",
    )


def report_error(exc: BaseException) -> int:
    """Print a user-facing error to stderr; returns the failure exit code."""
    print(f"error: {exc}", file=sys.stderr)
    return 1


def _serve() -> int:
    """`serve`: load the model and run uvicorn until interrupted."""
    from .inference import InferenceError
    from .server import run_server

    try:
        run_server()
    except (SettingsError, InferenceError) as exc:
        return report_error(exc)
    return 0


def _run(args: argparse.Namespace) -> int:
    """Execute the selected subcommand. Raises ModelError/SettingsError."""
    if args.command == "serve":
        return _serve()
    settings = load_settings()
    if args.command == "download":
        path = download_model(args.model, settings=settings)
        print(f"Downloaded to {path}")
        return 0
    if args.command == "list":
        models = list_models(settings)
        if not models:
            print(f"No models found in {models_root(settings)}.")
            print("Hint: make download MODEL=mlx-community/<repo>")
            return 0
        width = max(len(model.name) for model in models)
        for model in models:
            print(f"{model.name:<{width}}  {model.size:>10}  {model.path}")
        return 0
    path = delete_model(args.model, settings=settings)
    print(f"Deleted {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    try:
        return _run(args)
    except (ModelError, SettingsError) as exc:
        return report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
