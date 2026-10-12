"""Epic 6 - server startup (`make run`).

US 6.1: `make run` starts uvicorn on LLM_HOST:LLM_PORT with the model from
        LLM_MODEL; the model is loaded before requests are accepted and the
        startup log carries the model path and generation parameters, never
        the HF token.
US 6.2: startup failures (no model, invalid port, out of memory) print an
        actionable message and exit with a non-zero code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator

import pytest
from starlette.testclient import TestClient

from conftest import PROJECT_ROOT
from local_llm import server
from local_llm.cli import build_parser, main
from local_llm.config import Settings, SettingsError, load_settings
from local_llm.inference import (
    GenerationResult,
    InferenceError,
    TokenChunk,
    Usage,
)
from local_llm.server import is_out_of_memory, run_server, startup_banner

MAKEFILE = (PROJECT_ROOT / "Makefile").read_text()

SECRET_TOKEN = "hf_secret_must_not_be_logged"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Drop ambient LLM_*/HF_TOKEN vars and ignore any project .env file."""
    for name in list(os.environ):
        if name.startswith("LLM_") or name == "HF_TOKEN":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


class StubEngine:
    """Stand-in for local_llm.inference.Engine (the mocked inference layer)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        load_error: BaseException | None = None,
        model_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self.load_error = load_error
        self._model_path = model_path or Path("/models/TinyModel")
        self.load_calls = 0
        self._loaded = False

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> Path:
        return self._model_path

    def load(self) -> Path:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        self._loaded = True
        return self._model_path

    def generate(
        self, messages: list[dict[str, str]], params: Any = None
    ) -> GenerationResult:
        return GenerationResult(
            text="Hello from the stub!", finish_reason="stop", usage=Usage(5, 2)
        )

    def stream(
        self, messages: list[dict[str, str]], params: Any = None
    ) -> Iterator[TokenChunk]:
        yield TokenChunk(text="Hi", finish_reason="stop", usage=Usage(5, 1))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model="mlx-community/TinyModel",
        models_dir=tmp_path / "models",
        host="127.0.0.1",
        port=8000,
    )


@pytest.fixture
def engine(settings: Settings) -> StubEngine:
    return StubEngine(settings)


@pytest.fixture
def uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record uvicorn.run calls instead of starting a real server."""
    calls: list[dict[str, Any]] = []

    def fake_run(app: Any, **kwargs: Any) -> None:
        state = getattr(app.state, "engine", None)
        calls.append({"app": app, "loaded": getattr(state, "is_loaded", None), **kwargs})

    monkeypatch.setattr(server.uvicorn, "run", fake_run)
    return calls


def chat_payload() -> dict[str, Any]:
    return {"model": "TinyModel", "messages": [{"role": "user", "content": "hi"}]}


# --------------------------------------------------------------------- US 6.1


def test_make_run_target_starts_the_serve_command() -> None:
    block = MAKEFILE.split("## run:", 1)[1].split("## ", 1)[0]
    assert "-m local_llm serve" in block
    assert "check-venv" in block


def test_run_target_is_declared_phony() -> None:
    phony = MAKEFILE.split(".PHONY:", 1)[1].splitlines()[0]
    assert "run" in phony.split()


def test_cli_exposes_the_serve_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert build_parser().parse_args(["serve"]).command == "serve"
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("serve", "download", "list", "delete"):
        assert command in out


def test_serve_command_exits_zero_when_the_server_starts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    uvicorn_run: list[dict[str, Any]],
) -> None:
    engine = StubEngine(settings)
    monkeypatch.setattr(server, "Engine", lambda: engine)

    assert main(["serve"]) == 0

    assert len(uvicorn_run) == 1
    assert engine.load_calls == 1
    assert "error:" not in capsys.readouterr().err


def test_model_is_loaded_once_before_uvicorn_starts(
    uvicorn_run: list[dict[str, Any]], engine: StubEngine
) -> None:
    run_server(engine=engine)

    assert engine.load_calls == 1
    assert engine.is_loaded
    assert uvicorn_run[0]["loaded"] is True
    assert uvicorn_run[0]["app"].state.engine is engine


def test_server_binds_to_the_configured_host_and_port(
    uvicorn_run: list[dict[str, Any]], tmp_path: Path
) -> None:
    settings = Settings(
        model="mlx-community/TinyModel",
        models_dir=tmp_path / "models",
        host="0.0.0.0",
        port=9001,
    )
    run_server(engine=StubEngine(settings))

    assert uvicorn_run[0]["host"] == "0.0.0.0"
    assert uvicorn_run[0]["port"] == 9001


def test_loaded_app_answers_requests_through_testclient(
    uvicorn_run: list[dict[str, Any]], engine: StubEngine
) -> None:
    run_server(engine=engine)

    client = TestClient(uvicorn_run[0]["app"])
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "TinyModel"
    assert data["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello from the stub!",
    }
    assert data["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


def test_startup_log_contains_model_path_and_parameters(
    caplog: pytest.LogCaptureFixture,
    uvicorn_run: list[dict[str, Any]],
    engine: StubEngine,
) -> None:
    with caplog.at_level(logging.INFO, logger="local_llm.server"):
        run_server(engine=engine)

    text = caplog.text
    assert str(engine.model_path) in text
    assert "http://127.0.0.1:8000" in text
    assert "max_tokens=512" in text
    assert "temperature=0.7" in text
    assert "top_p=1.0" in text


def test_startup_log_never_contains_the_hf_token(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    uvicorn_run: list[dict[str, Any]],
) -> None:
    monkeypatch.setenv("HF_TOKEN", SECRET_TOKEN)
    monkeypatch.setenv("LLM_MODEL", "mlx-community/TinyModel")

    with caplog.at_level(logging.INFO, logger="local_llm.server"):
        run_server(engine=StubEngine())

    output = capsys.readouterr()
    assert "serving" in caplog.text
    assert SECRET_TOKEN not in caplog.text
    assert SECRET_TOKEN not in output.out
    assert SECRET_TOKEN not in output.err


def test_startup_banner_lists_model_path_and_parameters(settings: Settings) -> None:
    banner = startup_banner(settings, Path("/models/TinyModel"))

    assert "/models/TinyModel" in banner
    assert "http://127.0.0.1:8000" in banner
    assert "max_tokens=512" in banner
    assert "temperature=0.7" in banner
    assert "top_p=1.0" in banner


# --------------------------------------------------------------------- US 6.2


def test_missing_model_setting_exits_nonzero_with_a_hint(
    capsys: pytest.CaptureFixture[str], uvicorn_run: list[dict[str, Any]]
) -> None:
    assert main(["serve"]) == 1

    err = capsys.readouterr().err
    assert "error:" in err
    assert "LLM_MODEL" in err
    assert "make download" in err
    assert uvicorn_run == []


def test_missing_model_weights_point_to_download(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    uvicorn_run: list[dict[str, Any]],
) -> None:
    monkeypatch.setenv("LLM_MODEL", "mlx-community/Ghost")
    monkeypatch.setenv("LLM_MODELS_DIR", str(tmp_path / "models"))

    assert main(["serve"]) == 1

    err = capsys.readouterr().err
    assert "error:" in err
    assert "make download MODEL=mlx-community/Ghost" in err
    assert uvicorn_run == []


def test_invalid_port_exits_nonzero_before_the_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    uvicorn_run: list[dict[str, Any]],
) -> None:
    monkeypatch.setenv("LLM_PORT", "99999")
    engine = StubEngine()
    monkeypatch.setattr(server, "Engine", lambda: engine)

    assert main(["serve"]) == 1

    err = capsys.readouterr().err
    assert "error:" in err
    assert "Invalid configuration" in err
    assert "LLM_PORT" in err
    assert engine.load_calls == 0
    assert uvicorn_run == []


def test_invalid_port_is_a_settings_error_from_run_server(
    monkeypatch: pytest.MonkeyPatch,
    uvicorn_run: list[dict[str, Any]],
) -> None:
    monkeypatch.setenv("LLM_PORT", "0")
    engine = StubEngine()

    with pytest.raises(SettingsError, match="LLM_PORT"):
        run_server(engine=engine)

    assert engine.load_calls == 0
    assert uvicorn_run == []


def test_out_of_memory_exits_nonzero_with_a_hint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    uvicorn_run: list[dict[str, Any]],
) -> None:
    engine = StubEngine(settings, load_error=MemoryError())
    monkeypatch.setattr(server, "Engine", lambda: engine)

    assert main(["serve"]) == 1

    err = capsys.readouterr().err
    assert "error:" in err
    assert "Not enough memory" in err
    assert "make download" in err
    assert uvicorn_run == []


def test_out_of_memory_message_from_mlx_is_translated(
    uvicorn_run: list[dict[str, Any]], engine: StubEngine
) -> None:
    engine.load_error = InferenceError(
        "Failed to load model from /models/TinyModel: [Metal] OOM when allocating"
    )

    with pytest.raises(InferenceError) as excinfo:
        run_server(engine=engine)

    message = str(excinfo.value)
    assert "Not enough memory" in message
    assert "OOM" in message
    assert uvicorn_run == []


def test_plain_inference_error_passes_through_unchanged(
    uvicorn_run: list[dict[str, Any]], engine: StubEngine
) -> None:
    engine.load_error = InferenceError(
        "Model 'Ghost' not found in /models (LLM_MODELS_DIR). "
        "Run `make download MODEL=Ghost` first."
    )

    with pytest.raises(InferenceError, match="make download MODEL=Ghost"):
        run_server(engine=engine)

    assert uvicorn_run == []


def test_unexpected_load_failure_is_masked_but_logged(
    caplog: pytest.LogCaptureFixture,
    uvicorn_run: list[dict[str, Any]],
    engine: StubEngine,
) -> None:
    engine.load_error = ValueError("secret internals")

    with pytest.raises(InferenceError) as excinfo:
        run_server(engine=engine)

    message = str(excinfo.value)
    assert "ValueError" in message
    assert "secret internals" not in message
    assert "secret internals" in caplog.text
    assert uvicorn_run == []


@pytest.mark.parametrize(
    "text",
    [
        "[Metal] OOM when allocating buffer",
        "mlx error: out of memory",
        "Cannot allocate memory",
        "memory allocation failed",
    ],
)
def test_out_of_memory_messages_are_detected(text: str) -> None:
    assert is_out_of_memory(text)


@pytest.mark.parametrize(
    "text",
    [
        "Model 'Ghost' not found in /models",
        "there is room for one more",
        "permission denied",
    ],
)
def test_other_failures_are_not_treated_as_out_of_memory(text: str) -> None:
    assert not is_out_of_memory(text)
