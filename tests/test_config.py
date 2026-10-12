"""Epic 2 - configuration.

US 2.1: settings come from env vars only, defaults apply, invalid values fail
        fast with a readable message.
US 2.2: optional HF_TOKEN is forwarded to the Hugging Face SDK, never logged,
        with an actionable hint on 401/403.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import huggingface_hub
import pytest

from local_llm.config import (
    Settings,
    SettingsError,
    format_validation_error,
    hf_auth_kwargs,
    hf_error_hint,
    hf_token_value,
    load_settings,
)

@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Drop ambient LLM_*/HF_TOKEN vars and ignore any project .env file."""
    for name in list(os.environ):
        if name.startswith("LLM_") or name == "HF_TOKEN":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------- US 2.1


def test_defaults_apply_when_no_variables_are_set() -> None:
    settings = load_settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.model is None
    assert settings.models_dir == Path("./models")
    assert settings.max_tokens == 512
    assert settings.temperature == 0.7
    assert settings.top_p == 1.0
    assert hf_token_value(settings) is None


@pytest.mark.parametrize(
    ("var", "field", "value", "expected"),
    [
        ("LLM_HOST", "host", "0.0.0.0", "0.0.0.0"),
        ("LLM_PORT", "port", "9001", 9001),
        (
            "LLM_MODEL",
            "model",
            "mlx-community/Qwen3-0.6B-MLX-bf16",
            "mlx-community/Qwen3-0.6B-MLX-bf16",
        ),
        ("LLM_MODELS_DIR", "models_dir", "/tmp/models", Path("/tmp/models")),
        ("LLM_MAX_TOKENS", "max_tokens", "64", 64),
        ("LLM_TEMPERATURE", "temperature", "0.0", 0.0),
        ("LLM_TOP_P", "top_p", "0.5", 0.5),
    ],
)
def test_each_setting_is_overridden_via_env(
    monkeypatch: pytest.MonkeyPatch,
    var: str,
    field: str,
    value: str,
    expected: object,
) -> None:
    monkeypatch.setenv(var, value)
    assert getattr(load_settings(), field) == expected


def test_unknown_llm_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_NOT_A_SETTING", "whatever")
    assert load_settings().port == 8000


def test_dot_env_file_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_PORT=8123\nLLM_MODEL=acme/model\n")
    settings = load_settings()
    assert settings.port == 8123
    assert settings.model == "acme/model"


@pytest.mark.parametrize(
    ("var", "value", "field"),
    [
        ("LLM_PORT", "0", "port"),
        ("LLM_PORT", "65536", "port"),
        ("LLM_PORT", "not-a-number", "port"),
        ("LLM_TEMPERATURE", "-0.1", "temperature"),
        ("LLM_TEMPERATURE", "2.5", "temperature"),
        ("LLM_TOP_P", "1.5", "top_p"),
        ("LLM_TOP_P", "-0.01", "top_p"),
        ("LLM_MAX_TOKENS", "0", "max_tokens"),
        ("LLM_HOST", "   ", "host"),
    ],
)
def test_invalid_values_raise_readable_error(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str, field: str
) -> None:
    monkeypatch.setenv(var, value)
    with pytest.raises(SettingsError) as excinfo:
        load_settings()
    message = str(excinfo.value)
    assert message.startswith("Invalid configuration:")
    assert field in message
    assert var in message


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("LLM_PORT", "1"),
        ("LLM_PORT", "65535"),
        ("LLM_TEMPERATURE", "0"),
        ("LLM_TEMPERATURE", "2"),
        ("LLM_TOP_P", "0"),
        ("LLM_TOP_P", "1"),
        ("LLM_MAX_TOKENS", "1"),
    ],
)
def test_boundary_values_are_accepted(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    monkeypatch.setenv(var, value)
    assert load_settings()


def test_load_settings_error_hides_any_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_token")
    monkeypatch.setenv("LLM_PORT", "70000")
    with pytest.raises(SettingsError) as excinfo:
        load_settings()
    assert "hf_super_secret_token" not in str(excinfo.value)


def test_require_model_hints_at_download_when_unset() -> None:
    with pytest.raises(SettingsError) as excinfo:
        load_settings().require_model()
    message = str(excinfo.value)
    assert "LLM_MODEL" in message
    assert "make download" in message


def test_require_model_returns_stripped_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "  mlx-community/Qwen3  ")
    assert load_settings().require_model() == "mlx-community/Qwen3"


def test_blank_model_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "   ")
    settings = load_settings()
    assert settings.model is None
    with pytest.raises(SettingsError):
        settings.require_model()


def test_format_validation_error_names_the_env_variable() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        Settings(port=70000)  # type: ignore[call-arg]
    message = format_validation_error(excinfo.value)
    assert "port" in message
    assert "LLM_PORT" in message


# --------------------------------------------------------------------- US 2.2


def test_token_is_forwarded_to_hf_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    captured: dict[str, object] = {}

    def fake_snapshot_download(repo_id: str, *, token: str | None = None) -> str:
        captured["repo_id"] = repo_id
        captured["token"] = token
        return "/models/repo"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    settings = load_settings()
    huggingface_hub.snapshot_download("org/repo", **hf_auth_kwargs(settings))

    assert captured == {"repo_id": "org/repo", "token": "hf_from_env"}


def test_missing_token_means_anonymous_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(repo_id: str, *, token: str | None = None) -> str:
        captured["token"] = token
        return "/models/repo"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    settings = load_settings()
    huggingface_hub.snapshot_download("org/repo", **hf_auth_kwargs(settings))

    assert captured["token"] is None
    assert hf_token_value(settings) is None


def test_blank_token_is_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert hf_token_value(load_settings()) is None


def test_token_is_not_leaked_by_repr_or_serialisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf_should_never_appear"
    monkeypatch.setenv("HF_TOKEN", secret)
    settings = load_settings()
    assert hf_token_value(settings) == secret

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in settings.model_dump_json()
    assert secret not in str(settings.model_dump())


def test_token_is_not_written_to_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_log_secret")
    settings = load_settings()
    with caplog.at_level(logging.INFO):
        logging.getLogger("test-config").info("starting with %s", settings)
    assert "hf_log_secret" not in caplog.text


def test_token_is_not_printed_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_stdout_secret")
    settings = load_settings()
    print(settings)
    assert "hf_stdout_secret" not in capsys.readouterr().out


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_come_with_a_token_hint(status: int) -> None:
    class FakeHTTPError(Exception):
        status_code = status

    hint = hf_error_hint(FakeHTTPError("forbidden"))
    assert hint is not None
    assert "HF_TOKEN" in hint
    assert str(status) in hint


def test_error_hint_reads_status_from_response_object() -> None:
    class Response:
        status_code = 403

    class FakeHTTPError(Exception):
        response = Response()

    assert "HF_TOKEN" in (hf_error_hint(FakeHTTPError("nope")) or "")


def test_non_auth_errors_have_no_hint() -> None:
    assert hf_error_hint(RuntimeError("connection reset")) is None

    class NotFound(Exception):
        status_code = 404

    assert hf_error_hint(NotFound()) is None
