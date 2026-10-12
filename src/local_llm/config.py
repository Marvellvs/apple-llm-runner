"""Application settings loaded from environment variables (Epic 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "LLM_"
TOKEN_ENV = "HF_TOKEN"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODELS_DIR = "./models"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0


class SettingsError(RuntimeError):
    """User-facing configuration error; the message is safe to print at startup."""


class Settings(BaseSettings):
    """Environment-only configuration (see docs/PLAN.md for the variable table)."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    host: str = DEFAULT_HOST
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    model: str | None = None
    models_dir: Path = Path(DEFAULT_MODELS_DIR)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    hf_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices(TOKEN_ENV)
    )

    @field_validator("host")
    @classmethod
    def _host_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty host name or IP address")
        return value

    @field_validator("model")
    @classmethod
    def _blank_model_is_unset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("hf_token", mode="before")
    @classmethod
    def _blank_token_is_unset(
        cls, value: str | SecretStr | None
    ) -> str | SecretStr | None:
        if value is None or isinstance(value, SecretStr):
            return value
        return value.strip() or None

    def require_model(self) -> str:
        """Return the configured model or fail with an actionable hint."""
        if self.model is None:
            raise SettingsError(
                "LLM_MODEL is not set. Download a model first "
                "(`make download MODEL=mlx-community/<repo>`), then "
                "`export LLM_MODEL=<repo>` (or put it into .env)."
            )
        return self.model


def env_name(field: str) -> str:
    """Environment variable name that feeds the given settings field."""
    return TOKEN_ENV if field == "hf_token" else ENV_PREFIX + field.upper()


def format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic error as a short, human-readable startup message."""
    lines = ["Invalid configuration:"]
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"]) or "settings"
        lines.append(f"  - {field}: {error['msg']} (env: {env_name(field)})")
    return "\n".join(lines)


def load_settings() -> Settings:
    """Build settings from the environment; never expose secrets in errors."""
    try:
        return Settings()
    except ValidationError as exc:
        raise SettingsError(format_validation_error(exc)) from None


def hf_token_value(settings: Settings) -> str | None:
    """The raw HF token, or None when it is unset/blank (anonymous access)."""
    token = settings.hf_token
    if token is None:
        return None
    value = token.get_secret_value()
    return value or None


def hf_auth_kwargs(settings: Settings) -> dict[str, Any]:
    """Kwargs to forward to huggingface_hub so the token is passed explicitly."""
    return {"token": hf_token_value(settings)}


def hf_error_hint(exc: BaseException) -> str | None:
    """Actionable hint for HF 401/403 failures, None for anything else."""
    status = hf_status_code(exc)
    if status in (401, 403):
        return (
            f"Hugging Face denied access (HTTP {status}): check that HF_TOKEN "
            "is set, valid, and belongs to an account allowed to read this "
            "repository."
        )
    return None


def hf_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from a HF/hub exception, if any."""
    for attr in ("status_code", "statusCode"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None
