"""Pydantic models for the OpenAI-compatible chat completions API (Epic 5)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
OBJECT_ERROR = "invalid_request_error"
ERROR_INVALID_BODY = "invalid_body"
ERROR_MODEL_NOT_FOUND = "model_not_found"
ERROR_GENERATION_FAILED = "generation_failed"


class StrictIgnoresExtra:
    """Mixin: ignore unknown request fields instead of rejecting them."""

    model_config = ConfigDict(extra="ignore")


class ChatMessageIn(BaseModel, StrictIgnoresExtra):
    """Incoming chat message: a role plus its content."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def _flatten_content_parts(cls, value: Any) -> Any:
        """Accept OpenAI text-part arrays; reject non-text parts."""
        if not isinstance(value, list):
            return value
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(str(part.get("text", "")))
            else:
                raise ValueError("only text content parts are supported")
        return "".join(parts)


class ChatCompletionRequest(BaseModel, StrictIgnoresExtra):
    """POST /v1/chat/completions request body (OpenAI format)."""

    model: str
    messages: list[ChatMessageIn] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False

    @field_validator("model")
    @classmethod
    def _model_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty model name")
        return value


class UsageOut(BaseModel):
    """Token accounting reported by the API."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatMessageOut(BaseModel):
    """Assistant message returned in a completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChatMessageOut
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Non-streaming /v1/chat/completions response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageOut


class DeltaMessage(BaseModel):
    """Incremental message carried by a streaming chunk."""

    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    """A streaming choice: partial delta plus optional finish_reason."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """One SSE chunk of a streamed chat completion."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: UsageOut | None = None


class ErrorBody(BaseModel):
    """OpenAI-style error payload."""

    message: str
    type: str
    code: str


class ErrorResponse(BaseModel):
    """Envelope for every error the API returns."""

    error: ErrorBody


def error_response(message: str, type_: str, code: str) -> ErrorResponse:
    """Build the OpenAI-compatible error envelope."""
    return ErrorResponse(error=ErrorBody(message=message, type=type_, code=code))
