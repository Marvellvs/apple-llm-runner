"""Pydantic models for the OpenAI-compatible API (Epic 5 + Responses API)."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
OBJECT_ERROR = "invalid_request_error"
ERROR_INVALID_BODY = "invalid_body"
ERROR_MODEL_NOT_FOUND = "model_not_found"
ERROR_GENERATION_FAILED = "generation_failed"

# Responses API (Epic 8)
OBJECT_RESPONSE = "response"
RESPONSE_ID_PREFIX = "resp_"
ERROR_RESPONSE_NOT_FOUND = "response_not_found"
ERROR_UNSUPPORTED_TOOLS = "unsupported_tools"

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"
STATUS_QUEUED = "queued"

INCOMPLETE_MAX_OUTPUT_TOKENS = "max_output_tokens"


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


# ---------------------------------------------------- Responses API: input items


class InputTextPart(BaseModel, StrictIgnoresExtra):
    """Text part of a message in ``input`` (user/system/assistant text)."""

    type: Literal["input_text"] = "input_text"
    text: str = ""


class OutputTextPart(BaseModel, StrictIgnoresExtra):
    """``output_text`` part, as sent back by Responses clients."""

    type: Literal["output_text"] = "output_text"
    text: str = ""


class TextPart(BaseModel, StrictIgnoresExtra):
    """Plain ``text`` part accepted for compatibility with other clients."""

    type: Literal["text"] = "text"
    text: str = ""


InputContentPart = Annotated[
    Union[InputTextPart, OutputTextPart, TextPart],
    Field(discriminator="type"),
]


class InputMessageItem(BaseModel, StrictIgnoresExtra):
    """A message item inside ``input``."""

    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[InputContentPart | str] = ""


class FunctionCallItem(BaseModel, StrictIgnoresExtra):
    """An assistant function call replayed by the client."""

    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str = ""
    id: str | None = None
    status: str | None = None

    @field_validator("arguments", mode="before")
    @classmethod
    def _arguments_to_string(cls, value: Any) -> Any:
        """Accept an object where the contract expects a JSON string."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return value


class FunctionCallOutputItem(BaseModel, StrictIgnoresExtra):
    """The result of a function call, sent back by the client."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str | list[InputContentPart] | dict[str, Any] = ""


class ReasoningSummaryPart(BaseModel, StrictIgnoresExtra):
    """One part of a reasoning item's summary/content."""

    type: Literal["summary_text", "reasoning_text", "text"] = "summary_text"
    text: str = ""


class ReasoningItem(BaseModel, StrictIgnoresExtra):
    """A reasoning item replayed by the client."""

    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    summary: list[ReasoningSummaryPart] | None = None
    content: list[ReasoningSummaryPart] | None = None
    encrypted_content: str | None = None


ResponseInputItem = Annotated[
    Union[
        InputMessageItem,
        FunctionCallItem,
        FunctionCallOutputItem,
        ReasoningItem,
    ],
    Field(discriminator="type"),
]


# --------------------------------------------------- Responses API: tool config


class ToolDefinition(BaseModel, StrictIgnoresExtra):
    """A function tool (flat OpenAI Responses shape)."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None

    @field_validator("parameters", mode="before")
    @classmethod
    def _null_parameters_mean_empty(cls, value: Any) -> Any:
        """Some clients send ``parameters: null`` for parameterless tools."""
        return {} if value is None else value


class ReasoningConfig(BaseModel, StrictIgnoresExtra):
    """``reasoning`` request settings."""

    effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class TextFormat(BaseModel, StrictIgnoresExtra):
    """``text.format`` request settings."""

    type: Literal["text", "json_object", "json_schema"] = "text"


class TextConfig(BaseModel, StrictIgnoresExtra):
    """``text`` request settings."""

    format: TextFormat | None = None
    verbosity: str | None = None


INPUT_ITEM_TYPES = ("message", "function_call", "function_call_output", "reasoning")
INPUT_PART_TYPES = ("input_text", "output_text", "text")


def _normalize_input_item(item: Any, index: int) -> Any:
    """Fill in defaults so every item carries an unambiguous ``type``."""
    if not isinstance(item, dict):
        raise ValueError(f"input[{index}] must be an object or a string")
    item = dict(item)
    item_type = item.get("type")
    if item_type is None:
        item_type = "message" if "role" in item else None
    if not isinstance(item_type, str):
        raise ValueError(f"input[{index}] must have a string 'type'")
    if item_type not in INPUT_ITEM_TYPES:
        raise ValueError(
            f"input[{index}] has unsupported type {item_type!r}; supported types "
            f"are {', '.join(INPUT_ITEM_TYPES)}"
        )
    item["type"] = item_type
    content = item.get("content")
    if isinstance(content, list):
        parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part = dict(part)
                part.setdefault("type", "input_text")
                if part["type"] not in INPUT_PART_TYPES:
                    raise ValueError(
                        f"input[{index}] content part type {part['type']!r} is not "
                        f"supported; supported part types are "
                        f"{', '.join(INPUT_PART_TYPES)}"
                    )
                parts.append(part)
            else:
                raise ValueError(
                    f"input[{index}] content parts must be strings or objects"
                )
        item["content"] = parts
    return item


def _normalize_tool(tool: Any, index: int) -> Any:
    """Flatten the chat-completions ``{"function": {...}}`` tool shape."""
    if not isinstance(tool, dict):
        raise ValueError(f"tools[{index}] must be an object")
    tool = dict(tool)
    nested = tool.pop("function", None)
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update(tool)
        tool = merged
    tool.setdefault("type", "function")
    if tool.get("type") != "function":
        raise ValueError(
            f"tools[{index}]: unsupported tool type {tool.get('type')!r}; "
            "only 'function' tools are supported"
        )
    return tool


class ResponseRequest(BaseModel, StrictIgnoresExtra):
    """POST /v1/responses request body."""

    model: str
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    store: bool | None = None
    previous_response_id: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    reasoning: ReasoningConfig | None = None
    text: TextConfig | None = None
    background: bool = False
    parallel_tool_calls: bool | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("model")
    @classmethod
    def _model_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty model name")
        return value

    @field_validator("input", mode="before")
    @classmethod
    def _normalize_input(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [
                _normalize_input_item(item, index)
                for index, item in enumerate(value)
            ]
        return value

    @field_validator("input")
    @classmethod
    def _input_is_not_empty(cls, value: str | list[ResponseInputItem]) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be empty")
        if isinstance(value, list) and not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [_normalize_tool(tool, index) for index, tool in enumerate(value)]
        return value

    @field_validator("tool_choice")
    @classmethod
    def _valid_tool_choice(cls, value: str | dict[str, Any] | None) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            if value not in ("none", "auto", "required"):
                raise ValueError(
                    "must be 'none', 'auto', 'required' or a function object"
                )
            return value
        if isinstance(value, dict):
            if value.get("type") not in (None, "function"):
                raise ValueError("only tool_choice objects of type 'function'")
            return value
        raise ValueError("must be a string or an object")

    @field_validator("previous_response_id")
    @classmethod
    def _previous_id_is_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def _supported_combinations(self) -> ResponseRequest:
        if self.stream and self.background:
            raise ValueError("background=true cannot be combined with stream=true")
        if self.background and self.store is False:
            raise ValueError(
                "background=true requires store to be omitted or true "
                "(a background response is retrieved by id)"
            )
        if self.tool_choice not in (None, "none", "auto") and not self.tools:
            raise ValueError("tool_choice requires a non-empty 'tools' list")
        if self.text and self.text.format and self.text.format.type != "text":
            raise ValueError(
                f"text.format.type={self.text.format.type!r} is not supported; "
                "only 'text' is available"
            )
        return self


# -------------------------------------------------- Responses API: output items


class ResponseOutputText(BaseModel):
    """Text content part of a response message."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)
    logprobs: list[Any] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """An assistant message output item."""

    type: Literal["message"] = "message"
    id: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText] = Field(default_factory=list)


class ResponseReasoningSummaryPart(BaseModel):
    """A summary part of a reasoning output item."""

    type: Literal["summary_text"] = "summary_text"
    text: str = ""


class ResponseReasoningItem(BaseModel):
    """A reasoning (thinking) output item."""

    type: Literal["reasoning"] = "reasoning"
    id: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    summary: list[ResponseReasoningSummaryPart] = Field(default_factory=list)


class ResponseFunctionCall(BaseModel):
    """A function call output item."""

    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str = ""
    status: Literal["in_progress", "completed", "incomplete"] = "completed"


ResponseOutputItem = Annotated[
    Union[ResponseOutputMessage, ResponseReasoningItem, ResponseFunctionCall],
    Field(discriminator="type"),
]


class InputTokensDetails(BaseModel):
    """Breakdown of the input token usage."""

    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    """Breakdown of the output token usage."""

    reasoning_tokens: int = 0


class ResponsesUsage(BaseModel):
    """Usage accounting of a Responses API call."""

    input_tokens: int
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens: int
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails
    )
    total_tokens: int


class ResponseErrorBody(BaseModel):
    """Error attached to a failed Response object."""

    code: str | None = None
    message: str = ""


class IncompleteDetails(BaseModel):
    """Why a response is incomplete."""

    reason: str


class ResponseObject(BaseModel):
    """A Responses API Response object."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["in_progress", "completed", "incomplete", "failed", "queued"]
    error: ResponseErrorBody | None = None
    incomplete_details: IncompleteDetails | None = None
    model: str
    output: list[ResponseOutputItem] = Field(default_factory=list)
    usage: ResponsesUsage | None = None
    instructions: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    metadata: dict[str, Any] | None = None
    background: bool | None = None
    previous_response_id: str | None = None
    store: bool | None = None
