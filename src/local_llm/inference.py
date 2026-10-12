"""Inference layer over mlx_lm: load once, generate and stream (Epic 4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from mlx_lm import generate as mlx_generate
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate as mlx_stream_generate
from mlx_lm.sample_utils import make_sampler as mlx_make_sampler

from .config import Settings, load_settings
from .model_manager import models_root

Message = Mapping[str, Any]


class InferenceError(RuntimeError):
    """User-facing inference failure; the message is safe to show to clients."""


class ModelNotLoadedError(InferenceError):
    """Generation was requested before the model was loaded."""


class UnsupportedToolsError(InferenceError):
    """Tools were requested but the chat template cannot render them."""


# Markers used when the tokenizer does not advertise its own. They describe the
# common Qwen-family layout; every model-specific nuance is overridable via
# Engine.markers (see markers_for_tokenizer).
DEFAULT_REASONING_START = "\u003cthink\u003e"
DEFAULT_REASONING_END = "\u003c/think\u003e"
DEFAULT_TOOL_CALL_START = "\u003ctool_call\u003e"
DEFAULT_TOOL_CALL_END = "\u003c/tool_call\u003e"


@dataclass(frozen=True)
class ModelMarkers:
    """Text markers that separate reasoning / tool calls from plain output."""

    reasoning_start: str | None = None
    reasoning_end: str | None = None
    tool_start: str | None = None
    tool_end: str | None = None
    structural: tuple[str, ...] = ()
    introspected: bool = False

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning_start and self.reasoning_end)

    @property
    def has_tools(self) -> bool:
        return bool(self.tool_start and self.tool_end)


def default_markers() -> ModelMarkers:
    """The documented fallback marker set (used for unrecognised tokenizers)."""
    return ModelMarkers(
        reasoning_start=DEFAULT_REASONING_START,
        reasoning_end=DEFAULT_REASONING_END,
        tool_start=DEFAULT_TOOL_CALL_START,
        tool_end=DEFAULT_TOOL_CALL_END,
    )


def markers_for_tokenizer(tokenizer: Any) -> ModelMarkers:
    """Read the marker metadata an mlx_lm TokenizerWrapper advertises.

    Raw HF tokenizers expose none of these attributes; ``introspected`` stays
    False and callers may then fall back to :func:`default_markers`.
    """
    if tokenizer is None:
        return ModelMarkers()
    names = ("has_thinking", "think_start", "tool_call_start", "structural_markers")
    if not any(hasattr(tokenizer, name) for name in names):
        return ModelMarkers()
    reasoning_start = getattr(tokenizer, "think_start", None)
    reasoning_end = getattr(tokenizer, "think_end", None)
    tool_start = getattr(tokenizer, "tool_call_start", None)
    tool_end = getattr(tokenizer, "tool_call_end", None)
    if not getattr(tokenizer, "has_thinking", False):
        reasoning_start = reasoning_end = None
    if not getattr(tokenizer, "has_tool_calling", bool(tool_start)):
        tool_start = tool_end = None
    structural = tuple(getattr(tokenizer, "structural_markers", None) or ())
    return ModelMarkers(
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        tool_start=tool_start,
        tool_end=tool_end,
        structural=structural,
        introspected=True,
    )


@dataclass(frozen=True)
class Usage:
    """Token accounting for a prompt/completion pair."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class GenerationParams:
    """Sampling settings forwarded to mlx_lm for a single request."""

    max_tokens: int
    temperature: float
    top_p: float
    tools: Sequence[Mapping[str, Any]] | None = None
    enable_thinking: bool | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> GenerationParams:
        """Build the defaults that come from the LLM_* environment variables."""
        return cls(
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )


@dataclass(frozen=True)
class GenerationResult:
    """Complete non-streaming generation output."""

    text: str
    finish_reason: str
    usage: Usage


@dataclass(frozen=True)
class TokenChunk:
    """One streamed step: a text segment plus optional bookkeeping."""

    text: str
    finish_reason: str | None = None
    usage: Usage | None = None


def resolve_model_path(settings: Settings) -> Path:
    """Locate the weights folder for settings.model (LLM_MODELS_DIR first)."""
    model = settings.require_model()
    local = models_root(settings) / Path(model).name
    if local.is_dir():
        return local
    explicit = Path(model).expanduser()
    if explicit.is_dir():
        return explicit
    if explicit.is_absolute() or model.startswith("."):
        raise InferenceError(f"Model directory {explicit} does not exist.")
    raise InferenceError(
        f"Model {model!r} not found in {models_root(settings)} (LLM_MODELS_DIR). "
        f"Run `make download MODEL={model}` first."
    )


def has_chat_template(tokenizer: Any) -> bool:
    """True when the tokenizer can render a chat prompt (wrapper or raw HF)."""
    if getattr(tokenizer, "has_chat_template", False):
        return True
    return bool(getattr(tokenizer, "chat_template", None))


def render_messages(messages: Sequence[Message]) -> str:
    """Plain-text prompt used when the tokenizer has no chat template."""
    lines = [
        f"{message.get('role', 'user')}: {message.get('content') or ''}"
        for message in messages
    ]
    lines.append("assistant:")
    return "\n".join(lines)


def normalize_tools_for_template(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert tools to the chat-completions shape chat templates expect."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        mapping = (
            tool.model_dump(exclude_none=True)
            if hasattr(tool, "model_dump")
            else dict(tool)
        )
        if isinstance(mapping.get("function"), dict):
            normalized.append(mapping)
            continue
        function = {
            key: mapping[key]
            for key in ("name", "description", "parameters", "strict")
            if key in mapping
        }
        normalized.append({"type": "function", "function": function})
    return normalized


def build_prompt(
    tokenizer: Any,
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]] | None = None,
    enable_thinking: bool | None = None,
) -> tuple[str | list[int], int]:
    """Return the (prompt for mlx_lm, prompt token count) built from messages.

    When ``tools`` is given they are forwarded to the chat template. Models
    whose tokenizer does not advertise tool support, and templates that reject
    the ``tools`` keyword, make this an explicit UnsupportedToolsError instead
    of silently dropping the tools. ``enable_thinking`` is forwarded when given
    and silently dropped for templates that do not know the keyword.
    """
    if not messages:
        raise InferenceError("messages must not be empty")
    if tools and has_tool_support(tokenizer) is False:
        raise UnsupportedToolsError(_unsupported_tools_message())
    if has_chat_template(tokenizer):
        rendered = [dict(message) for message in messages]
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = normalize_tools_for_template(tools)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking

        def render() -> Any:
            return tokenizer.apply_chat_template(
                rendered, tokenize=True, add_generation_prompt=True, **kwargs
            )

        try:
            prompt = render()
        except TypeError as exc:
            message = str(exc).lower()
            if "enable_thinking" in kwargs and "thinking" in message:
                # the template does not know the keyword; thinking is off anyway
                kwargs.pop("enable_thinking")
                try:
                    prompt = render()
                except Exception as retry_exc:
                    raise InferenceError(
                        f"Failed to apply the chat template: {retry_exc}"
                    ) from None
            elif tools and "tool" in message:
                raise UnsupportedToolsError(_unsupported_tools_message()) from None
            else:
                raise InferenceError(
                    f"Failed to apply the chat template: {exc}"
                ) from None
        except Exception as exc:
            raise InferenceError(f"Failed to apply the chat template: {exc}") from None
        prompt = list(prompt)
        return prompt, len(prompt)
    if tools:
        raise UnsupportedToolsError(_unsupported_tools_message())
    text = render_messages(messages)
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def _unsupported_tools_message() -> str:
    """Explain why function calling is unavailable for the loaded model."""
    return (
        "The loaded model does not support tools (its tokenizer/chat template "
        "cannot render tool calls), so function calling is unavailable. Remove "
        "'tools' from the request or load a model with tool support."
    )


def has_tool_support(tokenizer: Any) -> bool | None:
    """True/False when the tokenizer advertises tools, None when unknown."""
    if tokenizer is None:
        return None
    value = getattr(tokenizer, "has_tool_calling", None)
    if value is None:
        return None
    return bool(value)


def count_tokens(tokenizer: Any, text: str) -> int:
    """Completion token count for generated text (0 for empty output)."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_chunk(response: Any, prompt_tokens: int) -> TokenChunk:
    """Convert one mlx_lm GenerationResponse into a TokenChunk."""
    completion = getattr(response, "generation_tokens", None)
    chunk_prompt = getattr(response, "prompt_tokens", None)
    usage = None
    if completion is not None:
        usage = Usage(
            prompt_tokens=prompt_tokens if chunk_prompt is None else int(chunk_prompt),
            completion_tokens=int(completion),
        )
    return TokenChunk(
        text=getattr(response, "text", "") or "",
        finish_reason=getattr(response, "finish_reason", None),
        usage=usage,
    )


class Engine:
    """Loads an MLX model once and runs chat generations against it."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_path: Path | None = None

    @property
    def settings(self) -> Settings:
        """Settings resolved from the environment on first access."""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Path | None:
        return self._model_path

    @property
    def markers(self) -> ModelMarkers:
        """Reasoning/tool markers of the loaded tokenizer (defaults if unknown)."""
        markers = markers_for_tokenizer(self._tokenizer)
        return markers if markers.introspected else default_markers()

    @property
    def tool_call_parser(self) -> Any:
        """The mlx_lm per-model tool-call parser, when the tokenizer has one."""
        return getattr(self._tokenizer, "tool_parser", None)

    def count_tokens(self, text: str) -> int:
        """Count tokens in generated text with the loaded tokenizer."""
        self.ensure_loaded()
        return count_tokens(self._tokenizer, text)

    def load(self) -> Path:
        """Load the weights into memory once; later calls return the path."""
        existing = self._model_path
        if existing is not None:
            return existing
        path = resolve_model_path(self.settings)
        try:
            model, tokenizer = mlx_load(str(path))
        except MemoryError:
            raise
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise InferenceError(f"Failed to load model from {path}: {detail}") from None
        self._model = model
        self._tokenizer = tokenizer
        self._model_path = path
        return path

    def ensure_loaded(self) -> None:
        """Raise ModelNotLoadedError unless load() already succeeded."""
        if self._model is None or self._tokenizer is None:
            raise ModelNotLoadedError(
                "The model is not loaded. Set LLM_MODEL, download the weights "
                "(`make download MODEL=mlx-community/<repo>`) and start the "
                "server with `make run`."
            )

    def generate(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        """One-shot generation with usage accounting and a finish reason."""
        self.ensure_loaded()
        try:
            params = params or GenerationParams.from_settings(self.settings)
            prompt, prompt_tokens = build_prompt(
                self._tokenizer,
                messages,
                tools=params.tools,
                enable_thinking=params.enable_thinking,
            )
            sampler = mlx_make_sampler(temp=params.temperature, top_p=params.top_p)
            text = mlx_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=params.max_tokens,
                sampler=sampler,
            )
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Generation failed: {exc}") from None
        text = text or ""
        completion_tokens = count_tokens(self._tokenizer, text)
        finish_reason = "length" if completion_tokens >= params.max_tokens else "stop"
        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
        )

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> Iterator[TokenChunk]:
        """Prepare a token generator; preparation errors surface immediately."""
        self.ensure_loaded()
        try:
            params = params or GenerationParams.from_settings(self.settings)
            prompt, prompt_tokens = build_prompt(
                self._tokenizer,
                messages,
                tools=params.tools,
                enable_thinking=params.enable_thinking,
            )
            sampler = mlx_make_sampler(temp=params.temperature, top_p=params.top_p)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Streaming generation failed: {exc}") from None
        return self._stream(prompt, prompt_tokens, params, sampler)

    def _stream(
        self,
        prompt: str | list[int],
        prompt_tokens: int,
        params: GenerationParams,
        sampler: Any,
    ) -> Iterator[TokenChunk]:
        """Yield TokenChunks from mlx_lm.stream_generate, wrapping failures."""
        try:
            responses = mlx_stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=params.max_tokens,
                sampler=sampler,
            )
            for response in responses:
                yield token_chunk(response, prompt_tokens)
        except Exception as exc:
            raise InferenceError(f"Streaming generation failed: {exc}") from None
