"""Epic 4 - inference layer over mlx_lm.

US 4.1: the model is loaded once at startup; a missing model gives an
        actionable error ("first run `make download`").
US 4.2: generation from chat messages applies the chat template and the
        max_tokens / temperature / top_p parameters, counts usage and
        reports a finish_reason.
US 4.3: streaming yields tokens one by one with a correct stop signal;
        a failure inside the generator surfaces as InferenceError so the
        server can keep running.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_llm import inference
from local_llm.config import Settings, SettingsError
from local_llm.inference import (
    Engine,
    GenerationParams,
    InferenceError,
    ModelNotLoadedError,
    TokenChunk,
    Usage,
    build_prompt,
    resolve_model_path,
)

MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Drop ambient LLM_*/HF_TOKEN vars and ignore any project .env file."""
    for name in list(os.environ):
        if name.startswith("LLM_") or name == "HF_TOKEN":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ fixtures


class FakeTokenizer:
    """Stand-in for mlx_lm's TokenizerWrapper / a raw HF tokenizer."""

    def __init__(self, chat_template: bool = True) -> None:
        self.chat_template = "chat-template" if chat_template else None
        self.template_calls: list[dict[str, Any]] = []
        self.encoded_texts: list[str] = []
        self.rendered = ""
        self.last_ids: list[int] = []
        self.template_error: BaseException | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> Any:
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "tools": kwargs.get("tools"),
                "enable_thinking": kwargs.get("enable_thinking"),
            }
        )
        if self.template_error is not None:
            raise self.template_error
        lines = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            lines.append("assistant:")
        self.rendered = "\n".join(lines)
        self.last_ids = list(range(1, len(self.rendered.split()) + 1))
        return self.last_ids if tokenize else self.rendered

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self.encoded_texts.append(text)
        return list(range(1, len(text.split()) + 1))


class FakeLoad:
    """Stand-in for mlx_lm.load."""

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls: list[str] = []
        self.error: BaseException | None = None

    def __call__(self, path: str) -> tuple[Any, Any]:
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        return "fake-model", self.tokenizer


class FakeGenerate:
    """Stand-in for mlx_lm.generate."""

    def __init__(self, text: str = "Hello world") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []
        self.error: BaseException | None = None

    def __call__(
        self,
        model: Any,
        tokenizer: Any,
        prompt: Any,
        *,
        max_tokens: int,
        sampler: Any,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
        )
        if self.error is not None:
            raise self.error
        return self.text


class FakeSamplerFactory:
    """Stand-in for mlx_lm.sample_utils.make_sampler."""

    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []
        self.sampler = object()

    def __call__(self, *, temp: float, top_p: float) -> Any:
        self.calls.append({"temp": temp, "top_p": top_p})
        return self.sampler


class FakeStreamGenerate:
    """Stand-in for mlx_lm.stream_generate; may yield exceptions to fail."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[Any] = []

    def __call__(
        self,
        model: Any,
        tokenizer: Any,
        prompt: Any,
        *,
        max_tokens: int,
        sampler: Any,
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
        )

        def iterate() -> Any:
            for item in self.responses:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return iterate()


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def load_mock(
    monkeypatch: pytest.MonkeyPatch, tokenizer: FakeTokenizer
) -> FakeLoad:
    fake = FakeLoad(tokenizer)
    monkeypatch.setattr(inference, "mlx_load", fake)
    return fake


@pytest.fixture
def generate_mock(monkeypatch: pytest.MonkeyPatch) -> FakeGenerate:
    fake = FakeGenerate()
    monkeypatch.setattr(inference, "mlx_generate", fake)
    return fake


@pytest.fixture
def sampler_mock(monkeypatch: pytest.MonkeyPatch) -> FakeSamplerFactory:
    fake = FakeSamplerFactory()
    monkeypatch.setattr(inference, "mlx_make_sampler", fake)
    return fake


@pytest.fixture
def stream_mock(monkeypatch: pytest.MonkeyPatch) -> FakeStreamGenerate:
    fake = FakeStreamGenerate()
    monkeypatch.setattr(inference, "mlx_stream_generate", fake)
    return fake


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    path = tmp_path / "models"
    (path / "TinyModel").mkdir(parents=True)
    return path


@pytest.fixture
def settings(models_dir: Path) -> Settings:
    return Settings(model="mlx-community/TinyModel", models_dir=models_dir)


@pytest.fixture
def engine(settings: Settings) -> Engine:
    return Engine(settings)


def make_response(
    text: str,
    finish_reason: str | None = None,
    prompt_tokens: int = 7,
    generation_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
    )


# --------------------------------------------------------------------- US 4.1


def test_model_loads_once_and_caches_the_path(
    engine: Engine, load_mock: FakeLoad, models_dir: Path
) -> None:
    first = engine.load()
    second = engine.load()

    expected = models_dir / "TinyModel"
    assert first == expected
    assert second == expected
    assert engine.is_loaded
    assert engine.model_path == expected
    assert load_mock.calls == [str(expected)]


def test_missing_model_folder_hints_at_download(
    tmp_path: Path, load_mock: FakeLoad
) -> None:
    engine = Engine(Settings(model="mlx-community/Ghost", models_dir=tmp_path / "m"))
    with pytest.raises(InferenceError) as excinfo:
        engine.load()
    assert "make download MODEL=mlx-community/Ghost" in str(excinfo.value)
    assert load_mock.calls == []


def test_missing_llm_model_setting_is_actionable(
    tmp_path: Path, load_mock: FakeLoad
) -> None:
    engine = Engine(Settings(models_dir=tmp_path / "m"))
    with pytest.raises(SettingsError) as excinfo:
        engine.load()
    assert "LLM_MODEL" in str(excinfo.value)
    assert load_mock.calls == []


def test_explicit_model_directory_is_used(
    tmp_path: Path, load_mock: FakeLoad
) -> None:
    custom = tmp_path / "custom-weights"
    custom.mkdir()
    engine = Engine(Settings(model=str(custom), models_dir=tmp_path / "m"))
    assert engine.load() == custom
    assert load_mock.calls == [str(custom)]


def test_mlx_load_failure_is_wrapped(engine: Engine, load_mock: FakeLoad) -> None:
    load_mock.error = RuntimeError("metal OOM")
    with pytest.raises(InferenceError) as excinfo:
        engine.load()
    assert "metal OOM" in str(excinfo.value)
    assert not engine.is_loaded


def test_memory_error_during_load_is_not_wrapped(
    engine: Engine, load_mock: FakeLoad
) -> None:
    load_mock.error = MemoryError()
    with pytest.raises(MemoryError):
        engine.load()
    assert not engine.is_loaded


def test_generate_before_load_raises(engine: Engine) -> None:
    with pytest.raises(ModelNotLoadedError) as excinfo:
        engine.generate(MESSAGES)
    assert isinstance(excinfo.value, InferenceError)
    assert "make download" in str(excinfo.value)


def test_engine_resolves_settings_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, load_mock: FakeLoad
) -> None:
    models = tmp_path / "env-models"
    (models / "EnvModel").mkdir(parents=True)
    monkeypatch.setenv("LLM_MODEL", "EnvModel")
    monkeypatch.setenv("LLM_MODELS_DIR", str(models))
    engine = Engine()
    assert not engine.is_loaded
    assert engine.load() == models / "EnvModel"


def test_resolve_model_path_rejects_missing_absolute_path(
    tmp_path: Path,
) -> None:
    settings = Settings(model=str(tmp_path / "nope"), models_dir=tmp_path / "m")
    with pytest.raises(InferenceError, match="does not exist"):
        resolve_model_path(settings)


# --------------------------------------------------------------------- US 4.2


def test_generate_applies_the_chat_template(
    engine: Engine,
    load_mock: FakeLoad,
    tokenizer: FakeTokenizer,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    engine.generate(MESSAGES, GenerationParams(32, 0.5, 0.9))
    call = tokenizer.template_calls[0]
    assert call["messages"] == MESSAGES
    assert call["tokenize"] is True
    assert call["add_generation_prompt"] is True
    assert generate_mock.calls[0]["prompt"] == tokenizer.last_ids


def test_generate_without_chat_template_renders_plain_text(
    engine: Engine,
    load_mock: FakeLoad,
    tokenizer: FakeTokenizer,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    tokenizer.chat_template = None
    engine.load()
    engine.generate(MESSAGES, GenerationParams(32, 0.5, 0.9))
    expected = "system: You are helpful.\nuser: Hi\nassistant:"
    assert generate_mock.calls[0]["prompt"] == expected
    assert tokenizer.encoded_texts[0] == expected
    assert "Hello world" in tokenizer.encoded_texts


def test_generate_forwards_sampling_parameters(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    engine.generate(MESSAGES, GenerationParams(64, 1.5, 0.42))
    assert sampler_mock.calls == [{"temp": 1.5, "top_p": 0.42}]
    assert generate_mock.calls[0]["max_tokens"] == 64
    assert generate_mock.calls[0]["sampler"] is sampler_mock.sampler
    assert generate_mock.calls[0]["model"] == "fake-model"


@pytest.mark.parametrize(
    ("temperature", "top_p"), [(0.0, 0.0), (2.0, 1.0)]
)
def test_generate_boundary_sampling_values_pass_through(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
    temperature: float,
    top_p: float,
) -> None:
    engine.load()
    engine.generate(MESSAGES, GenerationParams(8, temperature, top_p))
    assert sampler_mock.calls[-1] == {"temp": temperature, "top_p": top_p}


def test_generate_defaults_come_from_settings(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    engine.generate(MESSAGES)
    assert generate_mock.calls[0]["max_tokens"] == 512
    assert sampler_mock.calls[0] == {"temp": 0.7, "top_p": 1.0}


def test_generate_counts_usage_and_reports_stop(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
    tokenizer: FakeTokenizer,
) -> None:
    engine.load()
    generate_mock.text = "Hi there friend"
    result = engine.generate(MESSAGES, GenerationParams(10, 0.0, 1.0))
    assert result.text == "Hi there friend"
    assert result.finish_reason == "stop"
    assert result.usage.completion_tokens == 3
    assert result.usage.prompt_tokens == len(tokenizer.rendered.split())
    assert result.usage.total_tokens == result.usage.prompt_tokens + 3
    assert isinstance(result.usage, Usage)


def test_generate_reports_length_at_the_token_limit(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    generate_mock.text = "one two three"
    result = engine.generate(MESSAGES, GenerationParams(3, 0.0, 1.0))
    assert result.finish_reason == "length"


def test_generate_with_empty_output_counts_zero_tokens(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    generate_mock.text = ""
    result = engine.generate(MESSAGES, GenerationParams(8, 0.0, 1.0))
    assert result.text == ""
    assert result.usage.completion_tokens == 0
    assert result.finish_reason == "stop"


def test_generate_wraps_mlx_failures(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    generate_mock.error = RuntimeError("boom")
    with pytest.raises(InferenceError, match="boom"):
        engine.generate(MESSAGES)


def test_generate_rejects_empty_messages(
    engine: Engine, load_mock: FakeLoad
) -> None:
    engine.load()
    with pytest.raises(InferenceError, match="must not be empty"):
        engine.generate([])


def test_chat_template_failure_is_wrapped(
    engine: Engine,
    load_mock: FakeLoad,
    tokenizer: FakeTokenizer,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    tokenizer.template_error = ValueError("bad template")
    engine.load()
    with pytest.raises(InferenceError, match="bad template"):
        engine.generate(MESSAGES)


def test_chat_template_detection() -> None:
    class WrapperLike:
        has_chat_template = True
        chat_template = None

    class RawHF:
        chat_template = "template"

    class Plain:
        chat_template = None

    assert inference.has_chat_template(WrapperLike()) is True
    assert inference.has_chat_template(RawHF()) is True
    assert inference.has_chat_template(Plain()) is False


def test_mlx_sampler_boundary_is_callable() -> None:
    assert callable(inference.mlx_make_sampler(temp=0.0, top_p=1.0))


# --------------------------------------------------------------------- US 4.3


def test_stream_yields_tokens_one_by_one(
    engine: Engine,
    load_mock: FakeLoad,
    stream_mock: FakeStreamGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    stream_mock.responses = [
        make_response("Hel", generation_tokens=1),
        make_response("lo", generation_tokens=2),
        make_response("!", finish_reason="stop", generation_tokens=3),
    ]
    iterator = engine.stream(MESSAGES, GenerationParams(16, 0.3, 0.8))

    first = next(iterator)
    assert isinstance(first, TokenChunk)
    assert (first.text, first.finish_reason) == ("Hel", None)
    assert first.usage == Usage(prompt_tokens=7, completion_tokens=1)

    second = next(iterator)
    assert (second.text, second.finish_reason) == ("lo", None)
    assert second.usage == Usage(prompt_tokens=7, completion_tokens=2)

    third = next(iterator)
    assert (third.text, third.finish_reason) == ("!", "stop")
    assert third.usage == Usage(prompt_tokens=7, completion_tokens=3)

    with pytest.raises(StopIteration):
        next(iterator)
    assert stream_mock.calls[0]["max_tokens"] == 16
    assert sampler_mock.calls == [{"temp": 0.3, "top_p": 0.8}]


def test_stream_defaults_come_from_settings(
    engine: Engine,
    load_mock: FakeLoad,
    stream_mock: FakeStreamGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    stream_mock.responses = [make_response("x", finish_reason="stop", generation_tokens=1)]
    chunks = list(engine.stream(MESSAGES))
    assert len(chunks) == 1
    assert stream_mock.calls[0]["max_tokens"] == 512
    assert sampler_mock.calls[0] == {"temp": 0.7, "top_p": 1.0}


def test_stream_surfaces_generator_failure_as_inference_error(
    engine: Engine,
    load_mock: FakeLoad,
    stream_mock: FakeStreamGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    stream_mock.responses = [
        make_response("Hel", generation_tokens=1),
        RuntimeError("gpu gone"),
    ]
    iterator = engine.stream(MESSAGES)
    assert next(iterator).text == "Hel"
    with pytest.raises(InferenceError, match="gpu gone"):
        next(iterator)


def test_stream_requires_a_loaded_model(engine: Engine) -> None:
    with pytest.raises(ModelNotLoadedError):
        engine.stream(MESSAGES)


def test_stream_rejects_empty_messages(
    engine: Engine, load_mock: FakeLoad
) -> None:
    engine.load()
    with pytest.raises(InferenceError, match="must not be empty"):
        engine.stream([])


def test_stream_handles_responses_without_metadata(
    engine: Engine,
    load_mock: FakeLoad,
    stream_mock: FakeStreamGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    engine.load()
    stream_mock.responses = [SimpleNamespace(text="x")]
    chunks = list(engine.stream(MESSAGES))
    assert chunks == [TokenChunk(text="x", finish_reason=None, usage=None)]


def test_stream_prep_failure_is_wrapped(
    engine: Engine,
    load_mock: FakeLoad,
    stream_mock: FakeStreamGenerate,
    sampler_mock: FakeSamplerFactory,
    tokenizer: FakeTokenizer,
) -> None:
    tokenizer.template_error = ValueError("bad template")
    engine.load()
    with pytest.raises(InferenceError, match="bad template"):
        engine.stream(MESSAGES)


def test_stream_sampler_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine,
    load_mock: FakeLoad,
    stream_mock: FakeStreamGenerate,
    sampler_mock: FakeSamplerFactory,
) -> None:
    def explode(*, temp: float, top_p: float) -> Any:
        raise RuntimeError("sampler down")

    monkeypatch.setattr(inference, "mlx_make_sampler", explode)
    engine.load()
    with pytest.raises(InferenceError, match="sampler down"):
        engine.stream(MESSAGES)


# ------------------------------------------------- Responses API: tools/markers


def test_build_prompt_forwards_tools_in_the_chat_completions_shape() -> None:
    tokenizer = FakeTokenizer()
    tools = [
        {
            "type": "function",
            "name": "weather",
            "description": "d",
            "parameters": {"type": "object"},
        }
    ]

    build_prompt(tokenizer, MESSAGES, tools=tools)

    assert tokenizer.template_calls[-1]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_build_prompt_without_tools_omits_the_kwarg() -> None:
    tokenizer = FakeTokenizer()

    build_prompt(tokenizer, MESSAGES)

    assert tokenizer.template_calls[-1]["tools"] is None


def test_tools_are_rejected_when_the_tokenizer_advertises_no_support() -> None:
    class NoToolTokenizer(FakeTokenizer):
        has_tool_calling = False

    with pytest.raises(inference.UnsupportedToolsError, match="tools"):
        build_prompt(
            NoToolTokenizer(),
            MESSAGES,
            tools=[{"type": "function", "name": "f", "parameters": {}}],
        )


def test_tools_are_rejected_when_the_template_refuses_them() -> None:
    class StrictTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool = True,
            add_generation_prompt: bool = True,
            **kwargs: Any,
        ) -> Any:
            if "tools" in kwargs:
                raise TypeError(
                    "apply_chat_template() got an unexpected keyword argument 'tools'"
                )
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

    with pytest.raises(inference.UnsupportedToolsError):
        build_prompt(
            StrictTokenizer(),
            MESSAGES,
            tools=[{"type": "function", "name": "f", "parameters": {}}],
        )


def test_tools_without_a_chat_template_are_rejected() -> None:
    with pytest.raises(inference.UnsupportedToolsError):
        build_prompt(
            FakeTokenizer(chat_template=False),
            MESSAGES,
            tools=[{"type": "function", "name": "f", "parameters": {}}],
        )


def test_generate_forwards_tools_from_params(
    engine: Engine,
    load_mock: FakeLoad,
    generate_mock: FakeGenerate,
    sampler_mock: FakeSamplerFactory,
    tokenizer: FakeTokenizer,
) -> None:
    engine.load()
    params = GenerationParams(
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        tools=({"type": "function", "name": "f", "parameters": {}},),
    )

    result = engine.generate(MESSAGES, params)

    assert result.text == "Hello world"
    assert tokenizer.template_calls[-1]["tools"] == [
        {"type": "function", "function": {"name": "f", "parameters": {}}}
    ]


def test_stream_forwards_tools_from_params(
    engine: Engine,
    load_mock: FakeLoad,
    stream_mock: FakeStreamGenerate,
    sampler_mock: FakeSamplerFactory,
    tokenizer: FakeTokenizer,
) -> None:
    engine.load()
    params = GenerationParams(
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        tools=({"type": "function", "name": "f", "parameters": {}},),
    )

    list(engine.stream(MESSAGES, params))

    assert tokenizer.template_calls[-1]["tools"] == [
        {"type": "function", "function": {"name": "f", "parameters": {}}}
    ]


def test_engine_markers_default_for_an_unknown_tokenizer(engine: Engine) -> None:
    markers = engine.markers

    assert markers.has_reasoning
    assert markers.has_tools


def test_engine_markers_read_the_tokenizer_metadata(engine: Engine) -> None:
    tokenizer = SimpleNamespace(
        has_thinking=True,
        think_start="<T>",
        think_end="</T>",
        has_tool_calling=True,
        tool_call_start="<C>",
        tool_call_end="</C>",
        structural_markers=("<S>",),
    )
    engine._tokenizer = tokenizer

    markers = engine.markers

    assert markers.reasoning_start == "<T>"
    assert markers.reasoning_end == "</T>"
    assert markers.tool_start == "<C>"
    assert markers.tool_end == "</C>"
    assert markers.structural == ("<S>",)


def test_markers_for_a_tokenizer_without_thinking_or_tools() -> None:
    tokenizer = SimpleNamespace(
        has_thinking=False,
        think_start=None,
        think_end=None,
        has_tool_calling=False,
        tool_call_start=None,
        tool_call_end=None,
        structural_markers=(),
    )

    markers = inference.markers_for_tokenizer(tokenizer)

    assert markers.introspected is True
    assert markers.has_reasoning is False
    assert markers.has_tools is False


def test_engine_exposes_the_tokenizer_tool_parser(engine: Engine) -> None:
    parser = lambda body: {"name": "f", "arguments": {}}
    engine._tokenizer = SimpleNamespace(tool_parser=parser)

    assert engine.tool_call_parser is parser


def test_engine_count_tokens_uses_the_tokenizer(
    engine: Engine, load_mock: FakeLoad, tokenizer: FakeTokenizer
) -> None:
    engine.load()

    assert engine.count_tokens("two words") == 2
    assert tokenizer.encoded_texts[-1] == "two words"


def test_engine_count_tokens_requires_a_loaded_model(engine: Engine) -> None:
    with pytest.raises(ModelNotLoadedError):
        engine.count_tokens("x")


def test_build_prompt_forwards_enable_thinking() -> None:
    tokenizer = FakeTokenizer()

    build_prompt(tokenizer, MESSAGES, enable_thinking=False)

    assert tokenizer.template_calls[-1]["enable_thinking"] is False


def test_enable_thinking_is_dropped_for_templates_that_reject_it() -> None:
    class NoThinkingTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool = True,
            add_generation_prompt: bool = True,
            **kwargs: Any,
        ) -> Any:
            if "enable_thinking" in kwargs:
                raise TypeError(
                    "apply_chat_template() got an unexpected keyword argument "
                    "'enable_thinking'"
                )
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

    tokenizer = NoThinkingTokenizer()

    build_prompt(tokenizer, MESSAGES, enable_thinking=False)

    assert tokenizer.template_calls[-1]["enable_thinking"] is None
