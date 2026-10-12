"""Epic 5 - OpenAI-compatible API.

US 5.1: POST /v1/chat/completions accepts an OpenAI JSON body (without
        streaming) and answers with id / object / created / model /
        choices / usage; unknown fields are ignored; request parameters
        reach the inference engine.
US 5.2: stream=true answers with SSE: text/event-stream, data: {...}
        chunks with a delta, a final data: [DONE] and usage in the last
        chunk when available.
US 5.3: errors come back in the OpenAI envelope: 400 for an invalid
        body, 404 when the model is not loaded, 500 for generation
        failures - without leaking a traceback, and the server keeps
        serving subsequent requests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest
from starlette.testclient import TestClient

from local_llm.config import Settings, SettingsError
from local_llm.inference import (
    GenerationParams,
    GenerationResult,
    InferenceError,
    ModelNotLoadedError,
    TokenChunk,
    Usage,
)
from local_llm.schemas import ChatCompletionChunk, ChatCompletionResponse
from local_llm.server import create_app

DEFAULT_STREAM: list[TokenChunk] = [
    TokenChunk(text="Hel", usage=Usage(prompt_tokens=5, completion_tokens=1)),
    TokenChunk(
        text="lo", finish_reason="stop", usage=Usage(prompt_tokens=5, completion_tokens=2)
    ),
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Drop ambient LLM_*/HF_TOKEN vars and ignore any project .env file."""
    for name in list(os.environ):
        if name.startswith("LLM_") or name == "HF_TOKEN":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


class FakeEngine:
    """Stand-in for local_llm.inference.Engine (the mocked inference layer)."""

    def __init__(self, *, loaded: bool = True) -> None:
        self._settings: Settings | None = None
        self.loaded = loaded
        self.generate_result = GenerationResult(
            text="Hello!", finish_reason="stop", usage=Usage(5, 2)
        )
        self.stream_chunks: list[TokenChunk] = list(DEFAULT_STREAM)
        self.generate_error: BaseException | None = None
        self.stream_setup_error: BaseException | None = None
        self.stream_error: BaseException | None = None
        self.stream_error_after: int = 0
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def generate(
        self, messages: list[dict[str, str]], params: GenerationParams | None = None
    ) -> GenerationResult:
        self.generate_calls.append({"messages": messages, "params": params})
        if self.generate_error is not None:
            raise self.generate_error
        return self.generate_result

    def stream(
        self, messages: list[dict[str, str]], params: GenerationParams | None = None
    ) -> Iterator[TokenChunk]:
        self.stream_calls.append({"messages": messages, "params": params})
        if self.stream_setup_error is not None:
            raise self.stream_setup_error
        return self._iterate()

    def _iterate(self) -> Iterator[TokenChunk]:
        for index, chunk in enumerate(self.stream_chunks):
            if self.stream_error is not None and index >= self.stream_error_after:
                raise self.stream_error
            yield chunk


class BrokenSettingsEngine(FakeEngine):
    """Engine whose settings resolution fails at request time."""

    @property
    def settings(self) -> Settings:
        raise SettingsError("Invalid configuration:\n  - port: oops (env: LLM_PORT)")


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine) -> TestClient:
    return TestClient(create_app(engine=engine))  # type: ignore[arg-type]


def chat_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "TinyModel",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return payload


def sse_payloads(response: Any) -> list[dict[str, Any]]:
    """Parse a text/event-stream body into JSON payloads (skipping [DONE])."""
    payloads: list[dict[str, Any]] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads


# --------------------------------------------------------------------- US 5.1


def test_chat_completions_returns_openai_shaped_response(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 200
    data = response.json()
    ChatCompletionResponse.model_validate(data)
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert isinstance(data["created"], int) and data["created"] > 0
    assert data["model"] == "TinyModel"
    assert data["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ]
    assert data["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


def test_unknown_request_fields_are_ignored(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(n=1, tools=[], user="dev", response_format={"type": "text"}),
    )
    assert response.status_code == 200


def test_messages_and_parameters_reach_the_engine(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            temperature=0.5,
            top_p=0.8,
            max_tokens=42,
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        ),
    )

    assert response.status_code == 200
    call = engine.generate_calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert call["params"] == GenerationParams(42, 0.5, 0.8)


def test_missing_parameters_fall_back_to_settings(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 200
    assert engine.generate_calls[0]["params"] == GenerationParams(
        max_tokens=512, temperature=0.7, top_p=1.0
    )


def test_content_parts_are_flattened(client: TestClient, engine: FakeEngine) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hel"},
                        " plain",
                        {"type": "text", "text": "lo"},
                    ],
                }
            ]
        ),
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["messages"] == [
        {"role": "user", "content": "hel plainlo"}
    ]


def test_non_text_content_parts_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                }
            ]
        ),
    )
    assert response.status_code == 400
    assert "text content parts" in response.json()["error"]["message"]


# --------------------------------------------------------------------- US 5.2


def test_stream_answers_with_sse_chunks_and_done(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"

    payloads = sse_payloads(response)
    for payload in payloads:
        ChatCompletionChunk.model_validate(payload)

    assert payloads[0]["object"] == "chat.completion.chunk"
    assert payloads[0]["model"] == "TinyModel"
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert payloads[0]["choices"][0]["finish_reason"] is None

    contents = [
        choice["delta"].get("content")
        for payload in payloads
        for choice in payload["choices"]
    ]
    assert [text for text in contents if text] == ["Hel", "lo"]

    finished = [
        payload["choices"][0]["finish_reason"]
        for payload in payloads
        if payload["choices"] and payload["choices"][0]["finish_reason"]
    ]
    assert finished == ["stop"]

    assert payloads[-1]["choices"] == []
    assert payloads[-1]["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }
    assert len({payload["id"] for payload in payloads}) == 1
    assert engine.stream_calls[0]["messages"] == [
        {"role": "user", "content": "hi"}
    ]


def test_stream_without_usage_omits_the_usage_chunk(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [TokenChunk(text="x", finish_reason="stop")]
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    payloads = sse_payloads(response)
    assert all("usage" not in payload for payload in payloads)
    assert response.text.rstrip().endswith("data: [DONE]")


def test_stream_keeps_unicode_text_readable(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [
        TokenChunk(text="Привет", finish_reason="stop", usage=Usage(1, 1))
    ]
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    assert "Привет" in response.text
    assert "\\u041f" not in response.text


def test_stream_setup_failure_returns_an_openai_500(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_setup_error = InferenceError("no model")
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert error["message"] == "no model"
    assert error["code"] == "generation_failed"
    assert "Traceback" not in response.text


def test_stream_setup_model_not_loaded_maps_to_404(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_setup_error = ModelNotLoadedError("not loaded")
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_stream_failure_midway_keeps_the_server_running(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_error = InferenceError("gpu gone")
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    payloads = sse_payloads(response)
    errors = [payload for payload in payloads if "error" in payload]
    assert errors[0]["error"] == {
        "message": "gpu gone",
        "type": "server_error",
        "code": "generation_failed",
    }
    assert response.text.rstrip().endswith("data: [DONE]")
    assert "Traceback" not in response.text

    follow_up = client.post("/v1/chat/completions", json=chat_payload())
    assert follow_up.status_code == 200


def test_stream_unexpected_failure_is_masked(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_error = ValueError("secret internals")
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    payloads = sse_payloads(response)
    errors = [payload for payload in payloads if "error" in payload]
    assert errors[0]["error"]["message"] == "Internal server error"
    assert "secret internals" not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def test_stream_engine_setup_failure_masked_when_unexpected(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_setup_error = ValueError("secret internals")
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Internal server error"
    assert "secret internals" not in response.text


def test_stream_finish_reason_defaults_to_stop(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [TokenChunk(text="x", usage=Usage(2, 1))]
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    payloads = sse_payloads(response)
    finished = [
        payload["choices"][0]["finish_reason"]
        for payload in payloads
        if payload["choices"] and payload["choices"][0]["finish_reason"]
    ]
    assert finished == ["stop"]


# --------------------------------------------------------------------- US 5.3


def test_malformed_json_body_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        content="{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"]
    assert error["code"]
    assert "Traceback" not in response.text


def test_missing_messages_returns_400(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={"model": "TinyModel"})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "messages" in error["message"]
    assert error["code"] == "missing"


def test_empty_messages_returns_400(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload(messages=[]))
    assert response.status_code == 400
    assert "messages" in response.json()["error"]["message"]


def test_temperature_out_of_range_returns_400(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload(temperature=5))

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "temperature" in error["message"]


def test_blank_model_returns_400(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload(model="   "))
    assert response.status_code == 400
    assert "model" in response.json()["error"]["message"]


def test_unknown_role_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(messages=[{"role": "wizard", "content": "hi"}]),
    )
    assert response.status_code == 400
    assert "role" in response.json()["error"]["message"]


def test_unloaded_model_returns_404(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.loaded = False
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert "make download" in error["message"]
    assert engine.generate_calls == []


def test_unloaded_model_returns_json_404_for_streams(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.loaded = False
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "model_not_found"


def test_engine_model_not_loaded_error_maps_to_404(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = ModelNotLoadedError("not loaded")
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_generation_failure_returns_500_without_traceback(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = InferenceError("model exploded")
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert error["message"] == "model exploded"
    assert error["code"] == "generation_failed"
    assert "Traceback" not in response.text
    assert "local_llm/" not in response.text


def test_unexpected_generation_failure_is_masked(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = ValueError("secret internals")
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Internal server error"
    assert "secret internals" not in response.text
    assert "Traceback" not in response.text


def test_server_keeps_working_after_a_failed_request(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = InferenceError("boom")
    assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 500

    engine.generate_error = None
    assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 200


def test_broken_settings_return_a_server_error() -> None:
    client = TestClient(create_app(engine=BrokenSettingsEngine()))
    response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert "Invalid configuration" in error["message"]


def test_default_module_app_is_buildable() -> None:
    from local_llm import server

    client = TestClient(server.app)
    response = client.post("/v1/chat/completions", json=chat_payload())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
