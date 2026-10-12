"""Epic 8 - OpenAI Responses API.

POST /v1/responses answers with a Response object and, with stream=true, with
typed SSE events (response.created -> ... -> response.completed -> [DONE]).
GET /v1/responses/{id} serves stored responses; background=true stores an
in_progress response and completes it asynchronously.

All tests run on FakeEngine: no real MLX model is required.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from starlette.testclient import TestClient

from local_llm.config import Settings
from local_llm.inference import (
    GenerationParams,
    GenerationResult,
    InferenceError,
    ModelMarkers,
    ModelNotLoadedError,
    TokenChunk,
    UnsupportedToolsError,
    Usage,
    default_markers,
)
from local_llm.responses import ModelOutputParser, messages_from_input
from local_llm.schemas import ResponseObject, ResponseRequest
from local_llm.server import MAX_BACKGROUND_TASKS, TokenStream, create_app
from local_llm.store import ResponseStore, StoredResponse

LT = chr(0x3C)
GT = chr(0x3E)
THINK_START = f"{LT}think{GT}"
THINK_END = f"{LT}/think{GT}"
TOOL_START = f"{LT}tool_call{GT}"
TOOL_END = f"{LT}/tool_call{GT}"

DEFAULT_CHUNKS = [
    TokenChunk(text="Hel", usage=Usage(prompt_tokens=5, completion_tokens=1)),
    TokenChunk(
        text="lo",
        finish_reason="stop",
        usage=Usage(prompt_tokens=5, completion_tokens=2),
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
    """Stand-in for local_llm.inference.Engine."""

    def __init__(self, *, loaded: bool = True) -> None:
        self._settings: Settings | None = None
        self.loaded = loaded
        self.generate_result = GenerationResult(
            text="Hello!", finish_reason="stop", usage=Usage(5, 2)
        )
        self.stream_chunks: list[TokenChunk] = list(DEFAULT_CHUNKS)
        self.generate_error: BaseException | None = None
        self.stream_setup_error: BaseException | None = None
        self.stream_error: BaseException | None = None
        self.stream_error_after: int = 0
        self.stream_factory: Any = None
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self._markers = default_markers()
        self.tool_call_parser: Any = None
        self.token_counts: dict[str, int] = {}
        self.generate_delay = 0.0
        self.generate_gate: threading.Event | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def markers(self) -> ModelMarkers:
        return self._markers

    def count_tokens(self, text: str) -> int:
        if text in self.token_counts:
            return self.token_counts[text]
        return len(text.split())

    def generate(
        self, messages: list[dict[str, Any]], params: GenerationParams | None = None
    ) -> GenerationResult:
        self.generate_calls.append({"messages": messages, "params": params})
        if self.generate_gate is not None:
            self.generate_gate.wait(10)
        if self.generate_delay:
            time.sleep(self.generate_delay)
        if self.generate_error is not None:
            raise self.generate_error
        return self.generate_result

    def stream(
        self, messages: list[dict[str, Any]], params: GenerationParams | None = None
    ) -> Iterator[TokenChunk]:
        self.stream_calls.append({"messages": messages, "params": params})
        if self.stream_setup_error is not None:
            raise self.stream_setup_error
        if self.stream_factory is not None:
            return self.stream_factory(messages, params)
        return self._iterate()

    def _iterate(self) -> Iterator[TokenChunk]:
        for index, chunk in enumerate(self.stream_chunks):
            if self.stream_error is not None and index >= self.stream_error_after:
                raise self.stream_error
            yield chunk


class ClosableStream:
    """Token iterator that can be stopped and records when it is closed.

    The first chunks arrive immediately; afterwards the iterator blocks until
    the test releases the gate, so it is still running when the client
    disconnects.
    """

    def __init__(self, gate: threading.Event, closed: threading.Event) -> None:
        self._gate = gate
        self._closed = closed
        self._count = 0

    def __iter__(self) -> ClosableStream:
        return self

    def __next__(self) -> TokenChunk:
        if self._count == 2:
            if not self._gate.wait(10):  # safety net: never hang the test suite
                raise StopIteration
        if self._count >= 3:
            raise StopIteration
        self._count += 1
        return TokenChunk(text=f"chunk{self._count} ")

    def close(self) -> None:
        self._closed.set()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine) -> TestClient:
    return TestClient(create_app(engine=engine))  # type: ignore[arg-type]


def responses_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": "TinyModel", "input": "hi"}
    payload.update(overrides)
    return payload


def sse_events(response: Any) -> list[tuple[str, dict[str, Any]]]:
    """Parse ``event:``/``data:`` pairs; [DONE] comes back as ("done", {})."""
    events: list[tuple[str, dict[str, Any]]] = []
    name: str | None = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                events.append(("done", {}))
                continue
            payload = json.loads(data)
            events.append((name or payload.get("type", ""), payload))
            name = None
    return events


def event_names(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [name for name, _ in events]


def named(events: list[tuple[str, dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    return [payload for event, payload in events if event == name]


# ------------------------------------------------------- level A: plain answers


def test_plain_text_response_is_a_valid_response_object(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 200
    data = response.json()
    ResponseObject.model_validate(data)
    assert data["object"] == "response"
    assert data["id"].startswith("resp_")
    assert data["status"] == "completed"
    assert data["model"] == "TinyModel"
    assert data["error"] is None
    assert data["incomplete_details"] is None
    assert data["output"] == [
        {
            "type": "message",
            "id": data["output"][0]["id"],
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello!",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    ]
    assert data["usage"] == {
        "input_tokens": 5,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 7,
    }


def test_string_input_becomes_a_user_message(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(input="hello"))

    assert response.status_code == 200
    assert engine.generate_calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_message_array_preserves_order_and_roles(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            input=[
                {"role": "system", "content": "be brief"},
                {"role": "developer", "content": "rules"},
                {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "a"}],
                },
            ]
        ),
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_function_call_items_become_assistant_tool_calls(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            input=[
                {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "call_2", "name": "g", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "42"},
            ]
        ),
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "42", "tool_call_id": "call_1"},
    ]


def test_reasoning_items_are_replayed_as_reasoning_content(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            input=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "because"}],
                },
                {"role": "user", "content": "go on"},
            ]
        ),
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["messages"] == [
        {"role": "assistant", "content": "", "reasoning_content": "because"},
        {"role": "user", "content": "go on"},
    ]


def test_instructions_become_a_system_message(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(instructions="be terse")
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_temperature_top_p_and_token_limit_reach_the_engine(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(temperature=0.5, top_p=0.8, max_output_tokens=42),
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["params"] == GenerationParams(42, 0.5, 0.8)
    data = response.json()
    assert data["temperature"] == 0.5
    assert data["top_p"] == 0.8
    assert data["max_output_tokens"] == 42


def test_missing_parameters_fall_back_to_settings(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 200
    assert engine.generate_calls[0]["params"] == GenerationParams(
        max_tokens=512, temperature=0.7, top_p=1.0
    )


def test_tools_are_forwarded_to_the_engine(
    client: TestClient, engine: FakeEngine
) -> None:
    tools = [{"type": "function", "name": "weather", "parameters": {"type": "object"}}]
    response = client.post("/v1/responses", json=responses_payload(tools=tools))

    assert response.status_code == 200
    params = engine.generate_calls[0]["params"]
    assert params.tools is not None
    assert params.tools[0].name == "weather"
    assert response.json()["tools"][0]["name"] == "weather"


def test_unknown_fields_are_ignored(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(truncation="auto", user="dev", include=["x"], foo=1),
    )
    assert response.status_code == 200


def test_empty_output_produces_no_items(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_result = GenerationResult(
        text="", finish_reason="stop", usage=Usage(0, 0)
    )
    response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 200
    assert response.json()["output"] == []
    assert response.json()["status"] == "completed"


# ----------------------------------------------------------- level A: streaming


def test_stream_starts_with_created_and_in_progress(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert event_names(events)[:2] == ["response.created", "response.in_progress"]
    created = named(events, "response.created")[0]["response"]
    assert created["id"].startswith("resp_")
    assert created["created_at"] > 0
    assert created["model"] == "TinyModel"
    assert created["status"] == "in_progress"


def test_stream_event_order_for_text(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)

    assert event_names(events) == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
        "done",
    ]
    assert [payload["delta"] for payload in named(events, "response.output_text.delta")] == [
        "Hel",
        "lo",
    ]
    added = named(events, "response.output_item.added")[0]["item"]
    assert added["type"] == "message"
    assert added["role"] == "assistant"
    done = named(events, "response.output_item.done")[0]["item"]
    assert done["content"][0]["text"] == "Hello"


def test_stream_json_events_carry_the_type_field(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    for name, payload in sse_events(response):
        if name == "done":
            continue
        assert payload["type"] == name
        assert isinstance(payload["sequence_number"], int)


def test_no_delta_before_its_output_item_added(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [
        TokenChunk(text="a"),
        TokenChunk(text="b", finish_reason="stop", usage=Usage(2, 2)),
    ]
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)

    opened: dict[str, int] = {}
    delta_events: list[tuple[str, dict[str, Any]]] = [
        (name, payload)
        for name, payload in events
        if "delta" in name and name != "response.created"
    ]
    for name, payload in events:
        if name == "response.output_item.added":
            opened[payload["item"]["id"]] = payload["output_index"]
        elif name.endswith(".delta"):
            assert payload["item_id"] in opened, f"{name} before output_item.added"
            assert payload["output_index"] == opened[payload["item_id"]]
    assert delta_events


def test_output_index_is_stable_and_never_reused(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [
        TokenChunk(text="text "),
        TokenChunk(text=THINK_START + "why" + THINK_END),
        TokenChunk(text=TOOL_START + '{"name": "f", "arguments": {}}' + TOOL_END),
        TokenChunk(text="after", finish_reason="stop", usage=Usage(1, 1)),
    ]
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            stream=True, tools=[{"type": "function", "name": "f", "parameters": {}}]
        ),
    )
    events = sse_events(response)

    indexes: dict[str, int] = {}
    for name, payload in events:
        if name == "response.output_item.added":
            indexes.setdefault(payload["item"]["id"], payload["output_index"])
        if "output_index" in payload:
            if payload.get("item_id"):
                assert indexes[payload["item_id"]] == payload["output_index"]
    indexes_in_order = list(named(events, "response.output_item.added"))
    assert [item["output_index"] for item in indexes_in_order] == [0, 1, 2, 3]
    assert len({item["item"]["id"] for item in indexes_in_order}) == 4


def test_completed_event_carries_usage(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)
    completed = named(events, "response.completed")[0]["response"]

    assert completed["usage"] == {
        "input_tokens": 5,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 7,
    }
    assert completed["status"] == "completed"
    assert completed["output"][0]["content"][0]["text"] == "Hello"


def test_reasoning_tokens_are_reported_in_usage(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.token_counts = {"because": 7}
    engine.generate_result = GenerationResult(
        text=THINK_START + "because" + THINK_END + "answer",
        finish_reason="stop",
        usage=Usage(3, 9),
    )
    response = client.post("/v1/responses", json=responses_payload())

    usage = response.json()["usage"]
    assert usage["output_tokens_details"]["reasoning_tokens"] == 7


def test_stream_ends_with_done(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)

    assert events[-1] == ("done", {})
    assert response.text.rstrip().endswith("data: [DONE]")
    assert event_names(events).index("done") > event_names(events).index(
        "response.completed"
    )


def test_token_limit_produces_incomplete_and_no_completed(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [
        TokenChunk(text="cut", finish_reason="length", usage=Usage(1, 1))
    ]
    response = client.post(
        "/v1/responses", json=responses_payload(stream=True, max_output_tokens=1)
    )
    events = sse_events(response)

    assert "response.completed" not in event_names(events)
    terminal = named(events, "response.incomplete")[0]["response"]
    assert terminal["status"] == "incomplete"
    assert terminal["incomplete_details"] == {"reason": "max_output_tokens"}
    assert event_names(events)[-1] == "done"
    done_item = named(events, "response.output_item.done")[0]["item"]
    assert done_item["status"] == "incomplete"
    assert done_item["content"][0]["text"] == "cut"


def test_non_streaming_token_limit_is_incomplete(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_result = GenerationResult(
        text="cut", finish_reason="length", usage=Usage(1, 1)
    )
    response = client.post("/v1/responses", json=responses_payload())

    data = response.json()
    assert data["status"] == "incomplete"
    assert data["incomplete_details"] == {"reason": "max_output_tokens"}


def test_stream_output_keeps_unicode_readable(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [TokenChunk(text="Привет", finish_reason="stop")]
    response = client.post("/v1/responses", json=responses_payload(stream=True))

    assert "Привет" in response.text
    assert "\\u041f" not in response.text


# ----------------------------------------------- level A: reasoning and tools


def test_reasoning_is_parsed_into_a_reasoning_item(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_result = GenerationResult(
        text="answer " + THINK_START + "because" + THINK_END,
        finish_reason="stop",
        usage=Usage(1, 3),
    )
    response = client.post("/v1/responses", json=responses_payload())
    output = response.json()["output"]

    assert [item["type"] for item in output] == ["message", "reasoning"]
    assert output[0]["content"][0]["text"] == "answer "
    assert output[1]["summary"] == [{"type": "summary_text", "text": "because"}]


def test_reasoning_stream_events(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [
        TokenChunk(text=THINK_START + "why"),
        TokenChunk(text=THINK_END + "answer", finish_reason="stop", usage=Usage(1, 2)),
    ]
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)

    assert event_names(events) == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
        "done",
    ]
    reasoning_item = named(events, "response.output_item.added")[0]["item"]
    assert reasoning_item["type"] == "reasoning"
    assert named(events, "response.reasoning_summary_text.delta")[0]["delta"] == "why"
    assert named(events, "response.output_text.delta")[0]["delta"] == "answer"


def test_function_call_is_parsed_with_arguments(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_result = GenerationResult(
        text="calling " + TOOL_START + '{"name": "weather", "arguments": {"city": "Paris"}}' + TOOL_END,
        finish_reason="stop",
        usage=Usage(1, 3),
    )
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            tools=[{"type": "function", "name": "weather", "parameters": {}}]
        ),
    )
    output = response.json()["output"]

    assert [item["type"] for item in output] == ["message", "function_call"]
    call = output[1]
    assert call["name"] == "weather"
    assert json.loads(call["arguments"]) == {"city": "Paris"}
    assert call["call_id"].startswith("call_")
    assert call["status"] == "completed"


def test_function_call_stream_events(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_chunks = [
        TokenChunk(text=TOOL_START + '{"name": "weather", "argum'),
        TokenChunk(
            text='ents": {"city": "Paris"}}' + TOOL_END,
            finish_reason="stop",
            usage=Usage(1, 4),
        ),
    ]
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            stream=True, tools=[{"type": "function", "name": "weather", "parameters": {}}]
        ),
    )
    events = sse_events(response)

    assert event_names(events) == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
        "done",
    ]
    added = named(events, "response.output_item.added")[0]["item"]
    assert added["type"] == "function_call"
    assert added["name"] == "weather"
    delta = named(events, "response.function_call_arguments.delta")[0]
    assert json.loads(delta["delta"]) == {"city": "Paris"}
    done_item = named(events, "response.output_item.done")[0]["item"]
    assert json.loads(done_item["arguments"]) == {"city": "Paris"}


def test_model_specific_tool_call_parser_is_used(
    client: TestClient, engine: FakeEngine
) -> None:
    def parser(body: str) -> dict[str, Any]:
        return {"name": "custom", "arguments": {"raw": body}}

    engine.tool_call_parser = parser
    engine.generate_result = GenerationResult(
        text=TOOL_START + "not json at all" + TOOL_END,
        finish_reason="stop",
        usage=Usage(1, 1),
    )
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            tools=[{"type": "function", "name": "custom", "parameters": {}}]
        ),
    )
    call = response.json()["output"][0]

    assert call["name"] == "custom"
    assert json.loads(call["arguments"]) == {"raw": "not json at all"}


def test_unparsable_tool_block_is_returned_as_text(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_result = GenerationResult(
        text=TOOL_START + "not json" + TOOL_END,
        finish_reason="stop",
        usage=Usage(1, 1),
    )
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            tools=[{"type": "function", "name": "f", "parameters": {}}]
        ),
    )
    output = response.json()["output"]

    assert [item["type"] for item in output] == ["message"]
    assert "not json" in output[0]["content"][0]["text"]


def test_tools_without_model_support_return_400(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = UnsupportedToolsError("no tool support")
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            tools=[{"type": "function", "name": "f", "parameters": {}}]
        ),
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "unsupported_tools"
    assert "tool" in error["message"]


def test_tools_without_model_support_on_stream_returns_400(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_setup_error = UnsupportedToolsError("no tool support")
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            stream=True, tools=[{"type": "function", "name": "f", "parameters": {}}]
        ),
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "unsupported_tools"


def test_tool_choice_without_tools_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(tool_choice="required")
    )

    assert response.status_code == 400
    assert "tool_choice" in response.json()["error"]["message"]


# ------------------------------------------------------------------ A: parser


def test_marker_split_between_chunks_is_not_lost() -> None:
    parser = ModelOutputParser(enable_tools=True)
    events: list[Any] = []
    for chunk in ["Hello <thi", "nk>think</thi", "nk>world"]:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())

    assert [(event.kind, event.text) for event in events] == [
        ("text", "Hello "),
        ("reasoning", "think"),
        ("text", "world"),
    ]


def test_split_tool_call_marker_is_handled() -> None:
    parser = ModelOutputParser(enable_tools=True)
    events = parser.feed("<tool_ca")
    assert events == []
    events.extend(parser.feed('ll>{"name": "f", "arguments": {}}</tool_'))
    events.extend(parser.feed("call>"))
    events.extend(parser.finish())

    assert [event.kind for event in events] == ["tool_call"]
    assert events[0].call is not None
    assert events[0].call.name == "f"


def test_truncated_tool_call_is_never_a_fake_call() -> None:
    parser = ModelOutputParser(enable_tools=True)
    events = parser.feed(TOOL_START + '{"name": "f", "argum')
    events.extend(parser.finish())

    assert [event.kind for event in events] == ["text"]
    assert events[0].text.startswith(TOOL_START)
    assert "argum" in events[0].text


def test_tool_markers_stay_text_without_tools() -> None:
    parser = ModelOutputParser(enable_tools=False)
    events = parser.feed(TOOL_START + '{"name": "f", "arguments": {}}' + TOOL_END)
    events.extend(parser.finish())

    assert [event.kind for event in events] == ["text"]


def test_messages_from_input_rejects_unsupported_values() -> None:
    with pytest.raises(Exception):
        messages_from_input("   ")


# ----------------------------------------------------- level A: errors and 400


def test_generation_failure_returns_500_without_traceback(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = InferenceError("model exploded")
    response = client.post("/v1/responses", json=responses_payload())

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
    response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Internal server error"
    assert "secret internals" not in response.text


def test_stream_midway_failure_emits_error_then_done(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_error = InferenceError("gpu gone")
    response = client.post("/v1/responses", json=responses_payload(stream=True))

    assert response.status_code == 200
    events = sse_events(response)
    assert event_names(events)[-1] == "done"
    assert "response.completed" not in event_names(events)
    error = named(events, "error")[0]
    assert error["error"]["message"] == "gpu gone"
    assert error["code"] == "generation_failed"
    assert "Traceback" not in response.text

    follow_up = client.post("/v1/responses", json=responses_payload())
    assert follow_up.status_code == 200


def test_stream_unexpected_failure_is_masked(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_error = ValueError("secret internals")
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)

    assert named(events, "error")[0]["message"] == "Internal server error"
    assert "secret internals" not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def test_stream_setup_failure_returns_json_500(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_setup_error = InferenceError("no model")
    response = client.post("/v1/responses", json=responses_payload(stream=True))

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "generation_failed"


def test_unloaded_model_returns_404(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.loaded = False
    response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert engine.generate_calls == []


def test_engine_model_not_loaded_maps_to_404(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = ModelNotLoadedError("not loaded")
    response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_malformed_json_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        content="{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "Traceback" not in response.text


def test_empty_input_returns_400(client: TestClient) -> None:
    response = client.post("/v1/responses", json=responses_payload(input=[]))

    assert response.status_code == 400
    assert "input" in response.json()["error"]["message"]


def test_unsupported_input_item_type_returns_readable_400(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(input=[{"type": "web_search_call"}])
    )

    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "web_search_call" in message
    assert "supported types" in message


def test_unsupported_content_part_returns_readable_400(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "x"}],
                }
            ]
        ),
    )

    assert response.status_code == 400
    assert "input_image" in response.json()["error"]["message"]


def test_stream_and_background_together_return_400(client: TestClient) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(stream=True, background=True)
    )

    assert response.status_code == 400
    assert "background" in response.json()["error"]["message"]


def test_background_with_store_false_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(background=True, store=False)
    )

    assert response.status_code == 400
    assert "store" in response.json()["error"]["message"]


def test_unsupported_text_format_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(
            text={"format": {"type": "json_schema", "name": "x", "schema": {}}}
        ),
    )

    assert response.status_code == 400
    assert "text.format" in response.json()["error"]["message"]


def test_chat_completions_still_works_after_a_responses_error(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = InferenceError("boom")
    assert client.post("/v1/responses", json=responses_payload()).status_code == 500

    engine.generate_error = None
    chat = client.post(
        "/v1/chat/completions",
        json={"model": "TinyModel", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat.status_code == 200
    assert chat.json()["object"] == "chat.completion"
    assert client.post("/v1/responses", json=responses_payload()).status_code == 200


def test_client_disconnect_keeps_the_server_serving(engine: FakeEngine) -> None:
    gate = threading.Event()
    closed = threading.Event()
    engine.stream_factory = lambda messages, params: ClosableStream(gate, closed)
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]

    with client.stream(
        "POST", "/v1/responses", json=responses_payload(stream=True)
    ) as response:
        assert response.status_code == 200
        first_line = next(response.iter_lines())
        assert first_line.startswith("event: response.created")

    # the client is gone; let the aborted generation finish and release the queue
    gate.set()
    follow_up = client.post("/v1/responses", json=responses_payload())
    assert follow_up.status_code == 200
    assert follow_up.json()["status"] == "completed"


def test_token_stream_stops_and_closes_its_iterator() -> None:
    closed = threading.Event()

    class SlowStream:
        """Slow iterator, so it is still running when close() is called."""

        def __init__(self) -> None:
            self._count = 0

        def __iter__(self) -> SlowStream:
            return self

        def __next__(self) -> TokenChunk:
            if self._count >= 100:
                raise StopIteration
            self._count += 1
            time.sleep(0.02)
            return TokenChunk(text=f"chunk{self._count} ")

        def close(self) -> None:
            closed.set()

    stream = TokenStream(SlowStream())

    async def consume() -> list[str]:
        chunks: list[str] = []
        async for chunk in stream:
            chunks.append(chunk.text)
            if len(chunks) == 2:
                break
        return chunks

    assert asyncio.run(consume()) == ["chunk1 ", "chunk2 "]
    stream.close()
    stream.join(5)

    assert stream.stopped is True
    assert closed.is_set() is True


# ----------------------------------------------------- level B: response store


def test_store_get_and_previous_response_id(
    client: TestClient, engine: FakeEngine
) -> None:
    first = client.post("/v1/responses", json=responses_payload(input="first"))
    assert first.status_code == 200
    response_id = first.json()["id"]

    fetched = client.get(f"/v1/responses/{response_id}")
    assert fetched.status_code == 200
    assert fetched.json() == first.json()

    follow_up = client.post(
        "/v1/responses",
        json=responses_payload(input="second", previous_response_id=response_id),
    )
    assert follow_up.status_code == 200
    assert engine.generate_calls[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "second"},
    ]


def test_previous_response_id_does_not_duplicate_instructions(
    client: TestClient, engine: FakeEngine
) -> None:
    first = client.post(
        "/v1/responses", json=responses_payload(instructions="be terse")
    )
    follow_up = client.post(
        "/v1/responses",
        json=responses_payload(
            instructions="be terse",
            input="again",
            previous_response_id=first.json()["id"],
        ),
    )

    assert follow_up.status_code == 200
    messages = engine.generate_calls[1]["messages"]
    assert [message["content"] for message in messages] == [
        "be terse",
        "hi",
        "Hello!",
        "again",
    ]


def test_unknown_previous_response_id_returns_404(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(input="x", previous_response_id="resp_missing"),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "response_not_found"


def test_unknown_response_id_returns_404(client: TestClient) -> None:
    response = client.get("/v1/responses/resp_missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "response_not_found"
    assert "Traceback" not in response.text


def test_store_false_is_not_retrievable(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(store=False))
    response_id = response.json()["id"]

    assert client.get(f"/v1/responses/{response_id}").status_code == 404
    assert (
        client.post(
            "/v1/responses",
            json=responses_payload(input="x", previous_response_id=response_id),
        ).status_code
        == 404
    )
    # a stored response is still available in the same app
    stored = client.post("/v1/responses", json=responses_payload())
    assert client.get(f"/v1/responses/{stored.json()['id']}").status_code == 200


def test_stored_streaming_response_is_retrievable(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)
    response_id = named(events, "response.completed")[0]["response"]["id"]

    fetched = client.get(f"/v1/responses/{response_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "completed"
    assert fetched.json()["output"][0]["content"][0]["text"] == "Hello"


def test_failed_streaming_response_is_stored_as_failed(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.stream_error = InferenceError("gpu gone")
    response = client.post("/v1/responses", json=responses_payload(stream=True))
    events = sse_events(response)
    response_id = named(events, "error")[0]
    assert response_id["message"] == "gpu gone"
    created = named(events, "response.created")[0]["response"]["id"]

    fetched = client.get(f"/v1/responses/{created}").json()
    assert fetched["status"] == "failed"
    assert fetched["error"]["code"] == "generation_failed"


def test_store_ttl_expiry(engine: FakeEngine) -> None:
    clock = [1000.0]
    app = create_app(engine=engine)  # type: ignore[arg-type]
    app.state.response_store = ResponseStore(ttl_seconds=10, clock=lambda: clock[0])
    client = TestClient(app)

    response_id = client.post("/v1/responses", json=responses_payload()).json()["id"]
    assert client.get(f"/v1/responses/{response_id}").status_code == 200

    clock[0] += 11
    assert client.get(f"/v1/responses/{response_id}").status_code == 404


def test_store_max_entries_evicts_the_oldest(engine: FakeEngine) -> None:
    app = create_app(engine=engine)  # type: ignore[arg-type]
    app.state.response_store = ResponseStore(max_entries=1)
    client = TestClient(app)

    first = client.post("/v1/responses", json=responses_payload()).json()["id"]
    second = client.post("/v1/responses", json=responses_payload()).json()["id"]

    assert client.get(f"/v1/responses/{first}").status_code == 404
    assert client.get(f"/v1/responses/{second}").status_code == 200


def test_response_store_is_thread_safe() -> None:
    store = ResponseStore(max_entries=32, ttl_seconds=60)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for step in range(50):
                key = f"resp_{index}_{step}"
                store.put(key, StoredResponse(response={"status": "completed"}))
                store.get(key)
                len(store)
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store) <= 32


def test_get_unknown_response_returns_json_404(client: TestClient) -> None:
    response = client.get("/v1/responses/unknown-id")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


# --------------------------------------------------------- level C: background


def test_background_response_returns_in_progress_and_completes(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(input="bg", background=True)
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "in_progress"
    assert data["output"] == []
    assert data["id"].startswith("resp_")

    fetched = wait_for_status(client, data["id"], "completed")
    assert fetched["output"][0]["content"][0]["text"] == "Hello!"
    assert fetched["usage"]["total_tokens"] == 7
    assert fetched["background"] is True


def test_background_failure_is_recorded_not_raised(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = InferenceError("model exploded")
    response = client.post(
        "/v1/responses", json=responses_payload(background=True)
    )
    response_id = response.json()["id"]

    fetched = wait_for_status(client, response_id, "failed")
    assert fetched["error"]["code"] == "generation_failed"
    assert fetched["error"]["message"] == "model exploded"

    # the app still serves new requests
    engine.generate_error = None
    assert client.post("/v1/responses", json=responses_payload()).status_code == 200


def test_background_unexpected_failure_is_masked(
    client: TestClient, engine: FakeEngine
) -> None:
    engine.generate_error = ValueError("secret internals")
    response = client.post(
        "/v1/responses", json=responses_payload(background=True)
    )
    fetched = wait_for_status(client, response.json()["id"], "failed")

    assert fetched["error"]["message"] == "Internal server error"
    assert "secret internals" not in json.dumps(fetched)


def test_unknown_background_id_returns_404(client: TestClient) -> None:
    assert client.get("/v1/responses/resp_nope").status_code == 404


def test_background_survives_a_slow_generation(
    client: TestClient, engine: FakeEngine
) -> None:
    """The worker owns the generation, not the request's event loop."""
    engine.generate_delay = 0.3
    response = client.post(
        "/v1/responses", json=responses_payload(background=True)
    )
    response_id = response.json()["id"]

    fetched = wait_for_status(client, response_id, "completed", timeout=10)
    assert fetched["output"][0]["content"][0]["text"] == "Hello!"


def test_too_many_background_responses_returns_429(
    client: TestClient, engine: FakeEngine
) -> None:
    gate = threading.Event()
    engine.generate_gate = gate
    accepted = [
        client.post("/v1/responses", json=responses_payload(background=True))
        for _ in range(MAX_BACKGROUND_TASKS)
    ]
    extra = client.post("/v1/responses", json=responses_payload(background=True))

    assert [response.status_code for response in accepted] == [200] * MAX_BACKGROUND_TASKS
    assert extra.status_code == 429
    assert extra.json()["error"]["code"] == "too_many_background_responses"
    assert "limit" in extra.json()["error"]["message"]

    gate.set()
    for response in accepted:
        wait_for_status(client, response.json()["id"], "completed", timeout=10)


def wait_for_status(
    client: TestClient, response_id: str, status: str, timeout: float = 5.0
) -> dict[str, Any]:
    """Poll GET until the background response reaches a terminal status."""
    deadline = time.time() + timeout
    payload: dict[str, Any] = {}
    while time.time() < deadline:
        payload = client.get(f"/v1/responses/{response_id}").json()
        if payload["status"] == status:
            return payload
        if payload["status"] in ("failed", "completed", "incomplete"):
            return payload
        time.sleep(0.01)
    raise AssertionError(f"response {response_id} did not reach {status}: {payload}")


# ------------------------------------------------------------ schema-level unit


def test_response_request_rejects_bad_combinations() -> None:
    with pytest.raises(Exception):
        ResponseRequest.model_validate(
            {"model": "m", "input": "x", "background": True, "stream": True}
        )
    with pytest.raises(Exception):
        ResponseRequest.model_validate(
            {
                "model": "m",
                "input": "x",
                "tool_choice": "required",
            }
        )


def test_response_request_accepts_chat_style_tools() -> None:
    request = ResponseRequest.model_validate(
        {
            "model": "m",
            "input": "x",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    assert request.tools[0].name == "f"
    assert request.tools[0].description == "d"


def test_messages_from_input_string() -> None:
    assert messages_from_input("hello") == [{"role": "user", "content": "hello"}]


def test_models_markers_default() -> None:
    markers = default_markers()
    assert markers.has_reasoning
    assert markers.has_tools


def test_response_store_put_get_delete_clear() -> None:
    store = ResponseStore(max_entries=2, ttl_seconds=60)
    record = StoredResponse(response={"status": "completed"})
    store.put("resp_1", record)

    assert store.get("resp_1") is record
    assert "resp_1" in store
    assert len(store) == 1
    assert store.delete("resp_1") is True
    assert store.delete("resp_1") is False
    assert store.get("resp_1") is None

    store.put("resp_2", StoredResponse(response={"status": "completed"}))
    store.clear()
    assert len(store) == 0


def test_response_store_purges_expired_entries() -> None:
    clock = [0.0]
    store = ResponseStore(max_entries=10, ttl_seconds=5, clock=lambda: clock[0])
    store.put("resp_1", StoredResponse(response={"status": "completed"}))
    clock[0] = 6.0
    store.put("resp_2", StoredResponse(response={"status": "completed"}))

    assert store.get("resp_1") is None
    assert len(store) == 1
    clock[0] = 20.0
    assert store.purge_expired() == 1
    assert len(store) == 0


def test_response_store_validates_settings() -> None:
    with pytest.raises(ValueError):
        ResponseStore(max_entries=0)
    with pytest.raises(ValueError):
        ResponseStore(ttl_seconds=0)
    with pytest.raises(ValueError):
        ResponseStore(max_entries="many")  # type: ignore[arg-type]

    settings = Settings(responses_store_max=7, responses_store_ttl=30)
    store = ResponseStore.from_settings(settings)
    assert store.max_entries == 7
    assert store.ttl_seconds == 30


def test_response_store_keeps_creation_time_on_update() -> None:
    clock = [100.0]
    store = ResponseStore(max_entries=4, ttl_seconds=50, clock=lambda: clock[0])
    store.put("resp_1", StoredResponse(response={"status": "in_progress"}))
    created = store.get("resp_1").created  # type: ignore[union-attr]

    clock[0] = 120.0
    store.put("resp_1", StoredResponse(response={"status": "completed"}, created=created))

    assert store.get("resp_1").response["status"] == "completed"  # type: ignore[union-attr]
    assert store.get("resp_1").created == created  # type: ignore[union-attr]


def test_background_response_keeps_its_creation_time(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/v1/responses", json=responses_payload(background=True))
    response_id = response.json()["id"]
    created_at = response.json()["created_at"]

    fetched = wait_for_status(client, response_id, "completed")
    assert fetched["created_at"] == created_at


def test_reasoning_effort_none_disables_thinking(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(reasoning={"effort": "none"})
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["params"].enable_thinking is False


def test_reasoning_effort_is_advisory_for_other_values(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses", json=responses_payload(reasoning={"effort": "high"})
    )

    assert response.status_code == 200
    assert engine.generate_calls[0]["params"].enable_thinking is None


def test_reasoning_effort_none_reaches_streaming_too(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post(
        "/v1/responses",
        json=responses_payload(stream=True, reasoning={"effort": "none"}),
    )

    assert response.status_code == 200
    assert engine.stream_calls[0]["params"].enable_thinking is False
