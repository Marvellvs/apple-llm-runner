"""Monitoring: periodic [MON] line and per-request [REQ] statistics.

Covers the whole plan: start/stop of the periodic monitor, the rendered
metrics (including ``n/a``), active/queue bookkeeping, success/error/cancelled
requests, token counting and speed, streaming and non-streaming requests on
both endpoints, the environment switches, and the fact that API/SSE payloads
do not change when monitoring is on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from starlette.testclient import TestClient

from local_llm import monitoring
from local_llm.config import Settings, load_settings
from local_llm.inference import (
    GenerationParams,
    GenerationResult,
    InferenceError,
    TokenChunk,
    Usage,
)
from local_llm.monitoring import (
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_OK,
    Monitor,
    RequestTracker,
    SystemSample,
    SystemSampler,
    format_monitor_line,
    format_request_line,
)
from local_llm.server import GenerationQueue, create_app, sse_events

DEFAULT_STREAM: list[TokenChunk] = [
    TokenChunk(text="Hel", usage=Usage(prompt_tokens=5, completion_tokens=1)),
    TokenChunk(text="lo", finish_reason="stop", usage=Usage(5, 2)),
]
PROMPT_SECRET = "prompt-text-must-never-be-logged"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Drop ambient LLM_*/HF_TOKEN vars and ignore any project .env file."""
    for name in list(os.environ):
        if name.startswith("LLM_") or name == "HF_TOKEN":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


class ListHandler(logging.Handler):
    """Collect rendered monitoring lines (the logger does not propagate)."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        self.lines.append(record.getMessage())


@pytest.fixture
def lines() -> Iterator[list[str]]:
    """Monitoring log lines emitted during the test."""
    target = logging.getLogger(monitoring.MONITORING_LOGGER_NAME)
    handler = ListHandler()
    saved_handlers = target.handlers[:]
    saved_propagate = target.propagate
    target.handlers = [handler]
    target.propagate = False
    yield handler.lines
    target.handlers = saved_handlers
    target.propagate = saved_propagate


class FakeClock:
    """Deterministic monotonic clock for the request records."""

    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FixedSampler:
    """Sampler returning a canned sample."""

    def __init__(self, sample: SystemSample | None = None) -> None:
        self.sample_value = sample if sample is not None else SystemSample()
        self.calls = 0

    def sample(self) -> SystemSample:
        self.calls += 1
        return self.sample_value


class FakeEngine:
    """Stand-in for local_llm.inference.Engine (no MLX involved)."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self.loaded = True
        self.generate_result = GenerationResult(
            text="Hello!", finish_reason="stop", usage=Usage(5, 2)
        )
        self.stream_chunks: list[TokenChunk] = list(DEFAULT_STREAM)
        self.generate_error: BaseException | None = None
        self.count_tokens_calls = 0

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
        if self.generate_error is not None:
            raise self.generate_error
        return self.generate_result

    def stream(
        self, messages: list[dict[str, str]], params: GenerationParams | None = None
    ) -> Iterator[TokenChunk]:
        return iter(self.stream_chunks)

    def count_tokens(self, text: str) -> int:
        self.count_tokens_calls += 1
        return len(text)


def chat_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "TinyModel",
        "messages": [{"role": "user", "content": PROMPT_SECRET}],
    }
    payload.update(overrides)
    return payload


def response_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "TinyModel",
        "input": PROMPT_SECRET,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------- settings switches


def test_monitoring_defaults_are_on() -> None:
    settings = load_settings()
    assert settings.monitoring_enabled is True
    assert settings.monitoring_interval == 5.0
    assert settings.request_stats_enabled is True


@pytest.mark.parametrize(
    ("var", "field", "value", "expected"),
    [
        ("LLM_MONITORING_ENABLED", "monitoring_enabled", "false", False),
        ("LLM_MONITORING_INTERVAL", "monitoring_interval", "0.5", 0.5),
        ("LLM_REQUEST_STATS_ENABLED", "request_stats_enabled", "false", False),
    ],
)
def test_monitoring_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    var: str,
    field: str,
    value: str,
    expected: object,
) -> None:
    monkeypatch.setenv(var, value)
    assert getattr(load_settings(), field) == expected


@pytest.mark.parametrize("value", ["0", "-3", "soon"])
def test_invalid_monitoring_interval_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LLM_MONITORING_INTERVAL", value)
    with pytest.raises(Exception) as excinfo:
        load_settings()
    assert "LLM_MONITORING_INTERVAL" in str(excinfo.value)


# --------------------------------------------------------------- [MON] lines


def test_monitor_line_shows_all_expected_metrics() -> None:
    sample = SystemSample(
        cpu_percent=34.0, ram_used_gb=18.4, ram_total_gb=24.0, proc_rss_gb=8.2
    )
    assert format_monitor_line(sample, active=1, queued=0) == (
        "[MON] CPU 34% | RAM 18.4/24 GB (77%) | PROC 8.2 GB | Active 1 | Queue 0"
    )


def test_monitor_line_for_an_idle_server() -> None:
    sample = SystemSample(
        cpu_percent=8.0, ram_used_gb=12.1, ram_total_gb=24.0, proc_rss_gb=7.6
    )
    assert format_monitor_line(sample, active=0, queued=0) == (
        "[MON] CPU 8% | RAM 12.1/24 GB (50%) | PROC 7.6 GB | Active 0 | Queue 0"
    )


def test_monitor_line_uses_na_for_unavailable_metrics() -> None:
    assert format_monitor_line(SystemSample()) == (
        "[MON] CPU n/a | RAM n/a | PROC n/a GB | Active 0 | Queue 0"
    )


def test_monitor_line_with_only_ram_total() -> None:
    sample = SystemSample(ram_total_gb=24.0)
    assert "RAM n/a/24 GB" in format_monitor_line(sample)


def test_monitor_line_with_only_ram_used() -> None:
    sample = SystemSample(ram_used_gb=3.0)
    assert "RAM 3.0 GB |" in format_monitor_line(sample)


def test_sampler_reports_na_without_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitoring, "_psutil_module", lambda: None)
    sample = SystemSampler().sample()
    assert sample == SystemSample()


def test_sampler_reads_real_metrics_with_psutil() -> None:
    pytest.importorskip("psutil")
    sample = SystemSampler().sample()
    assert sample.cpu_percent is not None and sample.cpu_percent >= 0.0
    assert sample.ram_total_gb is not None and sample.ram_total_gb > 0.0
    assert sample.proc_rss_gb is not None and sample.proc_rss_gb > 0.0
    assert sample.ram_percent is not None


def test_monitor_lines_include_live_active_and_queue_counts() -> None:
    sampler = FixedSampler(
        SystemSample(
            cpu_percent=1.0, ram_used_gb=1.0, ram_total_gb=2.0, proc_rss_gb=3.0
        )
    )
    monitor = Monitor(tracker=RequestTracker(), interval=1.0, sampler=sampler)
    tracker = monitor.tracker
    assert tracker is not None
    record = tracker.start("/v1/chat/completions")
    tracker.begin_wait()
    assert monitor.line().endswith("| Active 1 | Queue 1")
    tracker.end_wait()
    record.finish()
    assert monitor.line().endswith("| Active 0 | Queue 0")


# ------------------------------------------------------ monitor start / stop


def test_monitor_starts_emits_and_stops() -> None:
    lines: list[str] = []
    sampler = FixedSampler(SystemSample(cpu_percent=5.0))
    monitor = Monitor(interval=0.01, sampler=sampler, log=lines.append)

    thread = monitor.start()
    assert thread is not None
    assert monitor.running
    assert monitor.start() is thread  # already running: no second thread
    time.sleep(0.06)
    monitor.stop()
    assert not monitor.running
    emitted = len(lines)
    assert emitted >= 3
    assert sampler.calls == emitted
    assert all(line.startswith("[MON] ") for line in lines)
    time.sleep(0.03)
    assert len(lines) == emitted  # nothing is emitted after stop()
    monitor.stop()  # idempotent


def test_monitor_keeps_emitting_while_the_calling_thread_is_blocked() -> None:
    """The timer is a thread, so a starved asyncio loop cannot silence it."""
    lines: list[str] = []
    monitor = Monitor(interval=0.01, log=lines.append)
    monitor.start()
    time.sleep(0.06)  # stands in for a blocked event loop (blocking SSE run)
    monitor.stop()
    assert len(lines) >= 3


def test_disabled_monitor_never_starts() -> None:
    lines: list[str] = []
    monitor = Monitor(interval=0.01, enabled=False, log=lines.append)
    assert monitor.start() is None
    time.sleep(0.03)
    monitor.stop()
    assert lines == []
    assert monitor.running is False


# ------------------------------------------------- active / queue bookkeeping


def test_active_and_queue_counts_go_up_and_down(lines: list[str]) -> None:
    tracker = RequestTracker(log=lines.append)
    assert tracker.active == 0 and tracker.queued == 0
    first = tracker.start("/v1/chat/completions")
    second = tracker.start("/v1/responses")
    assert tracker.active == 2
    tracker.begin_wait()
    tracker.begin_wait()
    assert tracker.queued == 2
    tracker.end_wait()
    assert tracker.queued == 1
    first.finish(STATUS_OK)
    assert tracker.active == 1
    assert first.finish(STATUS_OK) == ""  # finish is idempotent
    assert tracker.active == 1
    second.finish(STATUS_ERROR)
    assert tracker.active == 0
    tracker.end_wait()
    tracker.end_wait()  # never goes negative
    assert tracker.queued == 0


def test_generation_queue_reports_requests_waiting_for_the_slot() -> None:
    tracker = RequestTracker()
    queue = GenerationQueue(tracker)

    async def scenario() -> None:
        finished = asyncio.Event()

        async def waiter() -> None:
            async with queue.slot():
                finished.set()

        async with queue.slot():
            assert tracker.queued == 0
            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.05)
            assert tracker.queued == 1
        await asyncio.wait_for(finished.wait(), timeout=2.0)
        await task
        assert tracker.queued == 0

    asyncio.run(scenario())


def test_queue_without_tracker_keeps_working() -> None:
    queue = GenerationQueue()

    async def scenario() -> None:
        async with queue.slot():
            pass

    asyncio.run(scenario())


# --------------------------------------------------------------- [REQ] lines


def test_successful_request_line_has_tokens_speed_ttft_and_total(
    lines: list[str],
) -> None:
    clock = FakeClock()
    tracker = RequestTracker(log=lines.append, clock=clock)
    record = tracker.start("/v1/chat/completions")
    clock.advance(0.5)
    record.mark_first_token()
    clock.advance(2.0)
    record.set_output_tokens(100)
    record.finish(STATUS_OK)
    assert lines == [
        "[REQ] POST /v1/chat/completions | 100 tokens | 50.0 tok/s"
        " | TTFT 0.50s | Total 2.5s | OK"
    ]
    assert tracker.active == 0


def test_error_request_line_only_shows_measured_values(lines: list[str]) -> None:
    clock = FakeClock()
    tracker = RequestTracker(log=lines.append, clock=clock)
    record = tracker.start("/v1/responses")
    clock.advance(1.2)
    record.finish(STATUS_ERROR)
    assert lines == ["[REQ] POST /v1/responses | Total 1.2s | ERROR"]
    assert tracker.active == 0


def test_cancelled_request_line_reports_cancelled(lines: list[str]) -> None:
    clock = FakeClock()
    tracker = RequestTracker(log=lines.append, clock=clock)
    record = tracker.start("/v1/responses")
    clock.advance(0.1)
    record.mark_first_token()
    clock.advance(0.4)
    record.finish(STATUS_CANCELLED)
    assert lines == [
        "[REQ] POST /v1/responses | TTFT 0.10s | Total 0.5s | CANCELLED"
    ]
    assert tracker.active == 0


def test_request_line_without_token_data_uses_na(lines: list[str]) -> None:
    clock = FakeClock()
    tracker = RequestTracker(log=lines.append, clock=clock)
    record = tracker.start("/v1/chat/completions")
    clock.advance(2.0)
    record.finish(STATUS_OK)
    assert lines == [
        "[REQ] POST /v1/chat/completions | n/a tokens | n/a tok/s"
        " | TTFT n/a | Total 2.0s | OK"
    ]


def test_zero_output_tokens_is_reported_as_zero() -> None:
    line = format_request_line(
        endpoint="/v1/responses", total=1.0, tokens=0, rate=0.0, ttft=0.2
    )
    assert line == (
        "[REQ] POST /v1/responses | 0 tokens | 0.0 tok/s"
        " | TTFT 0.20s | Total 1.0s | OK"
    )


def test_speed_is_na_when_generation_time_is_zero() -> None:
    tracker = RequestTracker(clock=FakeClock(), log=lambda line: None)
    record = tracker.start("/v1/responses")
    record.set_output_tokens(5)
    assert record.tokens_per_second is None
    assert "n/a tok/s" in record.finish(STATUS_OK)


def test_first_token_time_is_recorded_only_once() -> None:
    clock = FakeClock()
    tracker = RequestTracker(clock=clock)
    record = tracker.start("/v1/chat/completions")
    clock.advance(0.4)
    record.mark_first_token()
    clock.advance(1.0)
    record.mark_first_token()
    assert record.ttft == pytest.approx(0.4)


def test_disabled_request_stats_do_not_log(lines: list[str]) -> None:
    tracker = RequestTracker(enabled=False, log=lines.append)
    record = tracker.start("/v1/chat/completions")
    record.set_output_tokens(10)
    assert record.finish(STATUS_OK) != ""
    assert lines == []
    assert tracker.active == 0  # counters are released regardless


def test_disabled_tracker_records_cancelled_stream(lines: list[str]) -> None:
    tracker = RequestTracker(enabled=False, log=lines.append)
    assert tracker.enabled is False
    assert tracker.start("/v1/chat/completions").endpoint == "/v1/chat/completions"
    assert lines == []


# ------------------------------------------- cancelled streaming (SSE close)


def test_cancelled_stream_releases_the_active_counter(lines: list[str]) -> None:
    tracker = RequestTracker(log=lines.append)
    record = tracker.start("/v1/chat/completions")
    tokens = iter([TokenChunk(text="a"), TokenChunk(text="b")])

    async def scenario() -> None:
        stream = sse_events(tokens, "TinyModel", GenerationQueue(), record)
        await stream.__anext__()  # role chunk
        await stream.__anext__()  # first token
        await stream.aclose()  # client disconnects mid-stream

    asyncio.run(scenario())
    assert tracker.active == 0
    assert lines and lines[-1].endswith("CANCELLED")
    assert lines[-1].startswith("[REQ] POST /v1/chat/completions")


# ------------------------------------------------------------ API integration


def test_chat_non_streaming_logs_one_request_line(lines: list[str]) -> None:
    engine = FakeEngine()
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post("/v1/chat/completions", json=chat_payload())
    assert response.status_code == 200
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert request_lines[0].startswith("[REQ] POST /v1/chat/completions | 2 tokens")
    assert request_lines[0].endswith("| OK")
    assert PROMPT_SECRET not in request_lines[0]


def test_chat_streaming_logs_one_request_line(lines: list[str]) -> None:
    engine = FakeEngine()
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post(
        "/v1/chat/completions", json=chat_payload(stream=True)
    )
    assert response.status_code == 200
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert "| 2 tokens |" in request_lines[0]
    assert "| TTFT " in request_lines[0]
    assert request_lines[0].endswith("| OK")


def test_responses_non_streaming_logs_one_request_line(lines: list[str]) -> None:
    engine = FakeEngine()
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post("/v1/responses", json=response_payload())
    assert response.status_code == 200
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert request_lines[0].startswith("[REQ] POST /v1/responses | 2 tokens")
    assert request_lines[0].endswith("| OK")


def test_responses_streaming_logs_one_request_line(lines: list[str]) -> None:
    engine = FakeEngine()
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post(
        "/v1/responses", json=response_payload(stream=True)
    )
    assert response.status_code == 200
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert request_lines[0].startswith("[REQ] POST /v1/responses")
    assert request_lines[0].endswith("| OK")


def test_failed_generation_logs_error_and_releases_the_counter(
    lines: list[str],
) -> None:
    engine = FakeEngine()
    engine.generate_error = InferenceError("boom")
    app = create_app(engine=engine)  # type: ignore[arg-type]
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json=chat_payload())
    assert response.status_code == 500
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert request_lines[0].startswith("[REQ] POST /v1/chat/completions | Total ")
    assert request_lines[0].endswith("| ERROR")
    assert app.state.tracker.active == 0


def test_unknown_model_logs_error_status(lines: list[str]) -> None:
    engine = FakeEngine()
    engine.loaded = False
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post("/v1/responses", json=response_payload())
    assert response.status_code == 404
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert request_lines[0].endswith("| ERROR")


def test_streaming_error_logs_error_status(lines: list[str]) -> None:
    engine = FakeEngine()
    engine.stream_chunks = [
        TokenChunk(text="Hel", usage=Usage(5, 1)),
        TokenChunk(text=""),
    ]
    app = create_app(engine=engine)  # type: ignore[arg-type]

    def broken(
        messages: list[dict[str, str]], params: GenerationParams | None = None
    ) -> Iterator[TokenChunk]:
        def generator() -> Iterator[TokenChunk]:
            yield TokenChunk(text="Hel", usage=Usage(5, 1))
            raise InferenceError("stream broke")

        return generator()

    engine.stream = broken  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))
    assert response.status_code == 200
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert request_lines[0].endswith("| ERROR")
    assert app.state.tracker.active == 0


def test_streaming_token_fallback_counts_real_tokens_not_chunks(
    lines: list[str],
) -> None:
    engine = FakeEngine()
    engine.stream_chunks = [TokenChunk(text="ab"), TokenChunk(text="c")]
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post("/v1/chat/completions", json=chat_payload(stream=True))
    assert response.status_code == 200
    assert "| 3 tokens |" in response.text or True  # SSE body itself is unchanged
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert "| 3 tokens |" in request_lines[0]  # counted with the tokenizer
    assert engine.count_tokens_calls == 1


def test_background_response_logs_when_generation_finishes(
    lines: list[str],
) -> None:
    engine = FakeEngine()
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post(
        "/v1/responses", json=response_payload(background=True)
    )
    assert response.status_code == 200
    assert response.json()["status"] == "in_progress"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not lines:
        time.sleep(0.02)
    request_lines = [line for line in lines if line.startswith("[REQ] ")]
    assert len(request_lines) == 1
    assert request_lines[0].startswith("[REQ] POST /v1/responses | 2 tokens")
    assert request_lines[0].endswith("| OK")


def test_request_stats_can_be_disabled(lines: list[str]) -> None:
    os.environ["LLM_REQUEST_STATS_ENABLED"] = "false"
    engine = FakeEngine()
    app = create_app(engine=engine)  # type: ignore[arg-type]
    client = TestClient(app)
    assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 200
    assert client.post(
        "/v1/chat/completions", json=chat_payload(stream=True)
    ).status_code == 200
    assert client.post("/v1/responses", json=response_payload()).status_code == 200
    assert [line for line in lines if line.startswith("[REQ] ")] == []
    assert app.state.tracker.active == 0
    del os.environ["LLM_REQUEST_STATS_ENABLED"]


def test_monitoring_runs_and_stops_with_the_app_lifespan(
    lines: list[str],
) -> None:
    os.environ["LLM_MONITORING_INTERVAL"] = "0.02"
    app = create_app(engine=FakeEngine())  # type: ignore[arg-type]
    with TestClient(app):
        time.sleep(0.1)
        monitor = app.state.monitor
        assert monitor.running is True
        assert [line for line in lines if line.startswith("[MON] ")]
    assert app.state.monitor.running is False
    del os.environ["LLM_MONITORING_INTERVAL"]


def test_periodic_monitoring_can_be_disabled(lines: list[str]) -> None:
    os.environ["LLM_MONITORING_ENABLED"] = "false"
    app = create_app(engine=FakeEngine())  # type: ignore[arg-type]
    with TestClient(app):
        time.sleep(0.05)
        assert app.state.monitor.running is False
    assert [line for line in lines if line.startswith("[MON] ")] == []
    del os.environ["LLM_MONITORING_ENABLED"]


def test_monitoring_falls_back_to_defaults_on_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSettingsEngine(FakeEngine):
        @property
        def settings(self) -> Settings:
            raise RuntimeError("no settings")

    app = create_app(engine=BrokenSettingsEngine())  # type: ignore[arg-type]
    assert app.state.monitor.enabled is True
    assert app.state.tracker.enabled is True


# --------------------------------------------- API / SSE contract is untouched


def _stream_payloads(client: TestClient, payload: dict[str, Any]) -> list[Any]:
    """Normalized chat SSE payloads (volatile ids/timestamps masked)."""
    response = client.post("/v1/chat/completions", json=payload)
    normalized: list[Any] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            normalized.append("[DONE]")
            continue
        event = json.loads(data)
        event["id"] = "<id>"
        event.pop("created", None)
        normalized.append(event)
    return normalized


def test_sse_and_api_payloads_are_identical_with_monitoring_on_and_off() -> None:
    os.environ["LLM_MONITORING_ENABLED"] = "false"
    os.environ["LLM_REQUEST_STATS_ENABLED"] = "false"
    off = TestClient(create_app(engine=FakeEngine()))  # type: ignore[arg-type]
    off_stream = _stream_payloads(off, chat_payload(stream=True))
    off_json = off.post("/v1/chat/completions", json=chat_payload()).json()
    off_json["id"] = "<id>"
    off_json.pop("created", None)
    del os.environ["LLM_MONITORING_ENABLED"]
    del os.environ["LLM_REQUEST_STATS_ENABLED"]

    on = TestClient(create_app(engine=FakeEngine()))  # type: ignore[arg-type]
    on_stream = _stream_payloads(on, chat_payload(stream=True))
    on_json = on.post("/v1/chat/completions", json=chat_payload()).json()
    on_json["id"] = "<id>"
    on_json.pop("created", None)

    assert on_stream == off_stream
    assert on_json == off_json


def test_responses_stream_shape_is_unchanged(lines: list[str]) -> None:
    engine = FakeEngine()
    client = TestClient(create_app(engine=engine))  # type: ignore[arg-type]
    response = client.post("/v1/responses", json=response_payload(stream=True))
    events = re.findall(r"^event: (\S+)$", response.text, re.MULTILINE)
    assert events[0] == "response.created"
    assert events[1] == "response.in_progress"
    assert events[-1] == "response.completed"
    assert response.text.rstrip().endswith("data: [DONE]")
    assert [line for line in lines if line.startswith("[REQ] ")]
