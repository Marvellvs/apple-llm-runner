"""FastAPI app and server startup: /v1/chat/completions (Epics 5 and 6)."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .config import Settings, SettingsError
from .monitoring import (
    DEFAULT_INTERVAL,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_OK,
    Monitor,
    RequestRecord,
    RequestTracker,
    SystemSampler,
)
from .inference import (
    Engine,
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
from .responses import (
    ResponseRequestError,
    ResponseStreamAssembler,
    build_messages,
    build_output_items,
    build_response_object,
    count_reasoning_tokens,
    follow_up_context,
    new_response_id,
    now_timestamp,
    parse_output,
    responses_usage,
    sse_done,
)
from .schemas import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_BODY,
    ERROR_MODEL_NOT_FOUND,
    ERROR_RESPONSE_NOT_FOUND,
    ERROR_UNSUPPORTED_TOOLS,
    OBJECT_CHAT_COMPLETION_CHUNK,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_INCOMPLETE,
    INCOMPLETE_MAX_OUTPUT_TOKENS,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseErrorBody,
    ResponseObject,
    ResponseRequest,
    UsageOut,
    error_response,
)
from .store import ResponseStore, StoredResponse

logger = logging.getLogger("local_llm.server")

SSE_MEDIA_TYPE = "text/event-stream"
SERVER_ERROR = "server_error"
INVALID_REQUEST = "invalid_request_error"
INTERNAL_ERROR_CODE = "internal_error"
INTERNAL_ERROR_MESSAGE = "Internal server error"

MEMORY_HINT = (
    "Not enough memory to load the model. Close other applications or "
    "download a smaller model (`make download MODEL=mlx-community/<repo>`)."
)
ENDPOINT_CHAT = "/v1/chat/completions"
ENDPOINT_RESPONSES = "/v1/responses"
MAX_BACKGROUND_TASKS = 8
ERROR_TOO_MANY_BACKGROUND = "too_many_background_responses"
MEMORY_PATTERN = re.compile(
    r"\b(?:oom|out of memory|not enough memory|cannot allocate memory|"
    r"can't allocate memory|memory (?:limit|exhausted|allocation failed))\b",
    re.IGNORECASE,
)


class GenerationQueue:
    """Serializes model access: one request at a time (MLX unified memory).

    A threading lock (not asyncio.Lock) keeps the queue usable from any event
    loop - TestClient and uvicorn may run requests on different loops, and
    background generations run on their own worker threads.

    An optional tracker only *observes* how many requests wait for the slot;
    it never changes the locking behaviour.
    """

    def __init__(self, tracker: RequestTracker | None = None) -> None:
        self._lock = threading.Lock()
        self._tracker = tracker

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        tracker = self._tracker
        if tracker is not None:
            tracker.begin_wait()
        try:
            await asyncio.to_thread(self._lock.acquire)
        finally:
            if tracker is not None:
                tracker.end_wait()
        try:
            yield
        finally:
            self._lock.release()

    @contextmanager
    def slot_sync(self) -> Iterator[None]:
        """Blocking variant used by background worker threads."""
        tracker = self._tracker
        if tracker is not None:
            tracker.begin_wait()
        try:
            self._lock.acquire()
        finally:
            if tracker is not None:
                tracker.end_wait()
        try:
            yield
        finally:
            self._lock.release()


def monitoring_options(engine: Engine) -> tuple[bool, float, bool]:
    """(periodic monitoring, interval, request stats); never fatal."""
    try:
        settings = engine.settings
    except Exception:
        logger.debug("monitoring falls back to its defaults", exc_info=True)
        return True, DEFAULT_INTERVAL, True
    return (
        settings.monitoring_enabled,
        settings.monitoring_interval,
        settings.request_stats_enabled,
    )


def complete_http_request(record: RequestRecord, response: Any) -> Any:
    """Finish request stats, deferring streaming/background work to its owner."""
    if isinstance(response, StreamingResponse):
        record.defer()
        return response
    if record.deferred:
        return response
    status = STATUS_OK
    status_code = getattr(response, "status_code", 200)
    if isinstance(status_code, int) and status_code >= 400:
        status = STATUS_ERROR
    record.finish(status)
    return response


def count_generated_tokens(
    counter: Callable[[str], int] | None, pieces: list[str]
) -> int | None:
    """Fallback token count for engines that report no usage while streaming."""
    if counter is None or not pieces:
        return None
    try:
        return counter("".join(pieces))
    except Exception:
        logger.debug("token counting for the request line failed", exc_info=True)
        return None


def params_for(request: ChatCompletionRequest, settings: Settings) -> GenerationParams:
    """Merge request overrides with the LLM_* defaults from the environment."""
    return GenerationParams(
        max_tokens=(
            request.max_tokens
            if request.max_tokens is not None
            else settings.max_tokens
        ),
        temperature=(
            request.temperature
            if request.temperature is not None
            else settings.temperature
        ),
        top_p=request.top_p if request.top_p is not None else settings.top_p,
    )


def invalid_body_message(exc: RequestValidationError) -> str:
    """Readable 400 message built from pydantic errors (no raw data embedded)."""
    details: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(
            str(part) for part in error.get("loc", ()) if part != "body"
        )
        message = str(error.get("msg", "invalid value")).removeprefix("Value error, ")
        details.append(f"{location}: {message}" if location else message)
    return "Invalid request body: " + "; ".join(details)


def invalid_body_code(exc: RequestValidationError) -> str:
    """Machine-readable 400 code taken from the first pydantic error."""
    errors = exc.errors()
    return str(errors[0].get("type") or ERROR_INVALID_BODY)


def error_json(status: int, message: str, type_: str, code: str) -> JSONResponse:
    """Serialize an OpenAI-style error envelope as an HTTP response."""
    return JSONResponse(status_code=status, content=error_response(message, type_, code).model_dump())


def model_not_found(model: str) -> JSONResponse:
    """404 telling the client how to get the requested model loaded."""
    message = (
        f"Model {model!r} is not loaded. Download the weights "
        "(`make download MODEL=mlx-community/<repo>`), set LLM_MODEL and "
        "restart the server with `make run`."
    )
    return error_json(404, message, INVALID_REQUEST, ERROR_MODEL_NOT_FOUND)


def usage_out(usage: Usage) -> UsageOut:
    """Convert engine usage accounting into the API schema."""
    return UsageOut(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def completion_response(model: str, result: GenerationResult) -> ChatCompletionResponse:
    """Wrap an engine result into the OpenAI completion envelope."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        usage=usage_out(result.usage),
    )


def sse(payload: dict[str, Any] | str) -> str:
    """Render one SSE event; strings (like [DONE]) pass through untouched."""
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


def chunk_payload(
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    """One chat.completion.chunk event body."""
    return {
        "id": chunk_id,
        "object": OBJECT_CHAT_COMPLETION_CHUNK,
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def usage_payload(
    chunk_id: str, created: int, model: str, usage: UsageOut
) -> dict[str, Any]:
    """Final chunk carrying usage (OpenAI stream_options semantics)."""
    payload = chunk_payload(chunk_id, created, model, {}, None)
    payload["choices"] = []
    payload["usage"] = usage.model_dump()
    return payload


async def sse_events(
    tokens: Iterator[TokenChunk],
    model: str,
    queue: GenerationQueue,
    record: RequestRecord | None = None,
    counter: Callable[[str], int] | None = None,
) -> AsyncIterator[str]:
    """Render token chunks as SSE; a mid-stream failure becomes an error event."""
    chunk_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    usage: UsageOut | None = None
    finish_reason: str | None = None
    failed = False
    outcome = STATUS_CANCELLED
    pieces: list[str] = []
    try:
        async with queue.slot():
            yield sse(
                chunk_payload(chunk_id, created, model, {"role": "assistant"}, None)
            )
            try:
                for token in tokens:
                    if token.usage is not None:
                        usage = usage_out(token.usage)
                    if token.text:
                        if record is not None:
                            record.mark_first_token()
                        if counter is not None:
                            pieces.append(token.text)
                        yield sse(
                            chunk_payload(
                                chunk_id, created, model, {"content": token.text}, None
                            )
                        )
                    if token.finish_reason is not None:
                        finish_reason = token.finish_reason
                        yield sse(
                            chunk_payload(chunk_id, created, model, {}, finish_reason)
                        )
                        break
            except InferenceError as exc:
                failed = True
                logger.warning("streaming generation failed: %s", exc)
                yield sse(
                    error_response(str(exc), SERVER_ERROR, ERROR_GENERATION_FAILED)
                    .model_dump()
                )
            except Exception:
                failed = True
                logger.exception("unexpected streaming failure")
                yield sse(
                    error_response(
                        INTERNAL_ERROR_MESSAGE, SERVER_ERROR, INTERNAL_ERROR_CODE
                    ).model_dump()
                )
            if failed:
                outcome = STATUS_ERROR
            else:
                if finish_reason is None:
                    yield sse(chunk_payload(chunk_id, created, model, {}, "stop"))
                if usage is not None:
                    yield sse(usage_payload(chunk_id, created, model, usage))
                outcome = STATUS_OK
            yield sse("[DONE]")
    except (asyncio.CancelledError, GeneratorExit):
        outcome = STATUS_CANCELLED
        raise
    finally:
        if record is not None:
            record.set_output_tokens(
                usage.completion_tokens
                if usage is not None
                else count_generated_tokens(counter, pieces)
            )
            record.finish(outcome)


async def stream_response(
    engine: Engine,
    model: str,
    messages: list[dict[str, str]],
    params: GenerationParams,
    queue: GenerationQueue,
    record: RequestRecord | None = None,
) -> StreamingResponse | JSONResponse:
    """Kick off SSE streaming; setup failures come back as JSON errors."""
    try:
        tokens = engine.stream(messages, params)
    except ModelNotLoadedError:
        return model_not_found(model)
    except InferenceError as exc:
        return error_json(500, str(exc), SERVER_ERROR, ERROR_GENERATION_FAILED)
    except Exception:
        logger.exception("failed to start streaming generation")
        return error_json(500, INTERNAL_ERROR_MESSAGE, SERVER_ERROR, INTERNAL_ERROR_CODE)
    return StreamingResponse(
        sse_events(tokens, model, queue, record, engine_token_counter(engine)),
        media_type=SSE_MEDIA_TYPE,
    )


# ------------------------------------------------------- Responses API (Epic 8)

BACKGROUND_MESSAGE = (
    "The response was not completed. Retrieve it with "
    "GET /v1/responses/{response_id} and check `error`."
)


def response_not_found(response_id: str) -> JSONResponse:
    """404 for an unknown or expired response id."""
    return error_json(
        404,
        f"Response {response_id!r} was not found. It is unknown or its "
        "in-memory TTL has expired; in-memory history is lost on restart.",
        INVALID_REQUEST,
        ERROR_RESPONSE_NOT_FOUND,
    )


def responses_params(request: ResponseRequest, settings: Settings) -> GenerationParams:
    """Merge Responses request overrides with the LLM_* defaults."""
    enable_thinking: bool | None = None
    if request.reasoning is not None and request.reasoning.effort == "none":
        enable_thinking = False
    return GenerationParams(
        max_tokens=(
            request.max_output_tokens
            if request.max_output_tokens is not None
            else settings.max_tokens
        ),
        temperature=(
            request.temperature
            if request.temperature is not None
            else settings.temperature
        ),
        top_p=request.top_p if request.top_p is not None else settings.top_p,
        tools=tuple(request.tools) or None,
        enable_thinking=enable_thinking,
    )


def get_response_store(app: FastAPI, settings: Settings) -> ResponseStore:
    """Lazily create the app-wide in-memory response store."""
    store = getattr(app.state, "response_store", None)
    if store is None:
        store = ResponseStore.from_settings(settings)
        app.state.response_store = store
    return store


def engine_markers(engine: Engine) -> ModelMarkers:
    """Reasoning/tool markers of the loaded model (engine or defaults)."""
    return getattr(engine, "markers", None) or default_markers()


def engine_tool_parser(engine: Engine) -> Any:
    return getattr(engine, "tool_call_parser", None)


def engine_token_counter(engine: Engine) -> Any:
    return getattr(engine, "count_tokens", None)


class TokenStream:
    """Bridge between a blocking token iterator and the event loop.

    MLX generation must run on a single thread, so the iterator is consumed on
    one dedicated worker thread; the event loop only awaits queue reads. The
    worker can be stopped from any thread through :meth:`close`, and it closes
    the underlying iterator itself (again on the same thread).
    """

    def __init__(self, tokens: Iterator[TokenChunk], maxsize: int = 16) -> None:
        self._tokens = tokens
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="local-llm-token-stream", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Ask the worker to stop; safe to call from any thread."""
        self._stop.set()

    async def __aiter__(self) -> AsyncIterator[TokenChunk]:
        while True:
            kind, value = await asyncio.to_thread(self._queue.get)
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to finish (used by tests)."""
        self._thread.join(timeout)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _put(self, item: tuple[str, Any]) -> bool:
        """Enqueue unless a stop was requested in the meantime."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            for chunk in self._tokens:
                if self._stop.is_set():
                    break
                if not self._put(("chunk", chunk)):
                    break
        except BaseException as exc:  # surfaced to the consumer as-is
            self._put(("error", exc))
        finally:
            if self._stop.is_set():
                close_iterator(self._tokens)
            try:
                self._queue.put(("end", None), timeout=1.0)
            except queue.Full:
                logger.debug("token stream queue stayed full at exit")


def close_iterator(tokens: Any) -> None:
    """Release a token generator (and its queue slot) on early exit."""
    close = getattr(tokens, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # closing a broken generator is not fatal
            logger.debug("failed to close the token generator", exc_info=True)


def save_stored_response(
    store: ResponseStore,
    response_id: str,
    response: ResponseObject,
    context: list[dict[str, Any]],
) -> None:
    """Persist a Response object plus the context for previous_response_id."""
    existing = store.get(response_id)
    store.put(
        response_id,
        StoredResponse(
            response=response.model_dump(),
            context=context,
            created=existing.created if existing is not None else 0.0,
        ),
    )


def build_non_streaming_response(
    *,
    engine: Engine,
    request: ResponseRequest,
    response_id: str,
    created_at: int,
    text: str,
    finish_reason: str,
    usage: Usage | None,
) -> ResponseObject:
    """Turn a complete generation into a Response object."""
    output = build_output_items(
        parse_output(
            text,
            markers=engine_markers(engine),
            enable_tools=bool(request.tools),
            tool_call_parser=engine_tool_parser(engine),
        )
    )
    status = STATUS_INCOMPLETE if finish_reason == "length" else STATUS_COMPLETED
    return build_response_object(
        request=request,
        response_id=response_id,
        created_at=created_at,
        status=status,
        output=output,
        usage=responses_usage(
            usage, count_reasoning_tokens(output, engine_token_counter(engine))
        ),
        incomplete_reason=(
            INCOMPLETE_MAX_OUTPUT_TOKENS if finish_reason == "length" else None
        ),
    )


def failed_response(
    request: ResponseRequest,
    response_id: str,
    created_at: int,
    code: str,
    message: str,
) -> ResponseObject:
    """A Response object in the ``failed`` terminal state."""
    return build_response_object(
        request=request,
        response_id=response_id,
        created_at=created_at,
        status=STATUS_FAILED,
        error=ResponseErrorBody(code=code, message=message),
    )


def background_workers(app: FastAPI) -> set[threading.Thread]:
    """Live background worker threads of this app (finished ones are pruned)."""
    workers = getattr(app.state, "background_workers", None)
    if workers is None:
        workers = set()
        app.state.background_workers = workers
    workers.intersection_update(thread for thread in workers if thread.is_alive())
    return workers


def start_background_response(
    app: FastAPI,
    *,
    engine: Engine,
    queue: GenerationQueue,
    store: ResponseStore,
    request: ResponseRequest,
    response_id: str,
    created_at: int,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    record: RequestRecord | None = None,
) -> bool:
    """Start a background generation in its own thread.

    The worker thread (not the request's event loop) owns the generation, so
    the work survives the request and stays on one thread - which MLX needs.
    The request record is handed over to the worker, which finishes it (and
    logs the ``[REQ]`` line) once the generation is done. Returns False when
    the concurrency limit is already reached.
    """
    workers = background_workers(app)
    if len(workers) >= MAX_BACKGROUND_TASKS:
        return False
    worker = threading.Thread(
        target=run_background_response,
        name="local-llm-background",
        daemon=True,
        kwargs={
            "engine": engine,
            "queue": queue,
            "store": store,
            "request": request,
            "response_id": response_id,
            "created_at": created_at,
            "messages": messages,
            "params": params,
            "record": record,
        },
    )
    workers.add(worker)
    worker.start()
    return True


def run_background_response(
    *,
    engine: Engine,
    queue: GenerationQueue,
    store: ResponseStore,
    request: ResponseRequest,
    response_id: str,
    created_at: int,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    record: RequestRecord | None = None,
) -> None:
    """Generate a background response and store its terminal state."""
    context: list[dict[str, Any]] = []
    outcome = STATUS_OK
    try:
        try:
            with queue.slot_sync():
                result = engine.generate(messages, params)
            if record is not None:
                record.set_output_tokens(result.usage.completion_tokens)
            response = build_non_streaming_response(
                engine=engine,
                request=request,
                response_id=response_id,
                created_at=created_at,
                text=result.text,
                finish_reason=result.finish_reason,
                usage=result.usage,
            )
            context = follow_up_context(messages, response.output)
        except UnsupportedToolsError as exc:
            outcome = STATUS_ERROR
            response = failed_response(
                request, response_id, created_at, ERROR_UNSUPPORTED_TOOLS, str(exc)
            )
        except ModelNotLoadedError as exc:
            outcome = STATUS_ERROR
            response = failed_response(
                request, response_id, created_at, ERROR_MODEL_NOT_FOUND, str(exc)
            )
        except InferenceError as exc:
            outcome = STATUS_ERROR
            logger.warning("background generation failed: %s", exc)
            response = failed_response(
                request, response_id, created_at, ERROR_GENERATION_FAILED, str(exc)
            )
        except Exception:
            outcome = STATUS_ERROR
            logger.exception("unexpected background generation failure")
            response = failed_response(
                request, response_id, created_at, INTERNAL_ERROR_CODE,
                INTERNAL_ERROR_MESSAGE,
            )
        save_stored_response(store, response_id, response, context)
    finally:
        if record is not None:
            record.finish(outcome)


async def responses_sse_events(
    *,
    assembler: ResponseStreamAssembler,
    tokens: TokenStream,
    queue: GenerationQueue,
    store: ResponseStore,
    should_store: bool,
    messages: list[dict[str, Any]],
    record: RequestRecord | None = None,
    counter: Callable[[str], int] | None = None,
) -> AsyncIterator[str]:
    """Render a Responses API stream; the queue slot is held while generating."""
    usage: Usage | None = None
    finish_reason: str | None = None
    terminal = False
    outcome = STATUS_CANCELLED
    pieces: list[str] = []
    try:
        async with queue.slot():
            for event in assembler.created_events():
                yield event.render()
            try:
                async for chunk in tokens:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.text:
                        if record is not None:
                            record.mark_first_token()
                        if counter is not None:
                            pieces.append(chunk.text)
                        for event in assembler.push(chunk.text):
                            yield event.render()
                    if chunk.finish_reason is not None:
                        finish_reason = chunk.finish_reason
                        break
                for event in assembler.finish(
                    finish_reason=finish_reason, usage=usage
                ):
                    yield event.render()
                terminal = True
                outcome = STATUS_OK
            except InferenceError as exc:
                logger.warning("responses streaming failed: %s", exc)
                for event in assembler.fail(str(exc), ERROR_GENERATION_FAILED):
                    yield event.render()
                terminal = True
                outcome = STATUS_ERROR
            except Exception:
                logger.exception("unexpected responses streaming failure")
                for event in assembler.fail(
                    INTERNAL_ERROR_MESSAGE, INTERNAL_ERROR_CODE
                ):
                    yield event.render()
                terminal = True
                outcome = STATUS_ERROR
    except (asyncio.CancelledError, GeneratorExit):
        outcome = STATUS_CANCELLED
        raise
    finally:
        close_iterator(tokens)
        if record is not None:
            record.set_output_tokens(
                usage.completion_tokens
                if usage is not None
                else count_generated_tokens(counter, pieces)
            )
            record.finish(outcome)
        if terminal and should_store:
            save_stored_response(
                store,
                assembler.response_id,
                assembler.response,
                follow_up_context(messages, assembler.output),
            )
    yield sse_done()


def create_app(engine: Engine | None = None) -> FastAPI:
    """Build the FastAPI app around an (optional) pre-built inference engine."""
    engine = engine if engine is not None else Engine()
    monitoring_enabled, monitoring_interval, request_stats = monitoring_options(engine)
    tracker = RequestTracker(enabled=request_stats)
    monitor = Monitor(
        tracker=tracker,
        interval=monitoring_interval,
        enabled=monitoring_enabled,
        sampler=SystemSampler(),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        monitor.start()
        try:
            yield
        finally:
            monitor.stop()

    app = FastAPI(
        title="local-llm",
        version=__version__,
        description="OpenAI-compatible chat completions served locally via MLX.",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.queue = GenerationQueue(tracker)
    app.state.tracker = tracker
    app.state.monitor = monitor

    @app.exception_handler(RequestValidationError)
    async def _invalid_body(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_json(
            400,
            invalid_body_message(exc),
            INVALID_REQUEST,
            invalid_body_code(exc),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        record = app.state.tracker.start(ENDPOINT_CHAT)
        try:
            response = await run_chat_completion(app, request, record)
        except Exception:
            record.finish(STATUS_ERROR)
            raise
        return complete_http_request(record, response)

    @app.post("/v1/responses")
    async def create_response(request: ResponseRequest) -> Any:
        record = app.state.tracker.start(ENDPOINT_RESPONSES)
        try:
            response = await run_response(app, request, record)
        except Exception:
            record.finish(STATUS_ERROR)
            raise
        return complete_http_request(record, response)

    @app.get("/v1/responses/{response_id}")
    async def get_response(response_id: str) -> Any:
        engine = app.state.engine
        try:
            settings = engine.settings
        except SettingsError as exc:
            return error_json(500, str(exc), SERVER_ERROR, INTERNAL_ERROR_CODE)
        record = get_response_store(app, settings).get(response_id)
        if record is None:
            return response_not_found(response_id)
        return JSONResponse(record.response)

    return app


async def run_chat_completion(
    app: FastAPI, request: ChatCompletionRequest, record: RequestRecord
) -> Any:
    """Handle POST /v1/chat/completions (monitoring-agnostic body)."""
    engine = app.state.engine
    if not engine.is_loaded:
        return model_not_found(request.model)
    try:
        params = params_for(request, engine.settings)
    except SettingsError as exc:
        return error_json(500, str(exc), SERVER_ERROR, INTERNAL_ERROR_CODE)
    messages = [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]
    if request.stream:
        return await stream_response(
            engine, request.model, messages, params, app.state.queue, record
        )
    async with app.state.queue.slot():
        try:
            result = await asyncio.to_thread(engine.generate, messages, params)
        except ModelNotLoadedError:
            return model_not_found(request.model)
        except InferenceError as exc:
            return error_json(500, str(exc), SERVER_ERROR, ERROR_GENERATION_FAILED)
        except Exception:
            logger.exception("generation failed")
            return error_json(
                500, INTERNAL_ERROR_MESSAGE, SERVER_ERROR, INTERNAL_ERROR_CODE
            )
    record.set_output_tokens(result.usage.completion_tokens)
    return completion_response(request.model, result)


async def run_response(
    app: FastAPI, request: ResponseRequest, record: RequestRecord
) -> Any:
    """Handle POST /v1/responses (monitoring-agnostic body)."""
    engine = app.state.engine
    if not engine.is_loaded:
        return model_not_found(request.model)
    try:
        settings = engine.settings
    except SettingsError as exc:
        return error_json(500, str(exc), SERVER_ERROR, INTERNAL_ERROR_CODE)
    store = get_response_store(app, settings)

    previous_context: list[dict[str, Any]] | None = None
    if request.previous_response_id:
        stored = store.get(request.previous_response_id)
        if stored is None:
            return response_not_found(request.previous_response_id)
        if stored.status not in (STATUS_COMPLETED, STATUS_INCOMPLETE):
            return error_json(
                400,
                f"previous response {request.previous_response_id!r} is not "
                f"complete (status: {stored.status})",
                INVALID_REQUEST,
                ERROR_RESPONSE_NOT_FOUND,
            )
        previous_context = stored.context
    try:
        messages = build_messages(request, previous_context)
    except ResponseRequestError as exc:
        return error_json(exc.status_code, exc.message, INVALID_REQUEST, exc.code)

    params = responses_params(request, settings)
    should_store = request.store is not False
    response_id = new_response_id()
    created_at = now_timestamp()
    markers = engine_markers(engine)
    tool_parser = engine_tool_parser(engine)

    if request.background:
        initial = build_response_object(
            request=request,
            response_id=response_id,
            created_at=created_at,
            status=STATUS_IN_PROGRESS,
        )
        store.put(
            response_id,
            StoredResponse(response=initial.model_dump(), context=[]),
        )
        started = start_background_response(
            app,
            engine=engine,
            queue=app.state.queue,
            store=store,
            request=request,
            response_id=response_id,
            created_at=created_at,
            messages=messages,
            params=params,
            record=record,
        )
        if not started:
            store.delete(response_id)
            return error_json(
                429,
                f"Too many background responses are already running "
                f"(limit: {MAX_BACKGROUND_TASKS}). Retry later.",
                "rate_limit_error",
                ERROR_TOO_MANY_BACKGROUND,
            )
        record.defer()  # the worker logs the [REQ] line when generation ends
        return JSONResponse(initial.model_dump())

    if request.stream:
        try:
            tokens = engine.stream(messages, params)
        except UnsupportedToolsError as exc:
            return error_json(
                400, str(exc), INVALID_REQUEST, ERROR_UNSUPPORTED_TOOLS
            )
        except ModelNotLoadedError:
            return model_not_found(request.model)
        except InferenceError as exc:
            return error_json(
                500, str(exc), SERVER_ERROR, ERROR_GENERATION_FAILED
            )
        except Exception:
            logger.exception("failed to start responses streaming")
            return error_json(
                500, INTERNAL_ERROR_MESSAGE, SERVER_ERROR, INTERNAL_ERROR_CODE
            )
        assembler = ResponseStreamAssembler(
            request=request,
            response_id=response_id,
            created_at=created_at,
            markers=markers,
            tool_call_parser=tool_parser,
            count_tokens=engine_token_counter(engine),
        )
        return StreamingResponse(
            responses_sse_events(
                assembler=assembler,
                tokens=TokenStream(tokens),
                queue=app.state.queue,
                store=store,
                should_store=should_store,
                messages=messages,
                record=record,
                counter=engine_token_counter(engine),
            ),
            media_type=SSE_MEDIA_TYPE,
        )

    async with app.state.queue.slot():
        try:
            result = await asyncio.to_thread(engine.generate, messages, params)
        except UnsupportedToolsError as exc:
            return error_json(
                400, str(exc), INVALID_REQUEST, ERROR_UNSUPPORTED_TOOLS
            )
        except ModelNotLoadedError:
            return model_not_found(request.model)
        except InferenceError as exc:
            return error_json(
                500, str(exc), SERVER_ERROR, ERROR_GENERATION_FAILED
            )
        except Exception:
            logger.exception("responses generation failed")
            return error_json(
                500, INTERNAL_ERROR_MESSAGE, SERVER_ERROR, INTERNAL_ERROR_CODE
            )
    record.set_output_tokens(result.usage.completion_tokens)
    response = build_non_streaming_response(
        engine=engine,
        request=request,
        response_id=response_id,
        created_at=created_at,
        text=result.text,
        finish_reason=result.finish_reason,
        usage=result.usage,
    )
    if should_store:
        save_stored_response(
            store,
            response_id,
            response,
            follow_up_context(messages, response.output),
        )
    return JSONResponse(response.model_dump())


app = create_app()


# --------------------------------------------------------------- Epic 6: run


def configure_logging() -> None:
    """Make INFO startup messages (the banner) visible on the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )


def is_out_of_memory(text: str) -> bool:
    """True when an error message reads like an out-of-memory failure."""
    return bool(MEMORY_PATTERN.search(text))


def out_of_memory_message(detail: str = "") -> str:
    """Actionable hint for a model load that failed on memory pressure."""
    if detail:
        return f"{MEMORY_HINT} Cause: {detail}"
    return MEMORY_HINT


def startup_banner(settings: Settings, model_path: Path) -> str:
    """One startup log line: address, model path and generation parameters."""
    params = GenerationParams.from_settings(settings)
    return (
        f"local-llm {__version__} serving {model_path} on "
        f"http://{settings.host}:{settings.port} "
        f"(max_tokens={params.max_tokens}, temperature={params.temperature}, "
        f"top_p={params.top_p})"
    )


def load_for_startup(engine: Engine) -> Path:
    """Load the weights before uvicorn accepts requests.

    Out-of-memory failures become an actionable InferenceError; other
    inference problems pass through unchanged and configuration errors
    (SettingsError) are never swallowed.
    """
    try:
        return engine.load()
    except SettingsError:
        raise
    except MemoryError:
        raise InferenceError(out_of_memory_message()) from None
    except InferenceError as exc:
        if is_out_of_memory(str(exc)):
            raise InferenceError(out_of_memory_message(str(exc))) from None
        raise
    except Exception as exc:
        logger.exception("unexpected failure while loading the model")
        raise InferenceError(
            f"Unexpected {type(exc).__name__} while loading the model; "
            "see the traceback above."
        ) from None


def run_server(engine: Engine | None = None) -> None:
    """Load the model, log the startup banner and serve with uvicorn.

    The model is loaded before uvicorn starts accepting requests. Startup
    failures raise SettingsError or InferenceError with a message that is
    safe to print to the user.
    """
    configure_logging()
    engine = engine if engine is not None else Engine()
    settings = engine.settings
    model_path = load_for_startup(engine)
    logger.info(startup_banner(settings, model_path))
    uvicorn.run(
        create_app(engine=engine), host=settings.host, port=settings.port
    )

