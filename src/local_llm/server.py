"""FastAPI app and server startup: /v1/chat/completions (Epics 5 and 6)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .config import Settings, SettingsError
from .inference import (
    Engine,
    GenerationParams,
    GenerationResult,
    InferenceError,
    ModelNotLoadedError,
    TokenChunk,
    Usage,
)
from .schemas import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_BODY,
    ERROR_MODEL_NOT_FOUND,
    OBJECT_CHAT_COMPLETION_CHUNK,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageOut,
    error_response,
)

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
MEMORY_PATTERN = re.compile(
    r"\b(?:oom|out of memory|not enough memory|cannot allocate memory|"
    r"can't allocate memory|memory (?:limit|exhausted|allocation failed))\b",
    re.IGNORECASE,
)


class GenerationQueue:
    """Serializes model access: one request at a time (MLX unified memory).

    A threading lock (not asyncio.Lock) keeps the queue usable from any event
    loop - TestClient and uvicorn may run requests on different loops.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await asyncio.to_thread(self._lock.acquire)
        try:
            yield
        finally:
            self._lock.release()


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
        message = str(error.get("msg", "invalid value"))
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
) -> AsyncIterator[str]:
    """Render token chunks as SSE; a mid-stream failure becomes an error event."""
    chunk_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    usage: UsageOut | None = None
    finish_reason: str | None = None
    failed = False
    async with queue.slot():
        yield sse(
            chunk_payload(chunk_id, created, model, {"role": "assistant"}, None)
        )
        try:
            for token in tokens:
                if token.usage is not None:
                    usage = usage_out(token.usage)
                if token.text:
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
        if not failed:
            if finish_reason is None:
                yield sse(chunk_payload(chunk_id, created, model, {}, "stop"))
            if usage is not None:
                yield sse(usage_payload(chunk_id, created, model, usage))
        yield sse("[DONE]")


async def stream_response(
    engine: Engine,
    model: str,
    messages: list[dict[str, str]],
    params: GenerationParams,
    queue: GenerationQueue,
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
    return StreamingResponse(sse_events(tokens, model, queue), media_type=SSE_MEDIA_TYPE)


def create_app(engine: Engine | None = None) -> FastAPI:
    """Build the FastAPI app around an (optional) pre-built inference engine."""
    app = FastAPI(
        title="local-llm",
        version=__version__,
        description="OpenAI-compatible chat completions served locally via MLX.",
    )
    app.state.engine = engine if engine is not None else Engine()
    app.state.queue = GenerationQueue()

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
                engine, request.model, messages, params, app.state.queue
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
        return completion_response(request.model, result)

    return app


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

