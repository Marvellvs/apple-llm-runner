"""Responses API logic: input normalization, output parsing, SSE assembly.

Three responsibilities deliberately kept apart:

1. normalization - Responses ``input`` items -> chat messages for the engine
   (:func:`messages_from_input`, :func:`build_messages`);
2. interpretation - raw model text -> text / reasoning / tool-call events
   (:class:`ModelOutputParser`);
3. rendering - those events -> Response object and typed SSE payloads
   (:class:`ResponseStreamAssembler`).

Nothing here knows about HTTP or about a concrete model family: markers come
from :class:`local_llm.inference.ModelMarkers`, and a model-specific tool-call
parser can be injected from the engine.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence
from uuid import uuid4

from .inference import ModelMarkers, Usage, default_markers
from .schemas import (
    INCOMPLETE_MAX_OUTPUT_TOKENS,
    OBJECT_RESPONSE,
    RESPONSE_ID_PREFIX,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_INCOMPLETE,
    FunctionCallItem,
    FunctionCallOutputItem,
    InputMessageItem,
    InputTextPart,
    OutputTextPart,
    ReasoningItem,
    ReasoningSummaryPart,
    ResponseErrorBody,
    ResponseFunctionCall,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryPart,
    ResponseRequest,
    ResponsesUsage,
    IncompleteDetails,
    InputTokensDetails,
    OutputTokensDetails,
    TextPart,
)

logger = logging.getLogger("local_llm.responses")

DEVELOPER_ROLE = "developer"
_ROLE_ALIASES = {"developer": "system"}


# --------------------------------------------------------------------- helpers


def new_response_id() -> str:
    """Unique ``resp_*`` identifier."""
    return f"{RESPONSE_ID_PREFIX}{uuid4().hex}"


def new_item_id(prefix: str) -> str:
    """Unique output-item identifier (``msg_`` / ``rs_`` / ``fc_``)."""
    return f"{prefix}{uuid4().hex}"


def new_call_id() -> str:
    """Unique function call id (``call_*``)."""
    return f"call_{uuid4().hex}"


class ResponseRequestError(Exception):
    """A request that must be answered with a specific HTTP error."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


# --------------------------------------------------------- input normalization


def _part_text(part: Any) -> str:
    """Text of one content part (typed part or bare string)."""
    if isinstance(part, str):
        return part
    if isinstance(part, (InputTextPart, OutputTextPart, TextPart)):
        return part.text
    raise ResponseRequestError(
        f"Unsupported content part: {part!r}. Supported part types are "
        "input_text, output_text and text."
    )


def _content_text(content: Any) -> str:
    """Flatten a message content value (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_part_text(part) for part in content)
    raise ResponseRequestError(f"Unsupported message content: {content!r}")


def _tool_output_text(output: Any) -> str:
    """Flatten a function_call_output value (string, parts or JSON object)."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _content_text(output)
    if isinstance(output, dict):
        return json.dumps(output, ensure_ascii=False)
    return str(output)


def _reasoning_text(item: ReasoningItem) -> str:
    """Readable text of a replayed reasoning item (summary and/or content)."""
    parts: list[str] = []
    for group in (item.summary, item.content):
        for part in group or ():
            if isinstance(part, ReasoningSummaryPart) and part.text:
                parts.append(part.text)
    return "".join(parts)


def messages_from_input(input_value: Any) -> list[dict[str, Any]]:
    """Convert Responses ``input`` into chat messages for the engine.

    Order and part order are preserved. Consecutive ``function_call`` items are
    merged into one assistant message with ``tool_calls``; ``function_call_output``
    becomes a ``tool`` message; ``reasoning`` becomes an assistant message with
    ``reasoning_content`` (ignored by templates that do not know the key).
    """
    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if pending_calls:
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": list(pending_calls)}
            )
            pending_calls.clear()

    if isinstance(input_value, str):
        if input_value.strip():
            messages.append({"role": "user", "content": input_value})
    else:
        for item in input_value:
            if isinstance(item, InputMessageItem):
                flush_calls()
                role = _ROLE_ALIASES.get(item.role, item.role)
                messages.append({"role": role, "content": _content_text(item.content)})
            elif isinstance(item, FunctionCallItem):
                pending_calls.append(
                    {
                        "id": item.call_id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": item.arguments},
                    }
                )
            elif isinstance(item, FunctionCallOutputItem):
                flush_calls()
                messages.append(
                    {
                        "role": "tool",
                        "content": _tool_output_text(item.output),
                        "tool_call_id": item.call_id,
                    }
                )
            elif isinstance(item, ReasoningItem):
                flush_calls()
                text = _reasoning_text(item)
                if text:
                    messages.append(
                        {"role": "assistant", "content": "", "reasoning_content": text}
                    )
            else:
                raise ResponseRequestError(
                    f"Unsupported input item: {type(item).__name__}"
                )
    flush_calls()
    if not messages:
        raise ResponseRequestError(
            "input does not contain any usable message; send a string, a message "
            "item or a supported function_call/function_call_output item"
        )
    return messages


def build_messages(
    request: ResponseRequest,
    previous_context: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Messages for the engine: prior context (if any) followed by new input.

    ``instructions`` are prepended as a system message unless the carried
    context already starts with exactly the same instruction, so a repeated
    ``instructions`` + ``previous_response_id`` pair does not duplicate history.
    """
    messages = messages_from_input(request.input)
    instructions = (request.instructions or "").strip()
    if instructions:
        context_head = previous_context[0] if previous_context else None
        already_present = (
            context_head is not None
            and context_head.get("role") == "system"
            and context_head.get("content") == request.instructions
        )
        if not already_present:
            messages.insert(0, {"role": "system", "content": request.instructions})
    return [dict(message) for message in (previous_context or ())] + messages


def follow_up_context(
    prompt_messages: Sequence[dict[str, Any]],
    output: Sequence[Any],
) -> list[dict[str, Any]]:
    """Context to store for ``previous_response_id`` (prompt + assistant reply)."""
    context = [dict(message) for message in prompt_messages]
    text = "".join(
        part.text
        for item in output
        if isinstance(item, ResponseOutputMessage)
        for part in item.content
    )
    calls = [item for item in output if isinstance(item, ResponseFunctionCall)]
    if calls:
        context.append(
            {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ],
            }
        )
    elif text:
        context.append({"role": "assistant", "content": text})
    return context


# ------------------------------------------------------------ model output parser


@dataclass(frozen=True)
class ParsedToolCall:
    """One function call extracted from the model output."""

    name: str
    arguments: str
    call_id: str | None = None


@dataclass(frozen=True)
class ParsedEvent:
    """A neutral event produced by :class:`ModelOutputParser`."""

    kind: Literal["text", "reasoning", "tool_call"]
    text: str = ""
    call: ParsedToolCall | None = None


ToolCallParser = Callable[[str], Any]


@dataclass
class _ScanMarker:
    marker: str
    kind: str
    next_state: str


class ModelOutputParser:
    """Incremental, marker-driven splitter of raw model text.

    Handles markers that arrive split across chunks: text that might still be
    the beginning of a marker is buffered, never emitted early. A tool-call
    block that never closes (e.g. truncated by max_output_tokens) is surfaced
    as plain text - no fake function call is ever produced.
    """

    def __init__(
        self,
        markers: ModelMarkers | None = None,
        *,
        enable_reasoning: bool = True,
        enable_tools: bool = False,
        tool_call_parser: ToolCallParser | None = None,
    ) -> None:
        self.markers = markers or default_markers()
        self.enable_reasoning = enable_reasoning and self.markers.has_reasoning
        self.enable_tools = enable_tools and self.markers.has_tools
        self._tool_call_parser = tool_call_parser
        self._state = "scan"
        self._buffer = ""
        self._scan: list[_ScanMarker] = []
        if self.enable_reasoning:
            self._scan.append(
                _ScanMarker(self.markers.reasoning_start or "", "reasoning", "reasoning")
            )
        if self.enable_tools:
            self._scan.append(
                _ScanMarker(self.markers.tool_start or "", "tool", "tool")
            )
        for marker in self.markers.structural:
            if marker:
                self._scan.append(_ScanMarker(marker, "structural", "scan"))

    @property
    def state(self) -> str:
        return self._state

    def feed(self, text: str) -> list[ParsedEvent]:
        """Consume one chunk of model text."""
        if self._state == "done":
            return []
        self._buffer += text or ""
        return self._drain(final=False)

    def finish(self) -> list[ParsedEvent]:
        """Flush the remaining buffer; no further events are produced after."""
        events = self._drain(final=True)
        self._state = "done"
        return events

    # ------------------------------------------------------------- internals

    def _drain(self, final: bool) -> list[ParsedEvent]:
        events: list[ParsedEvent] = []
        while True:
            buffer = self._buffer
            if self._state == "scan":
                match = self._find_marker(buffer)
                if match is None:
                    self._emit_scanned(buffer, final, events)
                    break
                index, marker = match
                before = buffer[:index]
                if before:
                    events.extend(self._text_events(self._drop_structural(before)))
                self._buffer = buffer[index + len(marker.marker) :]
                self._state = marker.next_state
                continue
            if self._state == "reasoning":
                end = self.markers.reasoning_end or ""
                index = buffer.find(end)
                if index == -1:
                    emit, hold = self._split(buffer, end, final)
                    if emit:
                        events.append(ParsedEvent("reasoning", emit))
                    self._buffer = hold
                    break
                if index:
                    events.append(ParsedEvent("reasoning", buffer[:index]))
                self._buffer = buffer[index + len(end) :]
                self._state = "scan"
                continue
            if self._state == "tool":
                end = self.markers.tool_end or ""
                index = buffer.find(end)
                if index == -1:
                    if final:
                        raw = self.markers.tool_start + buffer if buffer else ""
                        events.extend(self._text_events(raw))
                        self._buffer = ""
                        self._state = "scan"
                        break
                    # A tool call body may still be partial: keep it buffered.
                    break
                body = buffer[:index]
                self._buffer = buffer[index + len(end) :]
                self._state = "scan"
                calls = self._parse_calls(body)
                if calls is None:
                    events.extend(
                        self._text_events(
                            f"{self.markers.tool_start}{body}{self.markers.tool_end}"
                        )
                    )
                else:
                    events.extend(ParsedEvent("tool_call", call=call) for call in calls)
                continue
            break
        return events

    def _emit_scanned(
        self, buffer: str, final: bool, events: list[ParsedEvent]
    ) -> None:
        markers = [entry.marker for entry in self._scan]
        hold = 0 if final else _partial_marker_suffix(buffer, markers)
        emit = buffer[: len(buffer) - hold] if hold else buffer
        if emit:
            events.extend(self._text_events(self._drop_structural(emit)))
        self._buffer = buffer[len(buffer) - hold :] if hold else ""

    @staticmethod
    def _split(buffer: str, marker: str, final: bool) -> tuple[str, str]:
        """Split a buffer into emittable text and a held partial marker."""
        if final or not marker:
            return buffer, ""
        hold = _partial_marker_suffix(buffer, [marker])
        if not hold:
            return buffer, ""
        return buffer[: len(buffer) - hold], buffer[len(buffer) - hold :]

    def _find_marker(self, buffer: str) -> tuple[int, _ScanMarker] | None:
        best: tuple[int, _ScanMarker] | None = None
        for marker in self._scan:
            if not marker.marker:
                continue
            index = buffer.find(marker.marker)
            if index != -1 and (best is None or index < best[0]):
                best = (index, marker)
        return best

    def _drop_structural(self, text: str) -> str:
        for marker in self.markers.structural:
            if marker:
                text = text.replace(marker, "")
        return text

    @staticmethod
    def _text_events(text: str) -> list[ParsedEvent]:
        return [ParsedEvent("text", text)] if text else []

    def _parse_calls(self, body: str) -> list[ParsedToolCall] | None:
        """Parse a tool-call body; None means "not a tool call"."""
        payload: Any
        if self._tool_call_parser is not None:
            try:
                payload = self._tool_call_parser(body)
            except Exception as exc:  # a model-specific parser must not crash us
                logger.warning("tool-call parser failed: %s", exc)
                return None
        else:
            try:
                payload = json.loads(body.strip())
            except (ValueError, TypeError):
                return None
        entries = payload if isinstance(payload, list) else [payload]
        calls: list[ParsedToolCall] = []
        for entry in entries:
            call = _tool_call_from_mapping(entry)
            if call is not None:
                calls.append(call)
        return calls or None


def _partial_marker_suffix(buffer: str, markers: Iterable[str]) -> int:
    """Length of the longest suffix of ``buffer`` that may start a marker."""
    longest = 0
    for marker in markers:
        if not marker:
            continue
        limit = min(len(marker) - 1, len(buffer))
        for size in range(limit, 0, -1):
            if size <= longest:
                break
            if buffer.endswith(marker[:size]):
                longest = size
                break
    return longest


def _tool_call_from_mapping(entry: Any) -> ParsedToolCall | None:
    """Convert one parsed tool-call object into a ParsedToolCall."""
    if not isinstance(entry, dict):
        return None
    function = entry.get("function")
    if isinstance(function, dict):
        entry = {**function, **{k: v for k, v in entry.items() if k != "function"}}
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = entry.get("arguments")
    if isinstance(arguments, str):
        serialized = arguments
    elif arguments is None:
        serialized = "{}"
    else:
        serialized = json.dumps(arguments, ensure_ascii=False)
    call_id = entry.get("id") or entry.get("call_id")
    return ParsedToolCall(
        name=name.strip(),
        arguments=serialized,
        call_id=call_id if isinstance(call_id, str) and call_id else None,
    )


# ------------------------------------------------------------- object assembly


def responses_usage(usage: Usage | None, reasoning_tokens: int = 0) -> ResponsesUsage:
    """Map engine usage onto the Responses usage shape."""
    if usage is None:
        return ResponsesUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    return ResponsesUsage(
        input_tokens=usage.prompt_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=usage.completion_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=usage.total_tokens,
    )


def _message_item(text: str, status: str, item_id: str | None = None) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id or new_item_id("msg_"),
        status=status,  # type: ignore[arg-type]
        content=[ResponseOutputText(text=text)],
    )


def _reasoning_item(
    text: str, status: str, item_id: str | None = None
) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id=item_id or new_item_id("rs_"),
        status=status,  # type: ignore[arg-type]
        summary=[ResponseReasoningSummaryPart(text=text)] if text else [],
    )


def reasoning_text_of(output: Sequence[Any]) -> str:
    """Concatenated reasoning summary text of the output items."""
    return "".join(
        part.text
        for item in output
        if isinstance(item, ResponseReasoningItem)
        for part in item.summary
    )


def count_reasoning_tokens(
    output: Sequence[Any], counter: Callable[[str], int] | None
) -> int:
    """Reasoning token count when the engine can count tokens, else 0."""
    reasoning = reasoning_text_of(output)
    if not reasoning or counter is None:
        return 0
    try:
        return max(0, int(counter(reasoning)))
    except Exception:  # counting is best-effort only
        logger.debug("reasoning token counting failed", exc_info=True)
        return 0


def build_output_items(events: Iterable[ParsedEvent]) -> list[Any]:
    """Build non-streaming output items from parser events (text order kept)."""
    items: list[Any] = []
    for event in events:
        if event.kind == "text" and event.text:
            if isinstance(items[-1] if items else None, ResponseOutputMessage):
                items[-1].content[0].text += event.text
            else:
                items.append(_message_item(event.text, STATUS_COMPLETED))
        elif event.kind == "reasoning" and event.text:
            if isinstance(items[-1] if items else None, ResponseReasoningItem):
                items[-1].summary[0].text += event.text
            else:
                items.append(_reasoning_item(event.text, STATUS_COMPLETED))
        elif event.kind == "tool_call" and event.call is not None:
            items.append(
                ResponseFunctionCall(
                    id=new_item_id("fc_"),
                    call_id=event.call.call_id or new_call_id(),
                    name=event.call.name,
                    arguments=event.call.arguments,
                )
            )
    return items


def parse_output(
    text: str,
    *,
    markers: ModelMarkers | None = None,
    enable_reasoning: bool = True,
    enable_tools: bool = False,
    tool_call_parser: ToolCallParser | None = None,
) -> list[ParsedEvent]:
    """Parse a complete model output (non-streaming path)."""
    parser = ModelOutputParser(
        markers,
        enable_reasoning=enable_reasoning,
        enable_tools=enable_tools,
        tool_call_parser=tool_call_parser,
    )
    return parser.feed(text) + parser.finish()


def build_response_object(
    *,
    request: ResponseRequest,
    response_id: str,
    created_at: int,
    status: str,
    output: Sequence[Any] | None = None,
    usage: ResponsesUsage | None = None,
    error: ResponseErrorBody | None = None,
    incomplete_reason: str | None = None,
) -> ResponseObject:
    """Build a Response object from request metadata and generated output."""
    return ResponseObject(
        id=response_id,
        object=OBJECT_RESPONSE,  # type: ignore[arg-type]
        created_at=created_at,
        status=status,  # type: ignore[arg-type]
        error=error,
        incomplete_details=(
            IncompleteDetails(reason=incomplete_reason) if incomplete_reason else None
        ),
        model=request.model,
        output=list(output or ()),
        usage=usage,
        instructions=request.instructions,
        tools=list(request.tools),
        tool_choice=request.tool_choice,
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_output_tokens,
        parallel_tool_calls=request.parallel_tool_calls,
        metadata=request.metadata,
        background=request.background if request.background else None,
        previous_response_id=request.previous_response_id,
        store=request.store if request.store is not None else True,
    )


# ------------------------------------------------------------------- SSE events


@dataclass
class SseEvent:
    """One typed SSE event: the name plus the JSON payload."""

    type: str
    data: dict[str, Any]

    def render(self) -> str:
        """``event: <type>\\ndata: <json>\\n\\n`` (payload also carries ``type``)."""
        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.type}\ndata: {payload}\n\n"


class ResponseStreamAssembler:
    """Turns parser events into typed SSE events and the final Response object.

    Invariants: every delta is preceded by its ``*.output_item.added``, the
    ``output_index`` never changes within an item and is never reused, item ids
    are unique, and the terminal event is emitted exactly once.
    """

    def __init__(
        self,
        *,
        request: ResponseRequest,
        response_id: str,
        created_at: int,
        markers: ModelMarkers | None = None,
        tool_call_parser: ToolCallParser | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        self.request = request
        self.response_id = response_id
        self.created_at = created_at
        self._count_tokens = count_tokens
        self.parser = ModelOutputParser(
            markers,
            enable_reasoning=True,
            enable_tools=bool(request.tools),
            tool_call_parser=tool_call_parser,
        )
        self._sequence = 0
        self._next_output_index = 0
        self._items: list[Any] = []
        self._open: dict[str, Any] | None = None
        self._usage: ResponsesUsage | None = None
        self._status: str = STATUS_IN_PROGRESS
        self._error: ResponseErrorBody | None = None
        self._incomplete_reason: str | None = None
        self.response = self._response_object(STATUS_IN_PROGRESS)

    # ------------------------------------------------------------- properties

    @property
    def output(self) -> list[Any]:
        return list(self._items)

    @property
    def status(self) -> str:
        return self._status

    # ------------------------------------------------------------ start events

    def created_events(self) -> list[SseEvent]:
        """``response.created`` followed by ``response.in_progress``."""
        return [
            self._event("response.created", response=self._response_object(STATUS_IN_PROGRESS).model_dump()),
            self._event(
                "response.in_progress",
                response=self._response_object(STATUS_IN_PROGRESS).model_dump(),
            ),
        ]

    # -------------------------------------------------------------- token feed

    def push(self, text: str) -> list[SseEvent]:
        """Consume a model text chunk and render its events."""
        events: list[SseEvent] = []
        for parsed in self.parser.feed(text):
            events.extend(self._render(parsed))
        return events

    def finish(
        self, *, finish_reason: str | None, usage: Usage | None
    ) -> list[SseEvent]:
        """Close open items and emit the single terminal event."""
        events: list[SseEvent] = []
        for parsed in self.parser.finish():
            events.extend(self._render(parsed))
        status = (
            STATUS_INCOMPLETE if finish_reason == "length" else STATUS_COMPLETED
        )
        events.extend(self._close_open(status))
        reasoning_tokens = count_reasoning_tokens(self._items, self._count_tokens)
        self._usage = responses_usage(usage, reasoning_tokens)
        self._status = status
        self._incomplete_reason = (
            INCOMPLETE_MAX_OUTPUT_TOKENS if finish_reason == "length" else None
        )
        self.response = self._response_object(status)
        event_type = (
            "response.incomplete"
            if status == STATUS_INCOMPLETE
            else "response.completed"
        )
        events.append(self._event(event_type, response=self.response.model_dump()))
        return events

    def fail(self, message: str, code: str) -> list[SseEvent]:
        """Close open items and render the stream error event."""
        events = self._close_open(STATUS_INCOMPLETE)
        self._status = STATUS_FAILED
        self._error = ResponseErrorBody(code=code, message=message)
        self.response = self._response_object(STATUS_FAILED)
        events.append(
            self._event(
                "error",
                code=code,
                message=message,
                param=None,
                error={"type": "server_error", "code": code, "message": message},
            )
        )
        return events

    # ---------------------------------------------------------------- internals

    def _event(self, event_type: str, **fields: Any) -> SseEvent:
        payload = {"type": event_type, "sequence_number": self._sequence}
        payload.update(fields)
        self._sequence += 1
        return SseEvent(event_type, payload)

    def _response_object(self, status: str) -> ResponseObject:
        return build_response_object(
            request=self.request,
            response_id=self.response_id,
            created_at=self.created_at,
            status=status,
            output=self._items,
            usage=self._usage,
            error=self._error,
            incomplete_reason=self._incomplete_reason,
        )

    def _render(self, parsed: ParsedEvent) -> list[SseEvent]:
        if parsed.kind == "text" and parsed.text:
            return self._render_text(parsed.text)
        if parsed.kind == "reasoning" and parsed.text:
            return self._render_reasoning(parsed.text)
        if parsed.kind == "tool_call" and parsed.call is not None:
            return self._render_tool_call(parsed.call)
        return []

    def _render_text(self, text: str) -> list[SseEvent]:
        events: list[SseEvent] = []
        if self._open is None or self._open["kind"] != "message":
            events.extend(self._close_open(STATUS_COMPLETED))
            events.extend(self._open_message())
        self._open["text"] += text
        events.append(
            self._event(
                "response.output_text.delta",
                item_id=self._open["id"],
                output_index=self._open["index"],
                content_index=0,
                delta=text,
                logprobs=[],
            )
        )
        return events

    def _render_reasoning(self, text: str) -> list[SseEvent]:
        events: list[SseEvent] = []
        if self._open is None or self._open["kind"] != "reasoning":
            events.extend(self._close_open(STATUS_COMPLETED))
            events.extend(self._open_reasoning())
        self._open["text"] += text
        events.append(
            self._event(
                "response.reasoning_summary_text.delta",
                item_id=self._open["id"],
                output_index=self._open["index"],
                summary_index=0,
                delta=text,
            )
        )
        return events

    def _render_tool_call(self, call: ParsedToolCall) -> list[SseEvent]:
        events: list[SseEvent] = []
        events.extend(self._close_open(STATUS_COMPLETED))
        item = ResponseFunctionCall(
            id=new_item_id("fc_"),
            call_id=call.call_id or new_call_id(),
            name=call.name,
            arguments="",
            status=STATUS_IN_PROGRESS,  # type: ignore[arg-type]
        )
        details = {
            "kind": "function_call",
            "id": item.id,
            "index": self._take_index(),
            "item": item,
            "text": call.arguments,
        }
        self._open = details
        events.append(
            self._event(
                "response.output_item.added",
                output_index=details["index"],
                item=item.model_dump(),
            )
        )
        events.append(
            self._event(
                "response.function_call_arguments.delta",
                item_id=item.id,
                output_index=details["index"],
                delta=call.arguments,
            )
        )
        events.append(
            self._event(
                "response.function_call_arguments.done",
                item_id=item.id,
                output_index=details["index"],
                arguments=call.arguments,
            )
        )
        events.extend(self._close_open(STATUS_COMPLETED))
        return events

    def _take_index(self) -> int:
        index = self._next_output_index
        self._next_output_index += 1
        return index

    def _open_message(self) -> list[SseEvent]:
        item = ResponseOutputMessage(
            id=new_item_id("msg_"),
            status=STATUS_IN_PROGRESS,  # type: ignore[arg-type]
            content=[],
        )
        self._open = {
            "kind": "message",
            "id": item.id,
            "index": self._take_index(),
            "item": item,
            "text": "",
        }
        return [
            self._event(
                "response.output_item.added",
                output_index=self._open["index"],
                item=item.model_dump(),
            ),
            self._event(
                "response.content_part.added",
                item_id=item.id,
                output_index=self._open["index"],
                content_index=0,
                part=ResponseOutputText(text="").model_dump(),
            ),
        ]

    def _open_reasoning(self) -> list[SseEvent]:
        item = ResponseReasoningItem(
            id=new_item_id("rs_"),
            status=STATUS_IN_PROGRESS,  # type: ignore[arg-type]
            summary=[],
        )
        self._open = {
            "kind": "reasoning",
            "id": item.id,
            "index": self._take_index(),
            "item": item,
            "text": "",
        }
        return [
            self._event(
                "response.output_item.added",
                output_index=self._open["index"],
                item=item.model_dump(),
            ),
            self._event(
                "response.reasoning_summary_part.added",
                item_id=item.id,
                output_index=self._open["index"],
                summary_index=0,
                part=ResponseReasoningSummaryPart(text="").model_dump(),
            ),
        ]

    def _close_open(self, status: str) -> list[SseEvent]:
        """Close the currently open output item (if any)."""
        open_item = self._open
        if open_item is None:
            return []
        self._open = None
        events: list[SseEvent] = []
        text = open_item["text"]
        index = open_item["index"]
        item_id = open_item["id"]
        if open_item["kind"] == "message":
            item = _message_item(text, status, item_id)
            events.append(
                self._event(
                    "response.output_text.done",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    text=text,
                    logprobs=[],
                )
            )
            events.append(
                self._event(
                    "response.content_part.done",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    part=ResponseOutputText(text=text).model_dump(),
                )
            )
        elif open_item["kind"] == "reasoning":
            item = _reasoning_item(text, status, item_id)
            events.append(
                self._event(
                    "response.reasoning_summary_text.done",
                    item_id=item_id,
                    output_index=index,
                    summary_index=0,
                    text=text,
                )
            )
            events.append(
                self._event(
                    "response.reasoning_summary_part.done",
                    item_id=item_id,
                    output_index=index,
                    summary_index=0,
                    part=ResponseReasoningSummaryPart(text=text).model_dump(),
                )
            )
        else:
            item = open_item["item"].model_copy(
                update={"arguments": text, "status": status}
            )
        events.append(
            self._event(
                "response.output_item.done", output_index=index, item=item.model_dump()
            )
        )
        self._items.append(item)
        return events


def sse_done() -> str:
    """The final ``data: [DONE]`` line of a stream."""
    return "data: [DONE]\n\n"


def now_timestamp() -> int:
    """Unix timestamp used for ``created_at``."""
    return int(time.time())
