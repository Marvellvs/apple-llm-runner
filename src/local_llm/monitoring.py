"""Minimal monitoring of the local LLM server.

Two observers, both cheap and non-blocking:

* :class:`Monitor` prints one compact ``[MON] ...`` line every N seconds
  through its own named logger, whether the server is idle or busy.
* :class:`RequestTracker` counts in-flight and slot-waiting requests and
  prints one compact ``[REQ] ...`` line when a request finishes.

Monitoring is read-only: it never blocks the event loop, never touches the
generation queue semantics and never changes API responses or SSE payloads.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MONITORING_LOGGER_NAME = "monitoring"
DEFAULT_INTERVAL = 5.0
NA = "n/a"
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"
STATUS_CANCELLED = "CANCELLED"
BYTES_PER_GB = 1024**3

logger = logging.getLogger(MONITORING_LOGGER_NAME)


def configure_monitoring_logger(target: logging.Logger | None = None) -> logging.Logger:
    """Give the monitoring logger a compact handler of its own.

    The handler only prints the raw message (``[MON] ...`` / ``[REQ] ...``) and
    propagation is off, so uvicorn's own loggers and their format stay exactly
    as they were.
    """
    target = target if target is not None else logger
    if not target.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        target.addHandler(handler)
    target.propagate = False
    target.setLevel(logging.INFO)
    return target


configure_monitoring_logger()


# ------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class SystemSample:
    """One snapshot of system metrics; ``None`` means "not available"."""

    cpu_percent: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    proc_rss_gb: float | None = None

    @property
    def ram_percent(self) -> float | None:
        if self.ram_used_gb is None or not self.ram_total_gb:
            return None
        return self.ram_used_gb / self.ram_total_gb * 100.0


def _psutil_module() -> Any:
    """psutil when it is importable, else None (metrics become ``n/a``)."""
    try:
        import psutil
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return psutil


class SystemSampler:
    """Light system metrics for macOS on Apple Silicon (psutil-based).

    ``cpu_percent(interval=None)`` is non-blocking, so sampling costs nothing
    measurable and never delays the event loop. Without psutil every metric is
    reported as ``n/a`` instead of guessing a value.
    """

    def __init__(self) -> None:
        self._psutil = _psutil_module()
        self._process: Any = None
        if self._psutil is not None:
            try:
                self._process = self._psutil.Process(os.getpid())
                # Prime the CPU counters so the first sample is meaningful.
                self._psutil.cpu_percent(interval=None)
            except Exception:  # pragma: no cover - defensive
                self._process = None

    def sample(self) -> SystemSample:
        """Take one sample; a failing metric is reported as ``None``."""
        return SystemSample(
            cpu_percent=self.cpu_percent(),
            ram_used_gb=self.ram_used_gb(),
            ram_total_gb=self.ram_total_gb(),
            proc_rss_gb=self.proc_rss_gb(),
        )

    def cpu_percent(self) -> float | None:
        if self._psutil is None:
            return None
        try:
            return float(self._psutil.cpu_percent(interval=None))
        except Exception:  # pragma: no cover - defensive
            return None

    def ram_used_gb(self) -> float | None:
        if self._psutil is None:
            return None
        try:
            memory = self._psutil.virtual_memory()
        except Exception:  # pragma: no cover - defensive
            return None
        total = getattr(memory, "total", None)
        available = getattr(memory, "available", None)
        if total is None or available is None:
            return None
        return max(0.0, (total - available) / BYTES_PER_GB)

    def ram_total_gb(self) -> float | None:
        if self._psutil is None:
            return None
        try:
            return float(self._psutil.virtual_memory().total) / BYTES_PER_GB
        except Exception:  # pragma: no cover - defensive
            return None

    def proc_rss_gb(self) -> float | None:
        if self._process is None:
            return None
        try:
            return float(self._process.memory_info().rss) / BYTES_PER_GB
        except Exception:  # pragma: no cover - defensive
            return None


# ----------------------------------------------------------------- formatting


def _format_percent(value: float | None, digits: int = 0) -> str:
    if value is None:
        return NA
    return f"{value:.{digits}f}%"


def _format_gb(value: float | None, digits: int = 1) -> str:
    if value is None:
        return NA
    return f"{value:.{digits}f}"


def format_monitor_line(
    sample: SystemSample, *, active: int = 0, queued: int = 0
) -> str:
    """Render the periodic line, e.g. ``[MON] CPU 34% | RAM 18.4/24 GB (77%) ...``."""
    if sample.ram_used_gb is None and sample.ram_total_gb is None:
        ram = NA
    elif sample.ram_total_gb is None:
        ram = f"{_format_gb(sample.ram_used_gb)} GB"
    else:
        ram = (
            f"{_format_gb(sample.ram_used_gb)}/"
            f"{_format_gb(sample.ram_total_gb, 0)} GB"
            f" ({_format_percent(sample.ram_percent)})"
        )
    return (
        f"[MON] CPU {_format_percent(sample.cpu_percent)}"
        f" | RAM {ram}"
        f" | PROC {_format_gb(sample.proc_rss_gb)} GB"
        f" | Active {active} | Queue {queued}"
    )


def format_request_line(
    *,
    endpoint: str,
    total: float,
    status: str = STATUS_OK,
    tokens: int | None = None,
    rate: float | None = None,
    ttft: float | None = None,
) -> str:
    """Render one finished-request line.

    Successful requests always show the token fields (``n/a`` when the engine
    exposes no counter); failed and cancelled requests only show what was
    actually measured.
    """
    parts = [f"[REQ] POST {endpoint}"]
    if status == STATUS_OK:
        parts.append(f"{tokens} tokens" if tokens is not None else f"{NA} tokens")
        parts.append(
            f"{rate:.1f} tok/s" if rate is not None else f"{NA} tok/s"
        )
        parts.append(f"TTFT {ttft:.2f}s" if ttft is not None else f"TTFT {NA}")
    else:
        if tokens is not None:
            parts.append(f"{tokens} tokens")
        if rate is not None:
            parts.append(f"{rate:.1f} tok/s")
        if ttft is not None:
            parts.append(f"TTFT {ttft:.2f}s")
    parts.append(f"Total {total:.1f}s")
    parts.append(status)
    return " | ".join(parts)


# -------------------------------------------------------------- periodic line


class Monitor:
    """Prints one ``[MON]`` line every ``interval`` seconds while running.

    The timer lives on its own daemon thread (not on an asyncio task) on
    purpose: the SSE generators iterate a blocking MLX iterator, so the event
    loop can be starved for the whole duration of a streaming generation. A
    thread keeps the line periodic during active generation as well, and it
    never touches the event loop.
    """

    def __init__(
        self,
        *,
        tracker: "RequestTracker | None" = None,
        interval: float = DEFAULT_INTERVAL,
        enabled: bool = True,
        sampler: SystemSampler | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.interval = max(float(interval), 0.001)
        self.enabled = enabled
        self.sampler = sampler if sampler is not None else SystemSampler()
        self.tracker = tracker
        self._log = log if log is not None else logger.info
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def line(self) -> str:
        """Current ``[MON]`` line (also used by tests)."""
        active = self.tracker.active if self.tracker is not None else 0
        queued = self.tracker.queued if self.tracker is not None else 0
        return format_monitor_line(
            self.sampler.sample(), active=active, queued=queued
        )

    def emit(self) -> str:
        line = self.line()
        self._log(line)
        return line

    def start(self) -> threading.Thread | None:
        """Start emitting (a no-op when disabled or already running)."""
        if not self.enabled:
            return None
        if self.running:
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="local-llm-monitoring", daemon=True
        )
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        """Stop emitting; safe to call more than once."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.emit()
            self._stop.wait(self.interval)


# ------------------------------------------------------------ request tracker


class RequestTracker:
    """Counts in-flight requests and requests waiting for the generation slot.

    All mutations are cheap and guarded by a plain lock, so the tracker is safe
    from the event loop as well as from the background worker threads.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        log: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self._log = log if log is not None else logger.info
        self._clock = clock
        self._lock = threading.Lock()
        self._active = 0
        self._queued = 0

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def queued(self) -> int:
        with self._lock:
            return self._queued

    def now(self) -> float:
        return self._clock()

    def start(self, endpoint: str) -> "RequestRecord":
        """Begin tracking one request; the counter is released by finish()."""
        with self._lock:
            self._active += 1
        return RequestRecord(tracker=self, endpoint=endpoint, started=self.now())

    def begin_wait(self) -> None:
        with self._lock:
            self._queued += 1

    def end_wait(self) -> None:
        with self._lock:
            self._queued = max(0, self._queued - 1)

    def _release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def log(self, line: str) -> None:
        if self.enabled:
            self._log(line)


@dataclass
class RequestRecord:
    """Timing and token accounting of one request (one ``[REQ]`` line)."""

    tracker: RequestTracker
    endpoint: str
    started: float
    first_token_at: float | None = None
    output_tokens: int | None = None
    finished: bool = False
    deferred: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_first_token(self) -> None:
        """Record TTFT; later calls are ignored."""
        with self._lock:
            if self.first_token_at is None:
                self.first_token_at = self.tracker.now()

    def set_output_tokens(self, tokens: int | None) -> None:
        """Store the real token count from the engine (unknown stays unknown)."""
        if tokens is None:
            return
        with self._lock:
            self.output_tokens = int(tokens)

    def defer(self) -> None:
        """Hand completion over to a stream generator (it finishes the record)."""
        self.deferred = True

    @property
    def ttft(self) -> float | None:
        if self.first_token_at is None:
            return None
        return max(0.0, self.first_token_at - self.started)

    @property
    def total(self) -> float:
        return max(0.0, self.tracker.now() - self.started)

    @property
    def tokens_per_second(self) -> float | None:
        """Output tokens divided by generation time (TTFT excluded)."""
        tokens = self.output_tokens
        if tokens is None:
            return None
        generation = self.total - (self.ttft or 0.0)
        if generation <= 0:
            return None
        return tokens / generation

    def finish(self, status: str = STATUS_OK) -> str:
        """Release the active counter and log the line exactly once."""
        with self._lock:
            if self.finished:
                return ""
            self.finished = True
            tokens = self.output_tokens
        self.tracker._release()
        line = format_request_line(
            endpoint=self.endpoint,
            total=self.total,
            status=status,
            tokens=tokens,
            rate=self.tokens_per_second,
            ttft=self.ttft,
        )
        self.tracker.log(line)
        return line
