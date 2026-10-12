"""Thread-safe in-memory store for Responses API objects (level B).

The store keeps the public Response object plus the private chat context that
a follow-up request (``previous_response_id``) needs. Everything lives in
memory: restarting the server drops the whole history.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import (
    DEFAULT_RESPONSES_STORE_MAX,
    DEFAULT_RESPONSES_STORE_TTL,
    Settings,
)


@dataclass
class StoredResponse:
    """One stored response: the public object plus its private chat context."""

    response: dict[str, Any]
    context: list[dict[str, Any]] = field(default_factory=list)
    created: float = 0.0

    @property
    def status(self) -> str:
        return str(self.response.get("status", "completed"))


class ResponseStore:
    """In-memory, thread-safe, TTL- and size-limited response store."""

    def __init__(
        self,
        max_entries: int = DEFAULT_RESPONSES_STORE_MAX,
        ttl_seconds: int = DEFAULT_RESPONSES_STORE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if not isinstance(ttl_seconds, int) or ttl_seconds < 1:
            raise ValueError("ttl_seconds must be a positive integer")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, StoredResponse] = OrderedDict()

    @classmethod
    def from_settings(cls, settings: Settings) -> ResponseStore:
        """Build a store from the LLM_RESPONSES_STORE_* environment settings."""
        return cls(
            max_entries=settings.responses_store_max,
            ttl_seconds=settings.responses_store_ttl,
        )

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def put(self, response_id: str, record: StoredResponse) -> None:
        """Store (or replace) a response, evicting expired/oldest entries."""
        if not record.created:
            record.created = self._clock()
        with self._lock:
            self._purge_expired_locked()
            self._entries[response_id] = record
            self._entries.move_to_end(response_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def get(self, response_id: str) -> StoredResponse | None:
        """Return a live record, or None when unknown/expired."""
        with self._lock:
            record = self._entries.get(response_id)
            if record is None:
                return None
            if self._is_expired(record):
                del self._entries[response_id]
                return None
            return record

    def delete(self, response_id: str) -> bool:
        """Drop one entry; True when something was removed."""
        with self._lock:
            return self._entries.pop(response_id, None) is not None

    def purge_expired(self) -> int:
        """Drop every expired entry; returns how many were removed."""
        with self._lock:
            return self._purge_expired_locked()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._entries)

    def __contains__(self, response_id: object) -> bool:
        return isinstance(response_id, str) and self.get(response_id) is not None

    def _is_expired(self, record: StoredResponse) -> bool:
        return (self._clock() - record.created) > self._ttl_seconds

    def _purge_expired_locked(self) -> int:
        expired = [key for key, record in self._entries.items() if self._is_expired(record)]
        for key in expired:
            del self._entries[key]
        return len(expired)
