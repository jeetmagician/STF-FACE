"""Ephemeral in-memory session store.

Exists for exactly one reason: when several faces are found in an upload, the
user has to choose one, and re-uploading the photographs to express that choice
is wasteful. Entries live in process memory only.

Properties that matter for the privacy claim:
  * nothing is written to disk
  * entries expire after SESSION_TTL_SECONDS (default 5 minutes)
  * expired entries are purged on every access and by a background sweep
  * tokens are 256-bit random values, so entries are not enumerable
  * the store is capped, and the oldest entries are evicted when it is full
  * RETAIN_FOR_FACE_SELECTION=false disables it entirely, at the cost of
    requiring re-upload after face selection
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings


@dataclass
class SessionEntry:
    old_images: list[np.ndarray]
    new_images: list[np.ndarray]
    created_at: float = field(default_factory=time.monotonic)


class EphemeralStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._entries: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def _purge_locked(self) -> None:
        cutoff = time.monotonic() - self._settings.session_ttl_seconds
        expired = [k for k, v in self._entries.items() if v.created_at < cutoff]
        for key in expired:
            self._drop_locked(key)

    def _drop_locked(self, token: str) -> None:
        entry = self._entries.pop(token, None)
        if entry is None:
            return
        # Overwrite the pixel buffers before releasing the reference. This does
        # not defeat a determined memory forensic attack, but it shortens the
        # window in which recognisable imagery sits in the heap.
        for image in (*entry.old_images, *entry.new_images):
            try:
                image.fill(0)
            except (ValueError, AttributeError):  # pragma: no cover - defensive
                pass
        entry.old_images.clear()
        entry.new_images.clear()

    def put(self, old_images: list[np.ndarray], new_images: list[np.ndarray]) -> str:
        if not self._settings.retain_for_face_selection:
            return ""

        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge_locked()
            while len(self._entries) >= self._settings.session_max_entries:
                oldest = min(self._entries, key=lambda k: self._entries[k].created_at)
                self._drop_locked(oldest)
            self._entries[token] = SessionEntry(
                old_images=[img.copy() for img in old_images],
                new_images=[img.copy() for img in new_images],
            )
        return token

    def get(self, token: str) -> SessionEntry | None:
        if not token:
            return None
        with self._lock:
            self._purge_locked()
            return self._entries.get(token)

    def drop(self, token: str) -> None:
        with self._lock:
            self._drop_locked(token)

    def purge_expired(self) -> int:
        with self._lock:
            before = len(self._entries)
            self._purge_locked()
            return before - len(self._entries)

    def clear(self) -> None:
        with self._lock:
            for token in list(self._entries):
                self._drop_locked(token)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)
