# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wires-fork engine telemetry writer infrastructure (T-45.0).

Background-thread JSONL writer used by the four wires engine push
streams (E1 requests, E2 actions, E3 evictions, E4 segment_touches)
described in ``docs/v2/45_telemetry_proposal.md``.

Design goals (see proposal §5.2 hard rule):

* Default off. ``WIRES_TELEMETRY_ENABLED=1`` opts in. When off, the
  hot-path cost is one module-global attribute read plus a no-op
  ``append`` call (no dict lookup, no allocation, no IO).
* When on, ``append(row_dict)`` does exactly: lock acquire, deque
  append, lock release, return. No ``json.dumps``, no ``time.time()``,
  no file IO on the caller's thread.
* A daemon thread per stream drains the deque every 100 ms or when
  it crosses ~1 MB worth of buffered rows, whichever first. JSONL
  serialisation and file IO happen there.
* Bounded queue at 100k rows. Overflow drops OLDEST and bumps
  ``telemetry_dropped_total``; first drop and every 1000th drop log a
  warning to the vllm logger. Engine never blocks on a full queue.
* File rotation at 50 MB, up to 5 rolls (``.jsonl``, ``.jsonl.1``,
  ..., ``.jsonl.5``); oldest discarded. Same shape as
  ``logging.handlers.RotatingFileHandler`` but JSONL-aware.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

try:
    # The standard logger path for the wires fork. When running this
    # module under the production engine, ``vllm.logger`` is always
    # importable.
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # pragma: no cover - exercised only in stdlib-only test envs
    # Fallback so the module can be unit-tested without the full vLLM
    # runtime (e.g. CI lanes that have not installed torch). Functionally
    # identical for our purposes: a logger that emits to stderr.
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Module-level switches and constants. Resolved once at import; checked on
# the hot path via ``is_enabled()`` which compiles to one attribute read.
# ---------------------------------------------------------------------------

_ENABLED: bool = os.environ.get("WIRES_TELEMETRY_ENABLED") == "1"

# Hot-path queue cap; overflow drops oldest. Module-level so tests can
# monkeypatch per-instance via the ``max_queue`` constructor arg without
# touching this default.
DEFAULT_MAX_QUEUE: int = 100_000

# Background drain cadence (seconds) and approximate flush threshold (bytes).
DEFAULT_DRAIN_INTERVAL_S: float = 0.1
DEFAULT_FLUSH_BYTES: int = 1 * 1024 * 1024  # 1 MB

# File rotation: rotate when current file passes this many bytes; keep up
# to ``DEFAULT_ROTATE_KEEP`` historical rolls.
DEFAULT_ROTATE_BYTES: int = 50 * 1024 * 1024  # 50 MB
DEFAULT_ROTATE_KEEP: int = 5

# Approximate average row size used for the cheap byte estimate. Keeps the
# hot path free of per-row sizing work; the writer thread tightens the
# estimate by sampling 1-in-64 rows.
APPROX_ROW_BYTES: int = 300

# Module-level dropped-row counter. Shared across all writers so callers can
# poll a single number for sanity checks. Mutated only under each writer's
# own lock so a "telemetry_dropped_total == 0" assertion across all streams
# is meaningful even without atomic primitives.
telemetry_dropped_total: int = 0


def is_enabled() -> bool:
    """Return whether engine telemetry writers are active.

    Compiles to a single module-attribute read; safe to call from hot paths.
    """
    return _ENABLED


# ---------------------------------------------------------------------------
# Telemetry directory resolution.
# ---------------------------------------------------------------------------


def _resolve_telemetry_dir() -> Path:
    """Resolve the directory engine telemetry streams write into.

    Resolution order:

    1. ``WIRES_TELEMETRY_DIR`` env var (used as-is).
    2. ``$WIRES_SWEEP_OUT_DIR/engine_telemetry`` if the sweep var is set.
    3. ``/tmp/wires_telemetry`` with a warning, last-resort fallback.

    The chosen directory is created (parents included) if missing.
    """
    raw = os.environ.get("WIRES_TELEMETRY_DIR")
    if raw:
        path = Path(raw)
    else:
        sweep = os.environ.get("WIRES_SWEEP_OUT_DIR")
        if sweep:
            path = Path(sweep) / "engine_telemetry"
        else:
            path = Path("/tmp/wires_telemetry")
            logger.warning(
                "WIRES_TELEMETRY_DIR and WIRES_SWEEP_OUT_DIR both unset; "
                "engine telemetry falling back to %s",
                path,
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Background writer.
# ---------------------------------------------------------------------------


class BackgroundTelemetryWriter:
    """Daemon-thread JSONL writer for one stream.

    One instance per stream name (``requests``, ``actions``, ``evictions``,
    ``segment_touches``). Hot-path callers invoke ``append(row_dict)``;
    serialisation and file IO run on the writer thread.
    """

    def __init__(
        self,
        stream_name: str,
        directory: Path,
        *,
        max_queue: int = DEFAULT_MAX_QUEUE,
        drain_interval_s: float = DEFAULT_DRAIN_INTERVAL_S,
        flush_bytes: int = DEFAULT_FLUSH_BYTES,
        rotate_bytes: int = DEFAULT_ROTATE_BYTES,
        rotate_keep: int = DEFAULT_ROTATE_KEEP,
    ) -> None:
        self.stream_name = stream_name
        self.directory = directory
        self.path = directory / f"{stream_name}.jsonl"
        self._max_queue = max_queue
        self._drain_interval_s = drain_interval_s
        self._flush_bytes = flush_bytes
        self._rotate_bytes = rotate_bytes
        self._rotate_keep = rotate_keep

        # Single lock guards both the deque AND ``_dropped`` /
        # ``_buffered_bytes`` so the hot path takes exactly one mutex.
        self._lock = threading.Lock()
        self._queue: deque[dict[str, Any]] = deque()
        self._dropped: int = 0  # local count for warning cadence
        self._buffered_bytes: int = 0  # cheap running estimate
        self._sample_counter: int = 0

        # Separate write-side mutex so ``flush()`` can wait on a drain that
        # has already started (i.e. queue is empty under ``_lock`` but the
        # writer thread is mid-``json.dumps``). Cold-side only; never
        # touched by the hot path.
        self._write_lock = threading.Lock()

        # Wakes the writer when ``flush_bytes`` is crossed, and on shutdown.
        self._wake = threading.Event()
        self._shutdown = threading.Event()

        # Track the current file size to drive rotation without statting on
        # every write.
        self._current_size: int = self.path.stat().st_size if self.path.exists() else 0

        self._thread = threading.Thread(
            target=self._run,
            name=f"wires-telemetry-{stream_name}",
            daemon=True,
        )
        self._thread.start()

    # -- hot path ---------------------------------------------------------

    def append(self, row: dict[str, Any]) -> None:
        """Enqueue a row for background serialisation.

        Hot-path cost target: one lock acquire, one ``deque.append``, one
        lock release. No IO, no string formatting, no allocation beyond
        the row dict itself (provided by the caller).
        """
        global telemetry_dropped_total
        wake = False
        with self._lock:
            if len(self._queue) >= self._max_queue:
                # Overflow: drop oldest, count, log on cadence.
                try:
                    dropped_row = self._queue.popleft()
                except IndexError:
                    dropped_row = None
                if dropped_row is not None:
                    self._buffered_bytes = max(
                        0, self._buffered_bytes - APPROX_ROW_BYTES
                    )
                self._dropped += 1
                telemetry_dropped_total += 1
                if self._dropped == 1 or self._dropped % 1000 == 0:
                    # Log inside the lock to keep the warning ordered with
                    # the actual drop; warning is rare and short.
                    logger.warning(
                        "wires telemetry stream %s dropped row "
                        "(local_dropped=%d, total=%d, cap=%d)",
                        self.stream_name,
                        self._dropped,
                        telemetry_dropped_total,
                        self._max_queue,
                    )
            self._queue.append(row)
            self._buffered_bytes += APPROX_ROW_BYTES
            if self._buffered_bytes >= self._flush_bytes:
                wake = True
        if wake:
            self._wake.set()

    # -- public utilities -------------------------------------------------

    def dropped(self) -> int:
        """Return the number of rows this writer dropped due to overflow."""
        with self._lock:
            return self._dropped

    def queue_depth(self) -> int:
        """Return the current in-memory queue depth (debugging only)."""
        with self._lock:
            return len(self._queue)

    def flush(self, timeout_s: float = 5.0) -> None:
        """Drain the in-memory queue to disk synchronously.

        Used by ``flush_all`` at engine shutdown and by tests. Wait for
        any in-progress writer-thread drain to finish, then drain anything
        still in the queue ourselves so we have a hard "everything on
        disk" guarantee on return. Best-effort: bounded by ``timeout_s``.
        """
        if not self._thread.is_alive():
            # Thread is gone; do the whole drain ourselves.
            with self._write_lock:
                self._drain_inline()
            return
        deadline = time.monotonic() + timeout_s
        # Loop because between the writer thread releasing ``_write_lock``
        # and us reacquiring it, the hot path could enqueue more rows.
        while True:
            # Wait for any in-flight drain to finish, then take over.
            remaining = max(0.0, deadline - time.monotonic())
            acquired = self._write_lock.acquire(timeout=remaining)
            if not acquired:
                logger.warning(
                    "wires telemetry stream %s flush timed out waiting on "
                    "writer thread; queue_depth=%d",
                    self.stream_name,
                    self.queue_depth(),
                )
                return
            try:
                self._drain_inline_locked()
            finally:
                self._write_lock.release()
            with self._lock:
                if not self._queue:
                    return
            if time.monotonic() >= deadline:
                logger.warning(
                    "wires telemetry stream %s flush timed out with "
                    "%d rows still queued",
                    self.stream_name,
                    self.queue_depth(),
                )
                return

    def shutdown(self, timeout_s: float = 1.0) -> None:
        """Signal the writer thread to drain + exit, and join it.

        ``timeout_s`` bounds the join to avoid hanging shutdown if the
        writer thread is wedged on a slow filesystem.
        """
        self._shutdown.set()
        self._wake.set()
        self._thread.join(timeout=timeout_s)

    # -- writer thread ----------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._shutdown.is_set():
                self._wake.wait(timeout=self._drain_interval_s)
                self._wake.clear()
                self._drain_inline()
            # Final drain after shutdown signal.
            self._drain_inline()
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "wires telemetry writer thread for %s crashed",
                self.stream_name,
            )

    def _drain_inline(self) -> None:
        """Pull all queued rows under the write lock + the queue lock."""
        with self._write_lock:
            self._drain_inline_locked()

    def _drain_inline_locked(self) -> None:
        """Drain caller MUST already hold ``self._write_lock``."""
        with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
            self._buffered_bytes = 0

        # Serialise outside the lock so hot-path callers never wait on IO.
        try:
            lines = []
            sampled_total = 0
            sampled_count = 0
            for row in batch:
                line = json.dumps(row, separators=(",", ":"), default=str)
                lines.append(line)
                # Sample 1-in-64 to keep our row-size estimate honest
                # without paying per-row len cost.
                self._sample_counter += 1
                if self._sample_counter % 64 == 0:
                    sampled_total += len(line) + 1
                    sampled_count += 1
            if sampled_count:
                # Hint: future overflow accounting could use this, but for
                # now we keep ``APPROX_ROW_BYTES`` constant. Logging at
                # debug avoids leaking into normal sweeps.
                avg = sampled_total / sampled_count
                logger.debug(
                    "wires telemetry stream %s observed avg row %d B",
                    self.stream_name,
                    int(avg),
                )

            payload = ("\n".join(lines) + "\n").encode("utf-8")
            self._write_payload(payload)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "wires telemetry stream %s failed to flush %d rows",
                self.stream_name,
                len(batch),
            )

    def _write_payload(self, payload: bytes) -> None:
        # Open in append + binary so we control encoding and avoid the text
        # mode line-buffering surprise.
        with open(self.path, "ab") as fh:
            fh.write(payload)
            fh.flush()
        self._current_size += len(payload)
        # Rotate AFTER write: matches stdlib RotatingFileHandler semantics
        # and (importantly) handles the "single drain larger than rotate
        # threshold" case correctly. The new ``.jsonl`` will be empty
        # until the next drain.
        if self._rotate_bytes > 0 and self._current_size >= self._rotate_bytes:
            self._rotate()

    def _rotate(self) -> None:
        """Roll ``.jsonl`` -> ``.jsonl.1`` -> ... -> ``.jsonl.<keep>``."""
        if not self.path.exists():
            self._current_size = 0
            return
        # Drop the oldest if it exists.
        oldest = self.directory / f"{self.stream_name}.jsonl.{self._rotate_keep}"
        if oldest.exists():
            try:
                oldest.unlink()
            except OSError:
                logger.exception("wires telemetry rotation could not delete %s", oldest)
        # Shift .jsonl.k -> .jsonl.(k+1) for k = keep-1 .. 1.
        for i in range(self._rotate_keep - 1, 0, -1):
            src = self.directory / f"{self.stream_name}.jsonl.{i}"
            dst = self.directory / f"{self.stream_name}.jsonl.{i + 1}"
            if src.exists():
                try:
                    src.replace(dst)
                except OSError:
                    logger.exception(
                        "wires telemetry rotation could not move %s -> %s",
                        src,
                        dst,
                    )
        # Move the current .jsonl to .jsonl.1.
        rolled = self.directory / f"{self.stream_name}.jsonl.1"
        try:
            self.path.replace(rolled)
        except OSError:
            logger.exception("wires telemetry rotation could not rotate %s", self.path)
            return
        self._current_size = 0


# ---------------------------------------------------------------------------
# No-op writer for the disabled codepath.
# ---------------------------------------------------------------------------


class _NoopTelemetryWriter:
    """Stand-in returned when telemetry is disabled.

    Implements the hot-path surface so callers do not need an
    ``if writer is not None`` guard. Cost of ``append`` is one Python
    function call returning immediately.
    """

    __slots__ = ("stream_name",)

    def __init__(self, stream_name: str) -> None:
        self.stream_name = stream_name

    def append(self, row: dict[str, Any]) -> None:
        return None

    def dropped(self) -> int:
        return 0

    def queue_depth(self) -> int:
        return 0

    def flush(self, timeout_s: float = 5.0) -> None:
        return None

    def shutdown(self, timeout_s: float = 1.0) -> None:
        return None


# ---------------------------------------------------------------------------
# Writer registry.
# ---------------------------------------------------------------------------


_writers_lock = threading.Lock()
_writers: dict[str, BackgroundTelemetryWriter | _NoopTelemetryWriter] = {}
_resolved_dir: Path | None = None


def _get_resolved_dir() -> Path:
    """Cache the resolved telemetry directory across writer creations."""
    global _resolved_dir
    if _resolved_dir is None:
        _resolved_dir = _resolve_telemetry_dir()
    return _resolved_dir


def get_writer(
    stream_name: str,
) -> BackgroundTelemetryWriter | _NoopTelemetryWriter:
    """Return the cached writer for ``stream_name``.

    When telemetry is disabled, returns a shared no-op writer so the hot
    path cost is one attribute read + one no-op call. When enabled,
    returns a single ``BackgroundTelemetryWriter`` per stream name; the
    factory is idempotent across threads.
    """
    # Fast path: cache hit, no global lock churn beyond the dict read.
    cached = _writers.get(stream_name)
    if cached is not None:
        return cached
    with _writers_lock:
        cached = _writers.get(stream_name)
        if cached is not None:
            return cached
        if not _ENABLED:
            writer: BackgroundTelemetryWriter | _NoopTelemetryWriter = (
                _NoopTelemetryWriter(stream_name)
            )
        else:
            writer = BackgroundTelemetryWriter(stream_name, _get_resolved_dir())
        _writers[stream_name] = writer
        return writer


def flush_all(timeout_s: float = 5.0) -> None:
    """Best-effort drain of every active writer; called at engine shutdown."""
    with _writers_lock:
        writers = list(_writers.values())
    for w in writers:
        try:
            w.flush(timeout_s=timeout_s)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "wires telemetry flush_all failed for %s",
                getattr(w, "stream_name", "<unknown>"),
            )


# ---------------------------------------------------------------------------
# Test-only helpers.
# ---------------------------------------------------------------------------


def _reset_for_tests() -> None:
    """Tear down all writers and reset module state.

    Tests use this between cases to guarantee isolation; not part of the
    public API. Caller is responsible for setting / unsetting
    ``WIRES_TELEMETRY_ENABLED`` and re-importing if they want to flip the
    enabled state.
    """
    global telemetry_dropped_total, _resolved_dir
    with _writers_lock:
        writers = list(_writers.values())
        _writers.clear()
    for w in writers:
        with contextlib.suppress(Exception):
            w.shutdown(timeout_s=1.0)
    telemetry_dropped_total = 0
    _resolved_dir = None


def _set_enabled_for_tests(value: bool) -> None:
    """Flip ``_ENABLED`` without re-import; tests only."""
    global _ENABLED
    _ENABLED = value
