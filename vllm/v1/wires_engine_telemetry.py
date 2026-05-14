# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wires-fork engine telemetry emission helpers (T-45.4 ~ T-45.7).

Thin layer over ``vllm.v1.wires_telemetry`` that owns the per-stream
schemas described in ``docs/v2/45_telemetry_proposal.md`` §3. Each
emitter:

* No-ops in disabled mode (single ``is_enabled()`` attribute read; no
  dict construction, no allocation).
* In enabled mode, builds the row dict and hands it to the cached
  background writer (``deque.append`` under one mutex; serialisation
  + IO happen on the writer thread).
* Catches and swallows any exception so telemetry NEVER breaks
  serving — the worst-case telemetry bug is a silent missing row,
  never a crashed engine.

Public surface:

* ``emit_request_lifecycle(...)``   — E1, ``requests.jsonl``
* ``emit_action(...)``              — E2, ``actions.jsonl``
* ``emit_eviction(...)``            — E3, ``evictions.jsonl``
* ``emit_segment_touch(...)``       — E4, ``segment_touches.jsonl``
* ``emit_cache_stats(...)``         — E5, ``engine_cache_stats.jsonl``
* Constants: ``WRITER_KIND_*`` for the ``KVCacheBlock._last_writer_kind``
  enum the E4 cache_source decision walks over.

The split between this module and ``wires_telemetry`` is deliberate:
``wires_telemetry`` owns the *transport* (queue, drain thread, file
rotation, overflow accounting) and is reusable by any future engine
stream; this module owns the *shape* of each row and the call sites
that produce them.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

from vllm.v1.wires_telemetry import get_writer, is_enabled

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

# ---------------------------------------------------------------------------
# Stream names. These map 1:1 to ``engine_telemetry/<name>.jsonl``.
# ---------------------------------------------------------------------------

STREAM_REQUESTS = "requests"
STREAM_ACTIONS = "actions"
STREAM_EVICTIONS = "evictions"
STREAM_SEGMENT_TOUCHES = "segment_touches"
# M5(a): periodic engine-side cumulative cache-stats snapshot. Driven from
# the scheduler tick on a coarse interval so the JSONL stays low-volume
# (one row per second by default, gated by ``WIRES_TELEMETRY_ENABLED``).
STREAM_ENGINE_CACHE_STATS = "engine_cache_stats"

# ---------------------------------------------------------------------------
# Block writer-kind enum. Tagged onto KVCacheBlock at fill / re-fill time
# so E4 can answer "who last wrote this block" without a side table.
# ---------------------------------------------------------------------------

WRITER_KIND_CHAT_COMPLETION = "chat_completion"
WRITER_KIND_SEGMENT_PREPARE = "segment_prepare"
WRITER_KIND_SEGMENT_REFRESH = "segment_refresh"
WRITER_KIND_PREFIX_PREPARE = "prefix_prepare"

# ---------------------------------------------------------------------------
# E4 cache_source enum.
# ---------------------------------------------------------------------------

CACHE_SOURCE_COLD_PREFILL = "cold_prefill"
CACHE_SOURCE_PRE_PREPARED = "pre_prepared"
CACHE_SOURCE_PEER_CACHED = "peer_cached"
CACHE_SOURCE_SELF_HIT = "self_hit"

# pre_prepared freshness window: a block whose last writer was a
# segment_prepare / prefix_prepare action within the last
# ``WIRES_TELEMETRY_PRE_PREPARED_WINDOW_S`` seconds counts as
# pre_prepared; outside the window it falls back to peer_cached
# (the prepare write happened so long ago it might as well be
# any other peer's write).
DEFAULT_PRE_PREPARED_WINDOW_S = 60.0


def _resolve_pre_prepared_window_s() -> float:
    raw = os.environ.get("WIRES_TELEMETRY_PRE_PREPARED_WINDOW_S")
    if raw is None:
        return DEFAULT_PRE_PREPARED_WINDOW_S
    try:
        v = float(raw)
    except ValueError:
        return DEFAULT_PRE_PREPARED_WINDOW_S
    return max(0.0, v)


# Resolved once at import. Tests mutate via ``_set_pre_prepared_window_for_tests``.
_PRE_PREPARED_WINDOW_S: float = _resolve_pre_prepared_window_s()


def _set_pre_prepared_window_for_tests(value: float) -> None:
    """Tests-only hook to override the freshness window."""
    global _PRE_PREPARED_WINDOW_S
    _PRE_PREPARED_WINDOW_S = float(value)


# ---------------------------------------------------------------------------
# E1 — per-request lifecycle.
# ---------------------------------------------------------------------------


def emit_request_lifecycle(
    *,
    vllm_request_id: str,
    ts_arrived: float,
    ts_scheduled: float | None,
    ts_first_token: float | None,
    ts_finished: float,
    prompt_token_count: int,
    cached_token_count: int,
    output_token_count: int,
    blocks_allocated: int,
    blocks_cache_hit: int,
    evicts_caused: int,
    status: str,
) -> None:
    """Emit one E1 row when a request finishes (proposal §3.1).

    Hot path: scheduler ``_free_request``. Disabled-mode cost is a
    single ``is_enabled()`` attribute read.
    """
    if not is_enabled():
        return
    # Telemetry must never break serving; swallow any error in the
    # writer/queue path. ``contextlib.suppress`` keeps the hot path
    # short — no traceback construction on the success path.
    with contextlib.suppress(Exception):
        get_writer(STREAM_REQUESTS).append(
            {
                "ts_epoch": ts_finished,
                "vllm_request_id": vllm_request_id,
                "ts_arrived": ts_arrived,
                "ts_scheduled": ts_scheduled,
                "ts_first_token": ts_first_token,
                "ts_finished": ts_finished,
                "prompt_token_count": prompt_token_count,
                "cached_token_count": cached_token_count,
                "output_token_count": output_token_count,
                "blocks_allocated": blocks_allocated,
                "blocks_cache_hit": blocks_cache_hit,
                "evicts_caused": evicts_caused,
                "status": status,
            }
        )


# ---------------------------------------------------------------------------
# E2 — per-action handler.
# ---------------------------------------------------------------------------


# Outcome enum for ``emit_action``. Kept as constants so call sites
# don't drift into typos that silently break offline grouping.
ACTION_OUTCOME_PREPARED = "prepared"
ACTION_OUTCOME_NOOP_ALREADY_PRESENT = "noop_already_present"
ACTION_OUTCOME_REJECTED_DEDUP = "rejected_dedup"
ACTION_OUTCOME_REJECTED_BAD_INPUT = "rejected_bad_input"

ENDPOINT_SEGMENT_PREPARE = "segment_prepare"
ENDPOINT_SEGMENT_REFRESH = "segment_refresh"


def emit_action(
    *,
    endpoint: str,
    scope_key: str | None,
    blocks_touched: int,
    blocks_already_present: int,
    blocks_newly_written: int,
    outcome: str,
    elapsed_ms: float,
    ts_epoch: float | None = None,
) -> None:
    """Emit one E2 row when a coopt action handler exits (proposal §3.2).

    E2 is low-frequency; the schema-build cost is acceptable on the
    handler thread. ``ts_epoch`` falls back to ``time.time()`` so the
    caller doesn't have to plumb it for what amounts to a cold path.
    """
    if not is_enabled():
        return
    with contextlib.suppress(Exception):
        get_writer(STREAM_ACTIONS).append(
            {
                "ts_epoch": ts_epoch if ts_epoch is not None else time.time(),
                "endpoint": endpoint,
                "scope_key": scope_key,
                "blocks_touched": blocks_touched,
                "blocks_already_present": blocks_already_present,
                "blocks_newly_written": blocks_newly_written,
                "outcome": outcome,
                "elapsed_ms": elapsed_ms,
            }
        )


# ---------------------------------------------------------------------------
# E3 — per-evict.
# ---------------------------------------------------------------------------


EVICT_REASON_TTL_EXPIRED = "ttl_expired"
EVICT_REASON_LRU_PRESSURE = "lru_pressure"
EVICT_REASON_MUST_PRESSURE_BACKSTOP = "must_pressure_backstop"
EVICT_REASON_EXPLICIT_DEMOTE = "explicit_demote"

EVICT_POOL_MUST = "must"
EVICT_POOL_MAY = "may"
EVICT_POOL_NO = "no"


def emit_eviction(
    *,
    block_id: int,
    scope_key: str | None,
    pool_at_evict: str,
    source_class: str,
    hint: str,
    reason: str,
    ttl_at_demote_ms: float | None,
    must_hit_count: int,
    ref_count_at_evict: int,
    ts_epoch: float | None = None,
) -> None:
    """Emit one E3 row per block eviction (proposal §3.3).

    ``must_pool_evicted`` is derived from ``pool_at_evict``: a row
    with ``pool_at_evict=="must"`` IS the paper-§3 invariant
    violation. The field is on the row so it remains grep-detectable
    after offline aggregation collapses the pool field.
    """
    if not is_enabled():
        return
    with contextlib.suppress(Exception):
        get_writer(STREAM_EVICTIONS).append(
            {
                "ts_epoch": ts_epoch if ts_epoch is not None else time.time(),
                "block_id": block_id,
                "scope_key": scope_key,
                "pool_at_evict": pool_at_evict,
                "source_class": source_class,
                "hint": hint,
                "reason": reason,
                "ttl_at_demote_ms": ttl_at_demote_ms,
                "must_hit_count": must_hit_count,
                "ref_count_at_evict": ref_count_at_evict,
                "must_pool_evicted": pool_at_evict == EVICT_POOL_MUST,
            }
        )


# ---------------------------------------------------------------------------
# E4 — per-segment touch.
# ---------------------------------------------------------------------------


def decide_cache_source(
    blocks: list[KVCacheBlock],
    requesting_vllm_request_id: str | None,
    *,
    now_epoch: float | None = None,
) -> str:
    """Compute the E4 ``cache_source`` enum for one segment.

    Walks the block list, tallies a writer-kind/recency signal per
    block, and returns the dominant source by block count. Designed
    to be cheap enough to call in the allocate path:

    * No syscalls when ``now_epoch`` is supplied (allocate already
      knows the wall clock from the schedule tick).
    * One pass over ``blocks``; no allocations beyond the per-source
      counter dict.
    * Pure function over its inputs; trivially testable without a
      KVCacheManager.

    See proposal §3.4 cache_source enum semantics.
    """
    if not blocks:
        return CACHE_SOURCE_COLD_PREFILL
    now = now_epoch if now_epoch is not None else time.time()
    window = _PRE_PREPARED_WINDOW_S
    cold = 0
    pre_prepared = 0
    peer_cached = 0
    self_hit = 0
    for blk in blocks:
        kind = getattr(blk, "_last_writer_kind", None)
        if kind is None:
            cold += 1
            continue
        last_writer_id = getattr(blk, "_last_writer_request_id", None)
        last_writer_ts = getattr(blk, "_last_writer_ts", None) or 0.0
        if (
            kind in (WRITER_KIND_SEGMENT_PREPARE, WRITER_KIND_PREFIX_PREPARE)
            and (now - last_writer_ts) <= window
        ):
            pre_prepared += 1
        elif (
            requesting_vllm_request_id is not None
            and last_writer_id == requesting_vllm_request_id
        ):
            self_hit += 1
        else:
            peer_cached += 1
    # Dominant source by block count; ties broken by the priority
    # order self_hit > pre_prepared > peer_cached > cold (most
    # informative wins). Ties at zero default to cold_prefill,
    # which is the right answer for empty / freshly-allocated
    # segments.
    counts = {
        CACHE_SOURCE_SELF_HIT: self_hit,
        CACHE_SOURCE_PRE_PREPARED: pre_prepared,
        CACHE_SOURCE_PEER_CACHED: peer_cached,
        CACHE_SOURCE_COLD_PREFILL: cold,
    }
    # Pick the max-count entry; the dict insertion order above
    # encodes the tie-break priority.
    best_name = CACHE_SOURCE_COLD_PREFILL
    best_count = -1
    for name, count in counts.items():
        if count > best_count:
            best_count = count
            best_name = name
    return best_name


def emit_segment_touch(
    *,
    vllm_request_id: str | None,
    scope_key: str,
    segment_index: int,
    block_count: int,
    cache_source: str,
    lifecycle_hint: str,
    source_class: str,
    ts_epoch: float | None = None,
) -> None:
    """Emit one E4 row per segment touched at allocate time (proposal §3.4)."""
    if not is_enabled():
        return
    with contextlib.suppress(Exception):
        get_writer(STREAM_SEGMENT_TOUCHES).append(
            {
                "ts_epoch": ts_epoch if ts_epoch is not None else time.time(),
                "vllm_request_id": vllm_request_id,
                "scope_key": scope_key,
                "segment_index": segment_index,
                "block_count": block_count,
                "cache_source": cache_source,
                "lifecycle_hint": lifecycle_hint,
                "source_class": source_class,
            }
        )


# ---------------------------------------------------------------------------
# E5 — periodic engine cache-stats snapshot (M5(a)).
# ---------------------------------------------------------------------------
#
# Exports the cumulative counters tracked on ``FreeKVCacheBlockQueue``
# (must_pool_evicted, ttl_demoted, access_promoted, EMA sample tallies,
# speculation_*, lazy_flush_total_blocks, LRU walk-depth histogram) into
# a low-volume JSONL stream so offline analysis can plot them over wall
# clock without scraping engine logs.
#
# Driven from the scheduler tick (``Scheduler.schedule()``) on a coarse
# interval (``WIRES_TELEMETRY_CACHE_STATS_INTERVAL_S``, default 1.0).
# Disabled-mode cost is one ``is_enabled()`` attribute read; enabled-mode
# cost is one ``time.monotonic()`` + a small dict construction once per
# interval.

DEFAULT_CACHE_STATS_INTERVAL_S = 1.0


def _resolve_cache_stats_interval_s() -> float:
    raw = os.environ.get("WIRES_TELEMETRY_CACHE_STATS_INTERVAL_S")
    if raw is None:
        return DEFAULT_CACHE_STATS_INTERVAL_S
    try:
        v = float(raw)
    except ValueError:
        return DEFAULT_CACHE_STATS_INTERVAL_S
    # Guard a 0/negative value (would emit every tick); clamp to a small
    # positive minimum so the stream stays bounded even if a bad value
    # leaks in from env.
    return max(0.05, v)


_CACHE_STATS_INTERVAL_S: float = _resolve_cache_stats_interval_s()


def _set_cache_stats_interval_for_tests(value: float) -> None:
    """Tests-only hook to override the periodic emit interval."""
    global _CACHE_STATS_INTERVAL_S
    _CACHE_STATS_INTERVAL_S = float(value)


def get_cache_stats_interval_s() -> float:
    """Public accessor for the resolved periodic emit interval (seconds)."""
    return _CACHE_STATS_INTERVAL_S


def emit_cache_stats(
    snapshot: dict[str, Any],
    *,
    ts_epoch: float | None = None,
) -> None:
    """Emit one E5 row carrying the engine cache-stats snapshot
    (CODE_MISMATCH_NOTES.md M5(a)).

    ``snapshot`` is the dict returned by
    ``FreeKVCacheBlockQueue.cache_stats_snapshot()``; this helper just
    layers a wall-clock timestamp on top and hands the row to the
    background writer. Caller is responsible for cadence (typically the
    scheduler tick honoring ``WIRES_TELEMETRY_CACHE_STATS_INTERVAL_S``).
    """
    if not is_enabled():
        return
    with contextlib.suppress(Exception):
        row = {"ts_epoch": ts_epoch if ts_epoch is not None else time.time()}
        row.update(snapshot)
        get_writer(STREAM_ENGINE_CACHE_STATS).append(row)


# ---------------------------------------------------------------------------
# KVCacheBlock writer-tag helper.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-request evict attribution (T-45.4 helper).
# ---------------------------------------------------------------------------
#
# Engine core is single-threaded; ``allocate_slots`` calls into the
# block pool which may trigger evictions via ``popleft_n``. We need to
# attribute those evictions back to the request that caused them
# without rewiring the call chain. Module-level int is the cheapest
# correct answer (zero allocation, one pointer compare to skip when
# unset; the engine is single-threaded inside one core process).

_current_alloc_request: Any | None = None


def begin_request_alloc(request: Any) -> None:
    """Mark ``request`` as the request whose allocate path is on the
    stack. Subsequent ``note_evict_for_current_request()`` calls will
    bump ``request.num_evicts_caused_total``. Idempotent: nesting is
    not supported (the engine never re-enters allocate_slots).
    """
    global _current_alloc_request
    _current_alloc_request = request


def end_request_alloc() -> None:
    """Clear the current-allocate-request slot. Called at allocate exit."""
    global _current_alloc_request
    _current_alloc_request = None


def note_evict_for_current_request(count: int = 1) -> None:
    """Bump the current-allocate request's evict counter, if any.

    Hot path on the eviction codepath: zero cost when no allocate is
    on the stack (single attribute read + branch). Tolerates legacy
    request types that don't carry the counter via a ``getattr``
    pre-check.
    """
    req = _current_alloc_request
    if req is None:
        return
    with contextlib.suppress(AttributeError):
        req.num_evicts_caused_total += count


def stamp_block_writer(
    block: Any,
    *,
    kind: str,
    ts_epoch: float,
    request_id: str | None,
) -> None:
    """Stamp the per-block writer fields on ``block`` (T-45.7 prep).

    Tolerates the block lacking the slot attributes (for tests using
    a stand-in object): we ``setattr`` directly so attribute-only
    objects also work. KVCacheBlock declares the three fields with
    safe defaults so the assignment is a single store.
    """
    with contextlib.suppress(AttributeError):
        block._last_writer_kind = kind
        block._last_writer_ts = ts_epoch
        block._last_writer_request_id = request_id
