# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the wires-fork E3 evict + E4 segment-touch streams
(T-45.6 + T-45.7).

Self-contained: no GPU. We exercise the emitter shape contract
(``emit_eviction``, ``emit_segment_touch``) and the pure
``decide_cache_source`` decision function. The full
``KVCacheManager.allocate_slots`` integration is covered by the
existing v1/core test files; the unit-shape contracts are what this
file owns.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


def _load_module(qualname: str, relpath: str):
    try:
        return importlib.import_module(qualname)
    except Exception:
        module_path = Path(__file__).resolve().parents[2] / "vllm" / relpath
        spec = importlib.util.spec_from_file_location(qualname, module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        return module


wt = _load_module("vllm.v1.wires_telemetry", "v1/wires_telemetry.py")
wet = _load_module("vllm.v1.wires_engine_telemetry", "v1/wires_engine_telemetry.py")

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_module_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WIRES_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.delenv("WIRES_SWEEP_OUT_DIR", raising=False)
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)
    # Reset the pre_prepared window in case a prior test changed it.
    wet._set_pre_prepared_window_for_tests(wet.DEFAULT_PRE_PREPARED_WINDOW_S)
    yield
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)


# ---------------------------------------------------------------------------
# Test stand-in for KVCacheBlock; lets us exercise decide_cache_source
# without instantiating the slots-based class (which requires the v1
# core import chain).
# ---------------------------------------------------------------------------


class _FakeBlock:
    """Plain-attribute stand-in for KVCacheBlock for unit tests.

    Mirrors the four telemetry-relevant fields:
    ``_last_writer_kind`` / ``_last_writer_ts`` / ``_last_writer_request_id``
    / ``_segment_id``. Plain Python class so we can mutate freely
    without slots constraints.
    """

    def __init__(
        self,
        block_id: int = 0,
        *,
        kind: str | None = None,
        ts: float = 0.0,
        request_id: str | None = None,
        segment_id: str | None = None,
    ):
        self.block_id = block_id
        self._last_writer_kind = kind
        self._last_writer_ts = ts
        self._last_writer_request_id = request_id
        self._segment_id = segment_id


# ===========================================================================
# E3 (evictions) shape tests
# ===========================================================================


def test_e3_disabled_no_file(tmp_path: Path):
    wt._set_enabled_for_tests(False)
    for i in range(20):
        wet.emit_eviction(
            block_id=i,
            scope_key="seg-x",
            pool_at_evict="may",
            source_class="unstructured",
            hint="may",
            reason=wet.EVICT_REASON_LRU_PRESSURE,
            ttl_at_demote_ms=None,
            must_hit_count=0,
            ref_count_at_evict=0,
        )
    assert not (tmp_path / "evictions.jsonl").exists()


def test_e3_disabled_zero_cost():
    wt._set_enabled_for_tests(False)
    t0 = time.perf_counter()
    for _ in range(100_000):
        wet.emit_eviction(
            block_id=0,
            scope_key=None,
            pool_at_evict="may",
            source_class="unstructured",
            hint="may",
            reason=wet.EVICT_REASON_LRU_PRESSURE,
            ttl_at_demote_ms=None,
            must_hit_count=0,
            ref_count_at_evict=0,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"100k disabled E3 emits took {elapsed_ms:.2f} ms"


def test_e3_must_pool_evicted_is_grep_detectable(tmp_path: Path):
    """``must_pool_evicted=True`` rows are paper-§3 invariant
    violations — the field must be present, true, and easy to grep."""
    wt._set_enabled_for_tests(True)
    # One non-must (lru_pressure on may), one ttl_expired (pool="must"
    # so must_pool_evicted=True), one backstop (pool="must").
    fixtures = [
        dict(
            pool_at_evict="may",
            reason=wet.EVICT_REASON_LRU_PRESSURE,
            ttl=None,
            block_id=1,
        ),
        dict(
            pool_at_evict="must",
            reason=wet.EVICT_REASON_TTL_EXPIRED,
            ttl=5000.0,
            block_id=2,
        ),
        dict(
            pool_at_evict="must",
            reason=wet.EVICT_REASON_MUST_PRESSURE_BACKSTOP,
            ttl=300_000.0,
            block_id=3,
        ),
    ]
    for f in fixtures:
        wet.emit_eviction(
            block_id=f["block_id"],
            scope_key="system_prompt:v1",
            pool_at_evict=f["pool_at_evict"],
            source_class="structured",
            hint="must" if f["pool_at_evict"] == "must" else "may",
            reason=f["reason"],
            ttl_at_demote_ms=f["ttl"],
            must_hit_count=3,
            ref_count_at_evict=0,
        )
    wt.flush_all(timeout_s=5.0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "evictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 3
    # Every row carries the field; only the must rows are True.
    flags = [r["must_pool_evicted"] for r in rows]
    assert flags == [False, True, True]
    # Required keys present per §3.3.
    expected_keys = {
        "ts_epoch",
        "block_id",
        "scope_key",
        "pool_at_evict",
        "source_class",
        "hint",
        "reason",
        "ttl_at_demote_ms",
        "must_hit_count",
        "ref_count_at_evict",
        "must_pool_evicted",
    }
    for r in rows:
        assert set(r.keys()) == expected_keys


def test_e3_reason_enum_values(tmp_path: Path):
    """All four reason enum values must round-trip verbatim."""
    wt._set_enabled_for_tests(True)
    reasons = (
        wet.EVICT_REASON_TTL_EXPIRED,
        wet.EVICT_REASON_LRU_PRESSURE,
        wet.EVICT_REASON_MUST_PRESSURE_BACKSTOP,
        wet.EVICT_REASON_EXPLICIT_DEMOTE,
    )
    for i, reason in enumerate(reasons):
        wet.emit_eviction(
            block_id=i,
            scope_key=None,
            pool_at_evict="may",
            source_class="unstructured",
            hint="may",
            reason=reason,
            ttl_at_demote_ms=None,
            must_hit_count=0,
            ref_count_at_evict=0,
        )
    wt.flush_all(timeout_s=5.0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "evictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [r["reason"] for r in rows] == list(reasons)


# ===========================================================================
# E4 (segment_touches) shape tests
# ===========================================================================


def test_e4_disabled_no_file(tmp_path: Path):
    wt._set_enabled_for_tests(False)
    for i in range(20):
        wet.emit_segment_touch(
            vllm_request_id=f"chatcmpl-{i}",
            scope_key="seg-x",
            segment_index=0,
            block_count=4,
            cache_source=wet.CACHE_SOURCE_COLD_PREFILL,
            lifecycle_hint="may",
            source_class="unstructured",
        )
    assert not (tmp_path / "segment_touches.jsonl").exists()


def test_e4_disabled_zero_cost():
    wt._set_enabled_for_tests(False)
    t0 = time.perf_counter()
    for _ in range(100_000):
        wet.emit_segment_touch(
            vllm_request_id="chatcmpl",
            scope_key="seg",
            segment_index=0,
            block_count=1,
            cache_source=wet.CACHE_SOURCE_COLD_PREFILL,
            lifecycle_hint="may",
            source_class="unstructured",
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"100k disabled E4 emits took {elapsed_ms:.2f} ms"


def test_e4_shape(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    wet.emit_segment_touch(
        vllm_request_id="chatcmpl-9c1f",
        scope_key="system_prompt:reflexion:v1",
        segment_index=0,
        block_count=14,
        cache_source=wet.CACHE_SOURCE_PEER_CACHED,
        lifecycle_hint="must",
        source_class="structured",
        ts_epoch=1234.5,
    )
    wt.flush_all(timeout_s=5.0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "segment_touches.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    expected = {
        "ts_epoch",
        "vllm_request_id",
        "scope_key",
        "segment_index",
        "block_count",
        "cache_source",
        "lifecycle_hint",
        "source_class",
    }
    assert set(rows[0].keys()) == expected
    assert rows[0]["cache_source"] == "peer_cached"
    assert rows[0]["ts_epoch"] == 1234.5


# ===========================================================================
# decide_cache_source: the four cases + mixed-segment dominance
# ===========================================================================


def test_cache_source_cold_prefill():
    """All blocks have no writer → cold_prefill."""
    blocks = [_FakeBlock(i) for i in range(4)]
    assert (
        wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "cold_prefill"
    )


def test_cache_source_pre_prepared_recent():
    """All blocks last written by segment_prepare within window
    → pre_prepared."""
    blocks = [
        _FakeBlock(
            i,
            kind=wet.WRITER_KIND_SEGMENT_PREPARE,
            ts=100.0 - 5.0,  # 5 s ago, well within 60 s
            request_id="segment_prepare",
        )
        for i in range(4)
    ]
    assert (
        wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "pre_prepared"
    )


def test_cache_source_pre_prepared_window_expiry():
    """A segment_prepare write older than the freshness window is
    NOT pre_prepared anymore — it falls back to peer_cached."""
    blocks = [
        _FakeBlock(
            i,
            kind=wet.WRITER_KIND_SEGMENT_PREPARE,
            ts=100.0 - 1000.0,  # 1000 s ago, well past 60 s window
            request_id="segment_prepare",
        )
        for i in range(4)
    ]
    # request_id != "segment_prepare", and last_writer_request_id ==
    # "segment_prepare" sentinel != "chatcmpl-X" → peer_cached.
    assert (
        wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "peer_cached"
    )


def test_cache_source_self_hit():
    """All blocks last written by THIS chat completion → self_hit."""
    blocks = [
        _FakeBlock(
            i,
            kind=wet.WRITER_KIND_CHAT_COMPLETION,
            ts=100.0 - 5.0,
            request_id="chatcmpl-X",
        )
        for i in range(4)
    ]
    assert wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "self_hit"


def test_cache_source_peer_cached():
    """All blocks last written by a different chat completion."""
    blocks = [
        _FakeBlock(
            i,
            kind=wet.WRITER_KIND_CHAT_COMPLETION,
            ts=100.0 - 5.0,
            request_id="chatcmpl-PEER",
        )
        for i in range(4)
    ]
    assert (
        wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "peer_cached"
    )


def test_cache_source_mixed_dominant_wins():
    """Mixed-segment: 3 cold, 5 peer_cached → peer_cached (dominant)."""
    blocks = [_FakeBlock(i) for i in range(3)]  # 3 cold
    blocks += [
        _FakeBlock(
            i + 3,
            kind=wet.WRITER_KIND_CHAT_COMPLETION,
            ts=100.0 - 1.0,
            request_id="chatcmpl-PEER",
        )
        for i in range(5)
    ]
    assert (
        wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "peer_cached"
    )


def test_cache_source_mixed_tie_priority():
    """Tie: 2 self_hit + 2 peer_cached → self_hit wins
    (priority-tiebreak in the proposal: most informative bucket)."""
    blocks = [
        _FakeBlock(
            i,
            kind=wet.WRITER_KIND_CHAT_COMPLETION,
            ts=100.0 - 1.0,
            request_id="chatcmpl-X",
        )
        for i in range(2)
    ]
    blocks += [
        _FakeBlock(
            i + 2,
            kind=wet.WRITER_KIND_CHAT_COMPLETION,
            ts=100.0 - 1.0,
            request_id="chatcmpl-PEER",
        )
        for i in range(2)
    ]
    assert wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "self_hit"


def test_cache_source_window_env_override(monkeypatch: pytest.MonkeyPatch):
    """``WIRES_TELEMETRY_PRE_PREPARED_WINDOW_S`` is read at module
    import; once installed via the test hook the helper picks up
    the new value immediately."""
    wet._set_pre_prepared_window_for_tests(1.0)
    blocks = [
        _FakeBlock(
            i,
            kind=wet.WRITER_KIND_SEGMENT_PREPARE,
            ts=100.0 - 5.0,
            request_id="segment_prepare",
        )
        for i in range(4)
    ]
    # 5 s ago > 1 s window → no longer pre_prepared.
    assert (
        wet.decide_cache_source(blocks, "chatcmpl-X", now_epoch=100.0) == "peer_cached"
    )


# ===========================================================================
# stamp_block_writer + KVCacheBlock fields
# ===========================================================================


def test_stamp_block_writer_round_trips():
    """stamp_block_writer must populate the three fields atomically."""
    blk = _FakeBlock(7)
    wet.stamp_block_writer(
        blk,
        kind=wet.WRITER_KIND_SEGMENT_PREPARE,
        ts_epoch=1234.5,
        request_id="segment_prepare",
    )
    assert blk._last_writer_kind == "segment_prepare"
    assert blk._last_writer_ts == 1234.5
    assert blk._last_writer_request_id == "segment_prepare"


def test_kvcacheblock_field_defaults():
    """KVCacheBlock declares the three writer fields with safe
    defaults so non-WIRES blocks pay no construction cost beyond
    the slot — and the E4 decision sees them as ``None`` /
    ``cold_prefill``.

    Loaded via direct file-spec to dodge the torch import chain.
    """
    try:
        kcu = importlib.import_module("vllm.v1.core.kv_cache_utils")
    except Exception:
        pytest.skip("kv_cache_utils not importable in this env")
    blk = kcu.KVCacheBlock(block_id=0)
    assert blk._last_writer_kind is None
    assert blk._last_writer_ts == 0.0
    assert blk._last_writer_request_id is None
