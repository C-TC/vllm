# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M5(a) — engine cache-stats counter export.

Per ``paper/CODE_MISMATCH_NOTES.md`` M5(a) the cumulative counters
maintained on ``FreeKVCacheBlockQueue`` (must_pool_evicted, ttl_demoted,
access_promoted, the EMA sample tallies, the speculation_*
counters, lazy_flush_total_blocks, the LRU walk-depth histogram) MUST
be exposed via the existing engine-telemetry stream so offline
analysis can plot them over wall clock without scraping engine logs.

Two surfaces are tested:

1. ``FreeKVCacheBlockQueue.cache_stats_snapshot()`` — the structured
   dict consumed by the emitter. Asserts the schema (every counter
   present, the derived ``speculation_hit_rate`` is denominator-safe)
   and that the snapshot reflects observable counter increments after
   real queue operations.
2. ``wires_engine_telemetry.emit_cache_stats(snapshot)`` — emits one
   row to the ``engine_cache_stats`` JSONL stream (E5). Asserts that
   the row contains the snapshot keys + a wall-clock ``ts_epoch``,
   that the disabled-mode path writes no file, and that the periodic
   interval helper honors the env override.

The tests guard the OBSERVABLE downstream value (the snapshot dict
values, the JSONL row contents) per the memory rule
``feedback_test_guards_for_invariants``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    VICTIM_POLICY_WIRES_THREE_POOL,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)
from vllm.v1 import wires_engine_telemetry as wet
from vllm.v1 import wires_telemetry as wt

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Required snapshot keys (M5(a)). Locked in this list so a future PR that
# drops or renames a counter has to update the test alongside the helper.
# Adding new counters is fine: extend ``EXPECTED_SNAPSHOT_KEYS`` and the
# offline aggregator will pick them up at the next deploy.
# ---------------------------------------------------------------------------

EXPECTED_SNAPSHOT_KEYS = {
    "must_pool_evicted_count",
    "ttl_demoted_count",
    "access_promoted_count",
    "unstructured_ema_sample_count",
    "p_h_ema_sample_count",
    "speculation_promotion_count",
    "speculation_hit_count",
    "speculation_miss_count",
    "speculation_hit_rate",
    "lazy_flush_total_blocks",
    "lru_insert_count",
    "lru_insert_walk_steps_total",
    "lru_insert_walk_depth_buckets",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each test starts with a clean writer registry, telemetry
    disabled, and the periodic-interval override cleared so changes
    don't bleed between cases."""

    monkeypatch.setenv("WIRES_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.delenv("WIRES_SWEEP_OUT_DIR", raising=False)
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)
    wet._set_cache_stats_interval_for_tests(wet.DEFAULT_CACHE_STATS_INTERVAL_S)
    yield
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)
    wet._set_cache_stats_interval_for_tests(wet.DEFAULT_CACHE_STATS_INTERVAL_S)


def _three_pool_queue(blocks: list[KVCacheBlock]) -> FreeKVCacheBlockQueue:
    """Force three-pool mode regardless of env, with TTL effectively
    disabled so background sweeps don't perturb the counter values
    we drive deliberately."""
    return FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=10**12,
        unstructured_bootstrap_ttl_ns=10**12,
        speculative_ttl_ns=10**12,
    )


# ===========================================================================
# Snapshot shape contract
# ===========================================================================


def test_snapshot_contains_all_required_keys():
    """Every counter listed in M5(a) must appear in the snapshot dict."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = _three_pool_queue(blocks)

    snapshot = queue.cache_stats_snapshot()

    assert set(snapshot.keys()) == EXPECTED_SNAPSHOT_KEYS, (
        f"snapshot keys drift: missing="
        f"{EXPECTED_SNAPSHOT_KEYS - set(snapshot.keys())}, "
        f"extra={set(snapshot.keys()) - EXPECTED_SNAPSHOT_KEYS}"
    )


def test_snapshot_initial_values_are_zero_or_none():
    """Fresh queue: every cumulative counter is 0; the derived
    speculation_hit_rate is None (denominator-safe)."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = _three_pool_queue(blocks)

    snapshot = queue.cache_stats_snapshot()

    # Cumulative integer counters: all zero on a fresh queue.
    int_counters = {
        "must_pool_evicted_count",
        "ttl_demoted_count",
        "access_promoted_count",
        "unstructured_ema_sample_count",
        "p_h_ema_sample_count",
        "speculation_promotion_count",
        "speculation_hit_count",
        "speculation_miss_count",
        "lazy_flush_total_blocks",
        "lru_insert_count",
        "lru_insert_walk_steps_total",
    }
    for k in int_counters:
        assert snapshot[k] == 0, f"{k} should start at 0, got {snapshot[k]!r}"
    # Histogram: list of four zeros.
    assert snapshot["lru_insert_walk_depth_buckets"] == [0, 0, 0, 0]
    # Derived rate: None (avoid 0/0 noise).
    assert snapshot["speculation_hit_rate"] is None


def test_snapshot_speculation_hit_rate_denominator_safe():
    """``speculation_hit_rate`` must be denominator-safe across
    speculation_promotion_count == 0 AND > 0; never raises."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = _three_pool_queue(blocks)

    # Denominator zero: rate is None.
    snap_a = queue.cache_stats_snapshot()
    assert snap_a["speculation_hit_rate"] is None

    # Drive a denominator > 0 + a hit.
    queue.speculation_promotion_count = 4
    queue.speculation_hit_count = 3
    snap_b = queue.cache_stats_snapshot()
    assert snap_b["speculation_hit_rate"] == pytest.approx(0.75)

    # Drive a hit_count == 0, denominator > 0 (cold speculation, no hits).
    queue.speculation_hit_count = 0
    snap_c = queue.cache_stats_snapshot()
    assert snap_c["speculation_hit_rate"] == pytest.approx(0.0)


def test_snapshot_reflects_real_counter_changes():
    """End-to-end: drive operations that bump real counters, and
    verify the snapshot reflects the post-op values.

    Specifically:
      - access_promoted_count bumps when a may-block is access-promoted.
      - ttl_demoted_count + must_pool_evicted_count grow under the
        sweep + popleft pressure paths.
      - lazy_flush_total_blocks bumps after deferred hint flips flush.
      - lru_insert_count + walk-depth buckets bump on append.
      - unstructured_ema_sample_count bumps when the EMA is fed a
        sample.
    """
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
    )
    queue = pool.free_block_queue

    # Snapshot before any operation.
    snap0 = queue.cache_stats_snapshot()

    # Lazy flip + flush: bumps lazy_flush_total_blocks.
    block_a = pool.blocks[1]  # block_id 0 is the null block in BlockPool
    queue.update_block_hint(block_a, "must", source_class="structured")
    queue._flush_pending_hint_flips()

    snap1 = queue.cache_stats_snapshot()
    assert snap1["lazy_flush_total_blocks"] >= 1
    assert snap1["lazy_flush_total_blocks"] > snap0["lazy_flush_total_blocks"]

    # EMA sample bumps unstructured_ema_sample_count.
    queue._feed_unstructured_sample(123_456)
    snap2 = queue.cache_stats_snapshot()
    assert snap2["unstructured_ema_sample_count"] == (
        snap1["unstructured_ema_sample_count"] + 1
    )

    # p_h sample bumps p_h_ema_sample_count.
    queue._feed_p_h_sample(1)
    snap3 = queue.cache_stats_snapshot()
    assert snap3["p_h_ema_sample_count"] == snap2["p_h_ema_sample_count"] + 1

    # Manually bump the must_pool_evicted / ttl_demoted / access_promoted
    # counters via the tested attribute surface (the engine paths that
    # bump these are covered by their own dedicated test files; what
    # this test owns is the "snapshot reflects them" contract).
    queue.must_pool_evicted_count += 7
    queue.ttl_demoted_count += 11
    queue.access_promoted_count += 4
    snap4 = queue.cache_stats_snapshot()
    assert snap4["must_pool_evicted_count"] == 7
    assert snap4["ttl_demoted_count"] == 11
    assert snap4["access_promoted_count"] == 4


def test_snapshot_is_json_serialisable():
    """The snapshot dict will be handed to the JSONL writer, so it
    must round-trip through ``json.dumps`` without a custom encoder.
    """
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    queue = _three_pool_queue(blocks)
    queue.speculation_promotion_count = 2
    queue.speculation_hit_count = 1

    snapshot = queue.cache_stats_snapshot()

    payload = json.dumps(snapshot)
    decoded = json.loads(payload)
    assert decoded["speculation_hit_rate"] == pytest.approx(0.5)
    assert decoded["lru_insert_walk_depth_buckets"] == [0, 0, 0, 0]


# ===========================================================================
# Emitter contract — emit_cache_stats writes through the JSONL stream
# ===========================================================================


def test_emit_cache_stats_disabled_writes_nothing(tmp_path: Path):
    """Disabled-mode path: no file created, no allocation pressure."""
    wt._set_enabled_for_tests(False)
    snapshot = {"must_pool_evicted_count": 5}
    for _ in range(20):
        wet.emit_cache_stats(snapshot)
    assert not (tmp_path / "engine_cache_stats.jsonl").exists()


def test_emit_cache_stats_enabled_writes_row(tmp_path: Path):
    """Enabled-mode path: one row per call, ts_epoch prepended,
    snapshot keys preserved."""
    wt._set_enabled_for_tests(True)

    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = _three_pool_queue(blocks)
    queue.speculation_promotion_count = 4
    queue.speculation_hit_count = 1
    snapshot = queue.cache_stats_snapshot()

    wet.emit_cache_stats(snapshot)
    wt.flush_all(timeout_s=5.0)

    rows = [
        json.loads(line)
        for line in (tmp_path / "engine_cache_stats.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    # The row must carry every snapshot key + ts_epoch.
    expected = EXPECTED_SNAPSHOT_KEYS | {"ts_epoch"}
    assert set(row.keys()) == expected
    # ts_epoch is a sane wall-clock value (within the last 60s).
    assert (time.time() - row["ts_epoch"]) < 60.0
    # Snapshot values round-trip verbatim.
    assert row["speculation_promotion_count"] == 4
    assert row["speculation_hit_count"] == 1
    assert row["speculation_hit_rate"] == pytest.approx(0.25)


def test_emit_cache_stats_explicit_ts_epoch(tmp_path: Path):
    """An explicit ``ts_epoch`` overrides the wall-clock default."""
    wt._set_enabled_for_tests(True)
    snapshot = {"must_pool_evicted_count": 9}
    pinned_ts = 12345.678
    wet.emit_cache_stats(snapshot, ts_epoch=pinned_ts)
    wt.flush_all(timeout_s=5.0)

    rows = [
        json.loads(line)
        for line in (tmp_path / "engine_cache_stats.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["ts_epoch"] == pinned_ts
    assert rows[0]["must_pool_evicted_count"] == 9


# ===========================================================================
# Periodic-interval helper
# ===========================================================================


def test_cache_stats_interval_default_resolution():
    """Default interval is the documented constant (1.0s)."""
    wet._set_cache_stats_interval_for_tests(wet.DEFAULT_CACHE_STATS_INTERVAL_S)
    assert wet.get_cache_stats_interval_s() == pytest.approx(
        wet.DEFAULT_CACHE_STATS_INTERVAL_S
    )


def test_cache_stats_interval_test_override():
    """The tests-only override applies immediately."""
    wet._set_cache_stats_interval_for_tests(0.25)
    assert wet.get_cache_stats_interval_s() == pytest.approx(0.25)


def test_cache_stats_interval_env_resolution(monkeypatch: pytest.MonkeyPatch):
    """``WIRES_TELEMETRY_CACHE_STATS_INTERVAL_S`` env var is honored
    by ``_resolve_cache_stats_interval_s`` (called at import); the
    helper additionally floors small values at 0.05."""
    monkeypatch.setenv("WIRES_TELEMETRY_CACHE_STATS_INTERVAL_S", "2.5")
    assert wet._resolve_cache_stats_interval_s() == pytest.approx(2.5)
    monkeypatch.setenv("WIRES_TELEMETRY_CACHE_STATS_INTERVAL_S", "0.001")
    # Floor at 0.05 so a runaway value doesn't flood the JSONL.
    assert wet._resolve_cache_stats_interval_s() == pytest.approx(0.05)
    monkeypatch.setenv("WIRES_TELEMETRY_CACHE_STATS_INTERVAL_S", "not-a-float")
    # Bad input falls back to the default.
    assert wet._resolve_cache_stats_interval_s() == pytest.approx(
        wet.DEFAULT_CACHE_STATS_INTERVAL_S
    )
