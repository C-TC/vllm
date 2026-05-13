# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M17 (CODE_MISMATCH_NOTES.md): last_access_ns-correct LRU position
insertion for the may pool, plus walk-depth telemetry.

Asserts:
  - ``_PoolList.insert_by_last_access`` lands a fresh block at the MRU
    end (tail), an old block at the LRU end (head), and a middle block
    in the middle of an existing sorted-by-access pool.
  - ``_PoolList.merge_sorted_by_last_access`` preserves the ascending
    invariant when merging k blocks into n_existing.
  - The TTL sweep, when it demotes blocks from must -> may, lands them
    at their last_access_ns-correct position rather than the LRU end
    (the pre-M17 ``prepend_many`` behaviour).
  - The runner-driven hint flip (must -> may) routed through the M19
    lazy-flush path uses ``insert_by_last_access`` rather than the
    legacy ``append`` (MRU-end placement).
  - ``_record_lru_insert_walk`` bumps the cumulative-step counter and
    the four-bucket histogram correctly.
"""

from __future__ import annotations

import pytest

from vllm.v1.core.kv_cache_utils import (
    VICTIM_POLICY_WIRES_THREE_POOL,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    _PoolList,
)

pytestmark = pytest.mark.cpu_test


def _three_pool_queue(blocks: list[KVCacheBlock]) -> FreeKVCacheBlockQueue:
    return FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        # Long TTLs so the lazy sweep doesn't fire during these tests.
        structured_ttl_ns=10**12,
        unstructured_bootstrap_ttl_ns=10**12,
    )


def _pool_block_ids(pool: _PoolList) -> list[int]:
    return [b.block_id for b in pool.iter_blocks()]


def test_insert_by_last_access_fresh_block_lands_at_mru_end():
    """A block freshly accessed (largest last_access_ns) should end up
    adjacent to the tail, AFTER all existing entries."""
    pool = _PoolList()
    # Existing pool: ascending [10, 20, 30].
    for ts, bid in [(10, 0), (20, 1), (30, 2)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        pool.append(b)
    fresh = KVCacheBlock(block_id=99)
    fresh.last_access_ns = 100  # newer than every existing entry
    steps = pool.insert_by_last_access(fresh)
    # Insertion should be at the MRU end (no walk needed: the cursor
    # at the tail.prev already has last_access_ns = 30 <= 100).
    assert steps == 0
    assert _pool_block_ids(pool) == [0, 1, 2, 99]


def test_insert_by_last_access_old_block_lands_at_lru_end():
    """A block with the smallest last_access_ns lands at the LRU end."""
    pool = _PoolList()
    for ts, bid in [(10, 0), (20, 1), (30, 2)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        pool.append(b)
    old = KVCacheBlock(block_id=99)
    old.last_access_ns = 5  # older than every existing entry
    steps = pool.insert_by_last_access(old)
    # Walk passes 30, 20, 10 (3 cursor moves), settles before block 0.
    assert steps == 3
    assert _pool_block_ids(pool) == [99, 0, 1, 2]


def test_insert_by_last_access_middle_position():
    """A block whose last_access_ns falls between two existing entries
    lands at the right position."""
    pool = _PoolList()
    for ts, bid in [(10, 0), (20, 1), (30, 2), (40, 3)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        pool.append(b)
    mid = KVCacheBlock(block_id=99)
    mid.last_access_ns = 25  # between block 1 (20) and block 2 (30)
    steps = pool.insert_by_last_access(mid)
    # Walk passes 40, 30 (2 cursor moves), settles after block 1.
    assert steps == 2
    assert _pool_block_ids(pool) == [0, 1, 99, 2, 3]


def test_insert_by_last_access_into_empty_pool():
    """Inserting into an empty pool should land between the sentinels
    with zero walk steps."""
    pool = _PoolList()
    b = KVCacheBlock(block_id=42)
    b.last_access_ns = 50
    steps = pool.insert_by_last_access(b)
    assert steps == 0
    assert _pool_block_ids(pool) == [42]
    assert pool.size == 1


def test_merge_sorted_by_last_access_two_pointer():
    """Batched merge: two pre-sorted lists interleave correctly with one
    pass over the existing pool."""
    pool = _PoolList()
    # Existing pool: [10, 30, 50, 70].
    for ts, bid in [(10, 0), (30, 1), (50, 2), (70, 3)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        pool.append(b)
    # Incoming sorted: [20, 40, 60, 80].
    incoming: list[KVCacheBlock] = []
    for ts, bid in [(20, 100), (40, 101), (60, 102), (80, 103)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        incoming.append(b)
    steps = pool.merge_sorted_by_last_access(incoming)
    # Two-pointer cost: each existing block is walked past at most once.
    # We have 4 existing blocks; walking past all of them costs 4.
    assert steps == 4
    assert _pool_block_ids(pool) == [0, 100, 1, 101, 2, 102, 3, 103]


def test_merge_sorted_by_last_access_all_smaller():
    """Merging blocks all older than every existing entry: no cursor
    advance; insertions all land at the LRU end."""
    pool = _PoolList()
    for ts, bid in [(100, 0), (200, 1)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        pool.append(b)
    incoming: list[KVCacheBlock] = []
    for ts, bid in [(10, 50), (20, 51), (30, 52)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        incoming.append(b)
    steps = pool.merge_sorted_by_last_access(incoming)
    # Cursor never advances past block 0 (target is always smaller than 100).
    assert steps == 0
    assert _pool_block_ids(pool) == [50, 51, 52, 0, 1]


def test_merge_sorted_by_last_access_all_larger():
    """Merging blocks all newer than every existing entry: cursor
    advances through the entire pool once."""
    pool = _PoolList()
    for ts, bid in [(10, 0), (20, 1)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        pool.append(b)
    incoming: list[KVCacheBlock] = []
    for ts, bid in [(100, 50), (200, 51)]:
        b = KVCacheBlock(block_id=bid)
        b.last_access_ns = ts
        incoming.append(b)
    steps = pool.merge_sorted_by_last_access(incoming)
    # Cursor advances past blocks 0 and 1, then both incoming entries
    # land at the MRU end. After that, cursor is at tail and stays
    # there (no further advance) for the second incoming entry, so
    # total steps = 2.
    assert steps == 2
    assert _pool_block_ids(pool) == [0, 1, 50, 51]


def test_merge_sorted_by_last_access_empty_inputs():
    pool = _PoolList()
    pool.append(KVCacheBlock(block_id=0))
    steps = pool.merge_sorted_by_last_access([])
    assert steps == 0
    assert pool.size == 1


def test_telemetry_buckets_route_correctly():
    """_record_lru_insert_walk must bump the right bucket per the
    <10 / <100 / <1000 / >=1000 boundaries and accumulate totals."""
    queue = _three_pool_queue([])
    assert queue.lru_insert_count == 0
    assert queue.lru_insert_walk_steps_total == 0
    assert queue.lru_insert_walk_depth_buckets == [0, 0, 0, 0]

    # bucket 0: <10
    queue._record_lru_insert_walk(0)
    queue._record_lru_insert_walk(9)
    # bucket 1: <100
    queue._record_lru_insert_walk(10)
    queue._record_lru_insert_walk(99)
    # bucket 2: <1000
    queue._record_lru_insert_walk(100)
    queue._record_lru_insert_walk(999)
    # bucket 3: >=1000
    queue._record_lru_insert_walk(1000)
    queue._record_lru_insert_walk(5000)

    assert queue.lru_insert_count == 8
    expected_total = 0 + 9 + 10 + 99 + 100 + 999 + 1000 + 5000
    assert queue.lru_insert_walk_steps_total == expected_total
    assert queue.lru_insert_walk_depth_buckets == [2, 2, 2, 2]


def test_ttl_sweep_demoted_blocks_use_merge_not_prepend():
    """When _sweep_ttl_must demotes blocks from must -> may, they must
    land at LRU-position-correct positions (not all at the LRU head as
    the legacy prepend_many did)."""
    # Set up: must pool has 3 blocks with mid-ranged access times; may
    # pool has older blocks at the LRU end and newer blocks at the
    # MRU end. After demotion the must blocks should interleave by
    # last_access_ns, NOT all land at the LRU head.
    blocks = [KVCacheBlock(block_id=i) for i in range(7)]
    # Layout we want after init:
    #   may pool (sorted asc by last_access_ns):
    #     b0: ts=10  (oldest may)
    #     b1: ts=200 (newest may)
    #   must pool (will be demoted):
    #     b2: ts=50,  ttl_at_promotion=1ns (immediately expired)
    #     b3: ts=100, ttl_at_promotion=1ns
    #     b4: ts=150, ttl_at_promotion=1ns
    # Initially mark all "may"; set timestamps; then add to queue.
    blocks[0].last_access_ns = 10
    blocks[1].last_access_ns = 200
    blocks[2].last_access_ns = 50
    blocks[3].last_access_ns = 100
    blocks[4].last_access_ns = 150
    # Mark must blocks BEFORE handing to queue ctor so they land in must.
    for i in (2, 3, 4):
        blocks[i].lifecycle_hint = "must"
        blocks[i].source_class = "structured"
    queue = FreeKVCacheBlockQueue(
        [blocks[0], blocks[2], blocks[3], blocks[4], blocks[1]],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=1,  # 1 ns -> immediately expired on next sweep
        unstructured_bootstrap_ttl_ns=10**12,
    )
    # Confirm initial layout.
    assert _pool_block_ids(queue._pools["may"]) == [0, 1]
    assert _pool_block_ids(queue._pools["must"]) == [2, 3, 4]

    # Trigger TTL sweep via popleft_n (or call directly).
    demoted = queue._sweep_ttl_must()
    assert demoted == 3
    # Post-demote may pool should be sorted by last_access_ns ascending:
    #   b0(10), b2(50), b3(100), b4(150), b1(200)
    assert _pool_block_ids(queue._pools["may"]) == [0, 2, 3, 4, 1]
    # Walk-depth telemetry: one logical insert event was recorded.
    assert queue.lru_insert_count == 1
    # Merge cursor walk: starts at b0(10); for b2(50) advances past b0
    # to b1(200) (1 step), then b1 is bigger than every remaining
    # incoming entry (100, 150) so cursor never advances again.
    assert queue.lru_insert_walk_steps_total == 1
    assert queue.lru_insert_walk_depth_buckets[0] == 1


def test_runner_hint_demote_uses_lru_position_insertion():
    """A runner-driven must -> may demote routed via update_block_hint
    (which records pending and flushes at the next cache op) must land
    the demoted block at its access-time position, not at the MRU end."""
    blocks = [KVCacheBlock(block_id=i) for i in range(4)]
    # Existing may: b0(ts=10), b3(ts=200) (sorted ascending).
    blocks[0].last_access_ns = 10
    blocks[3].last_access_ns = 200
    # Existing must: b1(ts=50), b2(ts=100). Source class = structured
    # so the M14 p_h sample doesn't blow up the test (no must hits).
    blocks[1].last_access_ns = 50
    blocks[1].lifecycle_hint = "must"
    blocks[1].source_class = "structured"
    blocks[2].last_access_ns = 100
    blocks[2].lifecycle_hint = "must"
    blocks[2].source_class = "structured"
    queue = FreeKVCacheBlockQueue(
        [blocks[0], blocks[1], blocks[2], blocks[3]],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=10**12,
        unstructured_bootstrap_ttl_ns=10**12,
    )
    assert _pool_block_ids(queue._pools["may"]) == [0, 3]
    assert _pool_block_ids(queue._pools["must"]) == [1, 2]

    # Demote b1 (ts=50) to may. Pre-M17 this would land at the MRU end
    # of may, giving order [0, 3, 1] (wrong: b1 is older than b3).
    # Post-M17 it lands between b0 and b3, giving [0, 1, 3].
    queue.update_block_hint(blocks[1], "may")
    # Force the deferred move to apply.
    queue._flush_pending_hint_flips()
    assert _pool_block_ids(queue._pools["may"]) == [0, 1, 3]
    assert _pool_block_ids(queue._pools["must"]) == [2]
    # Walk-depth telemetry recorded one insert.
    assert queue.lru_insert_count == 1
    # Cursor walked: from tail.prev (b3, ts=200) -> b0 (ts=10), so 1 step.
    assert queue.lru_insert_walk_steps_total == 1


def test_promotion_to_must_keeps_mru_end_append():
    """Promotion (may -> must) is the OTHER direction; freshly promoted
    blocks land at the MRU end of the must pool (the existing
    behaviour, preserved by M17). The walk-depth telemetry should NOT
    fire for must-direction moves."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    blocks[0].last_access_ns = 50
    blocks[1].last_access_ns = 100
    queue = _three_pool_queue(blocks)
    assert _pool_block_ids(queue._pools["may"]) == [0, 1]

    # Promote b0 to must.
    queue.update_block_hint(blocks[0], "must", source_class="structured")
    queue._flush_pending_hint_flips()
    assert _pool_block_ids(queue._pools["must"]) == [0]
    # Walk-depth telemetry: no LRU-position insert happened (must pool
    # uses MRU-end append).
    assert queue.lru_insert_count == 0
    assert queue.lru_insert_walk_steps_total == 0
