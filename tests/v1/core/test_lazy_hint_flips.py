# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M19 (CODE_MISMATCH_NOTES.md): engine-side lazy hint flips.

Asserts:
  - ``update_block_hint`` records into ``_pending_hint_flips`` instead
    of moving the block immediately.
  - The flush at the entry of ``popleft_n``, ``append``, ``append_n``,
    ``remove``, and ``get_all_free_blocks`` materializes the move so
    callers see the latest pool placement.
  - Multiple flips on the same block within one window collapse to the
    final state (one pool move at the next cache op, not N).
  - ``block.lifecycle_hint`` is updated EAGERLY (lazy applies only to
    the doubly-linked list move, never to the metadata).
  - ``lazy_flush_total_blocks`` telemetry counts the cumulative
    flushed flips.
"""

from __future__ import annotations

import pytest

from vllm.v1.core.kv_cache_utils import (
    VICTIM_POLICY_PURE_LRU,
    VICTIM_POLICY_WIRES_THREE_POOL,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)

pytestmark = pytest.mark.cpu_test


def _three_pool_queue(blocks: list[KVCacheBlock]) -> FreeKVCacheBlockQueue:
    """Force three-pool mode regardless of env var, with TTL effectively
    disabled so deferred entries don't get clobbered by the sweep."""
    return FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        # Long TTLs keep the lazy-sweep paths inert during these unit tests.
        structured_ttl_ns=10**12,
        unstructured_bootstrap_ttl_ns=10**12,
    )


def test_update_block_hint_defers_pool_move():
    """The pool list move should NOT happen during update_block_hint;
    only the metadata flip and the pending-dict record should fire."""
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    # All start in may by default.
    queue = _three_pool_queue(blocks)
    assert queue.num_free_blocks_in_pool("may") == 3
    assert queue.num_free_blocks_in_pool("must") == 0

    # Promote blocks[0] to must.
    queue.update_block_hint(blocks[0], "must", source_class="structured")

    # Eager: lifecycle_hint updates immediately.
    assert blocks[0].lifecycle_hint == "must"
    # Lazy: the actual physical pool placement does NOT change yet.
    assert queue.num_free_blocks_in_pool("may") == 3
    assert queue.num_free_blocks_in_pool("must") == 0
    # Pending dict records the deferred move.
    assert blocks[0].block_id in queue._pending_hint_flips
    pending_block, pending_hint, pending_class = queue._pending_hint_flips[
        blocks[0].block_id
    ]
    assert pending_block is blocks[0]
    assert pending_hint == "must"
    assert pending_class == "structured"


def test_flush_at_popleft_n_materializes_moves():
    """popleft_n must flush pending flips so the physical pool order
    drives the eviction priority."""
    blocks = [KVCacheBlock(block_id=i) for i in range(4)]
    queue = _three_pool_queue(blocks)

    # Promote blocks[1] to must (deferred).
    queue.update_block_hint(blocks[1], "must", source_class="structured")
    # Demote blocks[2] to no (deferred).
    queue.update_block_hint(blocks[2], "no")
    # Pending dict has both entries.
    assert len(queue._pending_hint_flips) == 2

    # popleft_n triggers the flush. With the new placement, "no"
    # is highest priority -> blocks[2] should come out first.
    out = queue.popleft_n(1)
    assert out == [blocks[2]]
    # Pending dict is empty after flush.
    assert queue._pending_hint_flips == {}
    # Telemetry: 2 pending entries got flushed (blocks[1] -> must,
    # blocks[2] -> no).
    assert queue.lazy_flush_total_blocks == 2

    # Remaining pool layout: may has blocks[0], blocks[3]; must has
    # blocks[1]. popleft_n(2) should drain may in LRU order.
    out = queue.popleft_n(2)
    assert out == [blocks[0], blocks[3]]
    # Final: only must holds blocks[1].
    assert queue.num_free_blocks_in_pool("must") == 1
    out = queue.popleft_n(1)
    assert out == [blocks[1]]


def test_back_to_back_flips_on_same_block_collapse():
    """Multiple flips on the same block in one window must collapse to
    the final state (one pool move at flush, not N)."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = _three_pool_queue(blocks)

    # Eight back-to-back flips on blocks[0]: may -> no -> must -> may
    # -> no -> must -> may -> no -> must (final = must).
    transitions = ["no", "must", "may", "no", "must", "may", "no", "must"]
    for hint in transitions:
        queue.update_block_hint(blocks[0], hint, source_class="structured")

    # Pending dict only has 1 entry (collapsed) targeting the FINAL hint.
    assert len(queue._pending_hint_flips) == 1
    _, final_hint, _ = queue._pending_hint_flips[blocks[0].block_id]
    assert final_hint == "must"
    # Eager metadata: also reflects the final.
    assert blocks[0].lifecycle_hint == "must"

    # Flush via popleft_n. Only one pool move fires.
    queue.popleft_n(1)  # consumes the must block (blocks[0])
    # Telemetry: exactly one flip applied even though we called
    # update_block_hint 8 times.
    assert queue.lazy_flush_total_blocks == 1


def test_flip_to_current_pool_drops_pending_entry():
    """A flip whose target matches the block's current physical pool
    should drop any prior pending entry (no work to defer)."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = _three_pool_queue(blocks)
    # Promote blocks[0] to must, then immediately flip back to may
    # (still in may physically). Net effect: zero pool moves needed.
    queue.update_block_hint(blocks[0], "must", source_class="structured")
    assert blocks[0].block_id in queue._pending_hint_flips
    queue.update_block_hint(blocks[0], "may")
    # Pending entry dropped.
    assert blocks[0].block_id not in queue._pending_hint_flips
    # Flush: nothing to apply.
    applied = queue._flush_pending_hint_flips()
    assert applied == 0
    assert queue.lazy_flush_total_blocks == 0
    # Block is still in may physically.
    assert queue.num_free_blocks_in_pool("may") == 2


def test_flush_at_append_and_remove():
    """append, append_n, and remove must all flush before mutating."""
    blocks = [KVCacheBlock(block_id=i) for i in range(4)]
    queue = _three_pool_queue(blocks)

    # Promote blocks[0] to must (deferred).
    queue.update_block_hint(blocks[0], "must", source_class="structured")
    assert len(queue._pending_hint_flips) == 1

    # remove() should flush first so the pool layout matches metadata.
    # Use blocks[3] (a fresh may block) so we exercise remove without
    # touching the pending block.
    queue.remove(blocks[3])
    # Flush already happened.
    assert queue._pending_hint_flips == {}
    assert queue.lazy_flush_total_blocks == 1
    # blocks[0] should now physically be in must.
    assert queue.num_free_blocks_in_pool("must") == 1

    # Re-add blocks[3] via append (defaults to may pool). Layout
    # before re-add: may = [b1, b2], must = [b0]. After re-add:
    # may = [b1, b2, b3], must = [b0].
    queue.append(blocks[3])
    assert queue.num_free_blocks_in_pool("may") == 3
    assert queue.num_free_blocks_in_pool("must") == 1

    # Now create another deferred flip and verify append_n flushes.
    queue.update_block_hint(blocks[1], "must", source_class="structured")
    assert len(queue._pending_hint_flips) == 1
    # Pop blocks[3] back out (so we can append_n it).
    queue.remove(blocks[3])
    # Above remove already flushed blocks[1]'s pending entry.
    assert queue._pending_hint_flips == {}
    assert queue.lazy_flush_total_blocks == 2
    # append_n round-trip with a new pending entry.
    queue.update_block_hint(blocks[2], "no")
    assert len(queue._pending_hint_flips) == 1
    queue.append_n([blocks[3]])
    assert queue._pending_hint_flips == {}
    assert queue.lazy_flush_total_blocks == 3


def test_flush_at_get_all_free_blocks():
    """get_all_free_blocks returns the authoritative ordering, so it
    must flush deferred flips first."""
    blocks = [KVCacheBlock(block_id=i) for i in range(4)]
    queue = _three_pool_queue(blocks)

    # Move blocks[2] to no (deferred); without flush, get_all returns
    # the stale order [b0, b1, b2, b3] (all in may).
    queue.update_block_hint(blocks[2], "no")
    # Calling get_all should flush and return [b2 (no), b0, b1, b3 (may)].
    snapshot = queue.get_all_free_blocks()
    assert snapshot[0] is blocks[2]
    assert snapshot[1:] == [blocks[0], blocks[1], blocks[3]]
    assert queue._pending_hint_flips == {}
    assert queue.lazy_flush_total_blocks == 1


def test_lifecycle_hint_eager_update():
    """The metadata flip on block.lifecycle_hint must be eager (no lag)
    so external readers (e.g., block_pool.touch's combined check) see
    the latest value immediately."""
    block = KVCacheBlock(block_id=0)
    queue = _three_pool_queue([block])
    assert block.lifecycle_hint == "may"
    queue.update_block_hint(block, "must", source_class="structured")
    # Metadata: immediate.
    assert block.lifecycle_hint == "must"
    # Pool placement: deferred.
    assert queue.num_free_blocks_in_pool("must") == 0
    # source_class is captured in the pending tuple (the stamp at
    # flush time will set ``block.source_class``); the in-free-pool
    # path does NOT stamp the field eagerly because ``_stamp_must_promotion``
    # owns that side-effect and runs only when the pool move actually happens.
    _, _, pending_class = queue._pending_hint_flips[block.block_id]
    assert pending_class == "structured"
    # After flush, the stamp fires and the block reflects the new class.
    queue._flush_pending_hint_flips()
    assert block.source_class == "structured"
    assert queue.num_free_blocks_in_pool("must") == 1


def test_pure_lru_mode_skips_pending_dict():
    """In pure_lru mode the pending dict must stay empty (lifecycle is
    metadata-only and there's no second pool to move into)."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_PURE_LRU)
    queue.update_block_hint(blocks[0], "must", source_class="structured")
    assert blocks[0].lifecycle_hint == "must"
    assert queue._pending_hint_flips == {}
    # Pop order is preserved (FIFO) regardless of hint.
    out = queue.popleft_n(2)
    assert out == [blocks[0], blocks[1]]


def test_pending_entry_dropped_when_block_leaves_free_pool():
    """If a block's hint is updated while it's in-use (not in any free
    pool), no pending entry should be created. If a pending entry
    exists from a prior call and the block then leaves the free pool
    via a subsequent update, the entry is dropped."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = _three_pool_queue(blocks)
    # First flip while in free pool: pending entry created.
    queue.update_block_hint(blocks[0], "must", source_class="structured")
    assert blocks[0].block_id in queue._pending_hint_flips

    # Simulate the block being removed from the free queue (in-use
    # path). Use the queue's internal map directly to avoid triggering
    # the flush-on-remove path.
    queue._flush_pending_hint_flips()  # apply first, then test in-use case
    queue.remove(blocks[0])
    assert blocks[0].block_id not in queue._pool_of

    # Now the block is in-use. A subsequent hint flip should NOT
    # populate the pending dict (no pool move possible).
    queue.update_block_hint(blocks[0], "may")
    assert blocks[0].block_id not in queue._pending_hint_flips
    # Metadata still updated eagerly.
    assert blocks[0].lifecycle_hint == "may"


def test_no_op_flip_does_not_record_pending():
    """When the new hint matches the block's current lifecycle_hint,
    nothing should happen (early return)."""
    blocks = [KVCacheBlock(block_id=0)]
    queue = _three_pool_queue(blocks)
    # blocks[0] is already "may"; flipping to "may" is a no-op.
    queue.update_block_hint(blocks[0], "may")
    assert queue._pending_hint_flips == {}
    assert queue.lazy_flush_total_blocks == 0


def test_telemetry_counter_accumulates_across_flushes():
    """lazy_flush_total_blocks must accumulate across multiple flushes
    (it's a cumulative counter, not a per-flush value)."""
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    queue = _three_pool_queue(blocks)
    # Round 1: 3 flips.
    queue.update_block_hint(blocks[0], "no")
    queue.update_block_hint(blocks[1], "must", source_class="structured")
    queue.update_block_hint(blocks[2], "no")
    queue.popleft_n(1)
    assert queue.lazy_flush_total_blocks == 3
    # Round 2: 2 more flips.
    queue.update_block_hint(blocks[3], "must", source_class="structured")
    queue.update_block_hint(blocks[4], "no")
    queue.popleft_n(1)
    assert queue.lazy_flush_total_blocks == 5
    # Round 3: 1 flip.
    queue.update_block_hint(blocks[5], "must", source_class="structured")
    queue._flush_pending_hint_flips()
    assert queue.lazy_flush_total_blocks == 6
