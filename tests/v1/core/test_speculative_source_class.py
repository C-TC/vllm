# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M20 — speculative source class + short TTL backstop + class upgrade.

Per ``paper/CODE_MISMATCH_NOTES.md`` M20 (and the memory feedback rule
"test guards on the actual invariant"), each test asserts the
OBSERVABLE downstream value (block.source_class transitions, the three
speculation_* counters, pool membership) rather than just that a code
path was reached.

Three required tests per the spec:

1. Hit case: speculative block is reused before its TTL fires; class
   upgrades to "unstructured" and ``speculation_hit_count`` increments.
2. Miss case (reactive demote): the runner reactively demotes the
   block's segment before TTL fires; ``speculation_miss_count`` does
   NOT increment (the demote was reactive, not TTL-driven).
3. Miss case (TTL fallback): no reactive event arrives in time; the
   sweep demotes the block and bumps ``speculation_miss_count``; the
   block ends up in the may pool, not evicted.

The tests drive the engine-side primitive directly (the spec
explicitly leaves runner-side speculation patterns to future work).
"""

from __future__ import annotations

import time

import pytest

from vllm.entrypoints.openai.chat_completion import segment_actions
from vllm.entrypoints.openai.chat_completion.segment_actions import (
    SegmentEntry,
    mark_segment_blocks_speculative,
    reset_segment_registry_for_tests,
    set_block_hint_updater,
    set_speculative_block_promoter,
    update_segment_lifecycle_hint,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    VICTIM_POLICY_WIRES_THREE_POOL,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry_and_updaters_between_tests():
    """Each test starts with a clean global registry and unbound
    block-pool updaters. The local fixture rewires them to the test's
    own queue or block_pool."""

    reset_segment_registry_for_tests()
    set_block_hint_updater(None)
    set_speculative_block_promoter(None)
    yield
    reset_segment_registry_for_tests()
    set_block_hint_updater(None)
    set_speculative_block_promoter(None)


def _tag_block_with_segment_id(blk: KVCacheBlock, segment_id: str) -> None:
    """Tag a block with a segment id, supporting both the legacy single
    ``_segment_id`` slot and the M18 ``_segment_ids`` tuple if present.
    """

    if hasattr(blk, "_segment_ids"):
        existing = blk._segment_ids
        if segment_id not in existing:
            blk._segment_ids = existing + (segment_id,)
    else:
        blk._segment_id = segment_id  # noqa: SLF001


def _seed_segment_with_block(
    segment_id: str, queue: FreeKVCacheBlockQueue, block: KVCacheBlock
) -> None:
    """Wire the test fixture: register a SegmentEntry, tag the block
    with the segment id, and add it to the registry's reverse map.
    Also bind the speculative promoter to the queue under test so
    ``mark_segment_blocks_speculative`` flips real pool membership.
    """

    registry = segment_actions._registry
    registry._entries_by_segment_id[segment_id] = SegmentEntry(
        action_id="act-" + segment_id,
        action_kind="segment_prepare",
        segment_id=segment_id,
        family_id="fam-" + segment_id,
        parent_segment_id=None,
        token_hash="hash-" + segment_id,
        model="dummy-model",
        lifecycle_status="accepted",
        expected_consumers=1,
    )
    _tag_block_with_segment_id(block, segment_id)
    registry.register_block_for_segment(segment_id, block)

    def _structured_updater(blk, new_hint):
        queue.update_block_hint(blk, new_hint, source_class="structured")

    def _speculative_promoter(blk):
        queue.update_block_hint(blk, "must", source_class="speculative")

    set_block_hint_updater(_structured_updater)
    set_speculative_block_promoter(_speculative_promoter)


def _build_queue_with_block(
    *,
    speculative_ttl_ns: int = 30 * 1_000_000_000,
) -> tuple[FreeKVCacheBlockQueue, KVCacheBlock]:
    block = KVCacheBlock(block_id=7)  # default lifecycle_hint="may"
    queue = FreeKVCacheBlockQueue(
        [block],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        speculative_ttl_ns=speculative_ttl_ns,
    )
    return queue, block


# ---------------------------------------------------------------------------
# Test 1: HIT — speculative -> unstructured upgrade on first reuse
# ---------------------------------------------------------------------------


def test_speculative_hit_upgrades_class_and_increments_hit_count(monkeypatch):
    """Spec test 1.

    Steps:
    - Construct a speculative scenario via mark_segment_blocks_speculative.
    - Drive a touch() on the resulting block.

    Asserts:
    - Block initially has source_class == "speculative".
    - After touch, source_class upgrades to "unstructured".
    - speculation_promotion_count == 1, speculation_hit_count == 1.
    - The block's TTL is re-stamped to the unstructured value (not the
      30s speculative backstop).
    """

    monkeypatch.delenv("WIRES_KVCACHE_SPECULATIVE_TTL_MS", raising=False)
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
    )
    queue = pool.free_block_queue
    # Pick a non-null free block (block_id 0 is the null block).
    block = pool.blocks[1]
    assert block.lifecycle_hint == "may"

    _seed_segment_with_block("seg-spec-hit", queue, block)

    result = mark_segment_blocks_speculative("seg-spec-hit")
    assert result["updated"] == 1
    assert result["reject_reason"] is None

    # M19 lazy hint-flips defer the actual pool move + must-promotion
    # stamp until the next cache op. Force-apply so the observable
    # ``source_class`` snapshot below sees the speculative class.
    queue._flush_pending_hint_flips()

    # Observable: block is now in the must pool, classed speculative,
    # with the speculative-TTL backstop stamped.
    assert block.lifecycle_hint == "must"
    assert block.source_class == "speculative"
    assert queue.num_free_blocks_in_pool("must") == 1
    assert queue.speculation_promotion_count == 1
    assert queue.speculation_hit_count == 0
    assert queue.speculation_miss_count == 0
    expected_spec_ttl = queue._speculative_ttl_ns
    assert block.ttl_at_promotion_ns == expected_spec_ttl
    # Sanity: the speculative TTL is *much* shorter than the structured
    # one (the whole point of the M20 design).
    assert queue._speculative_ttl_ns < queue._structured_ttl_ns

    # Drive a cache-hit on the speculative block.
    pool.touch([block])

    # Observable upgrade: source_class became unstructured AND the TTL
    # snapshot was refreshed to the unstructured (EMA-bootstrap) value
    # so the block isn't still wearing the 30s backstop.
    assert block.source_class == "unstructured"
    assert queue.speculation_hit_count == 1
    assert queue.speculation_promotion_count == 1
    assert queue.speculation_miss_count == 0
    expected_unstructured_ttl = queue._unstructured_ttl_at_promotion_ns()
    assert block.ttl_at_promotion_ns == expected_unstructured_ttl
    assert block.ttl_at_promotion_ns != expected_spec_ttl
    # Derived metric is 1.0 after one hit on one promotion.
    assert queue.speculation_hit_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2: MISS via reactive demote — runner observes outcome before TTL
# ---------------------------------------------------------------------------


def test_speculative_reactive_demote_does_not_increment_miss_count():
    """Spec test 2.

    Steps:
    - Same speculative setup as test 1.
    - Drive a reactive demote via update_segment_lifecycle_hint(seg, "no",
      literal_no=True) — the loop-exit "force eviction" path that
      records a literal "no" contribution rather than collapsing to
      the empty-live-set default of "may".

    Asserts:
    - The block is demoted to "no" via the reactive path.
    - speculation_miss_count does NOT increment (reactive demotes are
      explicitly excluded from the miss counter; only TTL-fired demotes
      count).
    """

    queue, block = _build_queue_with_block()
    _seed_segment_with_block("seg-spec-react", queue, block)

    mark_segment_blocks_speculative("seg-spec-react")
    # Force-apply the M19-deferred speculative promotion so the
    # subsequent reactive-demote operates on a block that is actually
    # in the must pool with source_class == "speculative".
    queue._flush_pending_hint_flips()
    assert block.source_class == "speculative"
    assert block.lifecycle_hint == "must"
    assert queue.speculation_promotion_count == 1
    assert queue.num_free_blocks_in_pool("must") == 1

    # Reactive demote: monitor signals "force eviction" via the
    # literal-no path (loop-exit semantics, paper §3.5).
    result = update_segment_lifecycle_hint(
        "seg-spec-react", "no", literal_no=True
    )
    # Force-apply the M19-deferred reactive demote so the observable
    # pool count below reflects the move.
    queue._flush_pending_hint_flips()
    assert result.get("reject_reason") is None
    assert result.get("effective_hint") == "no"
    assert block.lifecycle_hint == "no"
    assert queue.num_free_blocks_in_pool("must") == 0
    assert queue.num_free_blocks_in_pool("no") == 1
    # Miss counter is for TTL-driven demotes only; reactive ones don't
    # count.
    assert queue.speculation_miss_count == 0
    assert queue.speculation_hit_count == 0
    assert queue.speculation_promotion_count == 1


# ---------------------------------------------------------------------------
# Test 3: MISS via TTL fallback — no reactive event, sweep demotes
# ---------------------------------------------------------------------------


def test_speculative_ttl_fallback_demotes_and_increments_miss_count():
    """Spec test 3.

    Steps:
    - Same speculative setup as test 1.
    - Do NOT drive any reactive event.
    - Backdate the block's promotion timestamp so its deadline is in
      the past, then trigger _sweep_ttl_must.

    Asserts:
    - TTL sweep demotes the block (return value == 1).
    - speculation_miss_count increments by 1.
    - Block ends up in the may pool (NOT evicted, NOT in no pool).
    """

    queue, block = _build_queue_with_block()
    _seed_segment_with_block("seg-spec-ttl", queue, block)

    mark_segment_blocks_speculative("seg-spec-ttl")
    # Force-apply the M19-deferred promotion so the block is actually
    # in the must pool when we backdate + sweep.
    queue._flush_pending_hint_flips()
    assert block.source_class == "speculative"
    assert block.lifecycle_hint == "must"
    assert queue.speculation_promotion_count == 1
    assert queue.speculation_miss_count == 0

    # Backdate the promotion timestamp so the deadline is already in
    # the past relative to monotonic_ns(). Same pattern as the
    # existing TTL tests in test_kv_cache_utils.py.
    block.last_promoted_ns = 1
    assert block.last_promoted_ns + block.ttl_at_promotion_ns < time.monotonic_ns()

    demoted = queue._sweep_ttl_must()

    # Observable: exactly one block demoted, miss count bumped, block
    # landed in may (not evicted, not in no).
    assert demoted == 1
    assert queue.speculation_miss_count == 1
    assert queue.speculation_hit_count == 0
    assert queue.speculation_promotion_count == 1
    assert block.lifecycle_hint == "may"
    assert queue.num_free_blocks_in_pool("must") == 0
    assert queue.num_free_blocks_in_pool("may") == 1
    assert queue.num_free_blocks_in_pool("no") == 0
    # Derived metric: 0 hits / 1 promotion = 0.0 (the bad-bet case).
    assert queue.speculation_hit_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Surface assertions on env / API (kept tight; the three above are the
# spec-mandated invariant tests).
# ---------------------------------------------------------------------------


def test_speculative_ttl_env_default_30s(monkeypatch):
    """Default speculative TTL is 30s = 30 * 1e9 ns when env unset."""

    monkeypatch.delenv("WIRES_KVCACHE_SPECULATIVE_TTL_MS", raising=False)
    queue = FreeKVCacheBlockQueue([], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue._speculative_ttl_ns == 30 * 1_000_000_000


def test_speculative_ttl_env_override(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_SPECULATIVE_TTL_MS", "5000")
    queue = FreeKVCacheBlockQueue([], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue._speculative_ttl_ns == 5 * 1_000_000_000


def test_speculative_source_class_validated_in_update_block_hint():
    """update_block_hint accepts the new "speculative" source_class."""

    blk = KVCacheBlock(block_id=99)
    queue = FreeKVCacheBlockQueue([blk], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    # Unknown values still raise (regression guard).
    with pytest.raises(ValueError, match="source_class"):
        queue.update_block_hint(blk, "must", source_class="bogus")
    # Speculative is accepted; the actual stamp lands at the next
    # flush (M19 lazy hint flips).
    queue.update_block_hint(blk, "must", source_class="speculative")
    queue._flush_pending_hint_flips()
    assert blk.source_class == "speculative"
    assert blk.ttl_at_promotion_ns == queue._speculative_ttl_ns
    assert queue.speculation_promotion_count == 1


def test_speculation_hit_rate_is_none_when_no_promotions():
    """Avoid 0/0: hit_rate is None until at least one speculative
    promotion has happened."""

    queue = FreeKVCacheBlockQueue([], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue.speculation_hit_rate is None
