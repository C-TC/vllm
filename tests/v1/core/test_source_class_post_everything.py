# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""source_class Option D (post-M17/M18/M19/M20).

The original Option A targeted only ``KVCacheManager.allocate_slots``.
After M17 (LRU-position insert), M18 (per-block segment_id tuples
with min-wins), M19 (engine-side lazy hint flips), and M20
(speculative source_class + short-TTL backstop) landed, the right
design has 3 write sites + a precedence rule + a runner-confirmed
upgrade rule:

source_class semantic = which policy promoted this block to the
must pool:
- ``"speculative"``  : runner spec-prepare with speculative=True
                       (M20: ``mark_segment_blocks_speculative``).
- ``"structured"``   : runner explicit hint=must (segment_prepare
                       with speculative=False, segment_refresh
                       hint=must, M13 per-segment hint=must).
- ``"unstructured"`` : engine access-based promotion (touch-driven).

Three write sites:

Site 1 (allocate-time): ``KVCacheManager.allocate_slots`` stamps
``"structured"`` on freshly-allocated blocks whose resolved hint is
``"must"`` (request-level OR M13 per-segment after M18 min-wins).
Speculative-class blocks are preserved.

Site 2 (deferred flip): ``FreeKVCacheBlockQueue._do_real_pool_move``
resolves the effective source_class with two precedence rules:
- Speculative + structured-intent must-promotion -> UPGRADE to
  structured (counts as a hit; re-stamps TTL to the 5min backstop).
- Speculative + non-structured promotion -> preserve speculative.
Demote (must -> may/no): structured -> unstructured (drop the stamp).

Site 3 (cache hit): ``BlockPool.touch`` upgrades speculative ->
unstructured on cache hit, bumping ``speculation_hit_count`` and
re-stamping TTL to the unstructured EMA value. M20 ships this; this
test file regression-guards that semantic.

Test 7 vs Test 8: both bump ``speculation_hit_count`` (both
confirm the speculation was useful). Test 7 leaves the block as
``"structured"`` (runner-confirmed); Test 8 leaves it as
``"unstructured"`` (access-confirmed).

Tests guard the OBSERVABLE downstream value (``block.source_class``,
``block.lifecycle_hint``, ``block.ttl_at_promotion_ns``,
``speculation_hit_count``) per the
``feedback_test_guards_for_invariants`` rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.v1.core.test_segment_lifecycle_hints import (
    _FakeBlock,
    _apply_lifecycle_hints_like_manager,
)
from vllm.entrypoints.openai.chat_completion import segment_actions
from vllm.entrypoints.openai.chat_completion.segment_actions import (
    SegmentEntry,
    mark_segment_blocks_speculative,
    reset_segment_registry_for_tests,
    set_block_hint_updater,
    set_speculative_block_promoter,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    VICTIM_POLICY_WIRES_THREE_POOL,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)

pytestmark = pytest.mark.cpu_test


# ===========================================================================
# Fixtures / helpers
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_registry_and_updaters_between_tests():
    """Each test starts with a clean global registry and unbound
    block-pool updaters. Tests that need them rebind to their own
    queue (mirrors the test_speculative_source_class.py pattern)."""
    reset_segment_registry_for_tests()
    set_block_hint_updater(None)
    set_speculative_block_promoter(None)
    yield
    reset_segment_registry_for_tests()
    set_block_hint_updater(None)
    set_speculative_block_promoter(None)


def _make_request(
    *,
    lifecycle_hint: str = "may",
    segment_lifecycle_hints: dict[str, str] | None = None,
):
    """Light request stand-in for the allocate-time mirror."""
    return SimpleNamespace(
        lifecycle_hint=lifecycle_hint,
        segment_lifecycle_hints=segment_lifecycle_hints,
    )


def _three_pool_queue(blocks: list[KVCacheBlock]) -> FreeKVCacheBlockQueue:
    """Three-pool queue with TTLs long enough that the lazy sweep stays
    inert during these unit tests (we want to observe the deferred
    flip + flush semantics, not race the sweep)."""
    return FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=10**12,
        unstructured_bootstrap_ttl_ns=10**12,
        speculative_ttl_ns=10**12,
    )


def _tag_block_with_segment_id(blk: KVCacheBlock, segment_id: str) -> None:
    """Tag a block with a segment id; supports the M18 ``_segment_ids``
    tuple."""
    if hasattr(blk, "_segment_ids"):
        existing = blk._segment_ids
        if segment_id not in existing:
            blk._segment_ids = existing + (segment_id,)
    else:
        blk._segment_id = segment_id


def _seed_segment_with_block(
    segment_id: str, queue: FreeKVCacheBlockQueue, block: KVCacheBlock
) -> None:
    """Register a SegmentEntry, tag the block, wire the structured +
    speculative updaters to the queue under test."""
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


# ===========================================================================
# Site 1 (allocate-time stamp): tests 1 through 4
# ===========================================================================


def test_allocate_request_hint_must_stamps_structured():
    """Site 1, request-level branch: ``lifecycle_hint = "must"`` flips
    every freshly-allocated block's ``source_class`` from the default
    ``"unstructured"`` to ``"structured"``."""
    req = _make_request(lifecycle_hint="must")
    blocks = [[_FakeBlock(), _FakeBlock(segment_id="seg-x")]]
    for blk in blocks[0]:
        # Pre-condition: default unstructured.
        assert blk.source_class == "unstructured"
        assert blk.lifecycle_hint == "may"

    _apply_lifecycle_hints_like_manager(req, blocks)

    for blk in blocks[0]:
        assert blk.lifecycle_hint == "must"
        assert blk.source_class == "structured"


def test_allocate_per_segment_hint_must_stamps_structured():
    """Site 1, per-segment branch (M13 + M18 min-wins): a per-segment
    override resolving to ``"must"`` stamps ``source_class =
    "structured"`` even when the request-level hint is the default
    ``"may"``. Untagged blocks keep their default class."""
    req = _make_request(
        lifecycle_hint="may",
        segment_lifecycle_hints={"seg-a": "must"},
    )
    tagged = _FakeBlock(segment_id="seg-a")
    untagged = _FakeBlock()
    blocks = [[tagged, untagged]]

    _apply_lifecycle_hints_like_manager(req, blocks)

    assert tagged.lifecycle_hint == "must"
    assert tagged.source_class == "structured"
    # Untagged stays at default (no per-segment match, no request-level
    # promotion).
    assert untagged.lifecycle_hint == "may"
    assert untagged.source_class == "unstructured"


def test_allocate_hint_may_keeps_unstructured():
    """Site 1 negative case: ``hint == "may"`` (or ``"no"``) does NOT
    promote ``source_class``. The default ``"unstructured"`` survives.
    The structured-TTL stamp is only meaningful for ``"must"`` hints."""
    req = _make_request(lifecycle_hint="may")
    blocks = [[_FakeBlock(), _FakeBlock(segment_id="seg-y")]]

    _apply_lifecycle_hints_like_manager(req, blocks)

    for blk in blocks[0]:
        assert blk.lifecycle_hint == "may"
        assert blk.source_class == "unstructured"


def test_allocate_speculative_preserved_against_structured_overwrite():
    """Site 1 + M20 precedence: a block already classed ``"speculative"``
    is NOT overwritten by the allocate-time structured stamp. The
    speculative short-TTL backstop (default 30s) is the whole point of
    M20 and must survive a hint=must allocate."""
    req = _make_request(lifecycle_hint="must")
    speculative_blk = _FakeBlock()
    speculative_blk.source_class = "speculative"  # set by M20 promoter
    structured_blk = _FakeBlock()  # default unstructured
    blocks = [[speculative_blk, structured_blk]]

    _apply_lifecycle_hints_like_manager(req, blocks)

    # Both blocks get lifecycle_hint=must (the request-level pass
    # touches every block uniformly).
    assert speculative_blk.lifecycle_hint == "must"
    assert structured_blk.lifecycle_hint == "must"
    # Speculative survives; only the default-unstructured block is
    # promoted to structured.
    assert speculative_blk.source_class == "speculative"
    assert structured_blk.source_class == "structured"


# ===========================================================================
# Site 2 (deferred flip stamp): tests 5 and 6
# ===========================================================================


def test_segment_refresh_hint_must_via_lazy_flush_stamps_structured():
    """Site 2, structured stamp: a deferred flip into the must pool
    via ``update_block_hint(must, source_class="structured")`` on a
    default-class block stamps ``source_class = "structured"`` AND
    snaps in the 300s structured TTL (NOT the 60s unstructured
    bootstrap)."""
    block = KVCacheBlock(block_id=11)  # default may + unstructured
    queue = _three_pool_queue([block])
    # Sanity: block sits in the may pool with the default class.
    assert queue._pool_of[block.block_id] == "may"
    assert block.source_class == "unstructured"

    # Enqueue a structured may->must flip (M19: deferred).
    queue.update_block_hint(block, "must", source_class="structured")

    # Eager: lifecycle_hint flipped, but source_class is set ONLY on
    # the actual stamp (which fires inside _do_real_pool_move).
    assert block.lifecycle_hint == "must"
    assert queue._pool_of[block.block_id] == "may"
    assert block.block_id in queue._pending_hint_flips

    # Force-apply the deferred flip.
    queue._flush_pending_hint_flips()

    # Site 2 observable: block landed in must pool, source_class
    # stamped to structured, TTL set to the 5min structured backstop.
    assert queue._pool_of[block.block_id] == "must"
    assert block.source_class == "structured"
    assert block.ttl_at_promotion_ns == queue._structured_ttl_ns
    # Counters: no speculation activity on this path.
    assert queue.speculation_promotion_count == 0
    assert queue.speculation_hit_count == 0


def test_segment_refresh_demote_clears_structured():
    """Site 2, demote branch: a block currently classed ``"structured"``
    in the must pool, when flipped to ``"may"`` via update_block_hint,
    has ``source_class`` reclassified to ``"unstructured"``. The
    structured class is meaningful only while the block sits in must;
    demoting drops the stamp so any future re-promotion has to earn
    the structured class anew."""
    block = KVCacheBlock(block_id=13)
    queue = _three_pool_queue([block])

    # Promote to structured + flush so the block is concretely in must
    # with source_class=structured.
    queue.update_block_hint(block, "must", source_class="structured")
    queue._flush_pending_hint_flips()
    assert block.source_class == "structured"
    assert queue._pool_of[block.block_id] == "must"

    # Now demote may; flush deferred move.
    queue.update_block_hint(block, "may", source_class="structured")
    queue._flush_pending_hint_flips()

    # Site 2 demote observable: pool moved to may, structured stamp
    # dropped (back to unstructured for any future re-promotion).
    assert block.lifecycle_hint == "may"
    assert queue._pool_of[block.block_id] == "may"
    assert block.source_class == "unstructured"
    # Counters: no speculation activity on this path.
    assert queue.speculation_promotion_count == 0
    assert queue.speculation_hit_count == 0
    assert queue.speculation_miss_count == 0


# ===========================================================================
# Tests 7 + 8: speculation confirmation by runner intent vs by access
# ===========================================================================


def test_speculative_upgrade_to_structured_on_runner_hint(monkeypatch):
    """Site 2 runner-confirmed upgrade: a block currently classed
    ``"speculative"`` (via ``mark_segment_blocks_speculative``), when a
    later runner hint=must lands as a deferred
    ``update_block_hint(must, source_class="structured")``, UPGRADES
    to ``"structured"`` + re-stamps TTL to the 5min structured
    backstop + bumps ``speculation_hit_count``.

    Rationale: speculation confirmed by runner intent counts as a hit,
    not a miss; symmetric to the touch-driven hit handled by
    ``BlockPool.touch`` (Site 3, test 8 below). The difference is the
    resulting class: structured (runner-confirmed) vs unstructured
    (access-confirmed)."""
    monkeypatch.delenv("WIRES_KVCACHE_SPECULATIVE_TTL_MS", raising=False)
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    queue = pool.free_block_queue
    block = pool.blocks[1]  # block 0 is the null block
    assert block.lifecycle_hint == "may"

    _seed_segment_with_block("seg-spec-runner", queue, block)

    # Step 1: speculative promotion + flush. Block is now in must
    # with source_class="speculative" + the 30s spec TTL backstop.
    mark_segment_blocks_speculative("seg-spec-runner")
    queue._flush_pending_hint_flips()
    assert block.lifecycle_hint == "must"
    assert block.source_class == "speculative"
    assert block.ttl_at_promotion_ns == queue._speculative_ttl_ns
    assert queue.speculation_promotion_count == 1
    assert queue.speculation_hit_count == 0
    speculative_promoted_at_ns = block.last_promoted_ns
    assert speculative_promoted_at_ns > 0

    # Step 2: runner-confirmed-must lands. The speculative block's
    # lifecycle_hint is already "must", so this exercises the
    # same-pool re-stamp path inside Site 2 (the upgrade short-circuit
    # in ``update_block_hint`` enqueues a pending flip; flush applies
    # the speculation_hit_count + structured TTL re-stamp).
    queue.update_block_hint(block, "must", source_class="structured")
    queue._flush_pending_hint_flips()

    # Site 2 upgrade observable:
    # - Class flipped speculative -> structured.
    assert block.source_class == "structured"
    # - TTL re-stamped to the 5min structured backstop (NOT the 30s
    #   speculative backstop it was wearing before the upgrade).
    assert block.ttl_at_promotion_ns == queue._structured_ttl_ns
    assert block.ttl_at_promotion_ns != queue._speculative_ttl_ns
    # - last_promoted_ns refreshed (a re-stamp event, not a stale
    #   timestamp from the original speculative promotion).
    assert block.last_promoted_ns >= speculative_promoted_at_ns
    # - Speculation hit counted (both confirmation paths bump this
    #   counter; see test 8 for the access-driven path).
    assert queue.speculation_hit_count == 1
    # - Promotion count unchanged (the original speculative promotion
    #   is already accounted; the upgrade doesn't double-count).
    assert queue.speculation_promotion_count == 1
    # - Miss count unchanged (this is a HIT, not a miss).
    assert queue.speculation_miss_count == 0
    # - Block stayed in the must pool throughout.
    assert queue._pool_of[block.block_id] == "must"


def test_speculative_upgrade_to_unstructured_on_touch(monkeypatch):
    """Site 3 (M20) regression guard: a block currently classed
    ``"speculative"``, when touched (cache hit), upgrades to
    ``"unstructured"`` + re-stamps TTL to the unstructured EMA value
    + bumps ``speculation_hit_count``.

    Companion to test 7: same hit-counter semantics, different
    resulting class. Access confirmation -> unstructured (the
    block joins the access-driven population). Runner confirmation
    (test 7) -> structured (the runner has explicit must intent)."""
    monkeypatch.delenv("WIRES_KVCACHE_SPECULATIVE_TTL_MS", raising=False)
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    queue = pool.free_block_queue
    block = pool.blocks[1]
    assert block.lifecycle_hint == "may"

    _seed_segment_with_block("seg-spec-touch", queue, block)

    # Speculative promotion + flush.
    mark_segment_blocks_speculative("seg-spec-touch")
    queue._flush_pending_hint_flips()
    assert block.source_class == "speculative"
    assert block.ttl_at_promotion_ns == queue._speculative_ttl_ns
    assert queue.speculation_hit_count == 0

    # Drive a cache hit (Site 3).
    pool.touch([block])

    # Site 3 observable: source_class upgraded to unstructured, TTL
    # re-stamped to the unstructured EMA value, hit count bumped.
    assert block.source_class == "unstructured"
    assert block.ttl_at_promotion_ns == queue._unstructured_ttl_at_promotion_ns()
    assert block.ttl_at_promotion_ns != queue._speculative_ttl_ns
    assert queue.speculation_hit_count == 1
    assert queue.speculation_promotion_count == 1
    assert queue.speculation_miss_count == 0
    # Derived metric reaches 1.0 after one hit on one promotion.
    assert queue.speculation_hit_rate == pytest.approx(1.0)
