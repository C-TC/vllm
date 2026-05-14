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

Test 7 vs Test 8: both confirm the speculation was useful, but
they bump SEPARATE counters (paper §3.6 fairness: the strict
M20 hit metric tracks access-driven reuse only). Test 7 (runner
confirmation, late-arrival hint=must) bumps
``speculation_runner_confirm_count`` and leaves the block as
``"structured"``. Test 8 (access confirmation, BlockPool.touch)
bumps ``speculation_hit_count`` and leaves the block as
``"unstructured"``. The combined "useful outcome" rate is
exposed via ``speculation_useful_rate``.

Tests guard the OBSERVABLE downstream value (``block.source_class``,
``block.lifecycle_hint``, ``block.ttl_at_promotion_ns``,
``speculation_hit_count``, ``speculation_runner_confirm_count``)
per the ``feedback_test_guards_for_invariants`` rule.
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
    backstop + bumps ``speculation_runner_confirm_count`` (NOT the
    strict ``speculation_hit_count``, which stays at 0 because no
    access-touch happened on this path).

    Rationale: speculation confirmed by a late-arriving runner hint
    is a useful outcome, but it is NOT what paper §3.6 measures
    (access-driven reuse predictive quality). The two counters are
    held separate so the strict M20 metric stays clean; the union
    is observable via ``speculation_useful_rate``."""
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
    assert queue.speculation_runner_confirm_count == 0
    speculative_promoted_at_ns = block.last_promoted_ns
    assert speculative_promoted_at_ns > 0

    # Step 2: runner-confirmed-must lands. The speculative block's
    # lifecycle_hint is already "must", so this exercises the
    # same-pool re-stamp path inside Site 2 (the upgrade short-circuit
    # in ``update_block_hint`` enqueues a pending flip; flush applies
    # the speculation_runner_confirm_count bump + structured TTL
    # re-stamp).
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
    # - Runner-confirm counter bumped; STRICT speculation_hit_count
    #   stays at 0 (no access-touch happened, so paper §3.6's metric
    #   correctly excludes this path).
    assert queue.speculation_runner_confirm_count == 1
    assert queue.speculation_hit_count == 0
    # - Promotion count unchanged (the original speculative promotion
    #   is already accounted; the upgrade doesn't double-count).
    assert queue.speculation_promotion_count == 1
    # - Miss count unchanged (this is a useful outcome, not a miss).
    assert queue.speculation_miss_count == 0
    # - Strict hit rate stays at 0/1 = 0; useful rate captures the
    #   runner-confirm at 1/1 = 1.
    assert queue.speculation_hit_rate == pytest.approx(0.0)
    assert queue.speculation_useful_rate == pytest.approx(1.0)
    # - Block stayed in the must pool throughout.
    assert queue._pool_of[block.block_id] == "must"


def test_speculative_upgrade_to_unstructured_on_touch(monkeypatch):
    """Site 3 (M20) STRICT regression guard: a block currently classed
    ``"speculative"``, when touched (cache hit), upgrades to
    ``"unstructured"`` + re-stamps TTL to the unstructured EMA value
    + bumps ``speculation_hit_count``. The runner-confirm counter
    stays at 0 because no runner hint=must arrived on this path.

    Companion to test 7: both confirm speculation, but they bump
    SEPARATE counters. Access confirmation -> unstructured + bumps
    the strict M20 hit counter (paper §3.6). Runner confirmation
    (test 7) -> structured + bumps the runner-confirm counter."""
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
    assert queue.speculation_runner_confirm_count == 0

    # Drive a cache hit (Site 3).
    pool.touch([block])

    # Site 3 observable: source_class upgraded to unstructured, TTL
    # re-stamped to the unstructured EMA value, STRICT hit count
    # bumped (paper §3.6 metric); runner-confirm stays at 0.
    assert block.source_class == "unstructured"
    assert block.ttl_at_promotion_ns == queue._unstructured_ttl_at_promotion_ns()
    assert block.ttl_at_promotion_ns != queue._speculative_ttl_ns
    assert queue.speculation_hit_count == 1
    assert queue.speculation_runner_confirm_count == 0
    assert queue.speculation_promotion_count == 1
    assert queue.speculation_miss_count == 0
    # Strict hit rate reaches 1.0 after one touch-driven hit on one
    # promotion; useful rate matches because no runner-confirm
    # happened.
    assert queue.speculation_hit_rate == pytest.approx(1.0)
    assert queue.speculation_useful_rate == pytest.approx(1.0)


# ===========================================================================
# Test 9: split-counter pin (paper §3.6 fairness, strict M20 vs Option D)
# ===========================================================================


def test_split_counters_pin_paper_metric_separation(monkeypatch):
    """Drive both confirmation paths in the SAME queue and pin the
    counter separation: one access-touch (Site 3, BlockPool.touch) +
    one runner-confirm (Site 2, deferred update_block_hint flush).
    The split is what protects paper §3.6 from the conflated bump
    that the original M20+Option D code shipped with.

    Expected: each counter sees exactly one event; the strict
    ``speculation_hit_rate`` is 1/2 (touch-only) and the union
    ``speculation_useful_rate`` is 2/2 (touch + runner-confirm).
    """
    monkeypatch.delenv("WIRES_KVCACHE_SPECULATIVE_TTL_MS", raising=False)
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    queue = pool.free_block_queue
    touch_block = pool.blocks[1]
    runner_block = pool.blocks[2]
    assert touch_block.lifecycle_hint == "may"
    assert runner_block.lifecycle_hint == "may"

    # Two segments, each with one speculative block; both seeded into
    # the same queue (the helper rebinds the global updaters per call,
    # which is fine because both segments end up driving the same
    # queue object).
    _seed_segment_with_block("seg-touch", queue, touch_block)
    mark_segment_blocks_speculative("seg-touch")
    queue._flush_pending_hint_flips()

    _seed_segment_with_block("seg-runner", queue, runner_block)
    mark_segment_blocks_speculative("seg-runner")
    queue._flush_pending_hint_flips()

    # After two speculative promotions: 2 promotions, 0 hits, 0
    # runner-confirms.
    assert queue.speculation_promotion_count == 2
    assert queue.speculation_hit_count == 0
    assert queue.speculation_runner_confirm_count == 0
    assert touch_block.source_class == "speculative"
    assert runner_block.source_class == "speculative"

    # Path 1: touch-driven hit on touch_block (Site 3).
    pool.touch([touch_block])
    assert touch_block.source_class == "unstructured"
    assert queue.speculation_hit_count == 1
    assert queue.speculation_runner_confirm_count == 0

    # Path 2: runner-confirmed-must on runner_block (Site 2 same-pool
    # re-stamp).
    queue.update_block_hint(runner_block, "must", source_class="structured")
    queue._flush_pending_hint_flips()
    assert runner_block.source_class == "structured"

    # Split pinned: each counter sees exactly one event of its kind.
    assert queue.speculation_hit_count == 1
    assert queue.speculation_runner_confirm_count == 1
    assert queue.speculation_promotion_count == 2
    assert queue.speculation_miss_count == 0

    # Derived rates:
    # - Strict speculation_hit_rate (paper §3.6) = touch hits / promotions
    #   = 1/2.
    assert queue.speculation_hit_rate == pytest.approx(0.5)
    # - Useful rate = (touch + runner-confirm) / promotions = 2/2.
    assert queue.speculation_useful_rate == pytest.approx(1.0)

    # Snapshot surface mirrors the in-memory counters.
    snap = queue.cache_stats_snapshot()
    assert snap["speculation_hit_count"] == 1
    assert snap["speculation_runner_confirm_count"] == 1
    assert snap["speculation_promotion_count"] == 2
    assert snap["speculation_hit_rate"] == pytest.approx(0.5)
    assert snap["speculation_useful_rate"] == pytest.approx(1.0)
