# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""source_class Option D (post-M17/M18/M19).

The original Option A targeted only ``KVCacheManager.allocate_slots``.
After M17 (LRU-position insert), M18 (per-block segment_id tuples
with min-wins), and M19 (engine-side lazy hint flips) landed, the
right design has 2 write sites:

source_class semantic = which policy promoted this block to the
must pool:
- ``"structured"``   : runner explicit hint=must (segment_prepare,
                       segment_refresh hint=must, M13 per-segment
                       hint=must).
- ``"unstructured"`` : engine access-based promotion (touch-driven).

Two write sites:

Site 1 (allocate-time): ``KVCacheManager.allocate_slots`` stamps
``"structured"`` on freshly-allocated blocks whose resolved hint is
``"must"`` (request-level OR M13 per-segment after M18 min-wins).

Site 2 (deferred flip): ``FreeKVCacheBlockQueue._do_real_pool_move``
stamps the incoming ``source_class`` on must-promotion. Demote
(must -> may/no): structured -> unstructured (drop the stamp).

Tests guard the OBSERVABLE downstream value (``block.source_class``,
``block.lifecycle_hint``, ``block.ttl_at_promotion_ns``) per the
``feedback_test_guards_for_invariants`` rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.v1.core.test_segment_lifecycle_hints import (
    _FakeBlock,
    _apply_lifecycle_hints_like_manager,
)
from vllm.entrypoints.openai.chat_completion.segment_actions import (
    reset_segment_registry_for_tests,
    set_block_hint_updater,
)
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
    block-pool updaters."""
    reset_segment_registry_for_tests()
    set_block_hint_updater(None)
    yield
    reset_segment_registry_for_tests()
    set_block_hint_updater(None)


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
    )


# ===========================================================================
# Site 1 (allocate-time stamp): tests 1 through 3
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
