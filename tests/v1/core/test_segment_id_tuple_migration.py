# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M18 (paper/CODE_MISMATCH_NOTES.md) — KVCacheBlock._segment_id
migration from ``str | None`` to ``_segment_ids: tuple[str, ...]``.

Setup
-----
A physical KV block (16 tokens by default in vLLM) may carry tokens
from multiple distinct segments when segment boundaries don't align
with block boundaries. Doc 32 §2.3 specs block-boundary "min wins":
the block's effective lifecycle hint is the MIN-priority hint
(``no`` < ``may`` < ``must``) across all overlapping segments so
the most-conservative caller's intent wins.

Before M18, ``KVCacheBlock._segment_id`` was a single ``str | None``
slot and ``tag_blocks_with_segment_id`` overwrote it
unconditionally. The previous segment's reverse-index entry
silently became stale, breaking the min-wins invariant (and
blocking M5 sub-gap (b) "min wins" recompute). M18 makes the field
a tuple, ``tag_blocks_with_segment_id`` appends-or-updates, and
``update_block_hints`` recomputes the per-block effective hint
across all overlapping segments.

These tests lock the behaviour:

1. Single-tag fast path — tag once with one segment_id; assert the
   block's tuple is ``(segment_id,)`` and the hint update is the
   single-source value.
2. Two overlapping segments — tag with A (must) then B (may);
   assert tuple is ``("A","B")`` and the per-block effective hint
   is ``may`` (min wins).
3. Three segments min wins — tag with (must, may, no); assert the
   per-block effective hint is ``no``.
4. Idempotent re-tag — tag with the same segment_id twice; assert
   the tuple stays ``(segment_id,)`` (no duplication).
5. Reverse-index correctness — both segment_ids' reverse-map
   entries contain the block.
6. Backward-compat property — ``block._segment_id`` returns the
   primary id (``_segment_ids[0]``) for callers that haven't
   migrated.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.openai.chat_completion.segment_actions import (
    SegmentEntry,
    SegmentRegistry,
    reset_segment_registry_for_tests,
    set_block_hint_updater,
    tag_blocks_with_segment_id,
    update_segment_lifecycle_hint,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry_between_tests():
    """Each test gets a clean registry; the production
    ``_block_hint_updater`` slot is shared module-level state and
    must also be cleared between tests so stale callbacks from a
    sibling test don't fire on this test's blocks."""

    reset_segment_registry_for_tests()
    set_block_hint_updater(None)
    yield
    reset_segment_registry_for_tests()
    set_block_hint_updater(None)


def _seed_entry(registry: SegmentRegistry, segment_id: str) -> None:
    """Register a SegmentEntry under ``segment_id`` directly via the
    secondary index so ``update_block_hints`` finds it (``apply_per_
    instance_hint`` requires an entry to record the runner / instance
    contribution; the min-wins recompute reads each overlapping
    segment's ``compute_effective_hint`` from its registry entry).
    """

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


# ---------------------------------------------------------------------------
# 1. Single-segment fast path
# ---------------------------------------------------------------------------


def test_single_segment_fast_path_tag_and_hint_update():
    """One tag => ``_segment_ids == (segment_id,)`` and the hint
    update is identical to the pre-M18 single-source path."""

    block = KVCacheBlock(block_id=0)  # default lifecycle_hint="may"
    assert block._segment_ids == ()
    tag_blocks_with_segment_id([block], "seg-A")
    assert block._segment_ids == ("seg-A",)

    # Run the runner-driven flip via the M3 entry point. With one
    # caller the effective hint = the call's hint = "must".
    from vllm.entrypoints.openai.chat_completion import segment_actions

    _seed_entry(segment_actions._registry, "seg-A")
    result = update_segment_lifecycle_hint(
        "seg-A", "must", runner_id="r1", instance_id="i1"
    )
    assert result["updated"] == 1
    assert result["effective_hint"] == "must"
    # Single-tag fast path: per-block effective hint = caller's
    # effective hint, with no min-wins recompute.
    assert block.lifecycle_hint == "must"


# ---------------------------------------------------------------------------
# 2. Two overlapping segments — different hints
# ---------------------------------------------------------------------------


def test_two_overlapping_segments_min_wins_may_under_must():
    """Block tagged with seg-A (must) and seg-B (may); the per-block
    effective hint is ``may`` (min priority wins)."""

    block = KVCacheBlock(block_id=0)
    tag_blocks_with_segment_id([block], "seg-A")
    tag_blocks_with_segment_id([block], "seg-B")
    assert block._segment_ids == ("seg-A", "seg-B")

    from vllm.entrypoints.openai.chat_completion import segment_actions

    registry = segment_actions._registry
    _seed_entry(registry, "seg-A")
    _seed_entry(registry, "seg-B")

    # First flip seg-A to "must"; with seg-B's effective still
    # default "may" (no live referencer), per-block min(must, may)=may.
    update_segment_lifecycle_hint(
        "seg-A", "must", runner_id="r1", instance_id="i1"
    )
    assert block.lifecycle_hint == "may"

    # Now flip seg-B to "may" explicitly via a real live referencer;
    # block stays "may" (min(must, may) = may).
    update_segment_lifecycle_hint(
        "seg-B", "may", runner_id="r2", instance_id="i2"
    )
    assert block.lifecycle_hint == "may"


# ---------------------------------------------------------------------------
# 3. Three segments — min wins drops to "no"
# ---------------------------------------------------------------------------


def test_three_overlapping_segments_min_wins_no_under_must_may_no():
    """Block tagged with three segments; one demoted to ``no`` (via
    literal_no=True) drops the per-block effective hint to ``no``
    (min priority across all overlapping segments)."""

    block = KVCacheBlock(block_id=0)
    tag_blocks_with_segment_id([block], "seg-must")
    tag_blocks_with_segment_id([block], "seg-may")
    tag_blocks_with_segment_id([block], "seg-no")
    assert block._segment_ids == ("seg-must", "seg-may", "seg-no")

    from vllm.entrypoints.openai.chat_completion import segment_actions

    registry = segment_actions._registry
    for seg in ("seg-must", "seg-may", "seg-no"):
        _seed_entry(registry, seg)

    # Establish each segment's effective hint via a live caller.
    update_segment_lifecycle_hint(
        "seg-must", "must", runner_id="r1", instance_id="i1"
    )
    update_segment_lifecycle_hint(
        "seg-may", "may", runner_id="r1", instance_id="i2"
    )
    # Use literal_no so the contribution is recorded as "no" rather
    # than removing the slot.
    update_segment_lifecycle_hint(
        "seg-no",
        "no",
        runner_id="r1",
        instance_id="i3",
        literal_no=True,
    )

    # Per-block min(must, may, no) = no.
    assert block.lifecycle_hint == "no"


# ---------------------------------------------------------------------------
# 4. Idempotent re-tag
# ---------------------------------------------------------------------------


def test_idempotent_re_tag_does_not_duplicate_entry():
    """Tagging the same segment_id twice on a block leaves the tuple
    as ``(segment_id,)`` (no duplication) and the reverse-index
    holds the block exactly once."""

    block = KVCacheBlock(block_id=0)
    tag_blocks_with_segment_id([block], "seg-A")
    tag_blocks_with_segment_id([block], "seg-A")  # second tag
    assert block._segment_ids == ("seg-A",)

    from vllm.entrypoints.openai.chat_completion import segment_actions

    rev = segment_actions._registry._blocks_by_segment_id.get("seg-A", [])
    # ``register_block_for_segment`` de-dups via identity check, so
    # the block appears once even after two tag calls.
    assert len(rev) == 1
    assert rev[0] is block


# ---------------------------------------------------------------------------
# 5. Reverse-index correctness on multi-tag
# ---------------------------------------------------------------------------


def test_reverse_index_holds_block_under_each_overlapping_segment():
    """After tagging a block with seg-A then seg-B, both entries in
    the reverse-index dict contain the block. Without M18 the
    second tag would have overwritten the first and seg-A's entry
    would be stale."""

    block = KVCacheBlock(block_id=0)
    tag_blocks_with_segment_id([block], "seg-A")
    tag_blocks_with_segment_id([block], "seg-B")

    from vllm.entrypoints.openai.chat_completion import segment_actions

    rev_a = segment_actions._registry._blocks_by_segment_id.get("seg-A", [])
    rev_b = segment_actions._registry._blocks_by_segment_id.get("seg-B", [])
    assert len(rev_a) == 1 and rev_a[0] is block, (
        "seg-A reverse-index entry was stale after seg-B tag (M18 regression)"
    )
    assert len(rev_b) == 1 and rev_b[0] is block, (
        "seg-B reverse-index entry missing"
    )


# ---------------------------------------------------------------------------
# 6. Backward-compat property
# ---------------------------------------------------------------------------


def test_backward_compat_segment_id_property_returns_primary():
    """``block._segment_id`` returns ``_segment_ids[0]`` for callers
    that haven't migrated to the new attribute yet. Empty tuple
    => None."""

    block = KVCacheBlock(block_id=0)
    assert block._segment_id is None
    tag_blocks_with_segment_id([block], "seg-primary")
    assert block._segment_id == "seg-primary"
    tag_blocks_with_segment_id([block], "seg-secondary")
    # Primary stays as the first-tagged id; secondary tag is appended.
    assert block._segment_id == "seg-primary"
    assert block._segment_ids == ("seg-primary", "seg-secondary")


# ---------------------------------------------------------------------------
# 7. reset_hash clears the segment tag tuple
# ---------------------------------------------------------------------------


def test_reset_hash_clears_segment_ids_for_recycled_slot():
    """When a block slot is recycled (``reset_hash``), its
    ``_segment_ids`` tuple must be cleared so the next content's
    tag does not union against stale segment ids."""

    block = KVCacheBlock(block_id=0)
    tag_blocks_with_segment_id([block], "seg-A")
    tag_blocks_with_segment_id([block], "seg-B")
    assert block._segment_ids == ("seg-A", "seg-B")

    block.reset_hash()
    assert block._segment_ids == ()
    # Backward-compat property follows the cleared tuple.
    assert block._segment_id is None
