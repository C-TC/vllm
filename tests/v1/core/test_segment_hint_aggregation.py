# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3 — engine-side cross-instance hint aggregation.

paper/CODE_MISMATCH_NOTES.md M3 entry: paper §3 + §3.5 say the engine's
cache policy reconciles per-segment hints across all sources sharing
the cache (multiple workflow instances on one runner, instances on
different runners, unrelated engine clients). Pre-M3 the engine flipped
``KVCacheBlock.lifecycle_hint`` last-write-wins on every
``segment_lifecycle_update`` call, regardless of who else still
referenced the segment. M3 routes hint updates through a per-segment
``live_hints_by_caller`` multiset and re-derives the effective hint
as ``max(must, may, no)`` over all live referencers.

Caller identity is the ``(runner_id, instance_id)`` tuple, NOT the
``instance_id`` alone: independent runners (different teams /
processes / orgs) do not coordinate the ``instance_id`` namespace,
so a collision on ``instance_id`` alone would silently merge
their entries — and a "no" from one would yank the other's
contribution. Keying on the tuple isolates each runner's
namespace.

These tests lock the contract:

1. Two instances with hints ``must`` + ``may`` resolve to ``must``.
2. The must-instance demoting itself to ``may`` keeps blocks at the
   max over the remaining live set (``may`` if that's all that's left).
3. Both instances dropping their contribution defaults the effective
   hint to ``may`` (empty live set).
4. ``literal_no=True`` records ``no`` as a contribution rather than
   removing the slot — useful for the loop-exit "force eviction" path.
5. ``release_segment_instance_contribution`` removes a slot
   explicitly (instance completion / crash / cancel).
6. Legacy callers without an ``instance_id`` fall through to the
   shared synthetic ``_anon`` slot — last-write-wins behavior is
   preserved.
7. Concurrent updates are serialized through the registry lock —
   final state is always one of the per-thread orderings, never a
   torn read.
8. Two runners with identical ``instance_id`` are tracked as
   independent entries; runner A releasing its contribution does
   NOT drop runner B's.
9. The HTTP route's default ``runner_id`` fallback ("default") keeps
   the pre-M3 single-tenant wire schema working unchanged.
10. Telemetry exposes both ``runner_id`` and ``instance_id`` per
    live entry as JSON-friendly records.
"""

from __future__ import annotations

import threading

import pytest

from vllm.entrypoints.openai.chat_completion.segment_actions import (
    SegmentEntry,
    SegmentRegistry,
    release_segment_instance_contribution,
    reset_segment_registry_for_tests,
    tag_blocks_with_segment_id,
    update_segment_lifecycle_hint,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeBlock:
    """Minimal stand-in for KVCacheBlock with the fields the registry
    needs to flip ``lifecycle_hint`` and stale-detect via
    ``_segment_id``. Same shape as the M13 fake in
    ``test_segment_lifecycle_hints.py``."""

    __slots__ = ("lifecycle_hint", "_segment_id", "block_id")

    def __init__(self, block_id: int, segment_id: str | None = None):
        self.block_id = block_id
        self.lifecycle_hint = "may"
        self._segment_id = segment_id


def _seed_segment(
    registry: SegmentRegistry,
    *,
    segment_id: str = "seg-shared",
    block_count: int = 2,
) -> list[_FakeBlock]:
    """Register a SegmentEntry + tagged blocks via the public APIs.

    Mirrors what the segment_prepare prefill helper does at runtime,
    but assembles the state directly from the test side so we don't
    have to spin up a real chat handler / engine."""

    registry._entries_by_segment_id[segment_id] = SegmentEntry(
        action_id="act-" + segment_id,
        action_kind="segment_prepare",
        segment_id=segment_id,
        family_id="fam-" + segment_id,
        parent_segment_id=None,
        token_hash="hash-" + segment_id,
        model="dummy-model",
        lifecycle_status="accepted",
        expected_consumers=2,
    )
    blocks = [_FakeBlock(i, segment_id=segment_id) for i in range(block_count)]
    for b in blocks:
        registry.register_block_for_segment(segment_id, b)
    return blocks


@pytest.fixture(autouse=True)
def _reset_registry_between_tests():
    reset_segment_registry_for_tests()
    yield
    reset_segment_registry_for_tests()


# ---------------------------------------------------------------------------
# SegmentEntry pure aggregation logic (no block flips)
# ---------------------------------------------------------------------------


def _new_entry(segment_id: str = "s") -> SegmentEntry:
    return SegmentEntry(
        action_id="a",
        action_kind="segment_prepare",
        segment_id=segment_id,
        family_id="f",
        parent_segment_id=None,
        token_hash="t",
        model="m",
        lifecycle_status="accepted",
        expected_consumers=0,
    )


def test_compute_effective_hint_empty_defaults_to_may():
    entry = _new_entry()
    assert entry.compute_effective_hint() == "may"


def test_compute_effective_hint_max_priority():
    """Priority order: must > may > no."""

    entry = _new_entry()
    entry.live_hints_by_caller = {
        ("default", "i1"): "may",
        ("default", "i2"): "no",
    }
    assert entry.compute_effective_hint() == "may"

    entry.live_hints_by_caller = {
        ("default", "i1"): "must",
        ("default", "i2"): "may",
    }
    assert entry.compute_effective_hint() == "must"

    entry.live_hints_by_caller = {
        ("default", "i1"): "no",
        ("default", "i2"): "no",
    }
    assert entry.compute_effective_hint() == "no"

    entry.live_hints_by_caller = {
        ("default", "i1"): "must",
        ("default", "i2"): "must",
    }
    assert entry.compute_effective_hint() == "must"


def test_record_instance_hint_no_removes_slot_by_default():
    """``new_hint=no`` without ``literal_no`` means "I'm done": drop
    the slot rather than contributing a literal ``no``."""

    entry = _new_entry()
    entry.record_instance_hint("default", "i1", "must")
    entry.record_instance_hint("default", "i2", "may")
    assert entry.compute_effective_hint() == "must"

    # i1 demotes by signaling "no" — its slot is dropped.
    entry.record_instance_hint("default", "i1", "no")
    assert ("default", "i1") not in entry.live_hints_by_caller
    # Only i2=may remains.
    assert entry.compute_effective_hint() == "may"

    # i2 also demotes — live set empty, default may.
    entry.record_instance_hint("default", "i2", "no")
    assert entry.live_hints_by_caller == {}
    assert entry.compute_effective_hint() == "may"


def test_set_instance_hint_literal_records_no_as_contribution():
    """``set_instance_hint_literal`` is the loop-exit "force evict"
    path: a literal ``no`` STAYS in the live set."""

    entry = _new_entry()
    entry.set_instance_hint_literal("default", "i1", "no")
    entry.set_instance_hint_literal("default", "i2", "may")
    # max(no, may) = may
    assert entry.compute_effective_hint() == "may"

    entry.set_instance_hint_literal("default", "i2", "no")
    # All literal no => effective no (no-one wants the block alive).
    assert entry.live_hints_by_caller == {
        ("default", "i1"): "no",
        ("default", "i2"): "no",
    }
    assert entry.compute_effective_hint() == "no"


def test_release_instance_contribution_idempotent():
    entry = _new_entry()
    entry.record_instance_hint("default", "i1", "must")
    entry.release_instance_contribution("default", "i1")
    assert entry.live_hints_by_caller == {}
    # Releasing again is a no-op, NOT an error.
    entry.release_instance_contribution("default", "i1")
    entry.release_instance_contribution("default", "nonexistent")
    assert entry.compute_effective_hint() == "may"


def test_anonymous_slot_collapses_to_last_write_wins():
    """Legacy callers without runner_id + instance_id share the
    ``("default", "_anon")`` slot."""

    entry = _new_entry()
    entry.record_instance_hint(None, None, "must")
    entry.record_instance_hint(None, None, "may")  # last-write-wins
    assert entry.live_hints_by_caller == {("default", "_anon"): "may"}
    assert entry.compute_effective_hint() == "may"

    entry.record_instance_hint("", "", "must")  # empty strings also collapse
    assert entry.live_hints_by_caller == {("default", "_anon"): "must"}


def test_two_runners_same_instance_id_are_independent_slots():
    """Cross-runner isolation: runner A's instance "alice-123" and
    runner B's instance "alice-123" must NOT collide. Each has its
    own slot in ``live_hints_by_caller`` keyed on the
    ``(runner_id, instance_id)`` tuple."""

    entry = _new_entry()
    entry.record_instance_hint("runner-A", "alice-123", "must")
    entry.record_instance_hint("runner-B", "alice-123", "may")
    # Both slots live, keyed on (runner_id, instance_id).
    assert entry.live_hints_by_caller == {
        ("runner-A", "alice-123"): "must",
        ("runner-B", "alice-123"): "may",
    }
    assert entry.compute_effective_hint() == "must"


def test_runner_a_release_does_not_drop_runner_b_slot():
    """Runner A demoting / releasing its instance "alice-123" must
    NOT drop runner B's instance "alice-123" — that would be the
    cross-runner coupling bug M3 fixes."""

    entry = _new_entry()
    entry.record_instance_hint("runner-A", "alice-123", "must")
    entry.record_instance_hint("runner-B", "alice-123", "must")
    assert entry.compute_effective_hint() == "must"

    # Runner A releases its contribution.
    entry.release_instance_contribution("runner-A", "alice-123")
    # Runner B's slot is intact.
    assert entry.live_hints_by_caller == {
        ("runner-B", "alice-123"): "must",
    }
    assert entry.compute_effective_hint() == "must"

    # Runner A demoting via "no" (without literal_no) is also
    # isolated: only A's slot is removed.
    entry.record_instance_hint("runner-A", "alice-123", "may")
    entry.record_instance_hint("runner-A", "alice-123", "no")
    assert entry.live_hints_by_caller == {
        ("runner-B", "alice-123"): "must",
    }
    assert entry.compute_effective_hint() == "must"


def test_default_runner_id_falls_back_when_field_omitted():
    """A caller that omits ``runner_id`` lands in the ``"default"``
    runner slot. This preserves back-compat for the pre-M3 single-
    tenant wire schema (where the wire only carried ``instance_id``)."""

    entry = _new_entry()
    # None runner_id resolves to "default"
    entry.record_instance_hint(None, "alice", "must")
    # Empty runner_id also resolves to "default"
    entry.record_instance_hint("", "bob", "may")
    # Explicit "default" lands in the same namespace, so submitting
    # for instance "alice" again under explicit "default" overwrites
    # the same slot.
    entry.record_instance_hint("default", "alice", "may")
    assert entry.live_hints_by_caller == {
        ("default", "alice"): "may",
        ("default", "bob"): "may",
    }
    assert entry.compute_effective_hint() == "may"


# ---------------------------------------------------------------------------
# End-to-end: registry + block flips through update_segment_lifecycle_hint
# ---------------------------------------------------------------------------


def test_e2e_two_instances_must_and_may_resolves_to_must():
    """Paper §3 example: instance A submits must for segment X, instance
    B refreshes the same segment as may. Engine flips blocks to must
    (max-priority)."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)

    # Instance A: must.
    res_a = update_segment_lifecycle_hint(
        "seg-shared", "must", instance_id="instance-A"
    )
    assert res_a["effective_hint"] == "must"
    assert res_a["live_referencer_count"] == 1
    assert res_a["updated"] == 2
    assert all(b.lifecycle_hint == "must" for b in blocks)

    # Instance B: may. Effective stays must (max(must, may) = must).
    res_b = update_segment_lifecycle_hint(
        "seg-shared", "may", instance_id="instance-B"
    )
    assert res_b["effective_hint"] == "must"
    assert res_b["live_referencer_count"] == 2
    assert all(b.lifecycle_hint == "must" for b in blocks)


def test_e2e_must_instance_demotes_other_still_may_drops_to_may():
    """Instance A demotes itself to may; instance B was already may.
    Effective hint becomes max(may, may) = may. The earlier ``must``
    state is NOT preserved by old state."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint("seg-shared", "must", instance_id="A")
    update_segment_lifecycle_hint("seg-shared", "may", instance_id="B")
    # State now: A=must, B=may, effective=must.

    # A demotes itself.
    res = update_segment_lifecycle_hint(
        "seg-shared", "may", instance_id="A"
    )
    assert res["effective_hint"] == "may"
    assert res["live_referencer_count"] == 2
    assert all(b.lifecycle_hint == "may" for b in blocks)


def test_e2e_other_instance_keeps_may_when_one_drops_out():
    """Lock the M3 contract head-on: if A is must and B is may, a
    pre-M3 last-write-wins implementation would have already lost
    must when B's may arrived. M3 keeps the must-effective. When A
    later drops out (no, no literal_no), B's may is what remains, so
    effective stays may rather than collapsing to no."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint("seg-shared", "may", instance_id="B")
    update_segment_lifecycle_hint("seg-shared", "must", instance_id="A")
    assert all(b.lifecycle_hint == "must" for b in blocks)

    # A drops out (instance completed) — segment_lifecycle_update with no.
    res = update_segment_lifecycle_hint(
        "seg-shared", "no", instance_id="A"
    )
    # B still references the segment as may, so effective = may
    # (NOT no, because A's "no" DROPS A's slot rather than recording
    # a literal no contribution).
    assert res["effective_hint"] == "may"
    assert res["live_referencer_count"] == 1
    assert all(b.lifecycle_hint == "may" for b in blocks)


def test_e2e_both_instances_drop_default_to_may():
    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint("seg-shared", "must", instance_id="A")
    update_segment_lifecycle_hint("seg-shared", "must", instance_id="B")
    assert all(b.lifecycle_hint == "must" for b in blocks)

    update_segment_lifecycle_hint("seg-shared", "no", instance_id="A")
    res = update_segment_lifecycle_hint(
        "seg-shared", "no", instance_id="B"
    )
    assert res["effective_hint"] == "may"
    assert res["live_referencer_count"] == 0
    assert all(b.lifecycle_hint == "may" for b in blocks)


def test_e2e_literal_no_drives_blocks_to_no():
    """Loop-exit force-eviction path: A literal-no claims the block is
    no longer needed; if all live contributions are literal no,
    effective is no, and blocks demote to no even if the live set is
    non-empty."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint(
        "seg-shared", "no", instance_id="A", literal_no=True
    )
    res = update_segment_lifecycle_hint(
        "seg-shared", "no", instance_id="B", literal_no=True
    )
    assert res["effective_hint"] == "no"
    assert res["live_referencer_count"] == 2
    assert all(b.lifecycle_hint == "no" for b in blocks)


def test_e2e_release_segment_instance_contribution_drops_slot():
    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint("seg-shared", "must", instance_id="A")
    update_segment_lifecycle_hint("seg-shared", "may", instance_id="B")
    assert all(b.lifecycle_hint == "must" for b in blocks)

    res = release_segment_instance_contribution(
        "seg-shared", None, "A"
    )
    assert res["effective_hint"] == "may"
    assert res["live_referencer_count"] == 1
    assert all(b.lifecycle_hint == "may" for b in blocks)


def test_e2e_anonymous_caller_falls_back_to_last_write_wins():
    """Legacy callers (no runner_id, no instance_id) share the
    ``("default", "_anon")`` slot, so consecutive anon calls just
    overwrite each other. This preserves backward compatibility with
    all callers that pre-date M3."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint("seg-shared", "must")
    update_segment_lifecycle_hint("seg-shared", "may")
    res = update_segment_lifecycle_hint("seg-shared", "must")
    assert res["effective_hint"] == "must"
    assert res["live_referencer_count"] == 1
    assert all(b.lifecycle_hint == "must" for b in blocks)


def test_e2e_anonymous_and_named_aggregate_independently():
    """An anon caller and a named instance ARE different slots. Tests
    the contract: a legacy anon caller setting must AND an M3 named
    caller setting may both contribute, and effective is must."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint("seg-shared", "must")  # anon
    res = update_segment_lifecycle_hint(
        "seg-shared", "may", instance_id="named-A"
    )
    assert res["effective_hint"] == "must"
    assert res["live_referencer_count"] == 2
    assert all(b.lifecycle_hint == "must" for b in blocks)


def test_e2e_two_runners_same_instance_isolated_at_block_level():
    """End-to-end version of the cross-runner isolation test: two
    runners with identical ``instance_id`` are tracked independently
    AND the block-flip path honors the aggregated effective hint."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    # Both runners use instance_id "alice-123".
    update_segment_lifecycle_hint(
        "seg-shared", "must", runner_id="runner-A", instance_id="alice-123"
    )
    res = update_segment_lifecycle_hint(
        "seg-shared", "may", runner_id="runner-B", instance_id="alice-123"
    )
    # Two independent slots — effective = max(must, may) = must.
    assert res["live_referencer_count"] == 2
    assert res["effective_hint"] == "must"
    assert all(b.lifecycle_hint == "must" for b in blocks)

    # Runner A drops out — runner B's "may" remains.
    res = update_segment_lifecycle_hint(
        "seg-shared", "no", runner_id="runner-A", instance_id="alice-123"
    )
    assert res["effective_hint"] == "may"
    assert res["live_referencer_count"] == 1
    assert all(b.lifecycle_hint == "may" for b in blocks)


def test_e2e_runner_a_release_does_not_yank_runner_b():
    """``release_segment_instance_contribution`` keyed on tuple: only
    runner A's slot for ``alice-123`` drops; runner B's stays."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint(
        "seg-shared", "must", runner_id="runner-A", instance_id="alice-123"
    )
    update_segment_lifecycle_hint(
        "seg-shared", "must", runner_id="runner-B", instance_id="alice-123"
    )

    res = release_segment_instance_contribution(
        "seg-shared", "runner-A", "alice-123"
    )
    assert res["effective_hint"] == "must"
    assert res["live_referencer_count"] == 1
    # Runner B's must-contribution still keeps blocks alive.
    assert all(b.lifecycle_hint == "must" for b in blocks)

    entry = segment_actions._registry._entry_by_segment_id_locked(
        "seg-shared"
    )
    assert entry is not None
    assert entry.live_hints_by_caller == {
        ("runner-B", "alice-123"): "must",
    }


def test_e2e_default_runner_back_compat_no_runner_id_field():
    """Pre-M3 wire schema sent only ``instance_id`` (no ``runner_id``).
    The HTTP route's "missing runner_id falls through to default"
    contract is exercised here at the function-level entry point:
    omitting ``runner_id`` from the kwargs is identical to passing
    ``runner_id="default"``."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    # First call omits runner_id (pre-M3 wire shape).
    update_segment_lifecycle_hint("seg-shared", "must", instance_id="alice")
    # Second call passes explicit runner_id="default" — same slot.
    res = update_segment_lifecycle_hint(
        "seg-shared", "may", runner_id="default", instance_id="alice"
    )
    # Same slot ⇒ overwritten ⇒ last-write-wins ⇒ may.
    assert res["live_referencer_count"] == 1
    assert res["effective_hint"] == "may"

    entry = segment_actions._registry._entry_by_segment_id_locked(
        "seg-shared"
    )
    assert entry is not None
    assert entry.live_hints_by_caller == {("default", "alice"): "may"}
    assert all(b.lifecycle_hint == "may" for b in blocks)


def test_invalid_hint_does_not_mutate_state():
    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint("seg-shared", "must", instance_id="A")

    res = update_segment_lifecycle_hint(
        "seg-shared", "bogus", instance_id="B"
    )
    assert res["reject_reason"] == "invalid_hint"
    # No new slot recorded; A's contribution intact.
    entry = segment_actions._registry._entry_by_segment_id_locked(
        "seg-shared"
    )
    assert entry is not None
    assert entry.live_hints_by_caller == {("default", "A"): "must"}
    assert all(b.lifecycle_hint == "must" for b in blocks)


def test_missing_segment_id_rejects_cleanly():
    res = update_segment_lifecycle_hint(None, "must", instance_id="A")
    assert res["reject_reason"] == "segment_id_missing"
    assert res["updated"] == 0


def test_unregistered_segment_returns_legacy_shape():
    """When apply_per_instance_hint runs against a segment with no
    registry entry (e.g., wire arrived before submit_prepare), the
    method falls back to treating the input as the effective hint
    directly. ``updated=0`` because no blocks are tagged yet, but
    no exception is raised."""

    res = update_segment_lifecycle_hint(
        "seg-not-registered", "must", instance_id="A"
    )
    assert res["effective_hint"] == "must"
    assert res["live_referencer_count"] == 0
    assert res["updated"] == 0


def test_live_hints_visible_in_telemetry_with_runner_id_records():
    """Telemetry must expose both ``runner_id`` AND ``instance_id``
    per live entry as JSON-friendly records (raw tuples are not
    JSON-serializable)."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    _seed_segment(segment_actions._registry)
    update_segment_lifecycle_hint(
        "seg-shared", "must", runner_id="runner-A", instance_id="alice"
    )
    update_segment_lifecycle_hint(
        "seg-shared", "may", runner_id="runner-B", instance_id="alice"
    )
    # Legacy anon caller too — exercises the default-runner fallback.
    update_segment_lifecycle_hint("seg-shared", "may")

    entry = segment_actions._registry._entry_by_segment_id_locked(
        "seg-shared"
    )
    assert entry is not None
    status = entry.redacted_status()
    assert status["live_referencer_count"] == 3
    assert status["effective_lifecycle_hint"] == "must"
    assert status["live_hints_by_caller"] == [
        {"runner_id": "default", "instance_id": "_anon", "hint": "may"},
        {"runner_id": "runner-A", "instance_id": "alice", "hint": "must"},
        {"runner_id": "runner-B", "instance_id": "alice", "hint": "may"},
    ]
    # Sanity: the JSON-friendly list survives a json round-trip.
    import json as _json

    _json.dumps(status["live_hints_by_caller"])


# ---------------------------------------------------------------------------
# tag_blocks_with_segment_id smoke test (sanity for the test fixture)
# ---------------------------------------------------------------------------


def test_tag_blocks_smoke_test_routes_through_registry():
    """Sanity that the test fixture pattern matches the engine's own
    block-tagging path; if the registry helper signature changes,
    this test surfaces it loudly."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    blocks = [_FakeBlock(i) for i in range(3)]
    tag_blocks_with_segment_id(blocks, "seg-tag")
    assert all(b._segment_id == "seg-tag" for b in blocks)
    # Reverse map should now hold all three blocks.
    assert (
        len(segment_actions._registry._blocks_by_segment_id["seg-tag"])
        == 3
    )


# ---------------------------------------------------------------------------
# Concurrency: simultaneous updates must not corrupt the live-hint set
# ---------------------------------------------------------------------------


def test_concurrent_updates_serialize_through_registry_lock():
    """Spin up several threads each repeatedly setting / dropping its
    own instance's hint. The final state must (a) never raise, and (b)
    reflect a coherent live set whose effective hint matches the
    max-priority over the recorded contributions."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    _seed_segment(segment_actions._registry, segment_id="seg-race")
    n_iters = 200
    instances = ["I1", "I2", "I3", "I4"]
    final_hints = ["must", "may", "must", "may"]

    def _worker(inst_id: str, final_hint: str) -> None:
        for _ in range(n_iters):
            update_segment_lifecycle_hint(
                "seg-race", "must", instance_id=inst_id
            )
            update_segment_lifecycle_hint(
                "seg-race", "may", instance_id=inst_id
            )
        # Land on the assigned final hint so the post-condition is
        # deterministic regardless of interleaving.
        update_segment_lifecycle_hint(
            "seg-race", final_hint, instance_id=inst_id
        )

    threads = [
        threading.Thread(target=_worker, args=(inst, hint))
        for inst, hint in zip(instances, final_hints)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entry = segment_actions._registry._entry_by_segment_id_locked(
        "seg-race"
    )
    assert entry is not None
    # Each instance's last write was its assigned final hint, so the
    # live set must reflect exactly those four contributions (under
    # the default runner) and the effective hint = max over them.
    expected = {
        ("default", inst): hint
        for inst, hint in zip(instances, final_hints)
    }
    assert entry.live_hints_by_caller == expected
    assert entry.compute_effective_hint() == "must"


def test_concurrent_updates_two_runners_same_instance_id():
    """Same instance_id on two runners must remain isolated under
    concurrent stress — proves the cross-runner contention guarantee
    is not just a single-thread invariant."""

    from vllm.entrypoints.openai.chat_completion import segment_actions

    _seed_segment(segment_actions._registry, segment_id="seg-cross")
    n_iters = 200

    def _worker(runner: str, final_hint: str) -> None:
        for _ in range(n_iters):
            update_segment_lifecycle_hint(
                "seg-cross",
                "must",
                runner_id=runner,
                instance_id="alice-123",
            )
            update_segment_lifecycle_hint(
                "seg-cross",
                "may",
                runner_id=runner,
                instance_id="alice-123",
            )
        update_segment_lifecycle_hint(
            "seg-cross",
            final_hint,
            runner_id=runner,
            instance_id="alice-123",
        )

    threads = [
        threading.Thread(target=_worker, args=("runner-A", "must")),
        threading.Thread(target=_worker, args=("runner-B", "may")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entry = segment_actions._registry._entry_by_segment_id_locked(
        "seg-cross"
    )
    assert entry is not None
    assert entry.live_hints_by_caller == {
        ("runner-A", "alice-123"): "must",
        ("runner-B", "alice-123"): "may",
    }
    assert entry.compute_effective_hint() == "must"
