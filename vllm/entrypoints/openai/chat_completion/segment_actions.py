# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Segment-granularity co-optimization action ingress (Phase E1).

Adds the engine-side counterpart of the runner's
``v2/backend/segment_action.py`` wire-action constructors:

* ``POST /v1/coopt/segment_prepare`` accepts a ``segment_prepare`` wire
  action, validates it, registers it under
  ``(family_id, token_hash)``, and (when chat handler exposes a
  ``submit_workflow_segment_prewarm`` hook) triggers an internal
  prefill-only request that warms the engine's KV cache for the
  segment's prefix tokens.
* ``POST /v1/coopt/segment_refresh`` accepts a ``segment_refresh`` wire
  action and, per docs/v2/17 LRU-touch invariant, asks the engine to
  re-touch the cached prefix lines via a "prefix + 1 dummy token"
  request that decodes one throwaway token. This moves the lines back
  to the MRU tail without disturbing the cached prefix payload.

Mirrors the existing ``prefix_prepare`` ingress at
``workflow_actions.py``: same observability-first stance, same
redaction contract (raw ``prompt_token_ids`` are accepted on the wire
but never serialized to telemetry or status responses), same lifecycle
state machine, same fail-closed env gating
(``VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS=1``).

Phase E1 deliberately stops at "register + trigger prefill"; the
block-pool eviction-policy hookup that lets ``lifecycle_hint`` actually
bias eviction is Phase E2 (parent's work, in
``vllm/v1/core/block_pool.py``). This module only stands up the wire
contract + registry + prefill trigger so runner-side R2 can talk to a
real engine immediately.

Internal prefill is driven through the chat handler's optional
``submit_workflow_segment_prewarm`` / ``submit_workflow_segment_refresh``
async methods. When those are absent the registry still accepts the
action and reports ``prewarm_status="prefill_only_unavailable"`` —
guaranteeing the server boots even before the engine-side prewarm
plumbing exists. Once it does exist, the registry transparently
routes through it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.chat_completion.workflow_actions import (
    workflow_coopt_actions_enabled,
)
from vllm.entrypoints.openai.chat_completion.workflow_test_hook import (
    record_workflow_segment_action,
)
from vllm.v1.wires_engine_telemetry import (
    ACTION_OUTCOME_NOOP_ALREADY_PRESENT,
    ACTION_OUTCOME_PREPARED,
    ACTION_OUTCOME_REJECTED_BAD_INPUT,
    ACTION_OUTCOME_REJECTED_DEDUP,
    ENDPOINT_SEGMENT_PREPARE,
    ENDPOINT_SEGMENT_REFRESH,
)
from vllm.v1.wires_engine_telemetry import (
    emit_action as _wires_emit_action,
)

__all__ = [
    "router",
    "SegmentRegistry",
    "submit_segment_prepare_action",
    "submit_segment_refresh_action",
    "get_segment_action",
    "reset_segment_registry_for_tests",
    "get_segment_telemetry",
    "record_segment_cache_hit",
    "record_segment_retention",
    "record_segment_eviction",
    "tag_blocks_with_segment_id",
    "update_segment_lifecycle_hint",
    "release_segment_instance_contribution",
    "mark_segment_prepare_prefilled",
    "mark_segment_blocks_speculative",
    "set_block_hint_updater",
    "set_speculative_block_promoter",
]


# WIRES Phase B: hint updates routed through the block_pool's queue.
#
# When a block is in the free queue, mutating ``block.lifecycle_hint``
# directly is INSUFFICIENT under the 3-pool victim policy
# (docs/v2/32 §2.3) — the block's pool membership won't change, so
# the new hint has no effect on eviction order.
#
# block_pool registers ``free_block_queue.update_block_hint`` here at
# startup. SegmentRegistry.update_block_hints calls this function
# (rather than direct field assignment) so hint flips correctly move
# blocks between pools.
#
# When unset (e.g. unit tests that don't construct a real block_pool),
# ``update_block_hints`` falls back to direct assignment — pool
# membership won't update, but that's a no-op concern in tests that
# don't exercise the queue.
_block_hint_updater: Callable[[Any, str], None] | None = None

# M20: callback that promotes a block to "must" with source_class
# "speculative" (short TTL backstop). Bound by ``BlockPool.__init__`` to
# a closure over ``free_block_queue.update_block_hint(blk, "must",
# source_class="speculative")``. Kept as a separate slot from
# ``_block_hint_updater`` (which is hardcoded to source_class
# "structured") so the speculative path doesn't risk changing the
# semantics of the existing structured-promotion callback.
_speculative_block_promoter: Callable[[Any], None] | None = None


def set_block_hint_updater(fn: Callable[[Any, str], None] | None) -> None:
    """Install (or clear) the callback used by ``SegmentRegistry.update_block_hints``
    to flip a block's lifecycle_hint via the block_pool's queue
    (which knows how to move it between pools).

    Called by ``BlockPool.__init__`` with
    ``self.free_block_queue.update_block_hint``. Multiple block_pool
    instances share this module-level slot — last writer wins.
    """
    global _block_hint_updater
    _block_hint_updater = fn


def set_speculative_block_promoter(fn: Callable[[Any], None] | None) -> None:
    """M20: install (or clear) the callback used by
    ``mark_segment_blocks_speculative`` to promote each tagged block to
    must with source_class "speculative" via the block_pool's queue.

    Called by ``BlockPool.__init__`` with a closure over
    ``free_block_queue.update_block_hint(blk, "must",
    source_class="speculative")``. Multiple block_pool instances share
    this module-level slot, last writer wins (mirrors
    ``_block_hint_updater`` semantics).
    """
    global _speculative_block_promoter
    _speculative_block_promoter = fn


# Action kinds (mirrors runner-side v2/backend/segment_action.py constants).
SEGMENT_PREPARE_ACTION_KIND = "segment_prepare"
SEGMENT_REFRESH_ACTION_KIND = "segment_refresh"

# Env names — independent of the prefix_prepare envs so segment-mode
# tuning does not overload prefix_prepare deployments.
_SEGMENT_TTL_MS_ENV = "WORKFLOW_SEGMENT_PREPARE_TTL_MS"
_SEGMENT_MAX_OUTSTANDING_ENV = "WORKFLOW_SEGMENT_MAX_OUTSTANDING"
_SEGMENT_PREPARE_MODE_ENV = "WORKFLOW_SEGMENT_PREPARE_MODE"

_DEFAULT_SEGMENT_TTL_MS = 30000
_DEFAULT_SEGMENT_MAX_OUTSTANDING = 256
_DEFAULT_SEGMENT_PREPARE_MODE = "tokenize_only"
_PREWARM_PREPARE_MODES = {"experimental_prewarm", "prewarm_observe"}

# Segment-status state machine. See docs/v2/29 §2.
_SEGMENT_STATUS_PENDING = "pending"
_SEGMENT_STATUS_PREFILLED = "prefilled"
_SEGMENT_STATUS_EVICTED = "evicted"
_SEGMENT_STATUS_EXPIRED = "expired"

# Refresh-lifecycle marker emitted on every refresh so the test hook can
# distinguish refresh events from prepare events even when both share the
# same ``(family_id, token_hash)`` registry key.
_REFRESH_PREWARM_STATUS_RETOUCHED = "retouched"

router = APIRouter()


def _now_unix_ms() -> int:
    import time

    return int(time.time() * 1000)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _segment_ttl_ms() -> int:
    return _env_int(_SEGMENT_TTL_MS_ENV, _DEFAULT_SEGMENT_TTL_MS)


def _segment_max_outstanding() -> int:
    return _env_int(_SEGMENT_MAX_OUTSTANDING_ENV, _DEFAULT_SEGMENT_MAX_OUTSTANDING)


def _segment_prepare_mode() -> str:
    mode = os.getenv(_SEGMENT_PREPARE_MODE_ENV, _DEFAULT_SEGMENT_PREPARE_MODE)
    if mode == "tokenize_only":
        return mode
    if mode in _PREWARM_PREPARE_MODES:
        return mode
    return _DEFAULT_SEGMENT_PREPARE_MODE


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _hash_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _action_kind_of(action: dict[str, Any]) -> str | None:
    """Accept either ``action_kind`` (canonical wire form) or ``kind``."""

    explicit = action.get("action_kind")
    if isinstance(explicit, str) and explicit:
        return explicit
    legacy = action.get("kind")
    if isinstance(legacy, str) and legacy:
        return legacy
    return None


# ----- Validators ----------------------------------------------------------


_REQUIRED_PREPARE_FIELDS = (
    "action_id",
    "segment_id",
    "token_hash",
    "model",
    "expected_consumers",
    "lifecycle_status",
)
_REQUIRED_REFRESH_FIELDS = (
    "action_id",
    "segment_id",
    "family_id",
    "token_hash",
    "model",
    "lifecycle_status",
)


def _missing_field_response(
    *,
    action: dict[str, Any],
    action_kind: str,
    field_name: str,
) -> dict[str, Any]:
    return {
        "action_id": _optional_str(action.get("action_id")),
        "action_kind": action_kind,
        "accepted": False,
        "lifecycle_status": "rejected",
        "reject_reason": f"missing_{field_name}",
        "engine_wire_status": "rejected",
        "prewarm_status": "not_attempted",
    }


def _normalize_token_hash(action: dict[str, Any]) -> str | None:
    """Resolve the canonical token hash from either wire field name.

    ``v2/backend/segment_action.py`` emits ``prompt_token_ids_hash``;
    the user-facing E1 spec called the field ``token_hash``. Accept
    either; downstream uses the resolved value as the registry key
    component.
    """

    explicit = action.get("token_hash")
    if isinstance(explicit, str) and explicit:
        return explicit
    fallback = action.get("prompt_token_ids_hash")
    if isinstance(fallback, str) and fallback:
        return fallback
    return None


def _resolve_family_id(action: dict[str, Any]) -> str | None:
    """For prepare we accept either ``family_id`` or fall back to ``segment_id``.

    The runner-side ``build_segment_prepare_wire_action`` does not currently
    set ``family_id`` (per-segment scope); ``segment_id`` uniquely identifies
    the prepare in that case. Refresh, in contrast, REQUIRES ``family_id`` —
    families are the unit at which refresh deduplicates across instances.
    """

    explicit = action.get("family_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    fallback = action.get("segment_id")
    if isinstance(fallback, str) and fallback:
        return fallback
    return None


def _validate_segment_prepare(action: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a segment_prepare wire action.

    Returns ``None`` if the action is well-formed; otherwise returns a
    rejection response in the same shape the prefix_prepare path uses
    (``accepted=False``, ``lifecycle_status="rejected"``,
    ``reject_reason=...``).
    """

    kind = _action_kind_of(action)
    if kind != SEGMENT_PREPARE_ACTION_KIND:
        return {
            "action_id": _optional_str(action.get("action_id")),
            "action_kind": kind,
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "unsupported_action_kind",
            "engine_wire_status": "rejected",
            "prewarm_status": "not_attempted",
        }
    for field_name in _REQUIRED_PREPARE_FIELDS:
        if field_name == "token_hash":
            if _normalize_token_hash(action) is None:
                return _missing_field_response(
                    action=action,
                    action_kind=SEGMENT_PREPARE_ACTION_KIND,
                    field_name="token_hash",
                )
            continue
        if action.get(field_name) is None:
            return _missing_field_response(
                action=action,
                action_kind=SEGMENT_PREPARE_ACTION_KIND,
                field_name=field_name,
            )
    if not isinstance(action.get("segment_id"), str) or not action["segment_id"]:
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_PREPARE_ACTION_KIND,
            field_name="segment_id",
        )
    if not isinstance(action.get("model"), str) or not action["model"]:
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_PREPARE_ACTION_KIND,
            field_name="model",
        )
    expected_consumers = action.get("expected_consumers")
    if not isinstance(expected_consumers, int) or expected_consumers < 0:
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_PREPARE_ACTION_KIND,
            field_name="expected_consumers",
        )
    prompt_token_ids = action.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(tok, int) for tok in prompt_token_ids
    ):
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_PREPARE_ACTION_KIND,
            field_name="prompt_token_ids",
        )
    if not prompt_token_ids:
        return {
            "action_id": _optional_str(action.get("action_id")),
            "action_kind": SEGMENT_PREPARE_ACTION_KIND,
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "empty_prompt_token_ids",
            "engine_wire_status": "rejected",
            "prewarm_status": "not_attempted",
        }
    if not isinstance(action.get("lifecycle_status"), str):
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_PREPARE_ACTION_KIND,
            field_name="lifecycle_status",
        )
    return None


_VALID_LIFECYCLE_HINTS = ("must", "may", "no")


def _validate_segment_refresh(action: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a segment_refresh wire action.

    Refresh adds a required ``family_id`` (cross-instance dedup key) and
    accepts:

    * an optional ``would_evict_without_refresh: bool`` (engine records
      it but does not gate on it; the decision was already made
      runner-side);
    * an optional ``lifecycle_hint`` field, one of ``"must"``,
      ``"may"``, ``"no"`` (M9 merge: when non-null this also flips the
      lifecycle hint on every block currently tagged with this
      ``segment_id``, replacing the legacy
      ``/v1/coopt/segment_lifecycle_update`` endpoint).

    M9 merge: ``prompt_token_ids`` is OPTIONAL (it used to be required).
    Two valid call shapes:

    1. Touch-only / touch + hint: ``prompt_token_ids`` present
       (non-empty list[int]); refresh re-MRUs the prefix lines via the
       prewarm hook, and (if ``lifecycle_hint`` is set) ALSO flips
       block hints by ``segment_id``. This is the legacy
       ``segment_refresh`` plus the new merged hint update.
    2. Hint-only (no LRU touch): ``prompt_token_ids`` absent or empty
       AND ``lifecycle_hint`` is non-null. Engine skips the prewarm
       and just flips block hints. This subsumes the legacy
       ``segment_lifecycle_update`` endpoint, including the
       loop-anchor-must-promotion path that has no token ids at hand.
    """

    kind = _action_kind_of(action)
    if kind != SEGMENT_REFRESH_ACTION_KIND:
        return {
            "action_id": _optional_str(action.get("action_id")),
            "action_kind": kind,
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "unsupported_action_kind",
            "engine_wire_status": "rejected",
            "prewarm_status": "not_attempted",
        }
    for field_name in _REQUIRED_REFRESH_FIELDS:
        if field_name == "token_hash":
            if _normalize_token_hash(action) is None:
                return _missing_field_response(
                    action=action,
                    action_kind=SEGMENT_REFRESH_ACTION_KIND,
                    field_name="token_hash",
                )
            continue
        if action.get(field_name) is None:
            return _missing_field_response(
                action=action,
                action_kind=SEGMENT_REFRESH_ACTION_KIND,
                field_name=field_name,
            )
    if not isinstance(action.get("segment_id"), str) or not action["segment_id"]:
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_REFRESH_ACTION_KIND,
            field_name="segment_id",
        )
    if not isinstance(action.get("family_id"), str) or not action["family_id"]:
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_REFRESH_ACTION_KIND,
            field_name="family_id",
        )
    if not isinstance(action.get("model"), str) or not action["model"]:
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_REFRESH_ACTION_KIND,
            field_name="model",
        )
    # M9: prompt_token_ids is now optional; when present, must be a
    # list[int]. When absent/empty, lifecycle_hint must be non-null
    # (hint-only mode).
    prompt_token_ids = action.get("prompt_token_ids")
    has_token_ids = (
        isinstance(prompt_token_ids, list)
        and len(prompt_token_ids) > 0
        and all(isinstance(tok, int) for tok in prompt_token_ids)
    )
    if prompt_token_ids is not None:
        if not isinstance(prompt_token_ids, list) or not all(
            isinstance(tok, int) for tok in prompt_token_ids
        ):
            return _missing_field_response(
                action=action,
                action_kind=SEGMENT_REFRESH_ACTION_KIND,
                field_name="prompt_token_ids",
            )
        if len(prompt_token_ids) == 0:
            return {
                "action_id": _optional_str(action.get("action_id")),
                "action_kind": SEGMENT_REFRESH_ACTION_KIND,
                "accepted": False,
                "lifecycle_status": "rejected",
                "reject_reason": "empty_prompt_token_ids",
                "engine_wire_status": "rejected",
                "prewarm_status": "not_attempted",
            }
    lifecycle_hint = action.get("lifecycle_hint")
    if lifecycle_hint is not None and lifecycle_hint not in _VALID_LIFECYCLE_HINTS:
        return {
            "action_id": _optional_str(action.get("action_id")),
            "action_kind": SEGMENT_REFRESH_ACTION_KIND,
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "invalid_lifecycle_hint",
            "engine_wire_status": "rejected",
            "prewarm_status": "not_attempted",
        }
    # Hint-only mode requires lifecycle_hint to be set; touch-only
    # mode requires prompt_token_ids. At least one must produce work.
    if not has_token_ids and lifecycle_hint is None:
        return {
            "action_id": _optional_str(action.get("action_id")),
            "action_kind": SEGMENT_REFRESH_ACTION_KIND,
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "missing_prompt_token_ids_or_lifecycle_hint",
            "engine_wire_status": "rejected",
            "prewarm_status": "not_attempted",
        }
    would_evict = action.get("would_evict_without_refresh")
    if would_evict is not None and not isinstance(would_evict, bool):
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_REFRESH_ACTION_KIND,
            field_name="would_evict_without_refresh",
        )
    if not isinstance(action.get("lifecycle_status"), str):
        return _missing_field_response(
            action=action,
            action_kind=SEGMENT_REFRESH_ACTION_KIND,
            field_name="lifecycle_status",
        )
    return None


# ----- Registry ------------------------------------------------------------


@dataclass
class SegmentEntry:
    """Engine-internal per-segment state.

    Lifecycle:
        ``pending`` (registered, no prefill yet) →
        ``prefilled`` (internal prefill submitted / completed) →
        ``evicted`` (block_pool dropped the cached prefix; recorded
        when E2 wires up the eviction callback) OR
        ``expired`` (TTL elapsed before any consumer matched).

    ``observed_consumers`` is incremented opportunistically by E4 when
    the per-segment cache-hit telemetry lands. For E1 it stays at 0.
    """

    action_id: str
    action_kind: str
    segment_id: str
    family_id: str
    parent_segment_id: str | None
    token_hash: str
    model: str
    lifecycle_status: str
    expected_consumers: int
    observed_consumers: int = 0
    prefill_token_count: int = 0
    prefill_status: str = "not_attempted"
    engine_seen_token_hash: str | None = None
    ttl_ms: int = _DEFAULT_SEGMENT_TTL_MS
    accepted_at_unix_ms: int = 0
    expires_at_unix_ms: int = 0
    status: str = _SEGMENT_STATUS_PENDING
    would_evict_without_refresh: bool | None = None
    last_refresh_at_unix_ms: int | None = None
    refresh_count: int = 0
    # WIRES Phase E4 — per-segment cache statistics. Populated by the
    # block-pool hooks in vllm/v1/core/block_pool.py whenever a block
    # tagged with this segment's id participates in a touch / allocation
    # / eviction event. None of these counters are used to drive
    # behavior; they are observability-only so we can verify the
    # 3-priority popleft + lifecycle_hint plumbing actually keeps
    # `must` segments resident.
    cache_hit_count: int = 0
    retention_count: int = 0
    evict_count_by_hint: dict[str, int] = field(default_factory=dict)
    # M3 (paper §3 + §3.5 cross-instance hint aggregation): per-caller
    # hint contributions for this segment. A "caller" is identified by
    # the tuple ``(runner_id, instance_id)``: independent runners
    # (different teams / processes / orgs) do not coordinate the
    # ``instance_id`` namespace, so a collision on ``instance_id``
    # alone would silently merge their entries — and a "no" from one
    # would yank the other's contribution. Keying on the ``runner_id``
    # tuple isolates each runner's instance namespace. The registry
    # aggregates contributions across all (runner_id, instance_id)
    # callers into the effective block hint by ``max(must, may, no)``
    # priority before pushing through to the block_pool. Empty dict =>
    # no live referencer => default ``"may"``. Anonymous (legacy)
    # callers that do not carry a ``runner_id`` fall back to the
    # default runner ``"default"``; callers that do not carry an
    # ``instance_id`` fall back to the synthetic ``"_anon"`` slot, so
    # legacy "last-write-wins" callers (single-tenant: one default
    # runner, no instance ids) keep working unchanged.
    live_hints_by_caller: dict[tuple[str, str], str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # M3: cross-instance hint aggregation.
    # Priority order: ``must`` > ``may`` > ``no``. ``no`` is the
    # lowest-priority observable hint; an empty live-hint dict means
    # no referencer is alive and the effective hint defaults to
    # ``"may"`` (matches the block-pool default).
    _HINT_PRIORITY = {"no": 0, "may": 1, "must": 2}
    # Default runner_id used when the wire payload omits the field.
    # Single-tenant deployments (one runner, no cross-runner sharing)
    # land all callers in this slot, preserving the original
    # "instance_id is the only namespace" behavior.
    _DEFAULT_RUNNER_ID = "default"
    # Synthetic instance_id used by callers that pass no instance_id
    # through the wire. All such calls (within the same runner) share
    # the same slot, so the aggregation reduces to last-write-wins
    # within a runner.
    _ANONYMOUS_INSTANCE_ID = "_anon"

    @staticmethod
    def _caller_key(runner_id: str | None, instance_id: str | None) -> tuple[str, str]:
        """Project ``(runner_id, instance_id)`` to the canonical caller key.

        Empty / non-string fields fall through to the default
        runner / anonymous-instance sentinels so the keyspace is
        always ``tuple[str, str]`` and lookups are consistent.
        """

        runner_slot = (
            runner_id
            if isinstance(runner_id, str) and runner_id
            else SegmentEntry._DEFAULT_RUNNER_ID
        )
        instance_slot = (
            instance_id
            if isinstance(instance_id, str) and instance_id
            else SegmentEntry._ANONYMOUS_INSTANCE_ID
        )
        return (runner_slot, instance_slot)

    def record_instance_hint(
        self,
        runner_id: str | None,
        instance_id: str | None,
        new_hint: str | None,
    ) -> str:
        """Record one caller's contribution and return the effective hint.

        ``runner_id is None`` (or empty) falls through to the
        ``_DEFAULT_RUNNER_ID`` sentinel; ``instance_id is None`` (or
        empty) falls through to the ``_ANONYMOUS_INSTANCE_ID``
        sentinel — together these preserve last-write-wins behavior
        for callers that pre-date M3 and do not carry either field.

        ``new_hint == "no"`` is treated as "this caller no longer
        needs the segment alive": the contribution is REMOVED from
        the live set rather than recorded as ``no``. This matches the
        runner-side "explicit demote on instance completion" pattern.
        Note: if all live referencers explicitly demote to ``no``,
        the effective hint snaps to the empty-set default ``may``
        rather than the literal ``no`` (because each caller dropping
        out means it stops contributing, not that it actively demands
        eviction). Callers that DO want the literal ``no`` (e.g. loop
        exit promotes-to-no on an in-flight block) should call
        ``set_instance_hint_literal`` instead.
        """

        slot = SegmentEntry._caller_key(runner_id, instance_id)
        if new_hint == "no":
            self.live_hints_by_caller.pop(slot, None)
        elif new_hint in ("must", "may"):
            self.live_hints_by_caller[slot] = new_hint
        else:
            # Unknown hint: do not mutate the live set, just return
            # the current effective hint so the caller can decide.
            pass
        return self.compute_effective_hint()

    def set_instance_hint_literal(
        self,
        runner_id: str | None,
        instance_id: str | None,
        new_hint: str | None,
    ) -> str:
        """Variant of ``record_instance_hint`` that records ``no`` as
        a literal contribution rather than removing the slot. Used
        for the loop-exit "promote to no" path where the caller wants
        the block evicted ASAP regardless of other live referencers.
        """

        slot = SegmentEntry._caller_key(runner_id, instance_id)
        if new_hint in ("must", "may", "no"):
            self.live_hints_by_caller[slot] = new_hint
        return self.compute_effective_hint()

    def release_instance_contribution(
        self, runner_id: str | None, instance_id: str | None
    ) -> str:
        """Remove ``(runner_id, instance_id)`` from the live-hint set
        (e.g. on instance completion) and return the new effective
        hint."""

        slot = SegmentEntry._caller_key(runner_id, instance_id)
        self.live_hints_by_caller.pop(slot, None)
        return self.compute_effective_hint()

    def compute_effective_hint(self) -> str:
        """Return ``max(live_hints_by_caller.values())`` by priority.

        Empty live set => default ``"may"`` (matches the block-pool
        default lifecycle_hint and the "no live referencer => no
        active claim" semantic).
        """

        if not self.live_hints_by_caller:
            return "may"
        return max(
            self.live_hints_by_caller.values(),
            key=lambda h: SegmentEntry._HINT_PRIORITY.get(h, -1),
        )

    def expire_if_due(self) -> None:
        if self.status in {_SEGMENT_STATUS_EVICTED, _SEGMENT_STATUS_EXPIRED}:
            return
        if self.expires_at_unix_ms and self.expires_at_unix_ms < _now_unix_ms():
            self.status = _SEGMENT_STATUS_EXPIRED

    def redacted_status(self) -> dict[str, Any]:
        self.expire_if_due()
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "segment_id": self.segment_id,
            "family_id": self.family_id,
            "parent_segment_id": self.parent_segment_id,
            "model": self.model,
            "accepted": True,
            "lifecycle_status": self.lifecycle_status,
            "engine_wire_status": "accepted",
            "expected_consumers": self.expected_consumers,
            "observed_consumers": self.observed_consumers,
            "prefill_token_count": self.prefill_token_count,
            "prefill_status": self.prefill_status,
            "engine_seen_token_hash": self.engine_seen_token_hash,
            "token_hash": self.token_hash,
            "ttl_ms": self.ttl_ms,
            "accepted_at_unix_ms": self.accepted_at_unix_ms,
            "expires_at_unix_ms": self.expires_at_unix_ms,
            "status": self.status,
            "would_evict_without_refresh": self.would_evict_without_refresh,
            "last_refresh_at_unix_ms": self.last_refresh_at_unix_ms,
            "refresh_count": self.refresh_count,
            "cache_hit_count": self.cache_hit_count,
            "retention_count": self.retention_count,
            "evict_count_by_hint": dict(self.evict_count_by_hint),
            "prewarm_status": self.prefill_status,
            # M3: cross-instance hint aggregation telemetry. Exposes
            # the live referencer count and the effective hint so
            # callers can verify the engine is doing the max-priority
            # combine rather than last-write-wins.
            #
            # ``live_hints_by_caller`` is exposed as a JSON-friendly
            # list of ``{runner_id, instance_id, hint}`` records (raw
            # tuple keys would not survive ``json.dumps``). The list
            # is sorted by ``(runner_id, instance_id)`` so the
            # serialization is deterministic across calls.
            "live_referencer_count": len(self.live_hints_by_caller),
            "effective_lifecycle_hint": self.compute_effective_hint(),
            "live_hints_by_caller": [
                {
                    "runner_id": runner_id,
                    "instance_id": instance_id,
                    "hint": hint,
                }
                for (runner_id, instance_id), hint in sorted(
                    self.live_hints_by_caller.items()
                )
            ],
        }


class SegmentRegistry:
    """In-memory registry of segment_prepare / segment_refresh actions.

    Keyed by ``(family_id, token_hash)`` per docs/v2/31 §E1. A second
    ``action_id`` index lets ``GET /v1/coopt/segment_prepare/{action_id}``
    look up the same entry.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries_by_key: OrderedDict[tuple[str, str], SegmentEntry] = OrderedDict()
        self._entries_by_action_id: dict[str, SegmentEntry] = {}
        # WIRES Phase E4 — secondary segment_id -> entry index used by
        # block-pool hooks (touch / eviction). The (family_id, token_hash)
        # key remains canonical; this mirror is rebuilt whenever an entry
        # is created or evicted so block_pool callbacks stay O(1).
        self._entries_by_segment_id: dict[str, SegmentEntry] = {}
        # WIRES Phase E2 step 3 — segment_id -> list of currently-tagged
        # KVCacheBlocks. Populated by tag_blocks_with_segment_id; queried
        # by the merged /v1/coopt/segment_refresh endpoint (M9: via the
        # body's lifecycle_hint field) when the monitor downgrades the
        # per-segment hint after a CFG event.
        # Strong refs (KVCacheBlock uses slots so weakref isn't free);
        # entries are pruned lazily when their _segment_id no longer
        # matches (block reused for a different segment).
        self._blocks_by_segment_id: dict[str, list[Any]] = {}

    def submit_prepare(
        self,
        action: dict[str, Any],
        *,
        prefill_status: str,
        prefill_token_count: int,
    ) -> dict[str, Any]:
        """Register a segment_prepare action.

        Recognised optional fields on ``action`` (in addition to the
        usual ``segment_id`` / ``token_hash`` / ``family_id`` / etc.):

          * ``speculative`` (bool, default False): M20 hook for the
            paper §3.2 future-work extension. When True, downstream
            block-pool operations on this segment's blocks should use
            ``source_class="speculative"`` (short TTL backstop, default
            30s, env ``WIRES_KVCACHE_SPECULATIVE_TTL_MS``). The actual
            stamping happens later when blocks are promoted via
            ``mark_segment_blocks_speculative`` (engine-side primitive
            this commit lands; runner-side patterns that *invoke* it
            are intentionally left to future work, see
            CODE_MISMATCH_NOTES.md M20).
        """
        token_hash = _normalize_token_hash(action) or ""
        family_id = _resolve_family_id(action) or ""
        segment_id = _optional_str(action.get("segment_id")) or ""
        action_id = _optional_str(action.get("action_id")) or ""
        ttl_ms = _optional_int(action.get("ttl_ms")) or _segment_ttl_ms()
        accepted_at = _now_unix_ms()
        entry = SegmentEntry(
            action_id=action_id,
            action_kind=SEGMENT_PREPARE_ACTION_KIND,
            segment_id=segment_id,
            family_id=family_id,
            parent_segment_id=_optional_str(action.get("parent_segment_id")),
            token_hash=token_hash,
            model=_optional_str(action.get("model")) or "",
            lifecycle_status=_optional_str(action.get("lifecycle_status"))
            or "accepted",
            expected_consumers=_optional_int(action.get("expected_consumers")) or 0,
            prefill_token_count=prefill_token_count,
            prefill_status=prefill_status,
            engine_seen_token_hash=token_hash,
            ttl_ms=ttl_ms,
            accepted_at_unix_ms=accepted_at,
            expires_at_unix_ms=accepted_at + ttl_ms,
            status=(
                _SEGMENT_STATUS_PREFILLED
                if prefill_status in {"prewarm_submitted", "prewarm_completed"}
                else _SEGMENT_STATUS_PENDING
            ),
        )
        with self._lock:
            key = (family_id, token_hash)
            self._entries_by_key[key] = entry
            self._entries_by_key.move_to_end(key)
            if action_id:
                self._entries_by_action_id[action_id] = entry
            if segment_id:
                self._entries_by_segment_id[segment_id] = entry
            self._evict_locked()
        return entry.redacted_status()

    def submit_refresh(
        self,
        action: dict[str, Any],
        *,
        prefill_status: str,
        prefill_token_count: int,
    ) -> dict[str, Any]:
        token_hash = _normalize_token_hash(action) or ""
        family_id = _resolve_family_id(action) or ""
        segment_id = _optional_str(action.get("segment_id")) or ""
        action_id = _optional_str(action.get("action_id")) or ""
        would_evict = _optional_bool(action.get("would_evict_without_refresh"))
        now = _now_unix_ms()
        with self._lock:
            key = (family_id, token_hash)
            entry = self._entries_by_key.get(key)
            if entry is None:
                # First time we see this family/hash — refresh creates
                # the registry record so subsequent lookups still find
                # it (mirrors prefix_prepare's "first-write wins" + later
                # supersede behavior).
                ttl_ms = _optional_int(action.get("ttl_ms")) or _segment_ttl_ms()
                entry = SegmentEntry(
                    action_id=action_id,
                    action_kind=SEGMENT_REFRESH_ACTION_KIND,
                    segment_id=segment_id,
                    family_id=family_id,
                    parent_segment_id=_optional_str(action.get("parent_segment_id")),
                    token_hash=token_hash,
                    model=_optional_str(action.get("model")) or "",
                    lifecycle_status=_optional_str(action.get("lifecycle_status"))
                    or "accepted",
                    expected_consumers=0,
                    prefill_token_count=prefill_token_count,
                    prefill_status=prefill_status,
                    engine_seen_token_hash=token_hash,
                    ttl_ms=ttl_ms,
                    accepted_at_unix_ms=now,
                    expires_at_unix_ms=now + ttl_ms,
                    status=_SEGMENT_STATUS_PREFILLED,
                    would_evict_without_refresh=would_evict,
                    last_refresh_at_unix_ms=now,
                    refresh_count=1,
                )
                self._entries_by_key[key] = entry
                self._entries_by_key.move_to_end(key)
            else:
                entry.last_refresh_at_unix_ms = now
                entry.refresh_count += 1
                entry.prefill_status = prefill_status
                entry.prefill_token_count = prefill_token_count
                entry.engine_seen_token_hash = token_hash
                entry.status = _SEGMENT_STATUS_PREFILLED
                # Refresh extends TTL (LRU touch invariant: a touched
                # block survives at least another TTL window).
                entry.expires_at_unix_ms = now + entry.ttl_ms
                entry.lifecycle_status = (
                    _optional_str(action.get("lifecycle_status"))
                    or entry.lifecycle_status
                )
                if would_evict is not None:
                    entry.would_evict_without_refresh = would_evict
                self._entries_by_key.move_to_end(key)
            if action_id:
                self._entries_by_action_id[action_id] = entry
            if segment_id:
                self._entries_by_segment_id[segment_id] = entry
            self._evict_locked()
            response = entry.redacted_status()
        # Override response action_kind to refresh so callers see what
        # they sent (the entry's action_kind reflects whichever wire
        # action created it first; refresh-on-existing keeps the prepare
        # kind. The per-call response should always identify the
        # received action.).
        response["action_kind"] = SEGMENT_REFRESH_ACTION_KIND
        response["refresh_observed_for_action_id"] = action_id
        return response

    def get_by_action_id(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries_by_action_id.get(action_id)
            if entry is None:
                return None
            return entry.redacted_status()

    def mark_prefilled(
        self,
        *,
        action_id: str,
        prefill_status: str,
        prefill_token_count: int | None,
    ) -> dict[str, Any]:
        """Phase E5 — flip an entry's prefill_status after the V1 prewarm
        prefill request completes.

        ``action_id`` must match the action that originally created the
        registry entry (via ``submit_prepare`` or ``submit_refresh``).
        Updates the per-entry ``prefill_status`` and (optionally) the
        ``prefill_token_count`` so segment_telemetry GETs return the
        real engine state.
        """

        with self._lock:
            entry = self._entries_by_action_id.get(action_id)
            if entry is None:
                return {"updated": False, "reject_reason": "unknown_action_id"}
            entry.prefill_status = prefill_status
            if isinstance(prefill_token_count, int) and prefill_token_count >= 0:
                entry.prefill_token_count = prefill_token_count
            return {
                "updated": True,
                "reject_reason": None,
                "segment_id": entry.segment_id,
                "prefill_status": prefill_status,
            }

    def reset_for_tests(self) -> None:
        with self._lock:
            self._entries_by_key.clear()
            self._entries_by_action_id.clear()
            self._entries_by_segment_id.clear()
            # Also clear the segment_id -> [block, ...] reverse map.
            # Without this, blocks tagged in an earlier test linger in
            # the registry and pollute later tests that re-tag under
            # the same segment_id (surfaced when adding the Phase B
            # test_segment_lifecycle_update_routes_through_block_pool_queue
            # alongside the existing fixed-id tests).
            self._blocks_by_segment_id.clear()

    # ----- Phase E4 per-segment cache telemetry --------------------------

    def register_block_for_segment(self, segment_id: str, block: Any) -> None:
        """WIRES Phase E2 step 3 — track which blocks belong to a segment.

        Called by ``tag_blocks_with_segment_id`` for every block the
        segment_prepare prefill helper allocates. The reverse map enables
        ``update_block_hints`` to find blocks by segment_id without an
        O(num_total_blocks) scan.

        Same block tagged twice for the same segment → de-duped via
        identity check at registration time.
        """

        if not isinstance(segment_id, str) or not segment_id:
            return
        with self._lock:
            blocks = self._blocks_by_segment_id.setdefault(segment_id, [])
            if all(b is not block for b in blocks):
                blocks.append(block)

    def apply_per_instance_hint(
        self,
        segment_id: str,
        runner_id: str | None,
        instance_id: str | None,
        new_hint: str,
        *,
        literal_no: bool = False,
    ) -> dict[str, Any]:
        """M3: record one caller's per-segment hint and push the
        max-priority effective hint through to the block_pool.

        This is the entry point that enforces paper §3 / §3.5's
        "engine's cache policy reconciles hints from all sources":
        the engine aggregates per-caller contributions, keyed on
        the ``(runner_id, instance_id)`` tuple, from multiple
        workflow instances (potentially across unrelated runners
        that share the engine's KV cache) into a single effective
        hint per segment, then flips block ``lifecycle_hint`` to
        that effective value.

        Independent runners (different teams / processes / orgs) do
        not coordinate the ``instance_id`` namespace; keying on
        ``(runner_id, instance_id)`` ensures a collision on
        ``instance_id`` alone never silently merges their entries
        and a "no" from one runner can never yank the other
        runner's contribution.

        Semantics
        ---------
        - ``new_hint == "no"`` AND ``literal_no=False`` (default):
          remove this caller's contribution from the live set. The
          new effective hint is computed across the remaining
          referencers; if none remain, defaults to ``"may"``.
        - ``new_hint == "no"`` AND ``literal_no=True``: record the
          literal ``no`` contribution. Useful for the loop-exit
          "force eviction" path where one caller wants the block
          gone even though others may still reference it.
        - ``new_hint in {"must", "may"}``: record the contribution.
          Effective hint becomes ``max(must, may, no)`` over all
          live contributions, by priority ``must > may > no``.

        ``runner_id is None`` falls back to the ``"default"``
        runner; ``instance_id is None`` falls back to the synthetic
        ``_anon`` slot. Single-tenant deployments that omit both
        therefore land in the ``("default", "_anon")`` slot —
        matching the legacy last-write-wins behavior.

        Returns the same shape as ``update_block_hints``, plus
        ``effective_hint`` and ``live_referencer_count`` for
        observability.
        """

        if not isinstance(segment_id, str) or not segment_id:
            return {
                "updated": 0,
                "skipped": 0,
                "reject_reason": "segment_id_missing",
            }
        if new_hint not in ("must", "may", "no"):
            return {
                "updated": 0,
                "skipped": 0,
                "reject_reason": "invalid_hint",
            }
        with self._lock:
            entry = self._entry_by_segment_id_locked(segment_id)
            if entry is not None:
                if literal_no and new_hint == "no":
                    effective = entry.set_instance_hint_literal(
                        runner_id, instance_id, "no"
                    )
                else:
                    effective = entry.record_instance_hint(
                        runner_id, instance_id, new_hint
                    )
                live_count = len(entry.live_hints_by_caller)
            else:
                # No registry entry yet (e.g. a hint update arrived for
                # a segment that was never registered via
                # segment_prepare). Fall back to treating the input as
                # the effective hint directly so legacy single-source
                # behavior is preserved.
                effective = new_hint
                live_count = 0
        # Push the effective hint to the block_pool. Reuses the
        # existing single-hint path so the 3-priority free-queue
        # plumbing kicks in unchanged.
        result = self.update_block_hints(segment_id, effective)
        result["effective_hint"] = effective
        result["live_referencer_count"] = live_count
        return result

    def update_block_hints(self, segment_id: str, new_hint: str) -> dict[str, Any]:
        """WIRES Phase E2 step 3 — flip lifecycle_hint on tagged blocks.

        Walks the registered blocks for ``segment_id``, validates each
        is still tagged with the same id (self-heals stale entries),
        and updates ``block.lifecycle_hint = new_hint``. Returns a
        small status dict for the HTTP endpoint.

        NOTE (M3): this is the LOW-LEVEL block-flip primitive. Callers
        with a per-instance hint should go through
        ``apply_per_instance_hint`` instead so the engine performs the
        cross-instance max-priority aggregation. This method remains
        public for legacy callers and tests that have already computed
        the effective hint.

        Stale blocks (``_segment_id`` no longer matches — e.g., evicted
        and reused) are removed from the reverse map.
        """

        # WIRES Phase B: route the hint flip through the block_pool's
        # queue so blocks currently in a free pool actually move to the
        # right pool (docs/v2/32 §2.3). Falls back to direct field
        # mutation when no updater is registered (e.g. unit tests
        # that don't construct a real block_pool); in that fallback,
        # pool membership is inconsistent — but those tests don't
        # exercise eviction order.
        updater = _block_hint_updater
        with self._lock:
            blocks = self._blocks_by_segment_id.get(segment_id, [])
            updated = 0
            kept: list[Any] = []
            skipped = 0
            for blk in blocks:
                # M18: stale-tag check now uses tuple membership.
                # Legacy fakes that only expose ``_segment_id`` (not
                # ``_segment_ids``) fall through the empty default and
                # are correctly treated as stale.
                tagged_ids = getattr(blk, "_segment_ids", ())
                if segment_id not in tagged_ids:
                    skipped += 1
                    continue
                # M18 (doc 32 §2.3 "min wins"): per-block effective
                # hint is the MIN-priority (no < may < must) hint
                # across all overlapping segments. Targeted
                # ``segment_id`` contributes ``new_hint``; other
                # overlapping ids contribute their current registry
                # effective hint (default ``"may"`` when unregistered).
                # Single-tag blocks (the common case) take a fast
                # path that bypasses the recompute and behaves
                # exactly like pre-M18 (the new hint is the only
                # contributor, so it IS the effective).
                if len(tagged_ids) == 1:
                    effective_for_block = new_hint
                else:
                    effective_for_block = self._compute_min_hint_locked(
                        tagged_ids, segment_id, new_hint
                    )
                try:
                    if updater is not None:
                        updater(blk, effective_for_block)
                    else:
                        blk.lifecycle_hint = effective_for_block
                    updated += 1
                    kept.append(blk)
                except AttributeError:
                    skipped += 1
            if kept:
                self._blocks_by_segment_id[segment_id] = kept
            else:
                self._blocks_by_segment_id.pop(segment_id, None)
            return {
                "updated": updated,
                "skipped": skipped,
                "reject_reason": None
                if updated > 0
                else ("no_blocks_for_segment" if not blocks else "all_stale"),
            }

    def _compute_min_hint_locked(
        self,
        tagged_ids: tuple[str, ...],
        target_segment_id: str,
        target_new_hint: str,
    ) -> str:
        """M18 helper: min-priority hint over all overlapping segments.

        Caller MUST hold ``self._lock``. ``target_segment_id`` is the
        segment whose hint is being flipped right now (use
        ``target_new_hint`` instead of the registry-stored value);
        every other id in ``tagged_ids`` contributes its current
        registry effective hint, defaulting to ``"may"`` if no entry
        exists yet (matches the empty-live-set default elsewhere).
        Priority order: ``no`` (0) < ``may`` (1) < ``must`` (2);
        "min wins" means we pick the LOWEST-priority hint.
        """

        priority = SegmentEntry._HINT_PRIORITY
        chosen: str | None = None
        for seg in tagged_ids:
            if seg == target_segment_id:
                contrib = target_new_hint
            else:
                other_entry = self._entries_by_segment_id.get(seg)
                contrib = (
                    other_entry.compute_effective_hint()
                    if other_entry is not None
                    else "may"
                )
            if chosen is None or priority.get(contrib, 1) < priority.get(chosen, 1):
                chosen = contrib
        # tagged_ids was non-empty by construction (called only on the
        # multi-tag branch); chosen is therefore guaranteed non-None.
        assert chosen is not None
        return chosen

    def _entry_by_segment_id_locked(self, segment_id: str) -> SegmentEntry | None:
        """Lookup a segment entry by its ``segment_id``.

        The (family_id, token_hash) keyspace is the canonical registry
        key, but block-pool hooks only know which segment they were
        allocated for. ``_entries_by_segment_id`` is a secondary index
        kept in lock-step with the primary one.
        """

        return self._entries_by_segment_id.get(segment_id)

    def get_telemetry(self, segment_id: str) -> dict[str, Any] | None:
        """Return the redacted full state for ``segment_id``.

        Returns ``None`` when no entry has been registered for that
        segment id. The returned dict is the same shape as
        ``redacted_status()`` plus the Phase E4 counters
        (already included by ``redacted_status``); never exposes raw
        token ids.
        """

        with self._lock:
            entry = self._entry_by_segment_id_locked(segment_id)
            if entry is None:
                return None
            return entry.redacted_status()

    def record_cache_hit(self, segment_id: str) -> None:
        """Increment cache_hit_count for ``segment_id`` if registered.

        No-op for unknown segment ids — block_pool hooks fire for every
        block and the vast majority of blocks are not WIRES-tagged, so
        this must be cheap and never raise.
        """

        with self._lock:
            entry = self._entry_by_segment_id_locked(segment_id)
            if entry is None:
                return
            entry.cache_hit_count += 1

    def record_retention(self, segment_id: str) -> None:
        """Increment retention_count for ``segment_id`` if registered.

        Called on each allocation pass where the block-pool's priority
        traversal kept this segment in cache (i.e. did NOT evict it).
        Together with ``record_eviction`` this lets us compute the
        survival ratio per lifecycle_hint class.
        """

        with self._lock:
            entry = self._entry_by_segment_id_locked(segment_id)
            if entry is None:
                return
            entry.retention_count += 1

    def record_eviction(self, segment_id: str, hint: str | None) -> None:
        """Increment ``evict_count_by_hint[hint]`` for ``segment_id``.

        ``hint`` defaults to ``"may"`` when not provided so the bucket
        names match ``KVCacheBlock.lifecycle_hint`` values verbatim.
        """

        bucket = hint if isinstance(hint, str) and hint else "may"
        with self._lock:
            entry = self._entry_by_segment_id_locked(segment_id)
            if entry is None:
                return
            entry.evict_count_by_hint[bucket] = (
                entry.evict_count_by_hint.get(bucket, 0) + 1
            )

    def _evict_locked(self) -> None:
        max_outstanding = _segment_max_outstanding()
        while len(self._entries_by_key) > max_outstanding:
            _evicted_key, evicted_entry = self._entries_by_key.popitem(last=False)
            self._entries_by_action_id.pop(evicted_entry.action_id, None)
            # Only drop the secondary index entry if it still maps to
            # this exact entry (a later submit_prepare for the same
            # segment_id would have superseded it).
            if (
                evicted_entry.segment_id
                and self._entries_by_segment_id.get(evicted_entry.segment_id)
                is evicted_entry
            ):
                self._entries_by_segment_id.pop(evicted_entry.segment_id, None)


_registry = SegmentRegistry()


def reset_segment_registry_for_tests() -> None:
    """Test-only helper to clear the module-level registry."""

    _registry.reset_for_tests()


# ----- Internal prefill trigger -------------------------------------------


async def _maybe_submit_segment_prewarm(
    action: dict[str, Any],
    *,
    chat_handler: Any | None,
    refresh: bool,
) -> tuple[str, int]:
    """Bridge to the engine's optional segment-prewarm hook.

    Returns ``(prefill_status, prefill_token_count)``. When the chat
    handler does not expose a segment-prewarm method, returns
    ``("prefill_only_unavailable", N)`` so the registry still
    accepts the action and the runner can validate the wire contract
    end-to-end.
    """

    prompt_token_ids = action.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list):
        prompt_token_ids = []
    prefill_token_count = len(prompt_token_ids)
    mode = _segment_prepare_mode()
    if mode == "tokenize_only":
        return "tokenize_only", prefill_token_count
    if mode not in _PREWARM_PREPARE_MODES:
        return "tokenize_only", prefill_token_count
    if chat_handler is None:
        return "prefill_only_unavailable", prefill_token_count
    method_name = (
        "submit_workflow_segment_refresh"
        if refresh
        else "submit_workflow_segment_prewarm"
    )
    submit = getattr(chat_handler, method_name, None)
    if not callable(submit):
        return "prefill_only_unavailable", prefill_token_count
    try:
        result = await submit(action)
    except Exception:  # noqa: BLE001 - defensive: server must keep serving
        return "prewarm_failed:submit_exception", prefill_token_count
    if not isinstance(result, dict):
        return "prewarm_failed:invalid_submit_response", prefill_token_count
    status = result.get("prewarm_status")
    if not isinstance(status, str) or not status:
        status = "prewarm_submitted"
    reported_count = result.get("prefill_token_count")
    if isinstance(reported_count, int) and reported_count >= 0:
        prefill_token_count = reported_count
    return status, prefill_token_count


# ----- Public entry points ------------------------------------------------


def _e2_decide_outcome(
    *,
    rejected: bool,
    reject_reason: str | None,
    prefill_status: str,
) -> str:
    """Map handler exit state to the E2 ``outcome`` enum (proposal §3.2).

    The four enum values are:
    * ``rejected_bad_input`` — validator returned a rejection.
    * ``rejected_dedup`` — registry-level dedup signalled by the
      ``reject_reason`` payload (``duplicate_*`` family).
    * ``noop_already_present`` — accepted, but no actual prefill was
      attempted (registry already had a hot entry; in M9 hint-only
      refresh this is also the normal path).
    * ``prepared`` — prefill / re-touch actually fired.
    """
    if rejected and reject_reason and reject_reason.startswith("duplicate"):
        return ACTION_OUTCOME_REJECTED_DEDUP
    if rejected:
        return ACTION_OUTCOME_REJECTED_BAD_INPUT
    if prefill_status in {
        "prewarm_submitted",
        "prewarm_completed",
        _REFRESH_PREWARM_STATUS_RETOUCHED,
    }:
        return ACTION_OUTCOME_PREPARED
    # not_attempted / hint_only / prefill_only_unavailable / etc.
    return ACTION_OUTCOME_NOOP_ALREADY_PRESENT


def _e2_emit(
    *,
    endpoint: str,
    action: dict[str, Any],
    response: dict[str, Any] | None,
    rejection: dict[str, Any] | None,
    prefill_status: str,
    prefill_token_count: int,
    started_at: float,
) -> None:
    """Build + push one E2 row. Pure cold-path: only fires after a
    ``/v1/coopt/segment_*`` POST handler has returned."""
    elapsed_ms = (time.monotonic() - started_at) * 1000.0
    rejected = rejection is not None
    reject_reason = (
        _optional_str(rejection.get("reject_reason")) if rejection is not None else None
    )
    outcome = _e2_decide_outcome(
        rejected=rejected,
        reject_reason=reject_reason,
        prefill_status=prefill_status,
    )
    scope_key = _optional_str(action.get("segment_id"))
    if outcome == ACTION_OUTCOME_PREPARED:
        blocks_newly_written = prefill_token_count
        blocks_already_present = 0
    else:
        blocks_newly_written = 0
        blocks_already_present = prefill_token_count
    blocks_touched = blocks_newly_written + blocks_already_present
    _wires_emit_action(
        endpoint=endpoint,
        scope_key=scope_key,
        blocks_touched=blocks_touched,
        blocks_already_present=blocks_already_present,
        blocks_newly_written=blocks_newly_written,
        outcome=outcome,
        elapsed_ms=elapsed_ms,
    )


async def submit_segment_prepare_action(
    action: dict[str, Any],
    *,
    chat_handler: Any | None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    rejection = _validate_segment_prepare(action)
    if rejection is not None:
        _emit_segment_telemetry(
            action_id=_optional_str(action.get("action_id")),
            action_kind=SEGMENT_PREPARE_ACTION_KIND,
            lifecycle_status=str(rejection.get("lifecycle_status") or "rejected"),
            reject_reason=_optional_str(rejection.get("reject_reason")),
            response=None,
            request_action=action,
        )
        _e2_emit(
            endpoint=ENDPOINT_SEGMENT_PREPARE,
            action=action,
            response=None,
            rejection=rejection,
            prefill_status="not_attempted",
            prefill_token_count=0,
            started_at=started_at,
        )
        return rejection
    prefill_status, prefill_token_count = await _maybe_submit_segment_prewarm(
        action,
        chat_handler=chat_handler,
        refresh=False,
    )
    response = _registry.submit_prepare(
        action,
        prefill_status=prefill_status,
        prefill_token_count=prefill_token_count,
    )
    _emit_segment_telemetry(
        action_id=_optional_str(response.get("action_id")),
        action_kind=SEGMENT_PREPARE_ACTION_KIND,
        lifecycle_status=str(response.get("lifecycle_status") or "accepted"),
        reject_reason=None,
        response=response,
        request_action=action,
    )
    _e2_emit(
        endpoint=ENDPOINT_SEGMENT_PREPARE,
        action=action,
        response=response,
        rejection=None,
        prefill_status=prefill_status,
        prefill_token_count=prefill_token_count,
        started_at=started_at,
    )
    return response


async def submit_segment_refresh_action(
    action: dict[str, Any],
    *,
    chat_handler: Any | None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    rejection = _validate_segment_refresh(action)
    if rejection is not None:
        _emit_segment_telemetry(
            action_id=_optional_str(action.get("action_id")),
            action_kind=SEGMENT_REFRESH_ACTION_KIND,
            lifecycle_status=str(rejection.get("lifecycle_status") or "rejected"),
            reject_reason=_optional_str(rejection.get("reject_reason")),
            response=None,
            request_action=action,
        )
        _e2_emit(
            endpoint=ENDPOINT_SEGMENT_REFRESH,
            action=action,
            response=None,
            rejection=rejection,
            prefill_status="not_attempted",
            prefill_token_count=0,
            started_at=started_at,
        )
        return rejection
    # M9 merge: detect mode from validated payload.
    raw_token_ids = action.get("prompt_token_ids")
    has_token_ids = isinstance(raw_token_ids, list) and len(raw_token_ids) > 0
    lifecycle_hint = action.get("lifecycle_hint")
    if has_token_ids:
        prefill_status, prefill_token_count = await _maybe_submit_segment_prewarm(
            action,
            chat_handler=chat_handler,
            refresh=True,
        )
        # Refresh always reports the dedicated retouched marker on
        # success to make telemetry distinguishable from the prepare
        # path even if the underlying chat-handler hook returns a
        # generic value.
        if prefill_status in {"prewarm_submitted", "prewarm_completed"}:
            prefill_status = _REFRESH_PREWARM_STATUS_RETOUCHED
    else:
        # Hint-only mode (M9): no LRU touch, just flip block hints.
        # Mark prefill as not-attempted so telemetry distinguishes the
        # two modes; the registry still tracks the refresh event.
        prefill_status = "hint_only"
        prefill_token_count = 0
    response = _registry.submit_refresh(
        action,
        prefill_status=prefill_status,
        prefill_token_count=prefill_token_count,
    )
    # M9 merge: when lifecycle_hint is non-null, also flip block hints
    # by segment_id (subsumes the legacy /v1/coopt/segment_lifecycle_update
    # endpoint).
    if isinstance(lifecycle_hint, str) and lifecycle_hint in _VALID_LIFECYCLE_HINTS:
        segment_id = _optional_str(action.get("segment_id"))
        # M3 (paper §3.5): extract optional cross-caller aggregation
        # fields from the refresh action body so the engine can key
        # per-caller hint contributions on (runner_id, instance_id).
        # Backward-compat: missing/empty falls through to the engine's
        # default ``"default"`` runner + ``_anon`` instance slot.
        # M3 ``literal_no``: when true, ``new_hint="no"`` records as a
        # literal "evict ASAP" contribution rather than dropping the
        # caller's slot from the live set. Defaults False.
        raw_runner_id = action.get("runner_id")
        runner_id = (
            raw_runner_id if isinstance(raw_runner_id, str) and raw_runner_id else None
        )
        raw_instance_id = action.get("instance_id")
        instance_id = (
            raw_instance_id
            if isinstance(raw_instance_id, str) and raw_instance_id
            else None
        )
        literal_no = bool(action.get("literal_no") or False)
        hint_result = update_segment_lifecycle_hint(
            segment_id,
            lifecycle_hint,
            runner_id=runner_id,
            instance_id=instance_id,
            literal_no=literal_no,
        )
        # Surface the hint-update result on the response so callers can
        # observe both effects in one call. Field names are scoped under
        # ``hint_update_*`` to avoid colliding with the refresh fields.
        response["hint_update_applied_hint"] = lifecycle_hint
        response["hint_update_updated"] = int(hint_result.get("updated") or 0)
        response["hint_update_skipped"] = int(hint_result.get("skipped") or 0)
        response["hint_update_reject_reason"] = hint_result.get("reject_reason")
    else:
        response["hint_update_applied_hint"] = None
        response["hint_update_updated"] = 0
        response["hint_update_skipped"] = 0
        response["hint_update_reject_reason"] = None
    _emit_segment_telemetry(
        action_id=_optional_str(response.get("action_id")),
        action_kind=SEGMENT_REFRESH_ACTION_KIND,
        lifecycle_status=str(response.get("lifecycle_status") or "accepted"),
        reject_reason=None,
        response=response,
        request_action=action,
    )
    _e2_emit(
        endpoint=ENDPOINT_SEGMENT_REFRESH,
        action=action,
        response=response,
        rejection=None,
        prefill_status=prefill_status,
        prefill_token_count=prefill_token_count,
        started_at=started_at,
    )
    return response


def get_segment_action(action_id: str) -> dict[str, Any] | None:
    return _registry.get_by_action_id(action_id)


# ----- Phase E4 cache telemetry helpers ----------------------------------


def get_segment_telemetry(segment_id: str) -> dict[str, Any] | None:
    """Module-level shortcut to ``_registry.get_telemetry`` for the
    ``GET /v1/coopt/segment_telemetry/{segment_id}`` route and any
    callers that hold a segment_id but not a registry handle.
    """

    return _registry.get_telemetry(segment_id)


def record_segment_cache_hit(segment_id: str | None) -> None:
    """Block-pool entry point: record a cache hit for ``segment_id``.

    Tolerates ``None`` so the block_pool hook can pass a block's
    ``_segment_id`` attribute unconditionally without branching.
    """

    if not isinstance(segment_id, str) or not segment_id:
        return
    _registry.record_cache_hit(segment_id)


def record_segment_retention(segment_id: str | None) -> None:
    """Record that this segment's blocks survived an allocation pass."""

    if not isinstance(segment_id, str) or not segment_id:
        return
    _registry.record_retention(segment_id)


def record_segment_eviction(segment_id: str | None, hint: str | None) -> None:
    """Record an eviction of a block tagged with ``segment_id``.

    ``hint`` is the block's ``lifecycle_hint`` at the moment the
    eviction fires; ``None`` is treated as the default ``"may"``.
    """

    if not isinstance(segment_id, str) or not segment_id:
        return
    _registry.record_eviction(segment_id, hint)


def tag_blocks_with_segment_id(blocks: Any, segment_id: str | None) -> None:
    """Stamp ``segment_id`` onto the iterable of KVCacheBlocks.

    Called from the segment_prepare prefill helper once
    ``kv_cache_manager.allocate_slots`` returns the freshly-allocated
    blocks for a WIRES segment. Non-WIRES blocks remain untagged
    (``_segment_id is None``) so the block-pool hooks short-circuit.

    Also registers each tagged block in the global
    ``segment_id -> blocks`` reverse map used by Phase E2 step 3's
    merged ``POST /v1/coopt/segment_refresh`` endpoint
    (M9: ``lifecycle_hint`` field) to find blocks by segment_id (e.g.,
    when the workflow monitor downgrades must -> no after a loop exit).

    Best-effort: silently ignores blocks that don't have the attribute
    (prevents type coupling to KVCacheBlock from this layer).
    """

    if not isinstance(segment_id, str) or not segment_id:
        return
    if blocks is None:
        return
    try:
        iterator = iter(blocks)
    except TypeError:
        return
    for blk in iterator:
        # M18: a block can carry MULTIPLE overlapping segment ids when
        # segment boundaries don't align with block boundaries (doc 32
        # §2.3 "min wins"). Append this segment_id to the block's
        # ``_segment_ids`` tuple if not already present (idempotent
        # re-tag) so the previous segment's reverse-index entry is
        # NOT lost on a second tag. ``_segment_ids[0]`` is the primary
        # tag (the first segment to claim the block); subsequent
        # tags accumulate.
        try:
            existing = getattr(blk, "_segment_ids", ())
            if segment_id not in existing:
                blk._segment_ids = existing + (segment_id,)  # noqa: SLF001
            _registry.register_block_for_segment(segment_id, blk)
        except AttributeError:
            continue


def mark_segment_blocks_speculative(segment_id: str | None) -> dict[str, Any]:
    """M20: speculative-promote every block currently tagged for
    ``segment_id``.

    Each block is moved to the must pool with
    ``source_class="speculative"`` (short TTL backstop, default 30s,
    env ``WIRES_KVCACHE_SPECULATIVE_TTL_MS``). The class upgrades back
    to ``"unstructured"`` on the first cache hit (see
    ``BlockPool.touch()``); if no hit lands before the TTL, the lazy
    sweep demotes the block and increments ``speculation_miss_count``.

    Returns ``{"updated": N, "skipped": K, "reject_reason": str|None}``.
    Engine-side primitive only: paper §3.2 leaves the runner-side
    speculation patterns to future work, so there is no built-in caller
    here. Tests and future runner code can invoke this directly after
    ``submit_segment_prepare_action`` has tagged the segment's blocks.
    """

    if not isinstance(segment_id, str) or not segment_id:
        return {"updated": 0, "skipped": 0, "reject_reason": "missing_segment_id"}
    promoter = _speculative_block_promoter
    with _registry._lock:
        blocks = _registry._blocks_by_segment_id.get(segment_id, [])
        updated = 0
        skipped = 0
        kept: list[Any] = []
        for blk in blocks:
            # Stale-tag check on M18 tuple field. The block is live for
            # this segment iff segment_id appears in the tuple. Test fakes
            # MUST expose ``_segment_ids`` (the legacy ``_segment_id``
            # backward-compat property was removed in the M18 cleanup).
            tagged_ids = getattr(blk, "_segment_ids", ())
            if segment_id not in tagged_ids:
                skipped += 1
                continue
            try:
                if promoter is not None:
                    promoter(blk)
                else:
                    # Fallback path (no block_pool wired, e.g. unit
                    # tests). Stamp the field directly so subsequent
                    # promote-on-append picks the speculative TTL.
                    blk.source_class = "speculative"
                    blk.lifecycle_hint = "must"
                updated += 1
                kept.append(blk)
            except AttributeError:
                skipped += 1
        if kept:
            _registry._blocks_by_segment_id[segment_id] = kept
        else:
            _registry._blocks_by_segment_id.pop(segment_id, None)
        return {
            "updated": updated,
            "skipped": skipped,
            "reject_reason": None
            if updated > 0
            else ("no_blocks_for_segment" if not blocks else "all_stale"),
        }


def mark_segment_prepare_prefilled(
    *,
    action_id: str | None,
    prefill_status: str,
    prefill_token_count: int | None = None,
) -> dict[str, Any]:
    """Phase E5 — record the result of an internal segment prewarm prefill.

    Called from the OpenAIServingChat ``_consume_workflow_segment_prewarm``
    background coroutine after the hidden ``workflow_prefill_only=True``
    request finishes (or fails). Locates the SegmentEntry by
    ``action_id`` and updates its ``prefill_status`` so subsequent
    ``GET /v1/coopt/segment_telemetry/{segment_id}`` calls report the
    real engine state ("prefilled" / "prewarm_failed:..." / etc.) rather
    than the placeholder ``"tokenize_only"`` the registry was initialised
    with.

    Returns ``{updated: bool, reject_reason: str | None}``. ``updated``
    is True when the entry was found and mutated; False when the
    action_id is unknown (e.g. the registry evicted the entry before
    the async prefill returned).
    """

    if not isinstance(action_id, str) or not action_id:
        return {"updated": False, "reject_reason": "missing_action_id"}
    if not isinstance(prefill_status, str) or not prefill_status:
        return {"updated": False, "reject_reason": "missing_prefill_status"}
    return _registry.mark_prefilled(
        action_id=action_id,
        prefill_status=prefill_status,
        prefill_token_count=prefill_token_count,
    )


def update_segment_lifecycle_hint(
    segment_id: str | None,
    new_hint: str | None,
    *,
    runner_id: str | None = None,
    instance_id: str | None = None,
    literal_no: bool = False,
) -> dict[str, Any]:
    """Phase E2 step 3 + M3 — flip the lifecycle_hint on tagged blocks.

    The workflow monitor (M9: now via the merged
    ``POST /v1/coopt/segment_refresh`` endpoint with
    ``lifecycle_hint`` set in the body) calls this after a CFG event
    narrows the segment's future consumer set (e.g., loop exit → "no",
    or loop continuation → "must"). The new hint propagates to all
    currently-tagged blocks for that segment_id; subsequent
    ``FreeKVCacheBlockQueue.popleft_n`` calls honor it via the
    3-priority traversal (no → may → must).

    M3 (paper §3 / §3.5 cross-instance aggregation): callers are
    identified by the ``(runner_id, instance_id)`` tuple. The engine
    records this as ONE caller's contribution and re-derives the
    effective block hint as ``max(must, may, no)`` over all currently-
    live referencers of the segment. Independent runners (different
    teams / processes / orgs) do not coordinate the ``instance_id``
    namespace; keying on ``(runner_id, instance_id)`` ensures runner
    A's "no" never yanks runner B's contribution even when they
    happen to share an ``instance_id`` value.

    Two callers each calling with hints ``must`` and ``may`` resolve
    to ``must``; if the must-caller later demotes to ``may``, the
    effective stays ``may``; if ALL callers drop their contribution
    (call with ``new_hint="no"`` and the default ``literal_no=False``),
    the segment's effective hint defaults to ``"may"``.

    When ``runner_id`` is omitted, falls back to the ``"default"``
    runner; when ``instance_id`` is omitted, falls back to the
    synthetic ``_anon`` slot. Single-tenant callers that omit both
    therefore reduce to the original last-write-wins behavior.

    ``literal_no``: if true, ``new_hint="no"`` is recorded as a LITERAL
    ``no`` contribution rather than removing the slot. Use for the
    loop-exit "force eviction" case where one caller wants the block
    evicted ASAP regardless of other live referencers.

    Self-healing on stale tags: if a block's ``_segment_id`` no longer
    matches (e.g., evicted and reused), it is silently skipped and
    removed from the registry's reverse map.

    Returns a small status dict: ``{updated: int, skipped: int,
    reject_reason: str | None, effective_hint: str,
    live_referencer_count: int}``. ``reject_reason`` is set if the
    segment_id is unknown or new_hint is invalid;
    ``updated``/``skipped`` are zero in those cases.
    ``effective_hint`` and ``live_referencer_count`` are present on
    every accepted call and let observability tools see what the
    engine actually decided after aggregation.
    """

    if not isinstance(segment_id, str) or not segment_id:
        return {
            "updated": 0,
            "skipped": 0,
            "reject_reason": "segment_id_missing",
        }
    if new_hint not in ("must", "may", "no"):
        return {
            "updated": 0,
            "skipped": 0,
            "reject_reason": "invalid_hint",
        }
    return _registry.apply_per_instance_hint(
        segment_id,
        runner_id,
        instance_id,
        new_hint,
        literal_no=literal_no,
    )


def release_segment_instance_contribution(
    segment_id: str | None,
    runner_id: str | None,
    instance_id: str | None,
) -> dict[str, Any]:
    """M3: explicitly drop one caller's contribution from a segment's
    live-hint set (e.g., when the instance completes / crashes /
    is cancelled). Caller identity is the ``(runner_id, instance_id)``
    tuple; runner A releasing its instance does NOT touch runner B's
    contribution even when both happen to use the same
    ``instance_id`` value.

    Re-derives the effective hint and pushes it through to the
    block_pool.

    Idempotent: dropping an unknown ``(runner_id, instance_id)``
    pair is a no-op that just returns the current effective hint."""

    if not isinstance(segment_id, str) or not segment_id:
        return {
            "updated": 0,
            "skipped": 0,
            "reject_reason": "segment_id_missing",
        }
    with _registry._lock:  # noqa: SLF001
        entry = _registry._entry_by_segment_id_locked(segment_id)  # noqa: SLF001
        if entry is None:
            return {
                "updated": 0,
                "skipped": 0,
                "reject_reason": "segment_not_found",
                "effective_hint": "may",
                "live_referencer_count": 0,
            }
        effective = entry.release_instance_contribution(runner_id, instance_id)
        live_count = len(entry.live_hints_by_caller)
    result = _registry.update_block_hints(segment_id, effective)
    result["effective_hint"] = effective
    result["live_referencer_count"] = live_count
    return result


# ----- Telemetry ----------------------------------------------------------


def _emit_segment_telemetry(
    *,
    action_id: str | None,
    action_kind: str,
    lifecycle_status: str,
    reject_reason: str | None,
    response: dict[str, Any] | None,
    request_action: dict[str, Any],
) -> None:
    response_view: dict[str, Any] = response if response is not None else {}
    expected_consumers = _optional_int(response_view.get("expected_consumers"))
    if expected_consumers is None:
        expected_consumers = _optional_int(request_action.get("expected_consumers"))
    # Phase E4: forward the engine-side cache counters into the test
    # hook record so a single workflow_segment_action event captures
    # both the wire-action lifecycle AND the latest cache-event totals.
    cache_hit_count = _optional_int(response_view.get("cache_hit_count"))
    retention_count = _optional_int(response_view.get("retention_count"))
    evict_by_hint = response_view.get("evict_count_by_hint")
    if isinstance(evict_by_hint, dict):
        evict_total: int | None = sum(
            value for value in evict_by_hint.values() if isinstance(value, int)
        )
    else:
        evict_total = None
    record_workflow_segment_action(
        action_id=action_id,
        action_kind=action_kind,
        lifecycle_status=lifecycle_status,
        reject_reason=reject_reason,
        segment_id=_optional_str(request_action.get("segment_id")),
        segment_family_id=_resolve_family_id(request_action),
        segment_parent_id=_optional_str(request_action.get("parent_segment_id")),
        segment_expected_consumers=expected_consumers,
        segment_observed_consumers=_optional_int(
            response_view.get("observed_consumers")
        ),
        segment_lifecycle_status=_optional_str(request_action.get("lifecycle_status")),
        segment_status=_optional_str(response_view.get("status")),
        segment_would_evict_without_refresh=_optional_bool(
            request_action.get("would_evict_without_refresh")
        ),
        segment_prefill_token_count=_optional_int(
            response_view.get("prefill_token_count")
        ),
        segment_prefill_status=_optional_str(response_view.get("prefill_status")),
        segment_engine_seen_token_hash=(
            _optional_str(response_view.get("engine_seen_token_hash"))
            or _normalize_token_hash(request_action)
        ),
        segment_ttl_ms=_optional_int(response_view.get("ttl_ms")),
        segment_expires_at_unix_ms=_optional_int(
            response_view.get("expires_at_unix_ms")
        ),
        engine_segment_cache_hit_count=cache_hit_count,
        engine_segment_retention_count=retention_count,
        engine_segment_evict_count=evict_total,
        model=_optional_str(request_action.get("model")),
    )


# ----- HTTP routes --------------------------------------------------------


def _disabled_response() -> JSONResponse:
    return JSONResponse(
        content={
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "workflow_coopt_actions_disabled",
            "engine_wire_status": "rejected",
        },
        status_code=404,
    )


def _bad_payload_response(reason: str) -> JSONResponse:
    return JSONResponse(
        content={
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": reason,
            "engine_wire_status": "rejected",
        },
        status_code=400,
    )


@router.post("/v1/coopt/segment_prepare")
async def http_submit_segment_prepare(raw_request: Request) -> JSONResponse:
    if not workflow_coopt_actions_enabled():
        return _disabled_response()
    try:
        payload = await raw_request.json()
    except Exception:  # noqa: BLE001
        return _bad_payload_response("invalid_json")
    if not isinstance(payload, dict):
        return _bad_payload_response("expected_object")
    chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
    response = await submit_segment_prepare_action(payload, chat_handler=chat_handler)
    status_code = 200 if response.get("accepted") is True else 400
    return JSONResponse(content=response, status_code=status_code)


@router.post("/v1/coopt/segment_refresh")
async def http_submit_segment_refresh(raw_request: Request) -> JSONResponse:
    if not workflow_coopt_actions_enabled():
        return _disabled_response()
    try:
        payload = await raw_request.json()
    except Exception:  # noqa: BLE001
        return _bad_payload_response("invalid_json")
    if not isinstance(payload, dict):
        return _bad_payload_response("expected_object")
    chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
    response = await submit_segment_refresh_action(payload, chat_handler=chat_handler)
    status_code = 200 if response.get("accepted") is True else 400
    return JSONResponse(content=response, status_code=status_code)


@router.get("/v1/coopt/segment_prepare/{action_id}")
async def http_get_segment_prepare(action_id: str) -> JSONResponse:
    if not workflow_coopt_actions_enabled():
        return _disabled_response()
    response = get_segment_action(action_id)
    if response is None:
        return JSONResponse(
            content={
                "action_id": action_id,
                "lifecycle_status": "missing",
                "reject_reason": "action_not_found",
                "engine_wire_status": "missing",
            },
            status_code=404,
        )
    return JSONResponse(content=response)


@router.get("/v1/coopt/segment_refresh/{action_id}")
async def http_get_segment_refresh(action_id: str) -> JSONResponse:
    if not workflow_coopt_actions_enabled():
        return _disabled_response()
    response = get_segment_action(action_id)
    if response is None:
        return JSONResponse(
            content={
                "action_id": action_id,
                "lifecycle_status": "missing",
                "reject_reason": "action_not_found",
                "engine_wire_status": "missing",
            },
            status_code=404,
        )
    return JSONResponse(content=response)


@router.get("/v1/coopt/segment_telemetry/{segment_id}")
async def http_get_segment_telemetry(segment_id: str) -> JSONResponse:
    """WIRES Phase E4 — per-segment cache telemetry surface.

    Returns the registry entry's ``redacted_status()`` payload, which
    includes the standard segment_prepare lifecycle fields PLUS the
    Phase E4 counters:

    * ``cache_hit_count`` — block-pool ``touch()`` hits attributed to
      this segment's cached blocks.
    * ``retention_count`` — number of allocation passes where this
      segment's blocks survived eviction.
    * ``evict_count_by_hint`` — dict keyed on the block's
      ``lifecycle_hint`` ("no" / "may" / "must") at the moment its
      block was evicted from the cache. Lets us verify ``must`` blocks
      rarely evict and ``no`` blocks evict first.

    Returns 404 when no segment with that id has been registered;
    raw token ids are never serialized (same redaction contract as
    ``GET /v1/coopt/segment_prepare/{action_id}``).
    """

    if not workflow_coopt_actions_enabled():
        return _disabled_response()
    response = get_segment_telemetry(segment_id)
    if response is None:
        return JSONResponse(
            content={
                "segment_id": segment_id,
                "lifecycle_status": "missing",
                "reject_reason": "segment_not_found",
                "engine_wire_status": "missing",
            },
            status_code=404,
        )
    return JSONResponse(content=response)


@router.post("/v1/coopt/segment_lifecycle_update")
async def http_segment_lifecycle_update(raw_request: Request) -> JSONResponse:
    """M9 (paper/CODE_MISMATCH_NOTES.md): MERGED into segment_refresh.

    The legacy
    ``POST /v1/coopt/segment_lifecycle_update {segment_id, new_hint}``
    endpoint is removed: per the paper §3.4 unified
    ``segment_refresh`` action (LRU touch + lifecycle hint update),
    callers must POST ``/v1/coopt/segment_refresh`` with
    ``lifecycle_hint`` set in the body. The merged endpoint accepts a
    "hint-only" mode (no ``prompt_token_ids``) so callers that have
    no token ids at hand (e.g. loop-anchor must-promotion at loop
    entry) can still flip block hints without performing an LRU touch.

    Returns HTTP 410 Gone with a structured error body so legacy
    callers fail loudly with a clear migration message rather than
    silently 404'ing or sending hint updates that never apply.
    """

    if not workflow_coopt_actions_enabled():
        return _disabled_response()
    # M9 (paper/CODE_MISMATCH_NOTES.md): legacy /v1/coopt/segment_lifecycle_update
    # endpoint REMOVED — its behavior is subsumed by /v1/coopt/segment_refresh
    # with ``lifecycle_hint`` in the body (hint-only mode). Return HTTP 410
    # Gone with redirect message so legacy clients fail loudly. The M3
    # runner_id / instance_id / literal_no extraction that previously lived
    # here now lives in the segment_refresh handler (see _validate_segment_refresh
    # + submit_segment_refresh_action).
    return JSONResponse(
        content={
            "error": (
                "merged into segment_refresh; pass lifecycle_hint in the "
                "POST /v1/coopt/segment_refresh body"
            ),
            "redirect_to": "/v1/coopt/segment_refresh",
            "redirect_body_field": "lifecycle_hint",
            "removed_at": "M9",
            "engine_wire_status": "gone",
            "lifecycle_status": "rejected",
            "reject_reason": "endpoint_merged_m9",
        },
        status_code=410,
    )
