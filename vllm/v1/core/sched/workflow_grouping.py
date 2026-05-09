# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

WORKFLOW_GROUP_AWARE_SCHEDULING_ENV = "WORKFLOW_GROUP_AWARE_SCHEDULING"
WORKFLOW_GROUP_AWARE_MAX_BURST_ENV = "WORKFLOW_GROUP_AWARE_MAX_BURST"
WORKFLOW_GROUP_AWARE_MAX_QUEUE_SCAN_ENV = "WORKFLOW_GROUP_AWARE_MAX_QUEUE_SCAN"
WORKFLOW_GROUP_AWARE_MAX_GROUP_DELAY_MS_ENV = (
    "WORKFLOW_GROUP_AWARE_MAX_GROUP_DELAY_MS"
)
WORKFLOW_GROUP_AWARE_UNGROUPED_MIN_SHARE_ENV = (
    "WORKFLOW_GROUP_AWARE_UNGROUPED_MIN_SHARE"
)
WORKFLOW_GROUP_AWARE_SCAN_TIME_BUDGET_US_ENV = (
    "WORKFLOW_GROUP_AWARE_SCAN_TIME_BUDGET_US"
)
WORKFLOW_JOIN_TAIL_SCHEDULING_ENV = "WORKFLOW_JOIN_TAIL_SCHEDULING"
DEFAULT_WORKFLOW_GROUP_AWARE_MAX_BURST = 4
DEFAULT_WORKFLOW_GROUP_AWARE_MAX_QUEUE_SCAN = 64
DEFAULT_WORKFLOW_GROUP_AWARE_MAX_GROUP_DELAY_MS = 200.0
DEFAULT_WORKFLOW_GROUP_AWARE_UNGROUPED_MIN_SHARE = 0.2
# 5 ms wall-clock budget per scheduler iteration for the LCP scan. Stops the
# block-split LCP from holding up the engine when the waiting queue is large
# or has long prompts, falling back to whatever best-so-far group has been
# found (None means stock ordering for that iteration).
DEFAULT_WORKFLOW_GROUP_AWARE_SCAN_TIME_BUDGET_US = 5000.0
_UNAVAILABLE_SINGLE_MODEL_ASSUMED = "unavailable_single_model_assumed"


@dataclass(frozen=True, slots=True)
class WorkflowGroupSelection:
    request: Any
    selected_rank: int
    group_key: str | None
    reason: str
    group_source: str | None = None
    token_lcp_len: int | None = None
    token_lcp_hash: str | None = None
    scan_count: int | None = None
    scan_us: float | None = None
    candidate_group_size: int | None = None
    fairness_guard_reason: str | None = None
    queue_head_delay_ms: float | None = None
    queue_head_delay_bucket: str | None = None
    join_tail_scheduling_enabled: bool | None = None
    workflow_join_tail_selected: bool | None = None
    workflow_join_tail_reason: str | None = None
    workflow_join_tail_scan_count: int | None = None
    workflow_join_tail_candidate_count: int | None = None
    workflow_join_tail_remaining_values: tuple[int, ...] | None = None
    workflow_join_tail_fairness_guard_reason: str | None = None


def workflow_group_aware_scheduling_enabled() -> bool:
    return os.getenv(WORKFLOW_GROUP_AWARE_SCHEDULING_ENV, "") == "1"


def workflow_join_tail_scheduling_enabled() -> bool:
    return os.getenv(WORKFLOW_JOIN_TAIL_SCHEDULING_ENV, "") == "1"


def workflow_group_aware_max_burst() -> int:
    raw_value = os.getenv(WORKFLOW_GROUP_AWARE_MAX_BURST_ENV, "")
    if not raw_value:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_BURST
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_BURST
    return max(1, value)


def workflow_group_aware_max_queue_scan() -> int:
    raw_value = os.getenv(WORKFLOW_GROUP_AWARE_MAX_QUEUE_SCAN_ENV, "")
    if not raw_value:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_QUEUE_SCAN
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_QUEUE_SCAN
    return max(1, value)


def workflow_group_aware_max_group_delay_ms() -> float:
    raw_value = os.getenv(WORKFLOW_GROUP_AWARE_MAX_GROUP_DELAY_MS_ENV, "")
    if not raw_value:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_GROUP_DELAY_MS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_GROUP_DELAY_MS
    return max(0.0, value)


def workflow_group_aware_ungrouped_min_share() -> float:
    raw_value = os.getenv(WORKFLOW_GROUP_AWARE_UNGROUPED_MIN_SHARE_ENV, "")
    if not raw_value:
        return DEFAULT_WORKFLOW_GROUP_AWARE_UNGROUPED_MIN_SHARE
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_WORKFLOW_GROUP_AWARE_UNGROUPED_MIN_SHARE
    return min(1.0, max(0.0, value))


def workflow_group_aware_scan_time_budget_us() -> float:
    raw_value = os.getenv(WORKFLOW_GROUP_AWARE_SCAN_TIME_BUDGET_US_ENV, "")
    if not raw_value:
        return DEFAULT_WORKFLOW_GROUP_AWARE_SCAN_TIME_BUDGET_US
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_WORKFLOW_GROUP_AWARE_SCAN_TIME_BUDGET_US
    return max(0.0, value)


def select_workflow_group_request(
    requests: Iterable[Any],
    *,
    last_group_key: str | None,
    group_burst: int,
    max_burst: int | None = None,
    max_queue_scan: int | None = None,
    max_group_delay_ms: float | None = None,
    ungrouped_min_share: float | None = None,
    block_size: int = 16,
    now_s: float | None = None,
) -> WorkflowGroupSelection | None:
    """Select a waiting request by redacted workflow grouping evidence.

    This helper never changes cache state and never looks at raw prompt text.
    Token-prefix grouping only uses concrete scheduler token ids to compute a
    redacted block-aligned LCP key; metadata grouping remains the fallback.
    """

    all_requests = tuple(requests)
    if not all_requests:
        return None

    max_burst = workflow_group_aware_max_burst() if max_burst is None else max_burst
    max_queue_scan = (
        workflow_group_aware_max_queue_scan()
        if max_queue_scan is None
        else max(1, max_queue_scan)
    )
    max_group_delay_ms = (
        workflow_group_aware_max_group_delay_ms()
        if max_group_delay_ms is None
        else max(0.0, max_group_delay_ms)
    )
    ungrouped_min_share = (
        workflow_group_aware_ungrouped_min_share()
        if ungrouped_min_share is None
        else min(1.0, max(0.0, ungrouped_min_share))
    )
    scan_count = min(len(all_requests), max_queue_scan)
    ordered_requests = all_requests[:scan_count]
    queue_head_delay_ms = _queue_head_delay_ms(
        all_requests[0],
        now_s=time.time() if now_s is None else now_s,
    )
    queue_head_delay_bucket = _queue_head_delay_bucket(queue_head_delay_ms)

    # Build per-call snapshots of validated request data so the inner LCP
    # scan does not re-tokenize/re-validate each request O(N^2) times. This
    # is the single biggest perf fix; before this, _request_prompt_token_ids
    # ran an isinstance(...) loop over every token in every request on every
    # pair iteration, which made the scan grow as O(N^3 * T).
    scan_start_ns = time.perf_counter_ns()
    snapshots = _build_request_snapshots(ordered_requests)
    scan_deadline_ns = scan_start_ns + int(
        workflow_group_aware_scan_time_budget_us() * 1000.0
    )
    candidate = _best_group_candidate_from_snapshots(
        snapshots,
        block_size=block_size,
        deadline_ns=scan_deadline_ns,
    )
    scan_us = round((time.perf_counter_ns() - scan_start_ns) / 1000.0, 6)
    if candidate is None:
        return WorkflowGroupSelection(
            request=all_requests[0],
            selected_rank=0,
            group_key=None,
            reason="stock_fallback_no_group",
            group_source=None,
            scan_count=scan_count,
            scan_us=scan_us,
            queue_head_delay_ms=queue_head_delay_ms,
            queue_head_delay_bucket=queue_head_delay_bucket,
        )

    selected_rank = candidate.selected_rank
    group_key = candidate.group_key
    if (
        selected_rank != 0
        and queue_head_delay_ms is not None
        and queue_head_delay_ms >= max_group_delay_ms
    ):
        fallback_keys = workflow_group_keys_for_request(all_requests[0])
        fallback_group_key = fallback_keys[0][1] if fallback_keys else None
        return WorkflowGroupSelection(
            request=all_requests[0],
            selected_rank=0,
            group_key=fallback_group_key,
            reason="fairness_max_group_delay",
            group_source=_group_source_for_key(fallback_group_key),
            scan_count=scan_count,
            scan_us=scan_us,
            candidate_group_size=candidate.group_size,
            fairness_guard_reason="max_group_delay_ms",
            queue_head_delay_ms=queue_head_delay_ms,
            queue_head_delay_bucket=queue_head_delay_bucket,
        )

    if (
        selected_rank != 0
        and _is_ungrouped_request(all_requests[0])
        and _ungrouped_guard_due(
            last_group_key=last_group_key,
            group_burst=group_burst,
            ungrouped_min_share=ungrouped_min_share,
        )
    ):
        return WorkflowGroupSelection(
            request=all_requests[0],
            selected_rank=0,
            group_key=None,
            reason="fairness_ungrouped_min_share",
            group_source=None,
            scan_count=scan_count,
            scan_us=scan_us,
            candidate_group_size=candidate.group_size,
            fairness_guard_reason="ungrouped_min_share",
            queue_head_delay_ms=queue_head_delay_ms,
            queue_head_delay_bucket=queue_head_delay_bucket,
        )

    if (
        last_group_key is not None
        and group_key == last_group_key
        and group_burst >= max_burst
    ):
        fairness_candidate = _first_request_outside_group(
            ordered_requests,
            candidate,
        )
        if fairness_candidate is not None:
            fairness_rank, fairness_group_key = fairness_candidate
            return WorkflowGroupSelection(
                request=ordered_requests[fairness_rank],
                selected_rank=fairness_rank,
                group_key=fairness_group_key,
                reason="fairness_burst_cap",
                group_source=_group_source_for_key(fairness_group_key),
                scan_count=scan_count,
                scan_us=scan_us,
                candidate_group_size=candidate.group_size,
                fairness_guard_reason="max_burst",
                queue_head_delay_ms=queue_head_delay_ms,
                queue_head_delay_bucket=queue_head_delay_bucket,
            )

    return WorkflowGroupSelection(
        request=ordered_requests[selected_rank],
        selected_rank=selected_rank,
        group_key=candidate.group_key,
        reason=candidate.reason,
        group_source=candidate.group_source,
        token_lcp_len=candidate.token_lcp_len,
        token_lcp_hash=candidate.token_lcp_hash,
        scan_count=scan_count,
        scan_us=scan_us,
        candidate_group_size=candidate.group_size,
        queue_head_delay_ms=queue_head_delay_ms,
        queue_head_delay_bucket=queue_head_delay_bucket,
    )


def select_workflow_join_tail_request(
    requests: Iterable[Any],
    *,
    last_group_key: str | None,
    group_burst: int,
    max_burst: int | None = None,
    max_queue_scan: int | None = None,
    max_group_delay_ms: float | None = None,
    block_size: int = 16,
    now_s: float | None = None,
) -> WorkflowGroupSelection | None:
    """Select a waiting request by workflow join-tail criticality evidence.

    This is intentionally separate from token-LCP grouping so that experiments
    can isolate workflow-criticality scheduling from prefix-locality scheduling.
    """

    del block_size
    all_requests = tuple(requests)
    if not all_requests:
        return None

    max_burst = workflow_group_aware_max_burst() if max_burst is None else max_burst
    max_queue_scan = (
        workflow_group_aware_max_queue_scan()
        if max_queue_scan is None
        else max(1, max_queue_scan)
    )
    max_group_delay_ms = (
        workflow_group_aware_max_group_delay_ms()
        if max_group_delay_ms is None
        else max(0.0, max_group_delay_ms)
    )
    scan_count = min(len(all_requests), max_queue_scan)
    ordered_requests = all_requests[:scan_count]
    queue_head_delay_ms = _queue_head_delay_ms(
        all_requests[0],
        now_s=time.time() if now_s is None else now_s,
    )
    queue_head_delay_bucket = _queue_head_delay_bucket(queue_head_delay_ms)
    scan_start_ns = time.perf_counter_ns()
    candidates = _join_tail_candidates(ordered_requests)
    scan_us = round((time.perf_counter_ns() - scan_start_ns) / 1000.0, 6)
    remaining_values = tuple(
        sorted(
            {
                value
                for candidate in candidates
                if isinstance((value := candidate.remaining_count), int)
            }
        )
    )
    if not candidates:
        return WorkflowGroupSelection(
            request=all_requests[0],
            selected_rank=0,
            group_key=None,
            reason="stock_fallback_no_join_tail",
            group_source=None,
            scan_count=scan_count,
            scan_us=scan_us,
            queue_head_delay_ms=queue_head_delay_ms,
            queue_head_delay_bucket=queue_head_delay_bucket,
            join_tail_scheduling_enabled=True,
            workflow_join_tail_selected=False,
            workflow_join_tail_reason="stock_fallback_no_join_tail",
            workflow_join_tail_scan_count=scan_count,
            workflow_join_tail_candidate_count=0,
            workflow_join_tail_remaining_values=remaining_values,
        )

    candidate = min(
        candidates,
        key=lambda item: (
            0 if item.request_would_release_join else 1,
            item.remaining_count if item.remaining_count is not None else 1_000_000,
            item.rank,
        ),
    )
    selected_rank = candidate.rank
    if (
        selected_rank != 0
        and queue_head_delay_ms is not None
        and queue_head_delay_ms >= max_group_delay_ms
    ):
        return WorkflowGroupSelection(
            request=all_requests[0],
            selected_rank=0,
            group_key=_join_tail_group_key_for_request(all_requests[0]),
            reason="fairness_max_group_delay",
            group_source="join_tail",
            scan_count=scan_count,
            scan_us=scan_us,
            candidate_group_size=len(candidates),
            fairness_guard_reason="max_group_delay_ms",
            queue_head_delay_ms=queue_head_delay_ms,
            queue_head_delay_bucket=queue_head_delay_bucket,
            join_tail_scheduling_enabled=True,
            workflow_join_tail_selected=False,
            workflow_join_tail_reason="fairness_max_group_delay",
            workflow_join_tail_scan_count=scan_count,
            workflow_join_tail_candidate_count=len(candidates),
            workflow_join_tail_remaining_values=remaining_values,
            workflow_join_tail_fairness_guard_reason="max_group_delay_ms",
        )
    if (
        selected_rank != 0
        and candidate.group_key == last_group_key
        and group_burst >= max_burst
    ):
        return WorkflowGroupSelection(
            request=all_requests[0],
            selected_rank=0,
            group_key=_join_tail_group_key_for_request(all_requests[0]),
            reason="fairness_burst_cap",
            group_source="join_tail",
            scan_count=scan_count,
            scan_us=scan_us,
            candidate_group_size=len(candidates),
            fairness_guard_reason="max_burst",
            queue_head_delay_ms=queue_head_delay_ms,
            queue_head_delay_bucket=queue_head_delay_bucket,
            join_tail_scheduling_enabled=True,
            workflow_join_tail_selected=False,
            workflow_join_tail_reason="fairness_burst_cap",
            workflow_join_tail_scan_count=scan_count,
            workflow_join_tail_candidate_count=len(candidates),
            workflow_join_tail_remaining_values=remaining_values,
            workflow_join_tail_fairness_guard_reason="max_burst",
        )

    reason = (
        "request_would_release_join"
        if candidate.request_would_release_join
        else "join_tail_phase"
    )
    return WorkflowGroupSelection(
        request=ordered_requests[selected_rank],
        selected_rank=selected_rank,
        group_key=candidate.group_key,
        reason=reason,
        group_source="join_tail",
        scan_count=scan_count,
        scan_us=scan_us,
        candidate_group_size=len(candidates),
        queue_head_delay_ms=queue_head_delay_ms,
        queue_head_delay_bucket=queue_head_delay_bucket,
        join_tail_scheduling_enabled=True,
        workflow_join_tail_selected=True,
        workflow_join_tail_reason=reason,
        workflow_join_tail_scan_count=scan_count,
        workflow_join_tail_candidate_count=len(candidates),
        workflow_join_tail_remaining_values=remaining_values,
    )


def workflow_group_keys_for_request(request: Any) -> tuple[tuple[str, str], ...]:
    """Return ordered grouping keys for a request.

    Earlier entries are stronger metadata grouping evidence. Token-verified
    LCP grouping is queue-level because it compares multiple concrete prompts.
    """

    xargs = _request_xargs(request)
    if not xargs:
        return ()

    keys: list[tuple[str, str]] = []
    for field in (
        "shared_prefill_group_id",
        "stable_prefix_group_id",
        "spawn_group_id",
        "cohort_group_id",
    ):
        value = xargs.get(field)
        if isinstance(value, str) and value:
            keys.append((field, f"{field}:{value}"))

    return tuple(keys)


@dataclass(frozen=True, slots=True)
class _Candidate:
    selected_rank: int
    group_key: str
    reason: str
    group_source: str
    token_lcp_len: int | None = None
    token_lcp_hash: str | None = None
    group_size: int | None = None


@dataclass(frozen=True, slots=True)
class _JoinTailCandidate:
    rank: int
    request: Any
    group_key: str
    remaining_count: int | None
    request_would_release_join: bool


def _join_tail_candidates(requests: tuple[Any, ...]) -> tuple[_JoinTailCandidate, ...]:
    candidates: list[_JoinTailCandidate] = []
    for rank, request in enumerate(requests):
        xargs = _request_xargs(request)
        if not xargs:
            continue
        release_join = _bool_xarg(xargs.get("request_would_release_join"))
        tail_phase = _bool_xarg(xargs.get("join_tail_phase"))
        remaining = _int_xarg(xargs.get("join_remaining_count"))
        if release_join is not True and tail_phase is not True:
            continue
        group_key = _join_tail_group_key_for_request(request)
        if group_key is None:
            continue
        candidates.append(
            _JoinTailCandidate(
                rank=rank,
                request=request,
                group_key=group_key,
                remaining_count=remaining,
                request_would_release_join=release_join is True,
            )
        )
    return tuple(candidates)


def _join_tail_group_key_for_request(request: Any) -> str | None:
    xargs = _request_xargs(request)
    if not xargs:
        return None
    join_group_id = xargs.get("join_group_id")
    if isinstance(join_group_id, str) and join_group_id:
        return f"join_tail:{join_group_id}"
    workflow_instance_id = xargs.get("workflow_instance_id")
    site_id = xargs.get("site_id")
    if (
        isinstance(workflow_instance_id, str)
        and workflow_instance_id
        and isinstance(site_id, str)
        and site_id
    ):
        return f"join_tail:{workflow_instance_id}:{site_id}"
    return None


@dataclass(frozen=True, slots=True)
class _RequestSnapshot:
    """Per-scan validated view of one waiting request.

    Built once at the start of select_workflow_group_request, then reused by
    every candidate-search subroutine. The snapshot is what makes the inner
    LCP scan O(N * T / B) instead of O(N^3 * T): without it, helpers like
    _request_prompt_token_ids and _request_xargs are called once per request
    pair per iteration, and each call walks every token in the prompt to
    type-check. Keep this dataclass cheap to construct; do not put any work
    here that is not also amortized by every candidate-search subroutine.
    """

    rank: int
    request: Any
    token_ids: tuple[int, ...]  # empty tuple when invalid / missing
    has_xargs: bool
    model_key: str
    token_domain_key: str
    workflow_keys: tuple[tuple[str, str], ...]


def _build_request_snapshots(
    requests: tuple[Any, ...],
) -> tuple[_RequestSnapshot, ...]:
    snapshots: list[_RequestSnapshot] = []
    for rank, request in enumerate(requests):
        xargs = _request_xargs(request)
        token_ids_list = _request_prompt_token_ids(request)
        token_ids: tuple[int, ...] = (
            tuple(token_ids_list) if token_ids_list else ()
        )
        snapshots.append(
            _RequestSnapshot(
                rank=rank,
                request=request,
                token_ids=token_ids,
                has_xargs=bool(xargs),
                model_key=_request_model_key(request),
                token_domain_key=_request_token_domain_key(request),
                workflow_keys=workflow_group_keys_for_request(request),
            )
        )
    return tuple(snapshots)


def _best_group_candidate_from_snapshots(
    snapshots: tuple[_RequestSnapshot, ...],
    *,
    block_size: int,
    deadline_ns: int | None = None,
) -> _Candidate | None:
    token_candidate = _best_token_lcp_candidate_from_snapshots(
        snapshots,
        block_size=block_size,
        deadline_ns=deadline_ns,
    )
    if token_candidate is not None:
        return token_candidate

    # Metadata grouping fallback: same algorithm as the legacy path but reads
    # from precomputed snapshot.workflow_keys instead of re-walking xargs.
    max_key_count = max(
        (len(snapshot.workflow_keys) for snapshot in snapshots),
        default=0,
    )
    for priority_index in range(max_key_count):
        ranks_by_key: dict[str, list[int]] = {}
        reason_by_key: dict[str, str] = {}
        for snapshot in snapshots:
            if priority_index >= len(snapshot.workflow_keys):
                continue
            reason, group_key = snapshot.workflow_keys[priority_index]
            ranks_by_key.setdefault(group_key, []).append(snapshot.rank)
            reason_by_key[group_key] = reason

        repeated_groups = {
            group_key: ranks
            for group_key, ranks in ranks_by_key.items()
            if len(ranks) > 1
        }
        if repeated_groups:
            group_key, ranks = min(
                repeated_groups.items(),
                key=lambda item: item[1][0],
            )
            reason = reason_by_key[group_key]
            return _Candidate(
                selected_rank=ranks[0],
                group_key=group_key,
                reason=reason,
                group_source=reason,
                group_size=len(ranks),
            )

    return None


def _best_token_lcp_candidate_from_snapshots(
    snapshots: tuple[_RequestSnapshot, ...],
    *,
    block_size: int,
    deadline_ns: int | None = None,
) -> _Candidate | None:
    """Find the longest block-aligned shared-prefix group across snapshots.

    Replaces the O(N^2 * T) pairwise LCP loop with an O(N * T / B) block
    radix split: bucket the snapshots by (model_key, token_domain_key), then
    repeatedly partition each bucket by the next ``block_size``-token slice
    of its members until no bucket retains >= 2 snapshots. The deepest
    surviving bucket wins; ties are broken by smallest first rank in queue
    so a token-aware decision is also queue-position aware.

    The deadline_ns argument enforces a wall-clock budget. When the budget
    is exceeded we return whatever best-so-far group has been observed
    (which may be None). Callers treat None as "fall back to stock head".
    """

    if not snapshots:
        return None

    # Eligible snapshots: must have xargs metadata and a non-empty validated
    # token sequence. Anything else cannot participate in a token-LCP group.
    by_domain: dict[tuple[str, str], list[_RequestSnapshot]] = {}
    for snapshot in snapshots:
        if not snapshot.has_xargs or not snapshot.token_ids:
            continue
        domain_key = (snapshot.model_key, snapshot.token_domain_key)
        by_domain.setdefault(domain_key, []).append(snapshot)

    # Best-so-far across all domains: (aligned_lcp_len, first_rank, hash, key, group_size).
    best: tuple[int, int, str, str, int] | None = None
    # Cache of hashes keyed by (id(token_ids), prefix_len). The same token
    # sequence is referenced across many bucket splits, and recomputing
    # sha1 over a multi-thousand-int json blob each time is the second
    # biggest cost after the type-check loop.
    hash_cache: dict[tuple[int, int], str] = {}

    for (model_key, token_domain_key), domain_snaps in by_domain.items():
        if len(domain_snaps) < 2:
            continue
        if deadline_ns is not None and time.perf_counter_ns() > deadline_ns:
            break
        domain_best = _block_split_best_lcp(
            domain_snaps,
            block_size=block_size,
            hash_cache=hash_cache,
            deadline_ns=deadline_ns,
        )
        if domain_best is None:
            continue
        aligned_lcp_len, first_rank, lcp_hash, group_size = domain_best
        group_key = (
            "token_verified_lcp:"
            f"model={model_key}:"
            f"token_domain={token_domain_key}:"
            f"len={aligned_lcp_len}:"
            f"hash={lcp_hash}"
        )
        if best is None or (aligned_lcp_len, -first_rank) > (best[0], -best[1]):
            best = (
                aligned_lcp_len,
                first_rank,
                lcp_hash,
                group_key,
                group_size,
            )

    if best is None:
        return None
    aligned_lcp_len, selected_rank, token_lcp_hash, group_key, group_size = best
    return _Candidate(
        selected_rank=selected_rank,
        group_key=group_key,
        reason="token_verified_lcp",
        group_source="token_verified_lcp",
        token_lcp_len=aligned_lcp_len,
        token_lcp_hash=token_lcp_hash,
        group_size=group_size,
    )


def _block_split_best_lcp(
    snapshots: list[_RequestSnapshot],
    *,
    block_size: int,
    hash_cache: dict[tuple[int, int], str],
    deadline_ns: int | None,
) -> tuple[int, int, str, int] | None:
    """Block-radix-split LCP search within one (model, token_domain) bucket.

    Returns (aligned_lcp_len, first_rank, lcp_hash, group_size) for the best
    surviving bucket of >= 2 snapshots, or None if no two snapshots share at
    least one full block_size of leading tokens.
    """

    if block_size <= 0:
        return None
    best: tuple[int, int, str, int] | None = None
    # State: list of (lcp_len_so_far, snapshots_in_bucket).
    current: list[tuple[int, list[_RequestSnapshot]]] = [(0, snapshots)]
    while current:
        if deadline_ns is not None and time.perf_counter_ns() > deadline_ns:
            return best
        next_buckets: list[tuple[int, list[_RequestSnapshot]]] = []
        for lcp_len, bucket in current:
            sub_groups: dict[tuple[int, ...], list[_RequestSnapshot]] = {}
            for snapshot in bucket:
                tokens = snapshot.token_ids
                if len(tokens) < lcp_len + block_size:
                    continue
                next_block = tokens[lcp_len : lcp_len + block_size]
                sub_groups.setdefault(next_block, []).append(snapshot)
            for next_block_key, sub_snaps in sub_groups.items():
                del next_block_key  # only used as dict key
                if len(sub_snaps) < 2:
                    continue
                new_len = lcp_len + block_size
                first_snap = sub_snaps[0]
                first_rank = first_snap.rank
                for snap in sub_snaps[1:]:
                    if snap.rank < first_rank:
                        first_rank = snap.rank
                cache_key = (id(first_snap.token_ids), new_len)
                lcp_hash = hash_cache.get(cache_key)
                if lcp_hash is None:
                    lcp_hash = _hash_token_ids(
                        list(first_snap.token_ids[:new_len])
                    )
                    hash_cache[cache_key] = lcp_hash
                if best is None or (new_len, -first_rank) > (best[0], -best[1]):
                    best = (new_len, first_rank, lcp_hash, len(sub_snaps))
                next_buckets.append((new_len, sub_snaps))
        current = next_buckets
    return best


def _first_request_outside_group(
    requests: tuple[Any, ...],
    candidate: _Candidate,
) -> tuple[int, str | None] | None:
    for rank, request in enumerate(requests):
        request_group_key = _candidate_group_key_for_request(request, candidate)
        if request_group_key != candidate.group_key:
            fallback_keys = workflow_group_keys_for_request(request)
            return rank, fallback_keys[0][1] if fallback_keys else None
    return None


def _candidate_group_key_for_request(
    request: Any,
    candidate: _Candidate,
) -> str | None:
    if candidate.group_source == "token_verified_lcp":
        token_ids = _request_prompt_token_ids(request)
        if (
            token_ids is None
            or candidate.token_lcp_len is None
            or candidate.token_lcp_hash is None
            or len(token_ids) < candidate.token_lcp_len
        ):
            return None
        if _hash_token_ids(token_ids[: candidate.token_lcp_len]) != (
            candidate.token_lcp_hash
        ):
            return None
        candidate_model_key = _model_key_from_group_key(candidate.group_key)
        candidate_token_domain_key = _token_domain_key_from_group_key(
            candidate.group_key
        )
        if (
            _request_model_key(request) != candidate_model_key
            or _request_token_domain_key(request) != candidate_token_domain_key
        ):
            return None
        return candidate.group_key

    for reason, group_key in workflow_group_keys_for_request(request):
        if reason == candidate.group_source:
            return group_key
    return None


def _group_source_for_key(group_key: str | None) -> str | None:
    if group_key is None:
        return None
    if ":" not in group_key:
        return None
    return group_key.split(":", 1)[0]


def _model_key_from_group_key(group_key: str) -> str:
    return _group_key_part(group_key, "model") or _UNAVAILABLE_SINGLE_MODEL_ASSUMED


def _token_domain_key_from_group_key(group_key: str) -> str:
    return (
        _group_key_part(group_key, "token_domain")
        or _UNAVAILABLE_SINGLE_MODEL_ASSUMED
    )


def _group_key_part(group_key: str, field: str) -> str | None:
    prefix = f"{field}="
    for part in group_key.split(":"):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _request_model_key(request: Any) -> str:
    xargs = _request_xargs(request) or {}
    for field in ("model_execution_key", "served_model_name", "model"):
        value = xargs.get(field)
        if isinstance(value, str) and value:
            return value
    value = getattr(request, "model", None)
    if isinstance(value, str) and value:
        return value
    return _UNAVAILABLE_SINGLE_MODEL_ASSUMED


def _request_token_domain_key(request: Any) -> str:
    xargs = _request_xargs(request) or {}
    for field in (
        "token_domain_id",
        "engine_token_domain_fingerprint",
        "tokenizer_id",
        "chat_template_id",
    ):
        value = xargs.get(field)
        if isinstance(value, str) and value:
            return value
    return _UNAVAILABLE_SINGLE_MODEL_ASSUMED


def _is_ungrouped_request(request: Any) -> bool:
    return not workflow_group_keys_for_request(request)


def _ungrouped_guard_due(
    *,
    last_group_key: str | None,
    group_burst: int,
    ungrouped_min_share: float,
) -> bool:
    if last_group_key is None or ungrouped_min_share <= 0.0:
        return False
    grouped_before_ungrouped = max(
        1,
        int((1.0 - ungrouped_min_share) / ungrouped_min_share),
    )
    return group_burst >= grouped_before_ungrouped


def _queue_head_delay_ms(request: Any, *, now_s: float) -> float | None:
    arrival_time = getattr(request, "arrival_time", None)
    if not isinstance(arrival_time, (int, float)):
        return None
    return max(0.0, (now_s - float(arrival_time)) * 1000.0)


def _queue_head_delay_bucket(queue_head_delay_ms: float | None) -> str | None:
    if queue_head_delay_ms is None:
        return None
    if queue_head_delay_ms < 50:
        return "lt_50ms"
    if queue_head_delay_ms < 200:
        return "lt_200ms"
    if queue_head_delay_ms < 1000:
        return "lt_1000ms"
    return "gte_1000ms"


def _request_xargs(request: Any) -> dict[str, Any] | None:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    return extra_args if isinstance(extra_args, dict) else None


def _bool_xarg(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def _int_xarg(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _request_prompt_token_ids(request: Any) -> list[int] | None:
    token_ids = getattr(request, "prompt_token_ids", None)
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        return None
    return token_ids


def _hash_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return f"sha1:{hashlib.sha1(payload).hexdigest()}"
