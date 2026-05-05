# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

WORKFLOW_GROUP_AWARE_SCHEDULING_ENV = "WORKFLOW_GROUP_AWARE_SCHEDULING"
WORKFLOW_GROUP_AWARE_MAX_BURST_ENV = "WORKFLOW_GROUP_AWARE_MAX_BURST"
DEFAULT_WORKFLOW_GROUP_AWARE_MAX_BURST = 4


@dataclass(frozen=True, slots=True)
class WorkflowGroupSelection:
    request: Any
    selected_rank: int
    group_key: str | None
    reason: str


def workflow_group_aware_scheduling_enabled() -> bool:
    return os.getenv(WORKFLOW_GROUP_AWARE_SCHEDULING_ENV, "") == "1"


def workflow_group_aware_max_burst() -> int:
    raw_value = os.getenv(WORKFLOW_GROUP_AWARE_MAX_BURST_ENV, "")
    if not raw_value:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_BURST
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_WORKFLOW_GROUP_AWARE_MAX_BURST
    return max(1, value)


def select_workflow_group_request(
    requests: Iterable[Any],
    *,
    last_group_key: str | None,
    group_burst: int,
    max_burst: int | None = None,
) -> WorkflowGroupSelection | None:
    """Select a waiting request by redacted workflow grouping metadata.

    This helper is intentionally metadata-only. It never changes cache state,
    never looks at raw prompts, and returns the stock head request when no
    repeated workflow grouping key is visible.
    """

    ordered_requests = tuple(requests)
    if not ordered_requests:
        return None

    max_burst = workflow_group_aware_max_burst() if max_burst is None else max_burst
    candidate = _best_group_candidate(ordered_requests)
    if candidate is None:
        return WorkflowGroupSelection(
            request=ordered_requests[0],
            selected_rank=0,
            group_key=None,
            reason="stock_fallback_no_group",
        )

    selected_rank, group_key, reason = candidate
    if (
        last_group_key is not None
        and group_key == last_group_key
        and group_burst >= max_burst
    ):
        fairness_candidate = _first_request_outside_group(
            ordered_requests,
            last_group_key,
        )
        if fairness_candidate is not None:
            fairness_rank, fairness_group_key = fairness_candidate
            return WorkflowGroupSelection(
                request=ordered_requests[fairness_rank],
                selected_rank=fairness_rank,
                group_key=fairness_group_key,
                reason="fairness_burst_cap",
            )

    return WorkflowGroupSelection(
        request=ordered_requests[selected_rank],
        selected_rank=selected_rank,
        group_key=group_key,
        reason=reason,
    )


def workflow_group_keys_for_request(request: Any) -> tuple[tuple[str, str], ...]:
    """Return ordered grouping keys for a request.

    Earlier entries are stronger grouping evidence. The prompt-token hash is
    redacted and only considered for sideband-bearing requests, so true vanilla
    traffic retains stock behavior.
    """

    xargs = _request_xargs(request)
    if not xargs:
        return ()

    keys: list[tuple[str, str]] = []
    prompt_hash = _prompt_token_ids_hash(request)
    if prompt_hash is not None:
        keys.append(("token_verified_prompt_hash", f"token_prompt:{prompt_hash}"))

    for field in (
        "stable_prefix_group_id",
        "spawn_group_id",
        "cohort_group_id",
    ):
        value = xargs.get(field)
        if isinstance(value, str) and value:
            keys.append((field, f"{field}:{value}"))

    return tuple(keys)


def _best_group_candidate(
    requests: tuple[Any, ...],
) -> tuple[int, str, str] | None:
    max_key_count = max(
        (len(workflow_group_keys_for_request(request)) for request in requests),
        default=0,
    )
    for priority_index in range(max_key_count):
        ranks_by_key: dict[str, list[int]] = {}
        reason_by_key: dict[str, str] = {}
        for rank, request in enumerate(requests):
            keys = workflow_group_keys_for_request(request)
            if priority_index >= len(keys):
                continue
            reason, group_key = keys[priority_index]
            ranks_by_key.setdefault(group_key, []).append(rank)
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
            return ranks[0], group_key, reason_by_key[group_key]

    return None


def _first_request_outside_group(
    requests: tuple[Any, ...],
    group_key: str,
) -> tuple[int, str | None] | None:
    for rank, request in enumerate(requests):
        request_group_keys = workflow_group_keys_for_request(request)
        if all(candidate_key != group_key for _, candidate_key in request_group_keys):
            return rank, request_group_keys[0][1] if request_group_keys else None
    return None


def _request_xargs(request: Any) -> dict[str, Any] | None:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    return extra_args if isinstance(extra_args, dict) else None


def _prompt_token_ids_hash(request: Any) -> str | None:
    token_ids = getattr(request, "prompt_token_ids", None)
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        return None
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return f"sha1:{hashlib.sha1(payload).hexdigest()}"
