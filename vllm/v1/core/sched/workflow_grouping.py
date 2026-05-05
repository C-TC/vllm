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
    group_source: str | None = None
    token_lcp_len: int | None = None
    token_lcp_hash: str | None = None


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
    block_size: int = 16,
) -> WorkflowGroupSelection | None:
    """Select a waiting request by redacted workflow grouping evidence.

    This helper never changes cache state and never looks at raw prompt text.
    Token-prefix grouping only uses concrete scheduler token ids to compute a
    redacted block-aligned LCP key; metadata grouping remains the fallback.
    """

    ordered_requests = tuple(requests)
    if not ordered_requests:
        return None

    max_burst = workflow_group_aware_max_burst() if max_burst is None else max_burst
    candidate = _best_group_candidate(ordered_requests, block_size=block_size)
    if candidate is None:
        return WorkflowGroupSelection(
            request=ordered_requests[0],
            selected_rank=0,
            group_key=None,
            reason="stock_fallback_no_group",
            group_source=None,
        )

    selected_rank = candidate.selected_rank
    group_key = candidate.group_key
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
            )

    return WorkflowGroupSelection(
        request=ordered_requests[selected_rank],
        selected_rank=selected_rank,
        group_key=candidate.group_key,
        reason=candidate.reason,
        group_source=candidate.group_source,
        token_lcp_len=candidate.token_lcp_len,
        token_lcp_hash=candidate.token_lcp_hash,
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


def _best_group_candidate(
    requests: tuple[Any, ...],
    *,
    block_size: int,
) -> _Candidate | None:
    token_candidate = _best_token_lcp_candidate(requests, block_size=block_size)
    if token_candidate is not None:
        return token_candidate

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
            reason = reason_by_key[group_key]
            return _Candidate(
                selected_rank=ranks[0],
                group_key=group_key,
                reason=reason,
                group_source=reason,
            )

    return None


def _best_token_lcp_candidate(
    requests: tuple[Any, ...],
    *,
    block_size: int = 16,
) -> _Candidate | None:
    best: tuple[int, int, str, str] | None = None
    for left_rank, left_request in enumerate(requests):
        left_tokens = _request_prompt_token_ids(left_request)
        if left_tokens is None or not _request_xargs(left_request):
            continue
        for right_rank in range(left_rank + 1, len(requests)):
            right_tokens = _request_prompt_token_ids(requests[right_rank])
            if right_tokens is None or not _request_xargs(requests[right_rank]):
                continue
            lcp_len = _common_prefix_len(left_tokens, right_tokens)
            aligned_lcp_len = (lcp_len // block_size) * block_size
            if aligned_lcp_len <= 0:
                continue
            lcp_hash = _hash_token_ids(left_tokens[:aligned_lcp_len])
            group_key = (
                "token_verified_lcp:"
                "model=unknown:"
                "token_domain=unknown:"
                f"len={aligned_lcp_len}:"
                f"hash={lcp_hash}"
            )
            group_ranks = _token_lcp_group_ranks(
                requests,
                token_lcp_len=aligned_lcp_len,
                token_lcp_hash=lcp_hash,
            )
            if len(group_ranks) < 2:
                continue
            group_first_rank = group_ranks[0]
            if best is None or (
                aligned_lcp_len,
                -group_first_rank,
            ) > (
                best[0],
                -best[1],
            ):
                best = (aligned_lcp_len, group_first_rank, lcp_hash, group_key)

    if best is None:
        return None
    token_lcp_len, selected_rank, token_lcp_hash, group_key = best
    return _Candidate(
        selected_rank=selected_rank,
        group_key=group_key,
        reason="token_verified_lcp",
        group_source="token_verified_lcp",
        token_lcp_len=token_lcp_len,
        token_lcp_hash=token_lcp_hash,
    )


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


def _token_lcp_group_ranks(
    requests: tuple[Any, ...],
    *,
    token_lcp_len: int,
    token_lcp_hash: str,
) -> list[int]:
    ranks: list[int] = []
    for rank, request in enumerate(requests):
        if not _request_xargs(request):
            continue
        token_ids = _request_prompt_token_ids(request)
        if token_ids is None or len(token_ids) < token_lcp_len:
            continue
        if _hash_token_ids(token_ids[:token_lcp_len]) == token_lcp_hash:
            ranks.append(rank)
    return ranks


def _request_xargs(request: Any) -> dict[str, Any] | None:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    return extra_args if isinstance(extra_args, dict) else None


def _request_prompt_token_ids(request: Any) -> list[int] | None:
    token_ids = getattr(request, "prompt_token_ids", None)
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        return None
    return token_ids


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    prefix_len = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix_len += 1
    return prefix_len


def _hash_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return f"sha1:{hashlib.sha1(payload).hexdigest()}"
