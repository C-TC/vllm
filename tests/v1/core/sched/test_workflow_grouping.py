# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.core.sched.workflow_grouping import (
    select_workflow_group_request,
    workflow_group_aware_max_burst,
    workflow_group_aware_max_group_delay_ms,
    workflow_group_aware_max_queue_scan,
    workflow_group_aware_scheduling_enabled,
    workflow_group_aware_ungrouped_min_share,
    workflow_group_keys_for_request,
)


def _request(
    request_id: str,
    *,
    xargs: dict[str, str] | None = None,
    prompt_token_ids: list[int] | None = None,
    arrival_time: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        sampling_params=SimpleNamespace(extra_args=xargs),
        prompt_token_ids=prompt_token_ids,
        arrival_time=arrival_time,
    )


def test_workflow_grouping_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("WORKFLOW_GROUP_AWARE_SCHEDULING", raising=False)
    monkeypatch.delenv("WORKFLOW_GROUP_AWARE_MAX_BURST", raising=False)
    monkeypatch.delenv("WORKFLOW_GROUP_AWARE_MAX_QUEUE_SCAN", raising=False)
    monkeypatch.delenv("WORKFLOW_GROUP_AWARE_MAX_GROUP_DELAY_MS", raising=False)
    monkeypatch.delenv("WORKFLOW_GROUP_AWARE_UNGROUPED_MIN_SHARE", raising=False)

    assert workflow_group_aware_scheduling_enabled() is False
    assert workflow_group_aware_max_burst() == 4
    assert workflow_group_aware_max_queue_scan() == 64
    assert workflow_group_aware_max_group_delay_ms() == 200.0
    assert workflow_group_aware_ungrouped_min_share() == 0.2


def test_workflow_grouping_prefers_token_verified_lcp() -> None:
    req_a0 = _request(
        "a0",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
    )
    req_b = _request(
        "b",
        xargs={"stable_prefix_group_id": "stable-b"},
        prompt_token_ids=[9, 9, 9, 9],
    )
    req_a1 = _request(
        "a1",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3, 4, 8, 8],
    )

    selection = select_workflow_group_request(
        [req_b, req_a0, req_a1],
        last_group_key=None,
        group_burst=0,
        block_size=4,
    )

    assert selection is not None
    assert selection.request is req_a0
    assert selection.selected_rank == 1
    assert selection.reason == "token_verified_lcp"
    assert selection.group_source == "token_verified_lcp"
    assert selection.token_lcp_len == 4
    assert selection.token_lcp_hash is not None
    assert selection.token_lcp_hash.startswith("sha1:")
    assert selection.group_key is not None
    assert selection.group_key.startswith("token_verified_lcp:")
    assert "model=unavailable_single_model_assumed" in selection.group_key
    assert "token_domain=unavailable_single_model_assumed" in selection.group_key
    assert selection.scan_count == 3
    assert selection.candidate_group_size == 2


def test_workflow_grouping_scan_cap_limits_token_lcp_search() -> None:
    req_b = _request(
        "b",
        xargs={"stable_prefix_group_id": "stable-b"},
        prompt_token_ids=[9, 9, 9, 9],
    )
    req_a0 = _request(
        "a0",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3, 4],
    )
    req_a1 = _request(
        "a1",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3, 4],
    )

    selection = select_workflow_group_request(
        [req_b, req_a0, req_a1],
        last_group_key=None,
        group_burst=0,
        block_size=4,
        max_queue_scan=2,
    )

    assert selection is not None
    assert selection.request is req_b
    assert selection.reason == "stock_fallback_no_group"
    assert selection.scan_count == 2


def test_workflow_grouping_falls_back_to_stable_prefix_group() -> None:
    req_a0 = _request(
        "a0",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3, 4],
    )
    req_b = _request(
        "b",
        xargs={"stable_prefix_group_id": "stable-b"},
        prompt_token_ids=[9, 9, 9, 9],
    )
    req_a1 = _request(
        "a1",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[5, 6, 7, 8],
    )

    selection = select_workflow_group_request(
        [req_b, req_a0, req_a1],
        last_group_key=None,
        group_burst=0,
        block_size=4,
    )

    assert selection is not None
    assert selection.request is req_a0
    assert selection.selected_rank == 1
    assert selection.group_key == "stable_prefix_group_id:stable-a"
    assert selection.reason == "stable_prefix_group_id"
    assert selection.group_source == "stable_prefix_group_id"


def test_workflow_grouping_prefers_shared_prefill_metadata() -> None:
    req_a0 = _request(
        "a0",
        xargs={
            "shared_prefill_group_id": "shared-a",
            "stable_prefix_group_id": "stable-b",
        },
        prompt_token_ids=[1, 2, 3, 4],
    )
    req_a1 = _request(
        "a1",
        xargs={
            "shared_prefill_group_id": "shared-a",
            "stable_prefix_group_id": "stable-c",
        },
        prompt_token_ids=[5, 6, 7, 8],
    )

    assert workflow_group_keys_for_request(req_a0)[0] == (
        "shared_prefill_group_id",
        "shared_prefill_group_id:shared-a",
    )
    selection = select_workflow_group_request(
        [req_a0, req_a1],
        last_group_key=None,
        group_burst=0,
        block_size=4,
    )

    assert selection is not None
    assert selection.group_key == "shared_prefill_group_id:shared-a"
    assert selection.reason == "shared_prefill_group_id"


def test_workflow_grouping_burst_cap_advances_other_group() -> None:
    req_a0 = _request("a0", xargs={"spawn_group_id": "spawn-a"})
    req_a1 = _request("a1", xargs={"spawn_group_id": "spawn-a"})
    req_b = _request("b", xargs={"spawn_group_id": "spawn-b"})

    selection = select_workflow_group_request(
        [req_a0, req_a1, req_b],
        last_group_key="spawn_group_id:spawn-a",
        group_burst=1,
        max_burst=1,
    )

    assert selection is not None
    assert selection.request is req_b
    assert selection.selected_rank == 2
    assert selection.reason == "fairness_burst_cap"
    assert selection.fairness_guard_reason == "max_burst"
    assert selection.group_key == "spawn_group_id:spawn-b"


def test_workflow_grouping_max_delay_protects_old_queue_head() -> None:
    req_b = _request(
        "b",
        xargs=None,
        prompt_token_ids=[9, 9, 9, 9],
        arrival_time=1.0,
    )
    req_a0 = _request(
        "a0",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3, 4],
        arrival_time=1.1,
    )
    req_a1 = _request(
        "a1",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[5, 6, 7, 8],
        arrival_time=1.2,
    )

    selection = select_workflow_group_request(
        [req_b, req_a0, req_a1],
        last_group_key=None,
        group_burst=0,
        max_group_delay_ms=200,
        block_size=4,
        now_s=1.3,
    )

    assert selection is not None
    assert selection.request is req_b
    assert selection.reason == "fairness_max_group_delay"
    assert selection.fairness_guard_reason == "max_group_delay_ms"
    assert selection.queue_head_delay_ms is not None
    assert 299 <= selection.queue_head_delay_ms <= 301
    assert selection.queue_head_delay_bucket == "lt_1000ms"


def test_workflow_grouping_ungrouped_share_guard() -> None:
    req_b = _request("b", xargs=None)
    req_a0 = _request("a0", xargs={"spawn_group_id": "spawn-a"})
    req_a1 = _request("a1", xargs={"spawn_group_id": "spawn-a"})

    selection = select_workflow_group_request(
        [req_b, req_a0, req_a1],
        last_group_key="spawn_group_id:spawn-a",
        group_burst=4,
        max_burst=99,
        ungrouped_min_share=0.2,
    )

    assert selection is not None
    assert selection.request is req_b
    assert selection.reason == "fairness_ungrouped_min_share"
    assert selection.fairness_guard_reason == "ungrouped_min_share"


def test_workflow_grouping_missing_sideband_keeps_stock_head() -> None:
    req_a = _request("a", xargs=None, prompt_token_ids=[1, 2, 3])
    req_b = _request("b", xargs={"spawn_group_id": "spawn-b"})

    assert workflow_group_keys_for_request(req_a) == ()
    selection = select_workflow_group_request(
        [req_a, req_b],
        last_group_key=None,
        group_burst=0,
    )

    assert selection is not None
    assert selection.request is req_a
    assert selection.selected_rank == 0
    assert selection.group_key is None
    assert selection.reason == "stock_fallback_no_group"
