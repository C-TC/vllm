# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.core.sched.workflow_grouping import (
    select_workflow_group_request,
    workflow_group_aware_max_burst,
    workflow_group_aware_scheduling_enabled,
    workflow_group_keys_for_request,
)


def _request(
    request_id: str,
    *,
    xargs: dict[str, str] | None = None,
    prompt_token_ids: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        sampling_params=SimpleNamespace(extra_args=xargs),
        prompt_token_ids=prompt_token_ids,
    )


def test_workflow_grouping_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("WORKFLOW_GROUP_AWARE_SCHEDULING", raising=False)
    monkeypatch.delenv("WORKFLOW_GROUP_AWARE_MAX_BURST", raising=False)

    assert workflow_group_aware_scheduling_enabled() is False
    assert workflow_group_aware_max_burst() == 4


def test_workflow_grouping_prefers_repeated_token_prompt_hash() -> None:
    req_a0 = _request(
        "a0",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3],
    )
    req_b = _request(
        "b",
        xargs={"stable_prefix_group_id": "stable-b"},
        prompt_token_ids=[9, 9],
    )
    req_a1 = _request(
        "a1",
        xargs={"stable_prefix_group_id": "stable-a"},
        prompt_token_ids=[1, 2, 3],
    )

    selection = select_workflow_group_request(
        [req_b, req_a0, req_a1],
        last_group_key=None,
        group_burst=0,
    )

    assert selection is not None
    assert selection.request is req_a0
    assert selection.selected_rank == 1
    assert selection.reason == "token_verified_prompt_hash"
    assert selection.group_key is not None
    assert selection.group_key.startswith("token_prompt:sha1:")


def test_workflow_grouping_falls_back_to_stable_prefix_group() -> None:
    req_a0 = _request("a0", xargs={"stable_prefix_group_id": "stable-a"})
    req_b = _request("b", xargs={"stable_prefix_group_id": "stable-b"})
    req_a1 = _request("a1", xargs={"stable_prefix_group_id": "stable-a"})

    selection = select_workflow_group_request(
        [req_b, req_a0, req_a1],
        last_group_key=None,
        group_burst=0,
    )

    assert selection is not None
    assert selection.request is req_a0
    assert selection.selected_rank == 1
    assert selection.group_key == "stable_prefix_group_id:stable-a"
    assert selection.reason == "stable_prefix_group_id"


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
    assert selection.group_key == "spawn_group_id:spawn-b"


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
