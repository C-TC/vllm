# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from vllm.entrypoints.openai.chat_completion.workflow_sideband import (
    validate_workflow_sideband,
)


def test_workflow_sideband_validation_accepts_current_contract() -> None:
    validation = validate_workflow_sideband(
        {
            "workflow_id": "demo",
            "workflow_instance_id": "wf-1",
            "graph_id": "main",
            "block_id": "map_worker",
            "op_id": "map_worker",
            "site_id": "main:map_worker:map_worker",
            "private_release_hint": "may",
            "effective_release_hint": "may",
            "release_hint": "may",
            "submit_group_id": "group-1",
            "spawn_group_id": "spawn-1",
            "spawn_index": "0",
            "spawn_size": "2",
            "shared_prefix_class_id": "shared_prefix:p1",
            "shared_prefill_group_id": "shared-prefill:group-1",
            "shared_prefill_group_size": "2",
            "stable_prefix_group_id": "group-1",
            "stable_prefix_group_size": "2",
            "cohort_group_id": "group-1:cohort:target-language",
            "cohort_group_size": "2",
            "cohort_shared_fields": "target_language",
            "pressure_score": "7",
            "sync_target_id": "main:reduce",
        }
    )

    assert validation is not None
    assert validation.valid is True
    assert validation.issues == ()
    assert "workflow_id" in validation.present_fields
    assert "shared_prefix_class_id" in validation.present_fields
    assert "shared_prefill_group_id" in validation.present_fields
    assert "stable_prefix_group_id" in validation.present_fields
    assert "cohort_group_id" in validation.present_fields


def test_workflow_sideband_validation_rejects_runtime_private_fields() -> None:
    validation = validate_workflow_sideband(
        {
            "workflow_id": "demo",
            "workflow_instance_id": "wf-1",
            "graph_id": "main",
            "block_id": "main",
            "op_id": "writer",
            "site_id": "main:main:writer",
            "private_release_hint": "may",
            "effective_release_hint": "may",
            "release_hint": "may",
            "engine_prompt_token_count": 12,
            "runner_token_measurement": {"source": "runner_advisory"},
            "stable_prefix_handle_id": "stable_prefix:abc",
            "prefix_fingerprint": "abc",
            "prompt_token_ids_hash": "sha1:abc",
        }
    )

    assert validation is not None
    assert validation.valid is False
    fields = {issue.field for issue in validation.issues}
    assert "stable_prefix_handle_id" in fields
    assert "prefix_fingerprint" in fields
    assert "engine_prompt_token_count" in fields
    assert "runner_token_measurement" in fields
    assert "prompt_token_ids_hash" in fields


def test_workflow_sideband_validation_distinguishes_absent_sideband() -> None:
    assert validate_workflow_sideband(None) is None
