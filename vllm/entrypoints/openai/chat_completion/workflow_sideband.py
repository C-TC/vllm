# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_WORKFLOW_SIDEBAND_FIELDS = (
    "workflow_id",
    "workflow_instance_id",
    "graph_id",
    "block_id",
    "op_id",
    "site_id",
    "private_release_hint",
    "effective_release_hint",
    "release_hint",
)

OPTIONAL_WORKFLOW_SIDEBAND_FIELDS = (
    "submit_group_id",
    "spawn_group_id",
    "spawn_index",
    "spawn_size",
    "shared_prefix_class_id",
    "shared_prefill_group_id",
    "shared_prefill_group_size",
    "stable_prefix_group_id",
    "stable_prefix_group_size",
    "cohort_group_id",
    "cohort_group_size",
    "cohort_shared_fields",
    "pressure_score",
    "sync_target_id",
    "criticality_source",
    "join_group_id",
    "join_width",
    "join_remaining_count",
    "request_would_release_join",
    "join_tail_phase",
)

ALLOWED_WORKFLOW_SIDEBAND_FIELDS = (
    REQUIRED_WORKFLOW_SIDEBAND_FIELDS + OPTIONAL_WORKFLOW_SIDEBAND_FIELDS
)

FORBIDDEN_WORKFLOW_SIDEBAND_FIELDS = (
    "backend_cache_key",
    "coopt_action_queue",
    "engine_prompt_token_count",
    "full_prompt_tokens",
    "prefix_class_id",
    "prefix_fingerprint",
    "prompt_token_ids_hash",
    "priority_class",
    "raw_atom_label",
    "runner_token_measurement",
    "serve_plan_atom",
    "stable_prefix_handle",
    "stable_prefix_handle_id",
    "stable_prefix_tokens",
)

_VALID_RELEASE_HINTS = {"must", "may", "no", "unsupported"}
_INTEGER_STRING_FIELDS = {
    "spawn_index",
    "spawn_size",
    "shared_prefill_group_size",
    "stable_prefix_group_size",
    "cohort_group_size",
    "pressure_score",
    "join_width",
    "join_remaining_count",
}
_BOOLEAN_STRING_FIELDS = {
    "request_would_release_join",
    "join_tail_phase",
}


@dataclass(slots=True, frozen=True)
class WorkflowSidebandIssue:
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "message": self.message,
        }


@dataclass(slots=True, frozen=True)
class WorkflowSidebandValidation:
    valid: bool
    present_fields: tuple[str, ...]
    missing_required_fields: tuple[str, ...]
    issues: tuple[WorkflowSidebandIssue, ...]

    def issues_as_dicts(self) -> list[dict[str, str]]:
        return [issue.as_dict() for issue in self.issues]


def validate_workflow_sideband(
    vllm_xargs: dict[str, Any] | None,
) -> WorkflowSidebandValidation | None:
    """Validate workflow metadata without changing scheduler behavior."""

    if vllm_xargs is None:
        return None
    present_fields = tuple(sorted(str(key) for key in vllm_xargs))
    issues: list[WorkflowSidebandIssue] = []
    missing = tuple(
        field
        for field in REQUIRED_WORKFLOW_SIDEBAND_FIELDS
        if not vllm_xargs.get(field)
    )
    for field in missing:
        issues.append(WorkflowSidebandIssue(field, "missing required sideband field"))
    for field in sorted(set(present_fields) - set(ALLOWED_WORKFLOW_SIDEBAND_FIELDS)):
        issues.append(
            WorkflowSidebandIssue(
                field,
                "field is not in the workflow sideband contract",
            )
        )
    for field in FORBIDDEN_WORKFLOW_SIDEBAND_FIELDS:
        if field in vllm_xargs:
            issues.append(
                WorkflowSidebandIssue(
                    field,
                    "field must not be sent in workflow sideband",
                )
            )
    for field in ("private_release_hint", "effective_release_hint", "release_hint"):
        value = vllm_xargs.get(field)
        if isinstance(value, str) and value and value not in _VALID_RELEASE_HINTS:
            issues.append(
                WorkflowSidebandIssue(field, f"invalid release hint {value!r}")
            )
    for field in _INTEGER_STRING_FIELDS:
        value = vllm_xargs.get(field)
        if isinstance(value, str) and value and not value.isdigit():
            issues.append(WorkflowSidebandIssue(field, "expected integer string"))
        elif isinstance(value, int) and value < 0:
            issues.append(
                WorkflowSidebandIssue(field, "expected non-negative integer")
            )
    for field in _BOOLEAN_STRING_FIELDS:
        value = vllm_xargs.get(field)
        if isinstance(value, str) and value not in {"true", "false"}:
            issues.append(WorkflowSidebandIssue(field, "expected boolean string"))
    if ("spawn_index" in vllm_xargs) != ("spawn_size" in vllm_xargs):
        issues.append(
            WorkflowSidebandIssue(
                "spawn_index/spawn_size",
                "spawn index and size must appear together",
            )
        )
    return WorkflowSidebandValidation(
        valid=not issues,
        present_fields=present_fields,
        missing_required_fields=missing,
        issues=tuple(issues),
    )
