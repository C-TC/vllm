# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter

from vllm.entrypoints.openai.chat_completion.workflow_sideband import (
    validate_workflow_sideband,
)

_MAX_RECORDS = 256
_WORKFLOW_TEST_HOOK_ENV = "WORKFLOW_TEST_HOOK"
_WORKFLOW_TEST_HOOK_FILE_ENV = "WORKFLOW_TEST_HOOK_FILE"
_DEFAULT_HOOK_FILE = "/tmp/vllm_workflow_test_hook.jsonl"
_TOKEN_PREFIX_FINGERPRINT_LEN = 16


@dataclass(slots=True, frozen=True)
class WorkflowTestHookRecord:
    source: str
    path: str | None
    request_id: str | None
    vllm_xargs: dict[str, Any] | None
    workflow_sideband_valid: bool | None = None
    workflow_sideband_present_fields: tuple[str, ...] | None = None
    workflow_sideband_missing_required_fields: tuple[str, ...] | None = None
    workflow_sideband_issues: list[dict[str, str]] | None = None
    engine_token_source: str | None = None
    model: str | None = None
    served_model_name: str | None = None
    tokenizer_id: str | None = None
    chat_template_id: str | None = None
    engine_prompt_token_count: int | None = None
    prompt_token_ids_hash: str | None = None
    prompt_token_ids_prefix_hash: str | None = None
    prompt_token_ids_prefix_len: int | None = None
    group_aware_scheduling_enabled: bool | None = None
    workflow_scheduler_group_key: str | None = None
    workflow_scheduler_selected_rank: int | None = None
    workflow_scheduler_reason: str | None = None
    workflow_scheduler_group_source: str | None = None
    workflow_scheduler_token_lcp_len: int | None = None
    workflow_scheduler_token_lcp_hash: str | None = None
    workflow_scheduler_scan_count: int | None = None
    workflow_scheduler_candidate_group_size: int | None = None
    workflow_scheduler_fairness_guard_reason: str | None = None
    workflow_scheduler_queue_head_delay_ms: float | None = None
    workflow_scheduler_queue_head_delay_bucket: str | None = None
    join_tail_scheduling_enabled: bool | None = None
    workflow_join_tail_selected: bool | None = None
    workflow_join_tail_reason: str | None = None
    workflow_join_tail_scan_count: int | None = None
    workflow_join_tail_candidate_count: int | None = None
    workflow_join_tail_remaining_values: tuple[int, ...] | None = None
    workflow_join_tail_fairness_guard_reason: str | None = None
    action_id: str | None = None
    action_kind: str | None = None
    lifecycle_status: str | None = None
    reject_reason: str | None = None
    prewarm_status: str | None = None
    prewarm_attempted: bool | None = None
    lease_status: str | None = None
    lease_reason: str | None = None
    lease_cleanup_reason: str | None = None
    lease_cleanup_count: int | None = None
    lease_token_count: int | None = None
    lease_full_block_count: int | None = None
    lease_ttl_ms: int | None = None
    prepared_prefix_object_status: str | None = None
    prepared_prefix_ref_status: str | None = None
    lease_event_status: str | None = None
    lease_event_reason: str | None = None
    lease_id_present: bool | None = None
    prefix_id_present: bool | None = None
    prefix_token_count: int | None = None
    prefix_token_hash: str | None = None
    prepared_prefix_match_status: str | None = None
    prepared_prefix_action_id: str | None = None
    prepared_prefix_token_count: int | None = None
    prepared_prefix_token_hash: str | None = None
    prepared_prefix_mismatch_reason: str | None = None
    prepared_prefix_lease_status: str | None = None
    prepared_prefix_lease_match_status: str | None = None
    prepared_prefix_cache_status: str | None = None
    prepared_prefix_num_cached_tokens: int | None = None
    prepared_prefix_recomputed_tokens: int | None = None
    prepared_prefix_cached_at_least_prefix: bool | None = None
    dp_rank: int | None = None
    client_index: int | None = None
    pid: int | None = None


router = APIRouter()
_records: deque[WorkflowTestHookRecord] = deque(maxlen=_MAX_RECORDS)
_records_lock = Lock()


def workflow_test_hook_enabled() -> bool:
    value = os.getenv(_WORKFLOW_TEST_HOOK_ENV, "")
    return value == "1"


def _hook_file_path() -> Path:
    return Path(os.getenv(_WORKFLOW_TEST_HOOK_FILE_ENV, _DEFAULT_HOOK_FILE))


def record_chat_request(
    *,
    path: str,
    request_id: str | None,
    vllm_xargs: dict[str, Any] | None,
    source: str = "api_server",
    prompt_token_ids: list[int] | None = None,
    engine_prompt_token_count: int | None = None,
    model: str | None = None,
    served_model_name: str | None = None,
    tokenizer_id: str | None = None,
    chat_template_id: str | None = None,
    prepared_prefix_match: dict[str, Any] | None = None,
    prepared_prefix_cache: dict[str, Any] | None = None,
) -> None:
    _record_event(
        source=source,
        path=path,
        request_id=request_id,
        vllm_xargs=vllm_xargs,
        prompt_token_ids=prompt_token_ids,
        engine_prompt_token_count=engine_prompt_token_count,
        engine_token_source=source,
        model=model,
        served_model_name=served_model_name,
        tokenizer_id=tokenizer_id,
        chat_template_id=chat_template_id,
        prepared_prefix_match=prepared_prefix_match,
        prepared_prefix_cache=prepared_prefix_cache,
    )


def record_scheduler_request(
    *,
    request_id: str | None,
    vllm_xargs: dict[str, Any] | None,
    dp_rank: int | None,
    client_index: int | None,
    prompt_token_ids: list[int] | None = None,
    engine_prompt_token_count: int | None = None,
    group_aware_scheduling_enabled: bool | None = None,
    workflow_scheduler_group_key: str | None = None,
    workflow_scheduler_selected_rank: int | None = None,
    workflow_scheduler_reason: str | None = None,
    workflow_scheduler_group_source: str | None = None,
    workflow_scheduler_token_lcp_len: int | None = None,
    workflow_scheduler_token_lcp_hash: str | None = None,
    workflow_scheduler_scan_count: int | None = None,
    workflow_scheduler_candidate_group_size: int | None = None,
    workflow_scheduler_fairness_guard_reason: str | None = None,
    workflow_scheduler_queue_head_delay_ms: float | None = None,
    workflow_scheduler_queue_head_delay_bucket: str | None = None,
    join_tail_scheduling_enabled: bool | None = None,
    workflow_join_tail_selected: bool | None = None,
    workflow_join_tail_reason: str | None = None,
    workflow_join_tail_scan_count: int | None = None,
    workflow_join_tail_candidate_count: int | None = None,
    workflow_join_tail_remaining_values: tuple[int, ...] | None = None,
    workflow_join_tail_fairness_guard_reason: str | None = None,
) -> None:
    internal_prefill_only_action_id = _internal_prefill_only_action_id(vllm_xargs)
    public_vllm_xargs = _public_workflow_xargs(vllm_xargs)
    _record_event(
        source="scheduler",
        path=None,
        request_id=request_id,
        # Internal PreparedPrefix prewarm requests use extra_args only as an
        # engine-private control plane. They are not ordinary workflow sideband,
        # so the hook should not validate those fields against the public
        # per-request contract.
        vllm_xargs=None
        if internal_prefill_only_action_id is not None
        else public_vllm_xargs,
        dp_rank=dp_rank,
        client_index=client_index,
        prompt_token_ids=prompt_token_ids,
        engine_prompt_token_count=engine_prompt_token_count,
        engine_token_source="scheduler",
        group_aware_scheduling_enabled=group_aware_scheduling_enabled,
        workflow_scheduler_group_key=workflow_scheduler_group_key,
        workflow_scheduler_selected_rank=workflow_scheduler_selected_rank,
        workflow_scheduler_reason=workflow_scheduler_reason,
        workflow_scheduler_group_source=workflow_scheduler_group_source,
        workflow_scheduler_token_lcp_len=workflow_scheduler_token_lcp_len,
        workflow_scheduler_token_lcp_hash=workflow_scheduler_token_lcp_hash,
        workflow_scheduler_scan_count=workflow_scheduler_scan_count,
        workflow_scheduler_candidate_group_size=workflow_scheduler_candidate_group_size,
        workflow_scheduler_fairness_guard_reason=(
            workflow_scheduler_fairness_guard_reason
        ),
        workflow_scheduler_queue_head_delay_ms=workflow_scheduler_queue_head_delay_ms,
        workflow_scheduler_queue_head_delay_bucket=(
            workflow_scheduler_queue_head_delay_bucket
        ),
        join_tail_scheduling_enabled=join_tail_scheduling_enabled,
        workflow_join_tail_selected=workflow_join_tail_selected,
        workflow_join_tail_reason=workflow_join_tail_reason,
        workflow_join_tail_scan_count=workflow_join_tail_scan_count,
        workflow_join_tail_candidate_count=workflow_join_tail_candidate_count,
        workflow_join_tail_remaining_values=workflow_join_tail_remaining_values,
        workflow_join_tail_fairness_guard_reason=(
            workflow_join_tail_fairness_guard_reason
        ),
        action_id=internal_prefill_only_action_id,
        action_kind="prefix_prepare"
        if internal_prefill_only_action_id is not None
        else None,
    )


def record_workflow_action(
    *,
    action_id: str | None,
    action_kind: str | None,
    lifecycle_status: str,
    reject_reason: str | None = None,
    prewarm_status: str | None = None,
    prewarm_attempted: bool | None = None,
    lease_status: str | None = None,
    lease_reason: str | None = None,
    lease_cleanup_reason: str | None = None,
    lease_cleanup_count: int | None = None,
    lease_token_count: int | None = None,
    lease_full_block_count: int | None = None,
    lease_ttl_ms: int | None = None,
    prepared_prefix_object_status: str | None = None,
    prepared_prefix_ref_status: str | None = None,
    lease_event_status: str | None = None,
    lease_event_reason: str | None = None,
    lease_id_present: bool | None = None,
    prefix_id_present: bool | None = None,
    prefix_token_count: int | None = None,
    prefix_token_hash: str | None = None,
    engine_token_source: str | None = None,
    model: str | None = None,
    served_model_name: str | None = None,
    tokenizer_id: str | None = None,
    chat_template_id: str | None = None,
) -> None:
    _record_event(
        source="workflow_action",
        path="/v1/workflow/coopt/actions",
        request_id=None,
        vllm_xargs=None,
        engine_token_source=engine_token_source,
        model=model,
        served_model_name=served_model_name,
        tokenizer_id=tokenizer_id,
        chat_template_id=chat_template_id,
        action_id=action_id,
        action_kind=action_kind,
        lifecycle_status=lifecycle_status,
        reject_reason=reject_reason,
        prewarm_status=prewarm_status,
        prewarm_attempted=prewarm_attempted,
        lease_status=lease_status,
        lease_reason=lease_reason,
        lease_cleanup_reason=lease_cleanup_reason,
        lease_cleanup_count=lease_cleanup_count,
        lease_token_count=lease_token_count,
        lease_full_block_count=lease_full_block_count,
        lease_ttl_ms=lease_ttl_ms,
        prepared_prefix_object_status=prepared_prefix_object_status,
        prepared_prefix_ref_status=prepared_prefix_ref_status,
        lease_event_status=lease_event_status,
        lease_event_reason=lease_event_reason,
        lease_id_present=lease_id_present,
        prefix_id_present=prefix_id_present,
        prefix_token_count=prefix_token_count,
        prefix_token_hash=prefix_token_hash,
    )


def _record_event(
    *,
    source: str,
    path: str | None,
    request_id: str | None,
    vllm_xargs: dict[str, Any] | None,
    dp_rank: int | None = None,
    client_index: int | None = None,
    prompt_token_ids: list[int] | None = None,
    engine_prompt_token_count: int | None = None,
    engine_token_source: str | None = None,
    model: str | None = None,
    served_model_name: str | None = None,
    tokenizer_id: str | None = None,
    chat_template_id: str | None = None,
    group_aware_scheduling_enabled: bool | None = None,
    workflow_scheduler_group_key: str | None = None,
    workflow_scheduler_selected_rank: int | None = None,
    workflow_scheduler_reason: str | None = None,
    workflow_scheduler_group_source: str | None = None,
    workflow_scheduler_token_lcp_len: int | None = None,
    workflow_scheduler_token_lcp_hash: str | None = None,
    workflow_scheduler_scan_count: int | None = None,
    workflow_scheduler_candidate_group_size: int | None = None,
    workflow_scheduler_fairness_guard_reason: str | None = None,
    workflow_scheduler_queue_head_delay_ms: float | None = None,
    workflow_scheduler_queue_head_delay_bucket: str | None = None,
    join_tail_scheduling_enabled: bool | None = None,
    workflow_join_tail_selected: bool | None = None,
    workflow_join_tail_reason: str | None = None,
    workflow_join_tail_scan_count: int | None = None,
    workflow_join_tail_candidate_count: int | None = None,
    workflow_join_tail_remaining_values: tuple[int, ...] | None = None,
    workflow_join_tail_fairness_guard_reason: str | None = None,
    action_id: str | None = None,
    action_kind: str | None = None,
    lifecycle_status: str | None = None,
    reject_reason: str | None = None,
    prewarm_status: str | None = None,
    prewarm_attempted: bool | None = None,
    lease_status: str | None = None,
    lease_reason: str | None = None,
    lease_cleanup_reason: str | None = None,
    lease_cleanup_count: int | None = None,
    lease_token_count: int | None = None,
    lease_full_block_count: int | None = None,
    lease_ttl_ms: int | None = None,
    prepared_prefix_object_status: str | None = None,
    prepared_prefix_ref_status: str | None = None,
    lease_event_status: str | None = None,
    lease_event_reason: str | None = None,
    lease_id_present: bool | None = None,
    prefix_id_present: bool | None = None,
    prefix_token_count: int | None = None,
    prefix_token_hash: str | None = None,
    prepared_prefix_match: dict[str, Any] | None = None,
    prepared_prefix_cache: dict[str, Any] | None = None,
) -> None:
    if not workflow_test_hook_enabled():
        return
    sideband_validation = validate_workflow_sideband(
        vllm_xargs if isinstance(vllm_xargs, dict) else None
    )
    record = WorkflowTestHookRecord(
        source=source,
        path=path,
        request_id=request_id,
        vllm_xargs=dict(vllm_xargs) if isinstance(vllm_xargs, dict) else None,
        workflow_sideband_valid=(
            sideband_validation.valid if sideband_validation is not None else None
        ),
        workflow_sideband_present_fields=(
            sideband_validation.present_fields
            if sideband_validation is not None
            else None
        ),
        workflow_sideband_missing_required_fields=(
            sideband_validation.missing_required_fields
            if sideband_validation is not None
            else None
        ),
        workflow_sideband_issues=(
            sideband_validation.issues_as_dicts()
            if sideband_validation is not None
            else None
        ),
        engine_token_source=engine_token_source,
        model=model,
        served_model_name=served_model_name,
        tokenizer_id=tokenizer_id,
        chat_template_id=chat_template_id,
        engine_prompt_token_count=_prompt_token_count(
            prompt_token_ids,
            engine_prompt_token_count,
        ),
        prompt_token_ids_hash=_hash_token_ids(prompt_token_ids),
        prompt_token_ids_prefix_hash=_hash_token_ids(
            prompt_token_ids[:_TOKEN_PREFIX_FINGERPRINT_LEN]
            if prompt_token_ids is not None
            else None
        ),
        prompt_token_ids_prefix_len=(
            min(len(prompt_token_ids), _TOKEN_PREFIX_FINGERPRINT_LEN)
            if prompt_token_ids is not None
            else None
        ),
        group_aware_scheduling_enabled=group_aware_scheduling_enabled,
        workflow_scheduler_group_key=workflow_scheduler_group_key,
        workflow_scheduler_selected_rank=workflow_scheduler_selected_rank,
        workflow_scheduler_reason=workflow_scheduler_reason,
        workflow_scheduler_group_source=workflow_scheduler_group_source,
        workflow_scheduler_token_lcp_len=workflow_scheduler_token_lcp_len,
        workflow_scheduler_token_lcp_hash=workflow_scheduler_token_lcp_hash,
        workflow_scheduler_scan_count=workflow_scheduler_scan_count,
        workflow_scheduler_candidate_group_size=workflow_scheduler_candidate_group_size,
        workflow_scheduler_fairness_guard_reason=(
            workflow_scheduler_fairness_guard_reason
        ),
        workflow_scheduler_queue_head_delay_ms=workflow_scheduler_queue_head_delay_ms,
        workflow_scheduler_queue_head_delay_bucket=(
            workflow_scheduler_queue_head_delay_bucket
        ),
        join_tail_scheduling_enabled=join_tail_scheduling_enabled,
        workflow_join_tail_selected=workflow_join_tail_selected,
        workflow_join_tail_reason=workflow_join_tail_reason,
        workflow_join_tail_scan_count=workflow_join_tail_scan_count,
        workflow_join_tail_candidate_count=workflow_join_tail_candidate_count,
        workflow_join_tail_remaining_values=workflow_join_tail_remaining_values,
        workflow_join_tail_fairness_guard_reason=(
            workflow_join_tail_fairness_guard_reason
        ),
        action_id=action_id,
        action_kind=action_kind,
        lifecycle_status=lifecycle_status,
        reject_reason=reject_reason,
        prewarm_status=prewarm_status,
        prewarm_attempted=prewarm_attempted,
        lease_status=lease_status,
        lease_reason=lease_reason,
        lease_cleanup_reason=lease_cleanup_reason,
        lease_cleanup_count=lease_cleanup_count,
        lease_token_count=lease_token_count,
        lease_full_block_count=lease_full_block_count,
        lease_ttl_ms=lease_ttl_ms,
        prepared_prefix_object_status=prepared_prefix_object_status,
        prepared_prefix_ref_status=prepared_prefix_ref_status,
        lease_event_status=lease_event_status,
        lease_event_reason=lease_event_reason,
        lease_id_present=lease_id_present,
        prefix_id_present=prefix_id_present,
        prefix_token_count=prefix_token_count,
        prefix_token_hash=prefix_token_hash,
        prepared_prefix_match_status=_prepared_prefix_match_str(
            prepared_prefix_match,
            "prepared_prefix_match_status",
        ),
        prepared_prefix_action_id=_prepared_prefix_match_str(
            prepared_prefix_match,
            "prepared_prefix_action_id",
        ),
        prepared_prefix_token_count=_prepared_prefix_match_int(
            prepared_prefix_match,
            "prepared_prefix_token_count",
        ),
        prepared_prefix_token_hash=_prepared_prefix_match_str(
            prepared_prefix_match,
            "prepared_prefix_token_hash",
        ),
        prepared_prefix_mismatch_reason=_prepared_prefix_match_str(
            prepared_prefix_match,
            "prepared_prefix_mismatch_reason",
        ),
        prepared_prefix_lease_status=_prepared_prefix_match_str(
            prepared_prefix_match,
            "prepared_prefix_lease_status",
        ),
        prepared_prefix_lease_match_status=_prepared_prefix_match_str(
            prepared_prefix_match,
            "prepared_prefix_lease_match_status",
        ),
        prepared_prefix_cache_status=_prepared_prefix_cache_str(
            prepared_prefix_cache,
            "prepared_prefix_cache_status",
        ),
        prepared_prefix_num_cached_tokens=_prepared_prefix_cache_int(
            prepared_prefix_cache,
            "prepared_prefix_num_cached_tokens",
        ),
        prepared_prefix_recomputed_tokens=_prepared_prefix_cache_int(
            prepared_prefix_cache,
            "prepared_prefix_recomputed_tokens",
        ),
        prepared_prefix_cached_at_least_prefix=_prepared_prefix_cache_bool(
            prepared_prefix_cache,
            "prepared_prefix_cached_at_least_prefix",
        ),
        dp_rank=dp_rank,
        client_index=client_index,
        pid=os.getpid(),
    )
    with _records_lock:
        _records.append(record)
    try:
        _append_record_to_file(record)
    except OSError:
        # The hook is observability-only and must never affect request serving.
        return


def _prompt_token_count(
    prompt_token_ids: list[int] | None,
    explicit_count: int | None,
) -> int | None:
    if prompt_token_ids is not None:
        return len(prompt_token_ids)
    return explicit_count


def _internal_prefill_only_action_id(vllm_xargs: dict[str, Any] | None) -> str | None:
    if not isinstance(vllm_xargs, dict):
        return None
    if vllm_xargs.get("workflow_prefill_only") is not True:
        return None
    action_id = vllm_xargs.get("workflow_prefix_prepare_action_id")
    return action_id if isinstance(action_id, str) and action_id else None


def _public_workflow_xargs(vllm_xargs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(vllm_xargs, dict):
        return None
    public = {
        key: value
        for key, value in vllm_xargs.items()
        if key != "workflow_prepared_prefix_matched_action_id"
    }
    return public or None


def _hash_token_ids(token_ids: list[int] | None) -> str | None:
    if token_ids is None:
        return None
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return f"sha1:{hashlib.sha1(payload).hexdigest()}"


def _prepared_prefix_match_str(
    match: dict[str, Any] | None,
    key: str,
) -> str | None:
    if not isinstance(match, dict):
        return None
    value = match.get(key)
    return value if isinstance(value, str) and value else None


def _prepared_prefix_match_int(
    match: dict[str, Any] | None,
    key: str,
) -> int | None:
    if not isinstance(match, dict):
        return None
    value = match.get(key)
    return value if isinstance(value, int) else None


def _prepared_prefix_cache_str(
    cache: dict[str, Any] | None,
    key: str,
) -> str | None:
    if not isinstance(cache, dict):
        return None
    value = cache.get(key)
    return value if isinstance(value, str) and value else None


def _prepared_prefix_cache_int(
    cache: dict[str, Any] | None,
    key: str,
) -> int | None:
    if not isinstance(cache, dict):
        return None
    value = cache.get(key)
    return value if isinstance(value, int) else None


def _prepared_prefix_cache_bool(
    cache: dict[str, Any] | None,
    key: str,
) -> bool | None:
    if not isinstance(cache, dict):
        return None
    value = cache.get(key)
    return value if isinstance(value, bool) else None


def _append_record_to_file(record: WorkflowTestHookRecord) -> None:
    hook_file = _hook_file_path()
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    with hook_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True))
        handle.write("\n")


def _read_file_records() -> list[dict[str, object]]:
    hook_file = _hook_file_path()
    if not hook_file.exists():
        return []
    records: list[dict[str, object]] = []
    with hook_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _clear_file_records() -> int:
    hook_file = _hook_file_path()
    if not hook_file.exists():
        return 0
    with hook_file.open("r", encoding="utf-8") as handle:
        cleared = sum(1 for _ in handle)
    hook_file.write_text("", encoding="utf-8")
    return cleared


@router.get("/debug/workflow_test_hook/records")
async def get_workflow_test_hook_records() -> dict[str, object]:
    with _records_lock:
        memory_records = [asdict(record) for record in _records]
    file_records = _read_file_records()
    return {
        "enabled": True,
        "memory_records": memory_records,
        "file_records": file_records,
        "records": memory_records + file_records,
    }


@router.delete("/debug/workflow_test_hook/records")
async def clear_workflow_test_hook_records() -> dict[str, object]:
    with _records_lock:
        cleared_memory = len(_records)
        _records.clear()
    cleared_file = _clear_file_records()
    return {
        "enabled": True,
        "cleared": cleared_memory + cleared_file,
        "cleared_memory": cleared_memory,
        "cleared_file": cleared_file,
    }
