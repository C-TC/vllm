# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi import FastAPI

from vllm.entrypoints.openai.chat_completion import api_router
from vllm.entrypoints.openai.chat_completion.workflow_test_hook import (
    _records,
    record_chat_request,
    record_scheduler_request,
    workflow_test_hook_enabled,
)


def _clear_memory_records() -> None:
    _records.clear()


def test_workflow_test_hook_disabled_by_default_is_noop(monkeypatch, tmp_path) -> None:
    hook_file = tmp_path / "hook.jsonl"
    _clear_memory_records()
    monkeypatch.delenv("WORKFLOW_TEST_HOOK", raising=False)
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(hook_file))

    assert workflow_test_hook_enabled() is False
    record_chat_request(
        path="/v1/chat/completions",
        request_id="req-1",
        vllm_xargs={"unexpected": "ignored"},
    )

    assert list(_records) == []
    assert not hook_file.exists()


def test_workflow_test_hook_route_is_only_attached_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("WORKFLOW_TEST_HOOK", raising=False)
    app = FastAPI()
    api_router.attach_router(app)
    route_paths = {route.path for route in app.routes}
    assert "/debug/workflow_test_hook/records" not in route_paths

    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    app = FastAPI()
    api_router.attach_router(app)
    route_paths = {route.path for route in app.routes}
    assert "/debug/workflow_test_hook/records" in route_paths


def test_workflow_test_hook_records_vanilla_request_without_sideband(
    monkeypatch,
    tmp_path,
) -> None:
    hook_file = tmp_path / "hook.jsonl"
    _clear_memory_records()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(hook_file))

    record_chat_request(
        path="/v1/chat/completions",
        request_id="req-vanilla",
        vllm_xargs=None,
    )

    assert len(_records) == 1
    record = _records[0]
    assert record.vllm_xargs is None
    assert record.workflow_sideband_valid is None
    assert record.workflow_sideband_present_fields is None
    assert '"request_id": "req-vanilla"' in hook_file.read_text(encoding="utf-8")


def test_workflow_test_hook_records_invalid_sideband_without_blocking(
    monkeypatch,
    tmp_path,
) -> None:
    hook_file = tmp_path / "hook.jsonl"
    _clear_memory_records()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(hook_file))

    record_chat_request(
        path="/v1/chat/completions",
        request_id="req-invalid",
        vllm_xargs={
            "workflow_id": "demo",
            "stable_prefix_handle_id": "stable_prefix:abc",
        },
    )

    assert len(_records) == 1
    record = _records[0]
    assert record.workflow_sideband_valid is False
    assert record.workflow_sideband_issues is not None
    issue_fields = {issue["field"] for issue in record.workflow_sideband_issues}
    assert "stable_prefix_handle_id" in issue_fields
    assert "site_id" in issue_fields
    assert '"request_id": "req-invalid"' in hook_file.read_text(encoding="utf-8")


def test_workflow_test_hook_records_redacted_engine_token_metrics(
    monkeypatch,
    tmp_path,
) -> None:
    hook_file = tmp_path / "hook.jsonl"
    _clear_memory_records()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(hook_file))

    record_chat_request(
        source="api_server_tokenized",
        path="/v1/chat/completions",
        request_id="req-tokenized",
        vllm_xargs=None,
        prompt_token_ids=[101, 202, 303],
        model="demo-model",
        served_model_name="served-demo",
        tokenizer_id="tok-demo",
        chat_template_id="sha1:template",
    )

    assert len(_records) == 1
    record = _records[0]
    assert record.source == "api_server_tokenized"
    assert record.engine_token_source == "api_server_tokenized"
    assert record.engine_prompt_token_count == 3
    assert record.prompt_token_ids_hash is not None
    assert record.prompt_token_ids_prefix_hash is not None
    assert record.prompt_token_ids_prefix_len == 3
    assert record.model == "demo-model"
    assert record.served_model_name == "served-demo"
    file_text = hook_file.read_text(encoding="utf-8")
    assert "prompt_token_ids_hash" in file_text
    assert "[101, 202, 303]" not in file_text
    assert '"prompt_token_ids"' not in file_text


def test_workflow_test_hook_records_scheduler_token_count_without_token_ids(
    monkeypatch,
    tmp_path,
) -> None:
    hook_file = tmp_path / "hook.jsonl"
    _clear_memory_records()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(hook_file))

    record_scheduler_request(
        request_id="req-scheduler",
        vllm_xargs=None,
        dp_rank=0,
        client_index=1,
        engine_prompt_token_count=42,
    )

    assert len(_records) == 1
    record = _records[0]
    assert record.source == "scheduler"
    assert record.engine_token_source == "scheduler"
    assert record.engine_prompt_token_count == 42
    assert record.prompt_token_ids_hash is None
    assert record.prompt_token_ids_prefix_hash is None


def test_workflow_test_hook_records_group_aware_scheduler_telemetry(
    monkeypatch,
    tmp_path,
) -> None:
    hook_file = tmp_path / "hook.jsonl"
    _clear_memory_records()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(hook_file))

    record_scheduler_request(
        request_id="req-grouped",
        vllm_xargs={"workflow_id": "wf"},
        dp_rank=0,
        client_index=1,
        prompt_token_ids=[10, 20, 30, 40],
        group_aware_scheduling_enabled=True,
        workflow_scheduler_group_key=(
            "token_verified_lcp:model=unavailable_single_model_assumed:"
            "token_domain=unavailable_single_model_assumed:"
            "len=4:hash=sha1:lcp"
        ),
        workflow_scheduler_selected_rank=2,
        workflow_scheduler_reason="token_verified_lcp",
        workflow_scheduler_group_source="token_verified_lcp",
        workflow_scheduler_token_lcp_len=4,
        workflow_scheduler_token_lcp_hash="sha1:lcp",
        workflow_scheduler_scan_count=8,
        workflow_scheduler_candidate_group_size=3,
        workflow_scheduler_fairness_guard_reason="max_burst",
        workflow_scheduler_queue_head_delay_ms=42.0,
        workflow_scheduler_queue_head_delay_bucket="lt_50ms",
    )

    assert len(_records) == 1
    record = _records[0]
    assert record.source == "scheduler"
    assert record.group_aware_scheduling_enabled is True
    assert record.workflow_scheduler_group_key == (
        "token_verified_lcp:model=unavailable_single_model_assumed:"
        "token_domain=unavailable_single_model_assumed:"
        "len=4:hash=sha1:lcp"
    )
    assert record.workflow_scheduler_selected_rank == 2
    assert record.workflow_scheduler_reason == "token_verified_lcp"
    assert record.workflow_scheduler_group_source == "token_verified_lcp"
    assert record.workflow_scheduler_token_lcp_len == 4
    assert record.workflow_scheduler_token_lcp_hash == "sha1:lcp"
    assert record.workflow_scheduler_scan_count == 8
    assert record.workflow_scheduler_candidate_group_size == 3
    assert record.workflow_scheduler_fairness_guard_reason == "max_burst"
    assert record.workflow_scheduler_queue_head_delay_ms == 42.0
    assert record.workflow_scheduler_queue_head_delay_bucket == "lt_50ms"
    file_text = hook_file.read_text(encoding="utf-8")
    assert "workflow_scheduler_group_key" in file_text
    assert "workflow_scheduler_token_lcp_len" in file_text
    assert "workflow_scheduler_scan_count" in file_text
    assert "sha1:lcp" in file_text
    assert "[10, 20, 30, 40]" not in file_text
    assert '"prompt_token_ids"' not in file_text


def test_workflow_test_hook_file_write_failure_does_not_block_request(
    monkeypatch,
    tmp_path,
) -> None:
    _clear_memory_records()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(tmp_path))

    record_chat_request(
        path="/v1/chat/completions",
        request_id="req-bad-hook-file",
        vllm_xargs=None,
    )

    assert len(_records) == 1
    assert _records[0].request_id == "req-bad-hook-file"
