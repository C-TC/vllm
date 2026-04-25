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
