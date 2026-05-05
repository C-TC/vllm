# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.chat_completion import api_router
from vllm.entrypoints.openai.chat_completion.workflow_actions import (
    workflow_coopt_actions_enabled,
)
from vllm.entrypoints.openai.chat_completion.workflow_test_hook import _records


def _valid_prefix_prepare_action(
    generation: int = 1,
    key_suffix: str = "demo",
) -> dict[str, object]:
    return {
        "action_id": f"coopt_action:prepare:{key_suffix}:{generation}",
        "action_kind": "prefix_prepare",
        "idempotency_key": f"prefix_prepare:{key_suffix}",
        "generation": generation,
        "virtual_request_id": f"wf:site:prefix_prepare:{key_suffix}",
        "site_id": "main:writer",
        "render_domain_id": "render",
        "token_domain_id": "token",
        "prefix_part_count": 2,
        "messages_prefix": [
            {"role": "system", "content": "private system prompt"},
            {"role": "user", "content": "private prefix"},
        ],
        "prefix_token_count_advisory": 64,
        "ttl_ms": 30000,
    }


def test_workflow_actions_route_is_only_attached_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", raising=False)
    app = FastAPI()
    api_router.attach_router(app)
    route_paths = {route.path for route in app.routes}
    assert workflow_coopt_actions_enabled() is False
    assert "/v1/workflow/coopt/actions" not in route_paths

    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    app = FastAPI()
    api_router.attach_router(app)
    route_paths = {route.path for route in app.routes}
    assert workflow_coopt_actions_enabled() is True
    assert "/v1/workflow/coopt/actions" in route_paths


def test_workflow_actions_accept_prefix_prepare_without_logging_raw_prompt(
    monkeypatch,
    tmp_path,
) -> None:
    _records.clear()
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(tmp_path / "hook.jsonl"))
    app = FastAPI()
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="accept"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["lifecycle_status"] == "accepted"
    assert payload["prewarm_status"] == "prewarm_not_implemented"
    assert "messages_prefix" not in payload
    assert _records
    record = _records[-1]
    assert record.source == "workflow_action"
    assert record.lifecycle_status == "accepted"
    assert record.prefix_token_hash is not None
    hook_text = (tmp_path / "hook.jsonl").read_text(encoding="utf-8")
    assert "workflow_action" in hook_text
    assert "private prefix" not in hook_text
    assert "messages_prefix" not in hook_text


def test_workflow_actions_reject_invalid_and_old_generation(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    app = FastAPI()
    api_router.attach_router(app)
    client = TestClient(app)

    invalid = _valid_prefix_prepare_action(key_suffix="invalid")
    invalid.pop("messages_prefix")
    invalid_response = client.post("/v1/workflow/coopt/actions", json=invalid)
    assert invalid_response.status_code == 400
    assert invalid_response.json()["reject_reason"] == "missing_messages_prefix"

    first = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(7, key_suffix="duplicate"),
    )
    old = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(6, key_suffix="duplicate"),
    )
    assert first.status_code == 200
    assert old.status_code == 400
    assert old.json()["reject_reason"] == "duplicate_or_old_generation"


def test_workflow_actions_can_ignore_short_advisory_prefix(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "32")
    app = FastAPI()
    api_router.attach_router(app)
    client = TestClient(app)
    action = _valid_prefix_prepare_action(99, key_suffix="short")
    action["prefix_token_count_advisory"] = 4

    response = client.post("/v1/workflow/coopt/actions", json=action)

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["lifecycle_status"] == "ignored"
    assert payload["prewarm_status"] == "ignored:prefix_too_short"
