# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.chat_completion import api_router
from vllm.entrypoints.openai.chat_completion.workflow_actions import (
    mark_workflow_prefix_prewarm_result,
    mark_workflow_prepared_prefix_lease_result,
    match_prepared_prefix_for_request,
    workflow_coopt_actions_enabled,
)
from vllm.entrypoints.openai.chat_completion.workflow_test_hook import (
    _records,
    record_chat_request,
    record_scheduler_request,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.scheduler import _workflow_prefill_only_ready_to_finish


class _FakeChatHandler:
    def __init__(self) -> None:
        self.prewarm_submissions: list[tuple[dict[str, object], dict[str, object]]] = []

    async def verify_workflow_prefix_prepare(self, action: dict[str, object]):
        assert action["model"] == "test-model"
        return {
            "ok": True,
            "prefix_token_ids": [101, 202, 303, 404, 505],
            "model": action["model"],
            "served_model_name": "test-model",
            "tokenizer_id": "fake-tokenizer",
            "chat_template_id": "sha1:fake-template",
        }

    async def submit_workflow_prefix_prewarm(
        self,
        action: dict[str, object],
        token_verification: dict[str, object],
    ):
        self.prewarm_submissions.append((action, token_verification))
        return {"prewarm_status": "prewarm_submitted"}


class _FakeChatHandlerWithoutPrewarm(_FakeChatHandler):
    submit_workflow_prefix_prewarm = None


def _kv_manager_with_cached_prefix(
    *,
    block_count: int = 2,
    hash_block_size: int = 2,
) -> tuple[KVCacheManager, SimpleNamespace]:
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.enable_caching = True
    manager.block_pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    manager.num_kv_cache_groups = 1
    manager._workflow_prepared_prefix_leases = {}
    block_hashes = []
    for index in range(block_count):
        block_hash = BlockHash(bytes([index + 1]) * 32)
        block = manager.block_pool.get_new_blocks(1)[0]
        block.block_hash = make_block_hash_with_group_id(block_hash, 0)
        manager.block_pool.cached_block_hash_to_block.insert(block.block_hash, block)
        manager.block_pool.free_blocks([block])
        block_hashes.append(block_hash)
    request = SimpleNamespace(block_hashes=block_hashes)
    return manager, request


def _valid_prefix_prepare_action(
    generation: int = 1,
    key_suffix: str = "demo",
) -> dict[str, object]:
    return {
        "action_id": f"coopt_action:prepare:{key_suffix}:{generation}",
        "action_kind": "prefix_prepare",
        "action_scope_key": f"workflow:wf:site:main:writer:prefix_prepare:{key_suffix}",
        "idempotency_key": f"prefix_prepare:{key_suffix}",
        "generation": generation,
        "virtual_request_id": f"wf:site:prefix_prepare:{key_suffix}",
        "model": "test-model",
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
    app.state.openai_serving_chat = _FakeChatHandler()
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
    app.state.openai_serving_chat = _FakeChatHandler()
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
    assert payload["prewarm_status"] == "tokenize_only"
    assert payload["prewarm_attempted"] is False
    assert payload["lease_status"] == "observe_only"
    assert payload["lease_reason"] == "retention_mode_observe"
    assert payload["prepared_prefix_ref_status"] == "observe_only"
    assert payload["lease_event_status"] == "observe_only"
    assert payload["lease_id_present"] is False
    assert payload["engine_token_source"] == "engine_authoritative"
    assert payload["prefix_token_count"] == 5
    assert payload["prefix_token_hash"].startswith("sha1:")
    assert payload["prefix_message_hash"].startswith("sha1:")
    assert payload["prepared_prefix_registered"] is True
    assert payload["prepared_prefix_match_status"] == "pending_request"
    assert payload["prepared_prefix_object_status"] == "registered"
    assert payload["tokenizer_id"] == "fake-tokenizer"
    assert "messages_prefix" not in payload
    assert _records
    record = _records[-1]
    assert record.source == "workflow_action"
    assert record.lifecycle_status == "accepted"
    assert record.engine_token_source == "engine_authoritative"
    assert record.lease_status == "observe_only"
    assert record.prepared_prefix_ref_status == "observe_only"
    assert record.prefix_token_count == 5
    assert record.prefix_token_hash is not None
    hook_text = (tmp_path / "hook.jsonl").read_text(encoding="utf-8")
    assert "workflow_action" in hook_text
    assert "private prefix" not in hook_text
    assert "messages_prefix" not in hook_text

    status_response = client.get(
        f"/v1/workflow/coopt/actions/{payload['action_id']}"
    )
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["lifecycle_status"] == "accepted"
    assert status_payload["prefix_token_count"] == 5
    assert status_payload["prepared_prefix_match_status"] == "pending_request"
    assert status_payload["prepared_prefix_object_status"] == "registered"
    assert "messages_prefix" not in status_payload


def test_workflow_actions_reject_prefix_prepare_without_scope_key(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandler()
    api_router.attach_router(app)
    client = TestClient(app)
    action = _valid_prefix_prepare_action(key_suffix="missing-scope")
    action.pop("action_scope_key")

    response = client.post("/v1/workflow/coopt/actions", json=action)

    assert response.status_code == 400
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["lifecycle_status"] == "rejected"
    assert payload["reject_reason"] == "missing_action_scope_key"


def test_workflow_actions_match_prepared_prefix_by_workflow_site(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandler()
    api_router.attach_router(app)
    client = TestClient(app)
    action = _valid_prefix_prepare_action(key_suffix="match")
    action["workflow_instance_id"] = "wf-inst"

    response = client.post("/v1/workflow/coopt/actions", json=action)

    assert response.status_code == 200
    match = match_prepared_prefix_for_request(
        vllm_xargs={
            "workflow_instance_id": "wf-inst",
            "site_id": "main:writer",
        },
        prompt_token_ids=[101, 202, 303, 404, 505, 606],
        model="test-model",
    )
    assert match is not None
    assert match["prepared_prefix_match_status"] == "matched"
    assert match["prepared_prefix_action_id"] == action["action_id"]
    assert match["prepared_prefix_token_count"] == 5
    assert match["prepared_prefix_token_hash"].startswith("sha1:")
    assert match["prepared_prefix_lease_status"] == "observe_only"
    assert match["prepared_prefix_lease_match_status"] == "observe_only"

    status_response = client.get(
        f"/v1/workflow/coopt/actions/{action['action_id']}"
    )
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["prepared_prefix_match_status"] == "matched"
    assert status["prepared_prefix_object_status"] == "matched"


def test_workflow_actions_report_prepared_prefix_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandler()
    api_router.attach_router(app)
    client = TestClient(app)
    action = _valid_prefix_prepare_action(key_suffix="mismatch")
    action["workflow_instance_id"] = "wf-inst"

    response = client.post("/v1/workflow/coopt/actions", json=action)

    assert response.status_code == 200
    match = match_prepared_prefix_for_request(
        vllm_xargs={
            "workflow_instance_id": "wf-inst",
            "site_id": "main:writer",
        },
        prompt_token_ids=[999, 202, 303, 404, 505],
        model="test-model",
    )
    assert match is not None
    assert match["prepared_prefix_match_status"] == "mismatch"
    assert match["prepared_prefix_mismatch_reason"] == "token_prefix_mismatch"


def test_workflow_actions_superseded_prefix_no_longer_matches(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandler()
    api_router.attach_router(app)
    client = TestClient(app)
    first_action = _valid_prefix_prepare_action(1, key_suffix="supersede")
    first_action["workflow_instance_id"] = "wf-inst"
    second_action = _valid_prefix_prepare_action(2, key_suffix="supersede")
    second_action["workflow_instance_id"] = "wf-inst"

    first = client.post("/v1/workflow/coopt/actions", json=first_action)
    second = client.post("/v1/workflow/coopt/actions", json=second_action)

    assert first.status_code == 200
    assert second.status_code == 200
    first_status = client.get(
        f"/v1/workflow/coopt/actions/{first_action['action_id']}"
    ).json()
    assert first_status["lifecycle_status"] == "superseded"
    assert first_status["prepared_prefix_match_status"] == "superseded"
    assert first_status["prepared_prefix_object_status"] == "superseded"
    assert first_status["lease_cleanup_reason"] == "superseded_registry_only"
    match = match_prepared_prefix_for_request(
        vllm_xargs={
            "workflow_instance_id": "wf-inst",
            "site_id": "main:writer",
        },
        prompt_token_ids=[101, 202, 303, 404, 505, 606],
        model="test-model",
    )
    assert match is not None
    assert match["prepared_prefix_match_status"] == "matched"
    assert match["prepared_prefix_action_id"] == second_action["action_id"]


def test_workflow_actions_reject_invalid_and_old_generation(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandler()
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
    app.state.openai_serving_chat = _FakeChatHandler()
    api_router.attach_router(app)
    client = TestClient(app)
    action = _valid_prefix_prepare_action(99, key_suffix="short")

    response = client.post("/v1/workflow/coopt/actions", json=action)

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["lifecycle_status"] == "ignored"
    assert payload["prewarm_status"] == "ignored:prefix_too_short"
    assert payload["prefix_token_count"] == 5


def test_workflow_actions_experimental_prewarm_submits_hidden_prewarm(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MODE", "experimental_prewarm")
    app = FastAPI()
    handler = _FakeChatHandler()
    app.state.openai_serving_chat = handler
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="experimental"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["lifecycle_status"] == "accepted"
    assert payload["prewarm_status"] == "prewarm_submitted"
    assert payload["prewarm_attempted"] is True
    assert isinstance(payload["accepted_at_unix_ms"], int)
    assert isinstance(payload["prewarm_submitted_at_unix_ms"], int)
    assert len(handler.prewarm_submissions) == 1
    _action, token_verification = handler.prewarm_submissions[0]
    assert "_prefix_token_ids" in token_verification

    mark_workflow_prefix_prewarm_result(
        action_id=str(payload["action_id"]),
        prewarm_status="prewarm_completed",
    )
    status_response = client.get(
        f"/v1/workflow/coopt/actions/{payload['action_id']}"
    )
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["prewarm_status"] == "prewarm_completed"
    assert isinstance(status["prewarm_finished_at_unix_ms"], int)


def test_workflow_actions_prewarm_observe_alias_submits_hidden_prewarm(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MODE", "prewarm_observe")
    app = FastAPI()
    handler = _FakeChatHandler()
    app.state.openai_serving_chat = handler
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="prewarm-observe"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prewarm_status"] == "prewarm_submitted"
    assert payload["prewarm_attempted"] is True
    assert len(handler.prewarm_submissions) == 1


def test_workflow_actions_lease_mode_reports_unavailable_without_exposing_handles(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MODE", "experimental_prewarm")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_RETENTION_MODE", "lease")
    app = FastAPI()
    handler = _FakeChatHandler()
    app.state.openai_serving_chat = handler
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="lease"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prewarm_status"] == "prewarm_submitted"
    assert payload["lease_status"] == "pending_prewarm"
    assert payload["lease_reason"] == "waiting_for_prewarm_completion"

    mark_workflow_prefix_prewarm_result(
        action_id=str(payload["action_id"]),
        prewarm_status="prewarm_completed",
    )
    status_response = client.get(
        f"/v1/workflow/coopt/actions/{payload['action_id']}"
    )
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["prewarm_status"] == "prewarm_completed"
    assert status["lease_status"] == "lease_unavailable"
    assert status["prepared_prefix_object_status"] == "lease_unavailable"
    assert status["prepared_prefix_ref_status"] == "lease_unavailable"
    assert status["lease_event_status"] == "lease_unavailable"
    assert status["lease_reason"] == "no_safe_internal_cache_lease_api"
    assert "kv" not in json.dumps(status).lower()


def test_workflow_actions_lease_mode_records_engine_core_lease_update(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MODE", "experimental_prewarm")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_RETENTION_MODE", "lease")
    app = FastAPI()
    handler = _FakeChatHandler()
    app.state.openai_serving_chat = handler
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="engine-lease"),
    )

    assert response.status_code == 200
    action_id = str(response.json()["action_id"])
    mark_workflow_prefix_prewarm_result(
        action_id=action_id,
        prewarm_status="prewarm_completed",
        lease_update={
            "lease_status": "leased",
            "lease_reason": "engine_core_cache_blocks_touched",
            "lease_token_count": 5,
            "lease_full_block_count": 1,
            "lease_ttl_ms": 30000,
            "prepared_prefix_ref_status": "leased",
            "lease_event_status": "leased",
            "lease_event_reason": "engine_core_cache_blocks_touched",
            "lease_id_present": True,
            "prefix_id_present": True,
        },
    )

    status_response = client.get(f"/v1/workflow/coopt/actions/{action_id}")

    assert status_response.status_code == 200
    status = status_response.json()
    assert status["prewarm_status"] == "prewarm_completed"
    assert status["lease_status"] == "lease_unavailable"
    assert status["prepared_prefix_object_status"] == "lease_unavailable"
    assert status["lease_reason"] == "no_safe_internal_cache_lease_api"
    assert status["lease_full_block_count"] == 1
    assert status["prepared_prefix_ref_status"] == "lease_unavailable"
    assert status["lease_event_status"] == "lease_unavailable"
    assert status["lease_event_reason"] == "no_safe_internal_cache_lease_api"
    assert status["lease_id_present"] is False
    assert status["prefix_id_present"] is False
    assert "block_id" not in json.dumps(status)
    assert "kv" not in json.dumps(status).lower()


def test_workflow_actions_records_lease_consumed_update(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MODE", "experimental_prewarm")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_RETENTION_MODE", "lease")
    app = FastAPI()
    handler = _FakeChatHandler()
    app.state.openai_serving_chat = handler
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="lease-consumed"),
    )
    action_id = str(response.json()["action_id"])
    mark_workflow_prefix_prewarm_result(
        action_id=action_id,
        prewarm_status="prewarm_completed",
        lease_update={
            "lease_status": "leased",
            "lease_reason": "engine_core_cache_blocks_touched",
            "lease_token_count": 5,
            "lease_full_block_count": 1,
            "lease_ttl_ms": 30000,
            "prepared_prefix_ref_status": "leased",
            "lease_event_status": "leased",
            "lease_event_reason": "engine_core_cache_blocks_touched",
            "lease_id_present": True,
            "prefix_id_present": True,
        },
    )

    mark_workflow_prepared_prefix_lease_result(
        action_id=action_id,
        lease_update={
            "lease_status": "lease_consumed",
            "lease_reason": "lease_ref_count_released",
            "lease_token_count": 5,
            "lease_full_block_count": 1,
            "lease_ttl_ms": 29999,
            "prepared_prefix_ref_status": "lease_consumed",
            "lease_event_status": "lease_consumed",
            "lease_event_reason": "lease_ref_count_released",
            "lease_id_present": True,
            "prefix_id_present": True,
        },
    )
    status_response = client.get(f"/v1/workflow/coopt/actions/{action_id}")

    assert status_response.status_code == 200
    status = status_response.json()
    assert status["lease_status"] == "lease_unavailable"
    assert status["prepared_prefix_object_status"] == "lease_unavailable"
    assert status["lease_reason"] == "no_safe_internal_cache_lease_api"
    assert status["prepared_prefix_ref_status"] == "lease_unavailable"
    assert status["lease_event_status"] == "lease_unavailable"
    assert "block_id" not in json.dumps(status)
    assert "kv" not in json.dumps(status).lower()


def test_workflow_kv_manager_lease_mode_does_not_touch_or_free_cached_blocks() -> None:
    manager, request = _kv_manager_with_cached_prefix()
    cached_block = manager.block_pool.get_cached_block(
        request.block_hashes[0],
        [0],
    )[0]
    assert cached_block.ref_cnt == 0

    result = manager.try_lease_workflow_prepared_prefix(
        request,
        action_id="action-lease",
        prefix_token_count=4,
        ttl_ms=30000,
    )

    assert result == {
        "lease_status": "lease_unavailable",
        "lease_reason": "no_safe_internal_cache_lease_api",
        "lease_token_count": 4,
        "lease_full_block_count": 2,
        "lease_ttl_ms": 30000,
        "prepared_prefix_ref_status": "lease_unavailable",
        "lease_event_status": "lease_unavailable",
        "lease_event_reason": "no_safe_internal_cache_lease_api",
        "lease_id_present": False,
        "prefix_id_present": False,
    }
    assert cached_block.ref_cnt == 0
    assert "action-lease" not in manager._workflow_prepared_prefix_leases

    release = manager.release_workflow_prepared_prefix_lease("action-lease")

    assert release is None
    assert cached_block.ref_cnt == 0


def test_workflow_kv_manager_expired_lease_cleanup_noops_without_owned_blocks(
) -> None:
    manager, request = _kv_manager_with_cached_prefix()
    cached_block = manager.block_pool.get_cached_block(
        request.block_hashes[0],
        [0],
        [0],
    )[0]

    result = manager.try_lease_workflow_prepared_prefix(
        request,
        action_id="action-expire",
        prefix_token_count=4,
        ttl_ms=1,
    )
    assert result["lease_status"] == "lease_unavailable"
    assert cached_block.ref_cnt == 0

    time.sleep(0.01)
    released = manager.release_expired_workflow_prepared_prefix_leases()

    assert released == ()
    assert cached_block.ref_cnt == 0
    assert "action-expire" not in manager._workflow_prepared_prefix_leases


def test_workflow_kv_manager_lease_reports_redacted_miss_reason() -> None:
    manager, request = _kv_manager_with_cached_prefix(block_count=1)

    result = manager.try_lease_workflow_prepared_prefix(
        request,
        action_id="action-miss",
        prefix_token_count=4,
        ttl_ms=30000,
    )

    assert result == {
        "lease_status": "lease_unavailable",
        "lease_reason": "cache_blocks_missing",
        "lease_token_count": 4,
        "lease_full_block_count": 2,
        "lease_ttl_ms": 30000,
        "prepared_prefix_ref_status": "lease_unavailable",
        "lease_event_status": "lease_unavailable",
        "lease_event_reason": "cache_blocks_missing",
        "lease_id_present": False,
        "prefix_id_present": False,
    }
    assert "block_id" not in json.dumps(result)


def test_workflow_actions_experimental_prewarm_reports_unavailable_without_hook(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MIN_TOKENS", "1")
    monkeypatch.setenv("WORKFLOW_PREFIX_PREPARE_MODE", "experimental_prewarm")
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandlerWithoutPrewarm()
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="experimental-unavailable"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["lifecycle_status"] == "accepted"
    assert payload["prewarm_status"] == "prefill_only_unavailable"
    assert payload["prewarm_attempted"] is False


def test_workflow_prefill_only_ready_to_finish_uses_internal_extra_args() -> None:
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={
                "workflow_prefill_only": True,
                "workflow_prefix_prepare_action_id": "action-1",
            }
        ),
        num_computed_tokens=5,
        num_prompt_tokens=5,
    )
    assert _workflow_prefill_only_ready_to_finish(request) is True

    request.num_computed_tokens = 4
    assert _workflow_prefill_only_ready_to_finish(request) is False

    request.num_computed_tokens = 5
    request.sampling_params.extra_args["workflow_prefill_only"] = False
    assert _workflow_prefill_only_ready_to_finish(request) is False


def test_workflow_hook_treats_prefill_only_extra_args_as_internal(
    monkeypatch,
) -> None:
    _records.clear()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")

    record_scheduler_request(
        request_id="workflow-prefix-prewarm-action-1",
        vllm_xargs={
            "workflow_prefill_only": True,
            "workflow_prefix_prepare_action_id": "action-1",
            "workflow_prefix_prepare_token_count": 5,
            "workflow_prefix_prepare_token_hash": "sha1:prefix",
        },
        dp_rank=0,
        client_index=0,
        prompt_token_ids=[1, 2, 3, 4, 5],
        engine_prompt_token_count=5,
    )

    assert _records
    record = _records[-1]
    assert record.source == "scheduler"
    assert record.vllm_xargs is None
    assert record.workflow_sideband_valid is None
    assert record.action_id == "action-1"
    assert record.action_kind == "prefix_prepare"
    assert record.engine_prompt_token_count == 5
    assert record.prompt_token_ids_hash is not None


def test_workflow_hook_strips_internal_matched_prefix_extra_arg(
    monkeypatch,
) -> None:
    _records.clear()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")

    record_scheduler_request(
        request_id="request-1",
        vllm_xargs={
            "workflow_id": "wf-def",
            "workflow_instance_id": "wf",
            "graph_id": "graph",
            "block_id": "main",
            "op_id": "writer",
            "site_id": "site",
            "private_release_hint": "may",
            "effective_release_hint": "may",
            "release_hint": "may",
            "workflow_prepared_prefix_matched_action_id": "action-1",
        },
        dp_rank=0,
        client_index=0,
        prompt_token_ids=[1, 2, 3],
        engine_prompt_token_count=3,
    )

    (record,) = tuple(_records)
    assert record.vllm_xargs == {
        "workflow_id": "wf-def",
        "workflow_instance_id": "wf",
        "graph_id": "graph",
        "block_id": "main",
        "op_id": "writer",
        "site_id": "site",
        "private_release_hint": "may",
        "effective_release_hint": "may",
        "release_hint": "may",
    }
    assert record.workflow_sideband_valid is True
    assert "action-1" not in json.dumps(record.vllm_xargs)


def test_workflow_hook_records_prepared_prefix_cache_observation(
    monkeypatch,
) -> None:
    _records.clear()
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")

    record_chat_request(
        source="api_server_complete",
        path="/v1/chat/completions",
        request_id="chatcmpl-cache",
        vllm_xargs={
            "workflow_id": "wf",
            "graph_id": "main",
            "site_id": "main:writer",
            "workflow_instance_id": "wf-inst",
        },
        prompt_token_ids=[1, 2, 3, 4, 5],
        prepared_prefix_match={
            "prepared_prefix_match_status": "matched",
            "prepared_prefix_action_id": "action-1",
            "prepared_prefix_token_count": 4,
            "prepared_prefix_token_hash": "sha1:prefix",
            "prepared_prefix_lease_status": "lease_unavailable",
            "prepared_prefix_lease_match_status": "lease_unavailable",
        },
        prepared_prefix_cache={
            "prepared_prefix_cache_status": "prefix_partially_cached",
            "prepared_prefix_num_cached_tokens": 2,
            "prepared_prefix_recomputed_tokens": 2,
            "prepared_prefix_cached_at_least_prefix": False,
        },
    )

    assert _records
    record = _records[-1]
    assert record.source == "api_server_complete"
    assert record.prepared_prefix_match_status == "matched"
    assert record.prepared_prefix_lease_status == "lease_unavailable"
    assert record.prepared_prefix_lease_match_status == "lease_unavailable"
    assert record.prepared_prefix_cache_status == "prefix_partially_cached"
    assert record.prepared_prefix_num_cached_tokens == 2
    assert record.prepared_prefix_recomputed_tokens == 2
    assert record.prepared_prefix_cached_at_least_prefix is False


def test_workflow_actions_reject_when_tokenizer_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    app = FastAPI()
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/workflow/coopt/actions",
        json=_valid_prefix_prepare_action(key_suffix="no-tokenizer"),
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["reject_reason"] == "tokenizer_unavailable"


def test_workflow_actions_reject_expired_ttl(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandler()
    api_router.attach_router(app)
    client = TestClient(app)
    action = _valid_prefix_prepare_action(key_suffix="expired")
    action["expires_at_unix_ms"] = 1

    response = client.post("/v1/workflow/coopt/actions", json=action)

    assert response.status_code == 400
    payload = response.json()
    assert payload["lifecycle_status"] == "expired"
    assert payload["reject_reason"] == "expired_ttl"
