# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in workflow co-optimization action ingress.

This module is intentionally observability-first. It accepts action metadata
plus an in-memory prefix payload for `prefix_prepare`, validates it, asks the
engine chat stack to render/tokenize the prefix, and records redacted lifecycle
telemetry. It does not expose raw prompt content in responses or hook records,
and it does not mutate KV/APC internals yet.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.chat_completion.workflow_test_hook import (
    record_workflow_action,
)

_ENABLE_ENV = "VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS"
_MIN_TOKENS_ENV = "WORKFLOW_PREFIX_PREPARE_MIN_TOKENS"
_TTL_MS_ENV = "WORKFLOW_PREFIX_PREPARE_TTL_MS"
_MAX_OUTSTANDING_ENV = "WORKFLOW_PREFIX_PREPARE_MAX_OUTSTANDING"
_DEFAULT_MIN_TOKENS = 32
_DEFAULT_TTL_MS = 30000
_DEFAULT_MAX_OUTSTANDING = 128

router = APIRouter()


class WorkflowActionRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._actions_by_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._generation_by_key: dict[str, int] = {}

    async def submit(
        self,
        action: dict[str, Any],
        *,
        chat_handler: Any | None,
    ) -> dict[str, Any]:
        validation = _validate_prefix_prepare(action)
        if validation is not None:
            return validation

        action_id = str(action["action_id"])
        generation = int(action["generation"])
        idempotency_key = str(action["idempotency_key"])
        registry_key = str(action.get("virtual_request_id") or idempotency_key)
        with self._lock:
            previous_generation = self._generation_by_key.get(registry_key)
            if previous_generation is not None and generation <= previous_generation:
                return _lifecycle_response(
                    action,
                    accepted=False,
                    lifecycle_status="rejected",
                    reject_reason="duplicate_or_old_generation",
                    prewarm_status="not_attempted",
                )
            self._generation_by_key[registry_key] = generation
            self._actions_by_id[action_id] = {
                "action_id": action_id,
                "action_kind": action["action_kind"],
                "generation": generation,
                "created_monotonic_s": time.monotonic(),
                "lifecycle_status": "created",
            }
            self._actions_by_id.move_to_end(action_id)
            while len(self._actions_by_id) > _max_outstanding():
                self._actions_by_id.popitem(last=False)

        token_verification = await _verify_prefix_tokens(action, chat_handler)
        if token_verification.get("ok") is not True:
            response = _lifecycle_response(
                action,
                accepted=False,
                lifecycle_status="rejected",
                reject_reason=str(
                    token_verification.get("reject_reason")
                    or "prefix_tokenization_failed"
                ),
                prewarm_status="not_attempted",
                token_verification=token_verification,
            )
            self._store_response(action_id, response)
            return response

        prefix_token_count = _optional_int(token_verification.get("prefix_token_count"))
        prefix_token_hash = _optional_str(token_verification.get("prefix_token_hash"))
        if prefix_token_count is not None and prefix_token_count < _min_tokens():
            response = _lifecycle_response(
                action,
                accepted=True,
                lifecycle_status="ignored",
                reject_reason=None,
                prewarm_status="ignored:prefix_too_short",
                prefix_token_count=prefix_token_count,
                prefix_token_hash=prefix_token_hash,
                token_verification=token_verification,
            )
            self._store_response(action_id, response)
            return response

        response = _lifecycle_response(
            action,
            accepted=True,
            lifecycle_status="accepted",
            reject_reason=None,
            prewarm_status="prewarm_not_implemented",
            prefix_token_count=prefix_token_count,
            prefix_token_hash=prefix_token_hash,
            token_verification=token_verification,
        )
        self._store_response(action_id, response)
        return response

    def get(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            action = self._actions_by_id.get(action_id)
            return dict(action) if action is not None else None

    def _store_response(self, action_id: str, response: dict[str, Any]) -> None:
        with self._lock:
            stored = self._actions_by_id.get(action_id)
            if stored is not None:
                stored.update(response)


_registry = WorkflowActionRegistry()


def workflow_coopt_actions_enabled() -> bool:
    return os.getenv(_ENABLE_ENV, "") == "1"


@router.post("/v1/workflow/coopt/actions")
async def submit_workflow_coopt_action(raw_request: Request) -> JSONResponse:
    if not workflow_coopt_actions_enabled():
        response = {
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "workflow_coopt_actions_disabled",
        }
        return JSONResponse(content=response, status_code=404)
    payload = await raw_request.json()
    if not isinstance(payload, dict):
        response = {
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "expected_object",
        }
        return JSONResponse(content=response, status_code=400)
    chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
    response = await _registry.submit(payload, chat_handler=chat_handler)
    status_code = 200 if response.get("accepted") is True else 400
    return JSONResponse(content=response, status_code=status_code)


@router.get("/v1/workflow/coopt/actions/{action_id}")
async def get_workflow_coopt_action(action_id: str) -> JSONResponse:
    if not workflow_coopt_actions_enabled():
        response = {
            "accepted": False,
            "lifecycle_status": "rejected",
            "reject_reason": "workflow_coopt_actions_disabled",
        }
        return JSONResponse(content=response, status_code=404)
    response = _registry.get(action_id)
    if response is None:
        return JSONResponse(
            content={
                "action_id": action_id,
                "lifecycle_status": "missing",
                "reject_reason": "action_not_found",
            },
            status_code=404,
        )
    return JSONResponse(content=response)


def _validate_prefix_prepare(action: dict[str, Any]) -> dict[str, Any] | None:
    if action.get("action_kind") != "prefix_prepare":
        return _lifecycle_response(
            action,
            accepted=False,
            lifecycle_status="rejected",
            reject_reason="unsupported_action_kind",
            prewarm_status="not_attempted",
        )
    required = (
        "action_id",
        "idempotency_key",
        "generation",
        "virtual_request_id",
        "model",
        "site_id",
        "render_domain_id",
        "token_domain_id",
        "prefix_part_count",
        "messages_prefix",
    )
    for field in required:
        if field not in action:
            return _lifecycle_response(
                action,
                accepted=False,
                lifecycle_status="rejected",
                reject_reason=f"missing_{field}",
                prewarm_status="not_attempted",
            )
    if not isinstance(action.get("generation"), int) or action["generation"] < 0:
        return _lifecycle_response(
            action,
            accepted=False,
            lifecycle_status="rejected",
            reject_reason="invalid_generation",
            prewarm_status="not_attempted",
        )
    model = action.get("model")
    if not isinstance(model, str) or not model:
        return _lifecycle_response(
            action,
            accepted=False,
            lifecycle_status="rejected",
            reject_reason="missing_model",
            prewarm_status="not_attempted",
        )
    messages_prefix = action.get("messages_prefix")
    if not isinstance(messages_prefix, list) or not messages_prefix:
        return _lifecycle_response(
            action,
            accepted=False,
            lifecycle_status="rejected",
            reject_reason="missing_prefix_payload",
            prewarm_status="not_attempted",
        )
    ttl_ms = action.get("ttl_ms", _default_ttl_ms())
    if not isinstance(ttl_ms, int) or ttl_ms <= 0:
        return _lifecycle_response(
            action,
            accepted=False,
            lifecycle_status="rejected",
            reject_reason="invalid_ttl_ms",
            prewarm_status="not_attempted",
        )
    expires_at_unix_ms = action.get("expires_at_unix_ms")
    if isinstance(expires_at_unix_ms, int) and expires_at_unix_ms < _now_unix_ms():
        return _lifecycle_response(
            action,
            accepted=False,
            lifecycle_status="expired",
            reject_reason="expired_ttl",
            prewarm_status="not_attempted",
        )
    return None


def _lifecycle_response(
    action: dict[str, Any],
    *,
    accepted: bool,
    lifecycle_status: str,
    reject_reason: str | None,
    prewarm_status: str,
    prefix_token_count: int | None = None,
    prefix_token_hash: str | None = None,
    token_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prefix_message_hash = _hash_prefix_messages(action.get("messages_prefix"))
    token_verification = token_verification or {}
    response = {
        "action_id": action.get("action_id"),
        "action_kind": action.get("action_kind"),
        "accepted": accepted,
        "lifecycle_status": lifecycle_status,
        "reject_reason": reject_reason,
        "prewarm_status": prewarm_status,
        "prewarm_attempted": prewarm_status
        not in {
            "not_attempted",
            "ignored:prefix_too_short",
            "prewarm_not_implemented",
        },
        "engine_token_source": token_verification.get("engine_token_source"),
        "prefix_token_count": prefix_token_count,
        "prefix_token_hash": prefix_token_hash,
        "prefix_message_hash": prefix_message_hash,
        "prefix_hash_present": prefix_token_hash is not None,
        "model": token_verification.get("model") or action.get("model"),
        "served_model_name": token_verification.get("served_model_name"),
        "tokenizer_id": token_verification.get("tokenizer_id"),
        "chat_template_id": token_verification.get("chat_template_id"),
    }
    action_id = _optional_str(response.get("action_id"))
    action_kind = _optional_str(response.get("action_kind"))
    record_workflow_action(
        action_id=action_id,
        action_kind=action_kind,
        lifecycle_status=lifecycle_status,
        reject_reason=reject_reason,
        prewarm_status=prewarm_status,
        prewarm_attempted=bool(response["prewarm_attempted"]),
        prefix_token_count=prefix_token_count,
        prefix_token_hash=prefix_token_hash,
        engine_token_source=_optional_str(response.get("engine_token_source")),
        model=_optional_str(response.get("model")),
        served_model_name=_optional_str(response.get("served_model_name")),
        tokenizer_id=_optional_str(response.get("tokenizer_id")),
        chat_template_id=_optional_str(response.get("chat_template_id")),
    )
    return response


async def _verify_prefix_tokens(
    action: dict[str, Any],
    chat_handler: Any | None,
) -> dict[str, Any]:
    if chat_handler is None or not hasattr(
        chat_handler, "verify_workflow_prefix_prepare"
    ):
        return {
            "ok": False,
            "reject_reason": "tokenizer_unavailable",
        }
    try:
        result = await chat_handler.verify_workflow_prefix_prepare(action)
    except Exception:  # noqa: BLE001
        return {
            "ok": False,
            "reject_reason": "prefix_tokenization_failed",
        }
    if not isinstance(result, dict):
        return {
            "ok": False,
            "reject_reason": "prefix_tokenization_failed",
        }
    if result.get("ok") is not True:
        return {
            "ok": False,
            "reject_reason": result.get("reject_reason")
            if isinstance(result.get("reject_reason"), str)
            else "prefix_tokenization_failed",
        }
    token_ids = result.get("prefix_token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        return {
            "ok": False,
            "reject_reason": "prefix_tokenization_failed",
        }
    verified = dict(result)
    verified["engine_token_source"] = "engine_authoritative"
    verified["prefix_token_count"] = len(token_ids)
    verified["prefix_token_hash"] = _hash_token_ids(token_ids)
    verified.pop("prefix_token_ids", None)
    return verified


def _hash_prefix_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    payload = json.dumps(
        messages,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha1:" + hashlib.sha1(payload).hexdigest()


def _hash_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return "sha1:" + hashlib.sha1(payload).hexdigest()


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _now_unix_ms() -> int:
    return int(time.time() * 1000)


def _min_tokens() -> int:
    return _env_int(_MIN_TOKENS_ENV, _DEFAULT_MIN_TOKENS)


def _default_ttl_ms() -> int:
    return _env_int(_TTL_MS_ENV, _DEFAULT_TTL_MS)


def _max_outstanding() -> int:
    return _env_int(_MAX_OUTSTANDING_ENV, _DEFAULT_MAX_OUTSTANDING)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
