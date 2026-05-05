# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in workflow co-optimization action ingress.

This module is intentionally observability-first.  It accepts redacted/action
metadata plus an in-memory prefix payload for `prefix_prepare`, validates it,
and records lifecycle telemetry.  It does not expose raw prompt content in
responses or hook records, and it does not mutate KV/APC internals yet.
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

    def submit(self, action: dict[str, Any]) -> dict[str, Any]:
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
            }
            self._actions_by_id.move_to_end(action_id)
            while len(self._actions_by_id) > _max_outstanding():
                self._actions_by_id.popitem(last=False)

        advisory_count = action.get("prefix_token_count_advisory")
        if isinstance(advisory_count, int) and advisory_count < _min_tokens():
            return _lifecycle_response(
                action,
                accepted=True,
                lifecycle_status="ignored",
                reject_reason=None,
                prewarm_status="ignored:prefix_too_short",
                prefix_token_count=advisory_count,
            )
        return _lifecycle_response(
            action,
            accepted=True,
            lifecycle_status="accepted",
            reject_reason=None,
            prewarm_status="prewarm_not_implemented",
        )


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
    response = _registry.submit(payload)
    status_code = 200 if response.get("accepted") is True else 400
    return JSONResponse(content=response, status_code=status_code)


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
    return None


def _lifecycle_response(
    action: dict[str, Any],
    *,
    accepted: bool,
    lifecycle_status: str,
    reject_reason: str | None,
    prewarm_status: str,
    prefix_token_count: int | None = None,
) -> dict[str, Any]:
    prefix_hash = _hash_prefix_messages(action.get("messages_prefix"))
    response = {
        "action_id": action.get("action_id"),
        "action_kind": action.get("action_kind"),
        "accepted": accepted,
        "lifecycle_status": lifecycle_status,
        "reject_reason": reject_reason,
        "prewarm_status": prewarm_status,
        "prewarm_attempted": prewarm_status
        not in {"not_attempted", "ignored:prefix_too_short"},
        "prefix_token_count": prefix_token_count,
        "prefix_token_hash": prefix_hash,
        "prefix_hash_present": prefix_hash is not None,
    }
    action_id = (
        response["action_id"] if isinstance(response["action_id"], str) else None
    )
    action_kind = (
        response["action_kind"] if isinstance(response["action_kind"], str) else None
    )
    record_workflow_action(
        action_id=action_id,
        action_kind=action_kind,
        lifecycle_status=lifecycle_status,
        reject_reason=reject_reason,
        prewarm_status=prewarm_status,
        prewarm_attempted=bool(response["prewarm_attempted"]),
        prefix_token_count=prefix_token_count,
        prefix_token_hash=prefix_hash,
    )
    return response


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
