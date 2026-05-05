# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in workflow co-optimization action ingress.

This module is intentionally observability-first. It accepts action metadata
plus an in-memory prefix payload for `prefix_prepare`, validates it, asks the
engine chat stack to render/tokenize the prefix, and records redacted lifecycle
telemetry. In `experimental_prewarm` mode it may ask the engine to submit a
hidden internal prefill-only request. It does not expose raw prompt content in
responses or hook records, and it never exposes KV handles.
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
_PREPARE_MODE_ENV = "WORKFLOW_PREFIX_PREPARE_MODE"
_RETENTION_MODE_ENV = "WORKFLOW_PREFIX_PREPARE_RETENTION_MODE"
_DEFAULT_MIN_TOKENS = 32
_DEFAULT_TTL_MS = 30000
_DEFAULT_MAX_OUTSTANDING = 128
_DEFAULT_PREPARE_MODE = "tokenize_only"
_DEFAULT_RETENTION_MODE = "observe"

router = APIRouter()


class WorkflowActionRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._actions_by_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._prepared_prefixes_by_action_id: OrderedDict[str, dict[str, Any]] = (
            OrderedDict()
        )
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
            self._evict_locked()

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

        self._store_prepared_prefix(
            action,
            token_verification,
            prefix_token_count=prefix_token_count,
            prefix_token_hash=prefix_token_hash,
        )
        prewarm_status = await _maybe_submit_prefix_prewarm(
            action,
            token_verification,
            chat_handler,
        )
        response = _lifecycle_response(
            action,
            accepted=True,
            lifecycle_status="accepted",
            reject_reason=None,
            prewarm_status=prewarm_status,
            prefix_token_count=prefix_token_count,
            prefix_token_hash=prefix_token_hash,
            token_verification=token_verification,
        )
        self._store_response(action_id, response)
        return response

    def get(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            action = self._actions_by_id.get(action_id)
            if action is None:
                return None
            response = dict(action)
            if _optional_str(response.get("lease_status")) == "leased":
                expires_at = response.get("expires_at_unix_ms")
                if isinstance(expires_at, int) and expires_at < _now_unix_ms():
                    response["lease_status"] = "lease_expired"
                    response["lease_reason"] = "ttl_expired"
            return response

    def _store_response(self, action_id: str, response: dict[str, Any]) -> None:
        with self._lock:
            stored = self._actions_by_id.get(action_id)
            if stored is not None:
                stored.update(response)

    def mark_prewarm_result(
        self,
        *,
        action_id: str,
        prewarm_status: str,
        reject_reason: str | None = None,
        lease_update: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            stored = self._actions_by_id.get(action_id)
            if stored is None:
                return None
            stored["prewarm_status"] = prewarm_status
            stored["prewarm_attempted"] = prewarm_status not in {
                "not_attempted",
                "ignored:prefix_too_short",
                "tokenize_only",
                "prefill_only_unavailable",
            }
            if reject_reason is not None:
                stored["reject_reason"] = reject_reason
            lease_update = _lease_update_for_prewarm_result(
                prewarm_status=prewarm_status,
                prefix_token_count=_optional_int(stored.get("prefix_token_count")),
                ttl_ms=_optional_int(stored.get("ttl_ms")),
                engine_lease_update=lease_update,
            )
            stored.update(lease_update)
            prepared = self._prepared_prefixes_by_action_id.get(action_id)
            if prepared is not None:
                prepared.update(lease_update)
            response = dict(stored)
        record_workflow_action(
            action_id=_optional_str(response.get("action_id")),
            action_kind=_optional_str(response.get("action_kind")),
            lifecycle_status=_optional_str(response.get("lifecycle_status"))
            or "accepted",
            reject_reason=_optional_str(response.get("reject_reason")),
            prewarm_status=prewarm_status,
            prewarm_attempted=bool(response.get("prewarm_attempted")),
            lease_status=_optional_str(response.get("lease_status")),
            lease_reason=_optional_str(response.get("lease_reason")),
            lease_token_count=_optional_int(response.get("lease_token_count")),
            lease_full_block_count=_optional_int(
                response.get("lease_full_block_count")
            ),
            lease_ttl_ms=_optional_int(response.get("lease_ttl_ms")),
            prefix_token_count=_optional_int(response.get("prefix_token_count")),
            prefix_token_hash=_optional_str(response.get("prefix_token_hash")),
            engine_token_source=_optional_str(response.get("engine_token_source")),
            model=_optional_str(response.get("model")),
            served_model_name=_optional_str(response.get("served_model_name")),
            tokenizer_id=_optional_str(response.get("tokenizer_id")),
            chat_template_id=_optional_str(response.get("chat_template_id")),
        )
        return response

    def _store_prepared_prefix(
        self,
        action: dict[str, Any],
        token_verification: dict[str, Any],
        *,
        prefix_token_count: int | None,
        prefix_token_hash: str | None,
    ) -> None:
        token_ids = token_verification.get("_prefix_token_ids")
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
            return
        action_id = str(action["action_id"])
        prepared = {
            "action_id": action_id,
            "virtual_request_id": str(action.get("virtual_request_id") or ""),
            "workflow_instance_id": _optional_str(action.get("workflow_instance_id")),
            "site_id": _optional_str(action.get("site_id")),
            "model": _optional_str(action.get("model")),
            "render_domain_id": _optional_str(action.get("render_domain_id")),
            "token_domain_id": _optional_str(action.get("token_domain_id")),
            "generation": int(action.get("generation") or 0),
            "created_monotonic_s": time.monotonic(),
            "expires_at_unix_ms": _expires_at_unix_ms(action),
            "prefix_token_count": prefix_token_count,
            "prefix_token_hash": prefix_token_hash,
            **_initial_lease_fields(
                prewarm_status=None,
                prefix_token_count=prefix_token_count,
                ttl_ms=_action_ttl_ms(action),
            ),
            "_prefix_token_ids": tuple(token_ids),
        }
        with self._lock:
            self._prepared_prefixes_by_action_id[action_id] = prepared
            self._prepared_prefixes_by_action_id.move_to_end(action_id)
            stored = self._actions_by_id.get(action_id)
            if stored is not None:
                stored.update(
                    {
                        "prepared_prefix_registered": True,
                        "prepared_prefix_match_status": "pending_request",
                        **{
                            key: prepared[key]
                            for key in (
                                "lease_status",
                                "lease_reason",
                                "lease_token_count",
                                "lease_full_block_count",
                                "lease_ttl_ms",
                            )
                            if key in prepared
                        },
                    }
                )
            self._evict_locked()

    def match_prepared_prefix(
        self,
        *,
        vllm_xargs: dict[str, Any] | None,
        prompt_token_ids: list[int] | None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        if not workflow_coopt_actions_enabled():
            return None
        if not isinstance(vllm_xargs, dict):
            return None
        if prompt_token_ids is None:
            return {
                "prepared_prefix_match_status": "unavailable",
                "prepared_prefix_mismatch_reason": "prompt_token_ids_unavailable",
            }
        workflow_instance_id = _optional_str(vllm_xargs.get("workflow_instance_id"))
        site_id = _optional_str(vllm_xargs.get("site_id"))
        if workflow_instance_id is None or site_id is None:
            return {
                "prepared_prefix_match_status": "unavailable",
                "prepared_prefix_mismatch_reason": "missing_workflow_site_identity",
            }
        now_ms = _now_unix_ms()
        best_match: dict[str, Any] | None = None
        best_mismatch: dict[str, Any] | None = None
        with self._lock:
            candidates = tuple(self._prepared_prefixes_by_action_id.values())
        for candidate in candidates:
            if candidate.get("workflow_instance_id") != workflow_instance_id:
                continue
            if candidate.get("site_id") != site_id:
                continue
            expires_at = candidate.get("expires_at_unix_ms")
            if isinstance(expires_at, int) and expires_at < now_ms:
                best_mismatch = {
                    "prepared_prefix_match_status": "mismatch",
                    "prepared_prefix_action_id": candidate.get("action_id"),
                    "prepared_prefix_mismatch_reason": "prepared_prefix_expired",
                }
                continue
            candidate_model = candidate.get("model")
            if isinstance(candidate_model, str) and model and candidate_model != model:
                best_mismatch = {
                    "prepared_prefix_match_status": "mismatch",
                    "prepared_prefix_action_id": candidate.get("action_id"),
                    "prepared_prefix_mismatch_reason": "model_mismatch",
                }
                continue
            token_ids = candidate.get("_prefix_token_ids")
            if not isinstance(token_ids, tuple) or not all(
                isinstance(token_id, int) for token_id in token_ids
            ):
                continue
            prefix_len = len(token_ids)
            candidate_match = {
                "prepared_prefix_action_id": candidate.get("action_id"),
                "prepared_prefix_token_count": candidate.get("prefix_token_count"),
                "prepared_prefix_token_hash": candidate.get("prefix_token_hash"),
                "prepared_prefix_lease_status": candidate.get("lease_status"),
                "prepared_prefix_lease_match_status": _lease_match_status(candidate),
            }
            if tuple(prompt_token_ids[:prefix_len]) == token_ids:
                if (
                    best_match is None
                    or int(candidate.get("prefix_token_count") or 0)
                    > int(best_match.get("prepared_prefix_token_count") or 0)
                ):
                    best_match = {
                        **candidate_match,
                        "prepared_prefix_match_status": "matched",
                    }
                continue
            best_mismatch = {
                **candidate_match,
                "prepared_prefix_match_status": "mismatch",
                "prepared_prefix_mismatch_reason": "token_prefix_mismatch",
            }
        if best_match is not None:
            self._mark_prepared_prefix_observed(best_match)
            return best_match
        if best_mismatch is not None:
            return best_mismatch
        return {
            "prepared_prefix_match_status": "missing",
            "prepared_prefix_mismatch_reason": "no_prepared_prefix_for_site",
        }

    def _mark_prepared_prefix_observed(self, match: dict[str, Any]) -> None:
        action_id = _optional_str(match.get("prepared_prefix_action_id"))
        if action_id is None:
            return
        with self._lock:
            stored = self._actions_by_id.get(action_id)
            if stored is not None:
                stored["prepared_prefix_match_status"] = "matched"

    def _evict_locked(self) -> None:
        while len(self._actions_by_id) > _max_outstanding():
            action_id, _stored = self._actions_by_id.popitem(last=False)
            self._prepared_prefixes_by_action_id.pop(action_id, None)
        while len(self._prepared_prefixes_by_action_id) > _max_outstanding():
            self._prepared_prefixes_by_action_id.popitem(last=False)


_registry = WorkflowActionRegistry()


def workflow_coopt_actions_enabled() -> bool:
    return os.getenv(_ENABLE_ENV, "") == "1"


def match_prepared_prefix_for_request(
    *,
    vllm_xargs: dict[str, Any] | None,
    prompt_token_ids: list[int] | None,
    model: str | None = None,
) -> dict[str, Any] | None:
    return _registry.match_prepared_prefix(
        vllm_xargs=vllm_xargs,
        prompt_token_ids=prompt_token_ids,
        model=model,
    )


def mark_workflow_prefix_prewarm_result(
    *,
    action_id: str,
    prewarm_status: str,
    reject_reason: str | None = None,
    lease_update: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return _registry.mark_prewarm_result(
        action_id=action_id,
        prewarm_status=prewarm_status,
        reject_reason=reject_reason,
        lease_update=lease_update,
    )


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
    lease_fields = _initial_lease_fields(
        prewarm_status=prewarm_status,
        prefix_token_count=prefix_token_count,
        ttl_ms=_action_ttl_ms(action),
    )
    response = {
        "action_id": action.get("action_id"),
        "action_kind": action.get("action_kind"),
        "ttl_ms": _action_ttl_ms(action),
        "accepted": accepted,
        "lifecycle_status": lifecycle_status,
        "reject_reason": reject_reason,
        "prewarm_status": prewarm_status,
        "prewarm_attempted": prewarm_status
        not in {
            "not_attempted",
            "ignored:prefix_too_short",
            "prewarm_not_implemented",
            "tokenize_only",
            "prefill_only_unavailable",
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
        "prepared_prefix_registered": (
            lifecycle_status == "accepted" and prefix_token_hash is not None
        ),
        "prepared_prefix_match_status": (
            "pending_request"
            if lifecycle_status == "accepted" and prefix_token_hash is not None
            else None
        ),
        **lease_fields,
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
        lease_status=_optional_str(response.get("lease_status")),
        lease_reason=_optional_str(response.get("lease_reason")),
        lease_token_count=_optional_int(response.get("lease_token_count")),
        lease_full_block_count=_optional_int(response.get("lease_full_block_count")),
        lease_ttl_ms=_optional_int(response.get("lease_ttl_ms")),
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
    verified["_prefix_token_ids"] = list(token_ids)
    verified["engine_token_source"] = "engine_authoritative"
    verified["prefix_token_count"] = len(token_ids)
    verified["prefix_token_hash"] = _hash_token_ids(token_ids)
    verified.pop("prefix_token_ids", None)
    return verified


async def _maybe_submit_prefix_prewarm(
    action: dict[str, Any],
    token_verification: dict[str, Any],
    chat_handler: Any | None,
) -> str:
    mode = _prefix_prepare_mode()
    if mode == "tokenize_only":
        return "tokenize_only"
    if mode != "experimental_prewarm":
        return "tokenize_only"
    submit_prewarm = (
        getattr(chat_handler, "submit_workflow_prefix_prewarm", None)
        if chat_handler is not None
        else None
    )
    if not callable(submit_prewarm):
        return "prefill_only_unavailable"
    try:
        result = await submit_prewarm(
            action,
            token_verification,
        )
    except Exception:  # noqa: BLE001
        return "prewarm_failed:submit_exception"
    if not isinstance(result, dict):
        return "prewarm_failed:invalid_submit_response"
    prewarm_status = result.get("prewarm_status")
    return prewarm_status if isinstance(prewarm_status, str) else "prewarm_submitted"


def _expires_at_unix_ms(action: dict[str, Any]) -> int:
    expires_at_unix_ms = action.get("expires_at_unix_ms")
    if isinstance(expires_at_unix_ms, int):
        return expires_at_unix_ms
    ttl_ms = action.get("ttl_ms", _default_ttl_ms())
    if not isinstance(ttl_ms, int) or ttl_ms <= 0:
        ttl_ms = _default_ttl_ms()
    return _now_unix_ms() + ttl_ms


def _action_ttl_ms(action: dict[str, Any]) -> int:
    ttl_ms = action.get("ttl_ms", _default_ttl_ms())
    return ttl_ms if isinstance(ttl_ms, int) and ttl_ms > 0 else _default_ttl_ms()


def _initial_lease_fields(
    *,
    prewarm_status: str | None,
    prefix_token_count: int | None,
    ttl_ms: int | None,
) -> dict[str, Any]:
    mode = _prefix_retention_mode()
    if mode == "observe":
        return {
            "lease_status": "observe_only",
            "lease_reason": "retention_mode_observe",
            "lease_token_count": prefix_token_count,
            "lease_full_block_count": None,
            "lease_ttl_ms": ttl_ms,
        }
    if prewarm_status == "prewarm_completed":
        return _lease_update_for_prewarm_result(
            prewarm_status=prewarm_status,
            prefix_token_count=prefix_token_count,
            ttl_ms=ttl_ms,
        )
    if prewarm_status == "prewarm_submitted":
        return {
            "lease_status": "pending_prewarm",
            "lease_reason": "waiting_for_prewarm_completion",
            "lease_token_count": prefix_token_count,
            "lease_full_block_count": None,
            "lease_ttl_ms": ttl_ms,
        }
    return {
        "lease_status": "not_attempted",
        "lease_reason": "prewarm_not_completed",
        "lease_token_count": prefix_token_count,
        "lease_full_block_count": None,
        "lease_ttl_ms": ttl_ms,
    }


def _lease_update_for_prewarm_result(
    *,
    prewarm_status: str,
    prefix_token_count: int | None,
    ttl_ms: int | None,
    engine_lease_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = _prefix_retention_mode()
    if mode == "observe":
        return {
            "lease_status": "observe_only",
            "lease_reason": "retention_mode_observe",
            "lease_token_count": prefix_token_count,
            "lease_full_block_count": None,
            "lease_ttl_ms": ttl_ms,
        }
    if prewarm_status == "prewarm_completed":
        if isinstance(engine_lease_update, dict):
            return {
                "lease_status": _optional_str(
                    engine_lease_update.get("lease_status")
                )
                or "lease_failed",
                "lease_reason": _optional_str(
                    engine_lease_update.get("lease_reason")
                )
                or "missing_engine_lease_reason",
                "lease_token_count": _optional_int(
                    engine_lease_update.get("lease_token_count")
                )
                or prefix_token_count,
                "lease_full_block_count": _optional_int(
                    engine_lease_update.get("lease_full_block_count")
                ),
                "lease_ttl_ms": _optional_int(engine_lease_update.get("lease_ttl_ms"))
                or ttl_ms,
            }
        return {
            "lease_status": "lease_unavailable",
            "lease_reason": "no_safe_internal_cache_lease_api",
            "lease_token_count": prefix_token_count,
            "lease_full_block_count": None,
            "lease_ttl_ms": ttl_ms,
        }
    return {
        "lease_status": "not_attempted",
        "lease_reason": prewarm_status,
        "lease_token_count": prefix_token_count,
        "lease_full_block_count": None,
        "lease_ttl_ms": ttl_ms,
    }


def _lease_match_status(candidate: dict[str, Any]) -> str:
    status = _optional_str(candidate.get("lease_status"))
    if status == "leased":
        expires_at = candidate.get("expires_at_unix_ms")
        if isinstance(expires_at, int) and expires_at < _now_unix_ms():
            return "lease_expired"
        return "matched_with_active_lease"
    if status in {
        "lease_unavailable",
        "lease_failed",
        "lease_expired",
        "lease_released",
        "observe_only",
        "pending_prewarm",
        "not_attempted",
    }:
        return status
    return "unknown"


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


def _prefix_prepare_mode() -> str:
    mode = os.getenv(_PREPARE_MODE_ENV, _DEFAULT_PREPARE_MODE)
    if mode == "tokenize_only":
        return mode
    if mode == "experimental_prewarm":
        return mode
    return _DEFAULT_PREPARE_MODE


def _prefix_retention_mode() -> str:
    mode = os.getenv(_RETENTION_MODE_ENV, _DEFAULT_RETENTION_MODE)
    if mode == "observe":
        return mode
    if mode == "lease":
        return mode
    return _DEFAULT_RETENTION_MODE


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
