# SPDX-License-Identifier: Apache-2.0

"""Smoke test for /v1/coopt/segment_prepare and /v1/coopt/segment_refresh.

Mirrors the prefix_prepare smoke test pattern in
``test_workflow_actions.py``: spins up a FastAPI app via
``api_router.attach_router``, posts wire-form actions, and verifies
the response shape + redaction contract.

Internal prefill is mocked via the ``_FakeChatHandler.submit_workflow_segment_prewarm``
hook; this lets us exercise the full registry / telemetry path without
touching the engine's real prefill scheduler (Phase E2 territory).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.chat_completion import api_router
from vllm.entrypoints.openai.chat_completion.segment_actions import (
    SEGMENT_PREPARE_ACTION_KIND,
    SEGMENT_REFRESH_ACTION_KIND,
    reset_segment_registry_for_tests,
)
from vllm.entrypoints.openai.chat_completion.workflow_test_hook import _records


_TOKEN_HASH = "a" * 64
_TOKEN_IDS = [101, 202, 303, 404, 505, 606, 707, 808]


class _FakeChatHandler:
    """Stub chat handler that records segment-prewarm submissions.

    Both ``submit_workflow_segment_prewarm`` and
    ``submit_workflow_segment_refresh`` return ``prewarm_submitted`` so
    the registry transitions to ``prefilled`` / ``retouched`` and the
    smoke test can assert the engine-side state machine wired through.
    """

    def __init__(self) -> None:
        self.prepare_submissions: list[dict[str, object]] = []
        self.refresh_submissions: list[dict[str, object]] = []

    async def submit_workflow_segment_prewarm(self, action):
        self.prepare_submissions.append(dict(action))
        return {
            "prewarm_status": "prewarm_submitted",
            "prefill_token_count": len(action.get("prompt_token_ids") or []),
        }

    async def submit_workflow_segment_refresh(self, action):
        self.refresh_submissions.append(dict(action))
        return {
            "prewarm_status": "prewarm_submitted",
            "prefill_token_count": len(action.get("prompt_token_ids") or []),
        }


def _segment_prepare_action(*, action_id: str = "segp:test:1") -> dict[str, object]:
    return {
        "action_kind": SEGMENT_PREPARE_ACTION_KIND,
        "action_id": action_id,
        "segment_id": "seg-abc",
        "family_id": "fam-1",
        "parent_segment_id": None,
        "token_hash": _TOKEN_HASH,
        "prompt_token_ids": list(_TOKEN_IDS),
        "model": "test-model",
        "expected_consumers": 3,
        "lifecycle_status": "created",
        "ttl_ms": 30000,
    }


def _segment_refresh_action(*, action_id: str = "segr:test:1") -> dict[str, object]:
    return {
        "action_kind": SEGMENT_REFRESH_ACTION_KIND,
        "action_id": action_id,
        "segment_id": "seg-abc",
        "family_id": "fam-1",
        "parent_segment_id": None,
        "token_hash": _TOKEN_HASH,
        "prompt_token_ids": list(_TOKEN_IDS),
        "model": "test-model",
        "lifecycle_status": "created",
        "would_evict_without_refresh": True,
        "ttl_ms": 30000,
        "dummy_token_id": 0,
    }


def _make_app(monkeypatch, tmp_path) -> tuple[TestClient, _FakeChatHandler]:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK", "1")
    monkeypatch.setenv("WORKFLOW_TEST_HOOK_FILE", str(tmp_path / "hook.jsonl"))
    monkeypatch.setenv("WORKFLOW_SEGMENT_PREPARE_MODE", "experimental_prewarm")
    reset_segment_registry_for_tests()
    _records.clear()
    app = FastAPI()
    handler = _FakeChatHandler()
    app.state.openai_serving_chat = handler
    api_router.attach_router(app)
    return TestClient(app), handler


def test_segment_routes_are_only_attached_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", raising=False)
    app = FastAPI()
    api_router.attach_router(app)
    paths = {route.path for route in app.routes}
    assert "/v1/coopt/segment_prepare" not in paths
    assert "/v1/coopt/segment_refresh" not in paths

    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    app = FastAPI()
    api_router.attach_router(app)
    paths = {route.path for route in app.routes}
    assert "/v1/coopt/segment_prepare" in paths
    assert "/v1/coopt/segment_refresh" in paths
    assert "/v1/coopt/segment_prepare/{action_id}" in paths
    assert "/v1/coopt/segment_refresh/{action_id}" in paths


def test_segment_prepare_accepts_wire_action(monkeypatch, tmp_path) -> None:
    client, handler = _make_app(monkeypatch, tmp_path)

    response = client.post(
        "/v1/coopt/segment_prepare",
        json=_segment_prepare_action(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["lifecycle_status"] in {"created", "accepted"}
    assert payload["engine_wire_status"] == "accepted"
    assert payload["action_kind"] == SEGMENT_PREPARE_ACTION_KIND
    assert payload["segment_id"] == "seg-abc"
    assert payload["family_id"] == "fam-1"
    assert payload["expected_consumers"] == 3
    assert payload["observed_consumers"] == 0
    assert payload["status"] == "prefilled"
    assert payload["prefill_status"] == "prewarm_submitted"
    assert payload["prefill_token_count"] == len(_TOKEN_IDS)
    assert payload["engine_seen_token_hash"] == _TOKEN_HASH
    assert payload["ttl_ms"] >= 1
    assert "prompt_token_ids" not in payload  # redaction contract

    # Engine submitted the prewarm with the wire payload intact.
    assert len(handler.prepare_submissions) == 1
    assert handler.prepare_submissions[0]["prompt_token_ids"] == list(_TOKEN_IDS)

    # Telemetry recorded a workflow_segment_action event with no raw prompt.
    assert _records, "expected at least one workflow_segment_action record"
    record = next(r for r in _records if r.source == "workflow_segment_action")
    assert record.action_kind == SEGMENT_PREPARE_ACTION_KIND
    assert record.segment_id == "seg-abc"
    assert record.segment_family_id == "fam-1"
    assert record.segment_engine_seen_token_hash == _TOKEN_HASH
    assert record.segment_prefill_status == "prewarm_submitted"
    assert record.segment_prefill_token_count == len(_TOKEN_IDS)

    hook_text = (tmp_path / "hook.jsonl").read_text(encoding="utf-8")
    assert '"prompt_token_ids":' not in hook_text
    assert "workflow_segment_action" in hook_text


def test_segment_refresh_extends_existing_entry_and_marks_retouched(
    monkeypatch, tmp_path
) -> None:
    client, handler = _make_app(monkeypatch, tmp_path)

    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200
    expires_after_prepare = prepare_resp.json()["expires_at_unix_ms"]

    refresh_resp = client.post(
        "/v1/coopt/segment_refresh", json=_segment_refresh_action()
    )

    assert refresh_resp.status_code == 200
    payload = refresh_resp.json()
    assert payload["accepted"] is True
    assert payload["action_kind"] == SEGMENT_REFRESH_ACTION_KIND
    assert payload["status"] == "prefilled"
    assert payload["refresh_count"] == 1
    assert payload["last_refresh_at_unix_ms"] is not None
    assert payload["would_evict_without_refresh"] is True
    # Refresh re-marks status using the dedicated retouched value so
    # consumers can distinguish it from prepare in telemetry.
    assert payload["prefill_status"] == "retouched"
    assert payload["expires_at_unix_ms"] >= expires_after_prepare
    assert "prompt_token_ids" not in payload

    # Lookup endpoint returns the same record.
    lookup = client.get(
        "/v1/coopt/segment_prepare/" + str(prepare_resp.json()["action_id"])
    )
    assert lookup.status_code == 200
    assert lookup.json()["refresh_count"] == 1

    assert len(handler.refresh_submissions) == 1
    assert handler.refresh_submissions[0]["prompt_token_ids"] == list(_TOKEN_IDS)


def test_segment_prepare_rejects_missing_required_fields(
    monkeypatch, tmp_path
) -> None:
    client, _handler = _make_app(monkeypatch, tmp_path)

    bad = _segment_prepare_action()
    bad.pop("expected_consumers")
    response = client.post("/v1/coopt/segment_prepare", json=bad)
    assert response.status_code == 400
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["lifecycle_status"] == "rejected"
    assert payload["reject_reason"] == "missing_expected_consumers"

    bad = _segment_prepare_action()
    bad.pop("token_hash")
    bad.pop("prompt_token_ids", None)  # leave nothing to derive a hash from
    response = client.post("/v1/coopt/segment_prepare", json=bad)
    assert response.status_code == 400
    assert response.json()["reject_reason"] == "missing_token_hash"

    bad = _segment_prepare_action()
    bad["action_kind"] = "prefix_prepare"
    response = client.post("/v1/coopt/segment_prepare", json=bad)
    assert response.status_code == 400
    assert response.json()["reject_reason"] == "unsupported_action_kind"


def test_segment_refresh_rejects_missing_family_id(monkeypatch, tmp_path) -> None:
    client, _handler = _make_app(monkeypatch, tmp_path)

    bad = _segment_refresh_action()
    bad.pop("family_id")
    response = client.post("/v1/coopt/segment_refresh", json=bad)
    assert response.status_code == 400
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["reject_reason"] == "missing_family_id"


def test_segment_prepare_falls_back_to_prompt_token_ids_hash(
    monkeypatch, tmp_path
) -> None:
    """Runner emits ``prompt_token_ids_hash`` (not ``token_hash``)."""

    client, _handler = _make_app(monkeypatch, tmp_path)
    action = _segment_prepare_action()
    action.pop("token_hash")
    action["prompt_token_ids_hash"] = _TOKEN_HASH
    response = client.post("/v1/coopt/segment_prepare", json=action)
    assert response.status_code == 200
    assert response.json()["engine_seen_token_hash"] == _TOKEN_HASH


def test_segment_endpoints_disabled_without_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", raising=False)
    app = FastAPI()
    app.state.openai_serving_chat = _FakeChatHandler()
    api_router.attach_router(app)
    client = TestClient(app)

    response = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    # Route is not attached at all -> FastAPI returns 404.
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Phase E4 — per-segment cache telemetry
# ---------------------------------------------------------------------------


def test_telemetry_endpoint_returns_zero_counters_initially(
    monkeypatch, tmp_path
) -> None:
    """Right after segment_prepare, the per-segment cache counters are zero.

    Verifies the telemetry endpoint shape + the documented invariant
    that the engine has not yet observed any cache hits / retentions /
    evictions for a freshly-registered segment.
    """

    client, _handler = _make_app(monkeypatch, tmp_path)

    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200

    telem = client.get("/v1/coopt/segment_telemetry/seg-abc")
    assert telem.status_code == 200
    payload = telem.json()
    # Per-segment lifecycle is preserved alongside the new counters.
    assert payload["segment_id"] == "seg-abc"
    assert payload["family_id"] == "fam-1"
    assert payload["status"] == "prefilled"
    # Phase E4 counters: present, zero / empty before any block-pool event.
    assert payload["cache_hit_count"] == 0
    assert payload["retention_count"] == 0
    assert payload["evict_count_by_hint"] == {}
    # Redaction contract: the telemetry surface never leaks token ids.
    assert "prompt_token_ids" not in payload

    # Unknown segment id returns 404 with a reject reason.
    missing = client.get("/v1/coopt/segment_telemetry/seg-does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["reject_reason"] == "segment_not_found"


def test_telemetry_records_eviction_with_hint(monkeypatch, tmp_path) -> None:
    """``record_segment_eviction`` increments ``evict_count_by_hint``.

    Drives the registry helper directly so the test exercises the
    counter logic without needing a live block_pool. Mirrors the way
    block_pool._maybe_evict_cached_block calls back into segment_actions.
    """

    client, _handler = _make_app(monkeypatch, tmp_path)
    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200

    from vllm.entrypoints.openai.chat_completion.segment_actions import (
        record_segment_eviction,
    )

    record_segment_eviction("seg-abc", "must")
    record_segment_eviction("seg-abc", "must")
    record_segment_eviction("seg-abc", "may")
    record_segment_eviction("seg-abc", None)  # treated as default "may"
    # No-op for unknown segments — must not raise + must not alter state.
    record_segment_eviction("seg-unknown", "must")
    record_segment_eviction(None, "must")

    telem = client.get("/v1/coopt/segment_telemetry/seg-abc")
    assert telem.status_code == 200
    payload = telem.json()
    assert payload["evict_count_by_hint"] == {"must": 2, "may": 2}
    # Eviction counters do not affect cache_hit / retention buckets.
    assert payload["cache_hit_count"] == 0
    assert payload["retention_count"] == 0


def test_telemetry_records_cache_hit(monkeypatch, tmp_path) -> None:
    """``record_segment_cache_hit`` + ``record_segment_retention`` increment.

    Verifies the two helpers used by ``BlockPool.touch`` increment
    independent counters and that the running totals are surfaced
    through both the telemetry endpoint and the workflow_test_hook
    record on the next wire-action emission.
    """

    client, _handler = _make_app(monkeypatch, tmp_path)
    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200

    from vllm.entrypoints.openai.chat_completion.segment_actions import (
        record_segment_cache_hit,
        record_segment_retention,
    )

    record_segment_cache_hit("seg-abc")
    record_segment_cache_hit("seg-abc")
    record_segment_cache_hit("seg-abc")
    record_segment_retention("seg-abc")
    record_segment_retention("seg-abc")
    record_segment_cache_hit("seg-unknown")  # silently ignored
    record_segment_cache_hit(None)

    telem = client.get("/v1/coopt/segment_telemetry/seg-abc")
    assert telem.status_code == 200
    payload = telem.json()
    assert payload["cache_hit_count"] == 3
    assert payload["retention_count"] == 2
    assert payload["evict_count_by_hint"] == {}

    # Subsequent wire action emission carries the counters through to
    # the workflow_test_hook record (engine_segment_* mirror fields).
    refresh_resp = client.post(
        "/v1/coopt/segment_refresh", json=_segment_refresh_action()
    )
    assert refresh_resp.status_code == 200
    refresh_record = next(
        record
        for record in reversed(_records)
        if record.source == "workflow_segment_action"
        and record.action_kind == SEGMENT_REFRESH_ACTION_KIND
    )
    assert refresh_record.engine_segment_cache_hit_count == 3
    assert refresh_record.engine_segment_retention_count == 2
    assert refresh_record.engine_segment_evict_count == 0


# ---------------------------------------------------------------------------
# M9 (paper/CODE_MISMATCH_NOTES.md): merged segment_refresh + lifecycle_hint
#
# These tests cover three call shapes for the merged endpoint:
#
# 1. Touch-only refresh: prompt_token_ids present, lifecycle_hint absent.
#    LRU touch via prefill; no block hint flip. Mirrors the legacy
#    segment_refresh contract.
# 2. Touch + hint update: both present. LRU touch AND flips block hints
#    by segment_id.
# 3. Hint-only: prompt_token_ids absent, lifecycle_hint present. No
#    LRU touch; just flips block hints. Subsumes the legacy
#    /v1/coopt/segment_lifecycle_update endpoint, including loop-anchor
#    must-promotion at loop entry that has no token ids at hand.
#
# Plus the 410 Gone redirect on the legacy endpoint URL.
# ---------------------------------------------------------------------------


def _hint_only_refresh_action(
    *, action_id: str = "segr:hint:1", lifecycle_hint: str = "must"
) -> dict[str, object]:
    """Build a hint-only segment_refresh payload (no prompt_token_ids).

    Mirrors the loop-anchor-must-promotion call site in v2/api.py:
    the runner has a segment_id but no token ids at the moment the
    promotion fires (token ids are owned by the prepare flow).
    """

    return {
        "action_kind": SEGMENT_REFRESH_ACTION_KIND,
        "action_id": action_id,
        "segment_id": "seg-abc",
        "family_id": "fam-1",
        "parent_segment_id": None,
        "token_hash": _TOKEN_HASH,
        "model": "test-model",
        "lifecycle_status": "created",
        "lifecycle_hint": lifecycle_hint,
        "would_evict_without_refresh": False,
        "ttl_ms": 30000,
    }


def test_m9_segment_refresh_without_hint_does_lru_touch_only(
    monkeypatch, tmp_path
) -> None:
    """M9 refresh shape #1: prompt_token_ids present + lifecycle_hint absent.

    Verifies that a refresh without ``lifecycle_hint`` performs the LRU
    touch (prefill via the chat handler) AND records hint_update_*
    fields with the not-applied sentinel values, so consumers can tell
    no block hints were flipped.
    """

    client, handler = _make_app(monkeypatch, tmp_path)

    # Prepare first so the (family_id, token_hash) entry exists.
    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200

    refresh_resp = client.post(
        "/v1/coopt/segment_refresh", json=_segment_refresh_action()
    )
    assert refresh_resp.status_code == 200
    body = refresh_resp.json()
    assert body["accepted"] is True
    # Touch happened (prewarm submitted -> retouched marker).
    assert body["prefill_status"] == "retouched"
    assert body["prefill_token_count"] == len(_TOKEN_IDS)
    # No hint update because lifecycle_hint is absent.
    assert body["hint_update_applied_hint"] is None
    assert body["hint_update_updated"] == 0
    assert body["hint_update_skipped"] == 0
    assert body["hint_update_reject_reason"] is None
    # Engine-side prewarm was invoked.
    assert len(handler.refresh_submissions) == 1


def test_m9_segment_refresh_with_hint_does_touch_and_flip(
    monkeypatch, tmp_path
) -> None:
    """M9 refresh shape #2: prompt_token_ids + lifecycle_hint both present.

    Verifies that a refresh with ``lifecycle_hint`` set performs BOTH
    the LRU touch AND flips block hints by segment_id. Replaces the
    legacy two-call sequence (refresh, then segment_lifecycle_update).
    """

    from vllm.entrypoints.openai.chat_completion.segment_actions import (
        tag_blocks_with_segment_id,
    )

    client, handler = _make_app(monkeypatch, tmp_path)

    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200

    class _FakeBlock:
        # M18: schema migrated from single ``_segment_id`` slot to
        # ``_segment_ids`` tuple. Test fakes that mimic KVCacheBlock
        # must expose the new attribute so tag_blocks_with_segment_id
        # can append the tag (the production setter writes the tuple).
        __slots__ = ("_segment_ids", "lifecycle_hint")

        def __init__(self):
            self._segment_ids = ()
            self.lifecycle_hint = "may"

    blk_a, blk_b = _FakeBlock(), _FakeBlock()
    tag_blocks_with_segment_id([blk_a, blk_b], "seg-abc")

    action = _segment_refresh_action()
    action["lifecycle_hint"] = "must"
    refresh_resp = client.post("/v1/coopt/segment_refresh", json=action)
    assert refresh_resp.status_code == 200
    body = refresh_resp.json()
    assert body["accepted"] is True
    # Touch happened.
    assert body["prefill_status"] == "retouched"
    # Hint flip happened too.
    assert body["hint_update_applied_hint"] == "must"
    assert body["hint_update_updated"] == 2
    assert body["hint_update_skipped"] == 0
    assert body["hint_update_reject_reason"] is None
    assert blk_a.lifecycle_hint == "must"
    assert blk_b.lifecycle_hint == "must"
    # The chat handler's prewarm hook was invoked exactly once (touch
    # piece), not twice.
    assert len(handler.refresh_submissions) == 1


def test_m9_segment_refresh_hint_only_skips_prewarm(
    monkeypatch, tmp_path
) -> None:
    """M9 refresh shape #3: lifecycle_hint without prompt_token_ids.

    Subsumes the legacy /v1/coopt/segment_lifecycle_update endpoint:
    when the loop-anchor must-promotion fires at loop entry, the
    runner has a segment_id but no token ids. The merged endpoint
    accepts that shape and just flips block hints (no prewarm).
    """

    from vllm.entrypoints.openai.chat_completion.segment_actions import (
        tag_blocks_with_segment_id,
    )

    client, handler = _make_app(monkeypatch, tmp_path)

    # Prepare so the segment is registered (sole purpose: lets us tag
    # blocks; the prepare also primes (family_id, token_hash) for the
    # hint-only refresh's registry lookup).
    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200

    class _FakeBlock:
        # M18: schema migrated from single ``_segment_id`` slot to
        # ``_segment_ids`` tuple. Test fakes that mimic KVCacheBlock
        # must expose the new attribute so tag_blocks_with_segment_id
        # can append the tag (the production setter writes the tuple).
        __slots__ = ("_segment_ids", "lifecycle_hint")

        def __init__(self):
            self._segment_ids = ()
            self.lifecycle_hint = "may"

    blk_a, blk_b, blk_c = _FakeBlock(), _FakeBlock(), _FakeBlock()
    tag_blocks_with_segment_id([blk_a, blk_b, blk_c], "seg-abc")

    refresh_resp = client.post(
        "/v1/coopt/segment_refresh",
        json=_hint_only_refresh_action(lifecycle_hint="must"),
    )
    assert refresh_resp.status_code == 200, refresh_resp.text
    body = refresh_resp.json()
    assert body["accepted"] is True
    # Hint-only mode marker on prefill_status — no LRU touch.
    assert body["prefill_status"] == "hint_only"
    assert body["prefill_token_count"] == 0
    # Hint flip happened.
    assert body["hint_update_applied_hint"] == "must"
    assert body["hint_update_updated"] == 3
    for blk in (blk_a, blk_b, blk_c):
        assert blk.lifecycle_hint == "must"
    # Crucially: engine-side prewarm hook was NOT invoked (no token ids).
    assert len(handler.refresh_submissions) == 0


def test_m9_segment_refresh_hint_only_unknown_segment_is_soft_noop(
    monkeypatch, tmp_path
) -> None:
    """Hint-only mode targeting an un-tagged segment: 200, hint_update_updated=0.

    Mirrors the legacy endpoint's soft-no-op semantics for speculative
    promotions (docs/v2/32 §4 OQ6 #1) that target may segments that
    haven't had blocks tagged yet.
    """

    client, _handler = _make_app(monkeypatch, tmp_path)

    # Don't tag any blocks; just send a hint-only refresh.
    action = _hint_only_refresh_action(lifecycle_hint="no")
    action["segment_id"] = "unknown-seg"
    # family_id is required so we can find / create the registry entry;
    # but no blocks are tagged, so the hint update is a soft no-op.
    refresh_resp = client.post("/v1/coopt/segment_refresh", json=action)
    assert refresh_resp.status_code == 200, refresh_resp.text
    body = refresh_resp.json()
    assert body["accepted"] is True
    assert body["prefill_status"] == "hint_only"
    assert body["hint_update_applied_hint"] == "no"
    assert body["hint_update_updated"] == 0
    assert body["hint_update_reject_reason"] in {
        "no_blocks_for_segment",
        "all_stale",
    }


def test_m9_segment_refresh_rejects_invalid_lifecycle_hint(
    monkeypatch, tmp_path
) -> None:
    """An unknown ``lifecycle_hint`` value must yield 400 invalid_lifecycle_hint."""

    client, _handler = _make_app(monkeypatch, tmp_path)

    action = _segment_refresh_action()
    action["lifecycle_hint"] = "INVALID"  # not in must|may|no
    resp = client.post("/v1/coopt/segment_refresh", json=action)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["accepted"] is False
    assert body["reject_reason"] == "invalid_lifecycle_hint"


def test_m9_segment_refresh_rejects_neither_token_ids_nor_hint(
    monkeypatch, tmp_path
) -> None:
    """A refresh that lacks BOTH prompt_token_ids and lifecycle_hint is malformed."""

    client, _handler = _make_app(monkeypatch, tmp_path)

    action = _hint_only_refresh_action()
    action.pop("lifecycle_hint")  # leave nothing to do
    resp = client.post("/v1/coopt/segment_refresh", json=action)
    assert resp.status_code == 400, resp.text
    assert (
        resp.json()["reject_reason"]
        == "missing_prompt_token_ids_or_lifecycle_hint"
    )


def test_m9_legacy_segment_lifecycle_update_returns_410_gone(
    monkeypatch, tmp_path
) -> None:
    """The legacy /v1/coopt/segment_lifecycle_update endpoint is removed.

    Returns HTTP 410 Gone with a structured error body so callers using
    the old path fail loudly with a clear migration message rather than
    silently 404'ing or sending hint updates that never apply.
    """

    client, _handler = _make_app(monkeypatch, tmp_path)
    resp = client.post(
        "/v1/coopt/segment_lifecycle_update",
        json={"segment_id": "seg-abc", "new_hint": "no"},
    )
    assert resp.status_code == 410, resp.text
    body = resp.json()
    assert "merged into segment_refresh" in body["error"]
    assert body["redirect_to"] == "/v1/coopt/segment_refresh"
    assert body["redirect_body_field"] == "lifecycle_hint"
    assert body["removed_at"] == "M9"
    assert body["reject_reason"] == "endpoint_merged_m9"


def test_m9_segment_refresh_hint_routes_through_block_pool_queue(
    monkeypatch, tmp_path
) -> None:
    """WIRES Phase B (preserved across M9): when block_pool registers
    update_block_hint as the updater, the merged segment_refresh endpoint
    flips block hints AND moves them between pools in the 3-pool queue
    (docs/v2/32 §2.3 + §2.7).
    """

    from vllm.entrypoints.openai.chat_completion.segment_actions import (
        reset_segment_registry_for_tests,
        set_block_hint_updater,
        tag_blocks_with_segment_id,
    )
    from vllm.v1.core.kv_cache_utils import (
        VICTIM_POLICY_WIRES_THREE_POOL,
        FreeKVCacheBlockQueue,
        KVCacheBlock,
    )

    client, _handler = _make_app(monkeypatch, tmp_path)
    prepare_resp = client.post(
        "/v1/coopt/segment_prepare",
        json=_segment_prepare_action(action_id="m9-wire-prepare"),
    )
    assert prepare_resp.status_code == 200

    seg_id = "seg-m9-wire"

    real_blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    queue = FreeKVCacheBlockQueue(
        real_blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL
    )
    set_block_hint_updater(queue.update_block_hint)
    try:
        assert queue.num_free_blocks_in_pool("may") == 3
        assert queue.num_free_blocks_in_pool("must") == 0

        tag_blocks_with_segment_id(real_blocks, seg_id)

        # Promote via the merged refresh endpoint, hint-only mode.
        action = _hint_only_refresh_action(
            action_id="m9-wire-promote",
            lifecycle_hint="must",
        )
        action["segment_id"] = seg_id
        resp = client.post("/v1/coopt/segment_refresh", json=action)
        assert resp.status_code == 200, resp.text
        assert resp.json()["hint_update_updated"] == 3

        assert queue.num_free_blocks_in_pool("may") == 0
        assert queue.num_free_blocks_in_pool("must") == 3
        for blk in real_blocks:
            assert blk.lifecycle_hint == "must"
    finally:
        set_block_hint_updater(None)
        reset_segment_registry_for_tests()


def test_m9_segment_refresh_hint_only_self_heals_stale_tag(
    monkeypatch, tmp_path
) -> None:
    """A block whose ``_segment_id`` was reassigned mid-flight is skipped
    by the merged refresh endpoint (preserved Phase E2 step 3 invariant).
    """

    from vllm.entrypoints.openai.chat_completion.segment_actions import (
        tag_blocks_with_segment_id,
    )

    client, _handler = _make_app(monkeypatch, tmp_path)
    prepare_resp = client.post(
        "/v1/coopt/segment_prepare", json=_segment_prepare_action()
    )
    assert prepare_resp.status_code == 200

    class _FakeBlock:
        # M18: schema migrated from single ``_segment_id`` slot to
        # ``_segment_ids`` tuple. Test fakes that mimic KVCacheBlock
        # must expose the new attribute so tag_blocks_with_segment_id
        # can append the tag (the production setter writes the tuple).
        __slots__ = ("_segment_ids", "lifecycle_hint")

        def __init__(self):
            self._segment_ids = ()
            self.lifecycle_hint = "may"

    blk_a, blk_b = _FakeBlock(), _FakeBlock()
    tag_blocks_with_segment_id([blk_a, blk_b], "seg-abc")
    blk_b._segment_ids = ("seg-other",)  # simulate eviction + reuse

    resp = client.post(
        "/v1/coopt/segment_refresh",
        json=_hint_only_refresh_action(lifecycle_hint="must"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["hint_update_updated"] == 1
    assert body["hint_update_skipped"] == 1
    assert blk_a.lifecycle_hint == "must"
    assert blk_b.lifecycle_hint == "may"
