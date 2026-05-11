# SPDX-License-Identifier: Apache-2.0

"""Test guards for OpenAIServingChat.submit_workflow_segment_prewarm.

Phase E5 (docs/v2/31). The Stage D verification surfaced that the
hook ``OpenAIServingChat.submit_workflow_segment_prewarm`` was
missing entirely — every segment_prepare action accepted by the
endpoint silently fell back to ``prefill_status="tokenize_only"``
because ``_maybe_submit_segment_prewarm`` could not find the method
to delegate to. Existing smoke tests in test_workflow_segment_actions.py
stub the hook on a fake handler, so they did NOT catch the gap.

These tests exercise the REAL ``OpenAIServingChat.submit_workflow_segment_prewarm``
method (via duck-typed self to avoid the full constructor surface) and
assert on the OBSERVABLE downstream behaviour:

1. The submitted internal request carries the right
   ``SamplingParams.extra_args`` (lifecycle_hint, workflow_segment_id,
   workflow_segment_action_id).
2. The submitted request is prefill-only (max_tokens=1, temperature=0).
3. After the consume coroutine drains, the SegmentRegistry entry's
   ``prefill_status`` flipped from the initial ``"tokenize_only"``
   placeholder to ``"prefilled"``.
4. The hook returns ``{"prewarm_status": "prewarm_submitted", ...}``.

Without these guards, a future refactor could silently drop any of
those wires and Stage D / E sweeps would again show wires_full
trailing stock for no apparent reason.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

from vllm.entrypoints.openai.chat_completion.segment_actions import (
    SEGMENT_PREPARE_ACTION_KIND,
    _registry,
    reset_segment_registry_for_tests,
    submit_segment_prepare_action,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


_TOKEN_HASH = "a" * 64
_TOKEN_IDS = [10, 20, 30, 40, 50]


class _FakeOutput:
    def __init__(self) -> None:
        self.token_ids: list[int] = []
        self.text: str = ""


class _FakeRequestOutput:
    def __init__(self) -> None:
        self.outputs = [_FakeOutput()]
        self.kv_transfer_params: dict[str, Any] | None = None


class _RecordingEngineClient:
    """Records ``generate(...)`` calls; yields a single fake output."""

    def __init__(self) -> None:
        self.errored = False
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        priority: int = 0,
        **_kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
                "priority": priority,
            }
        )

        async def _gen() -> Any:
            yield _FakeRequestOutput()

        return _gen()


def _make_fake_self() -> types.SimpleNamespace:
    """Build a minimal duck-typed `self` for the bound submit_* method.

    OpenAIServingChat.submit_workflow_segment_prewarm only references
    ``self.engine_client`` and the internal ``_consume_workflow_segment_prewarm``
    method — both of which we can attach to a SimpleNamespace.
    """

    fake_self = types.SimpleNamespace(engine_client=_RecordingEngineClient())
    fake_self._consume_workflow_segment_prewarm = (
        OpenAIServingChat._consume_workflow_segment_prewarm.__get__(
            fake_self, type(fake_self)
        )
    )
    return fake_self


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_WORKFLOW_COOPT_ACTIONS", "1")
    monkeypatch.setenv("WORKFLOW_SEGMENT_PREPARE_MODE", "experimental_prewarm")
    reset_segment_registry_for_tests()


def _segment_action(*, action_id: str = "segp:test:1") -> dict[str, Any]:
    return {
        "action_kind": SEGMENT_PREPARE_ACTION_KIND,
        "action_id": action_id,
        "segment_id": "seg-test-abc",
        "family_id": "fam-test-1",
        "parent_segment_id": None,
        "token_hash": _TOKEN_HASH,
        "prompt_token_ids": list(_TOKEN_IDS),
        "model": "test-model",
        "expected_consumers": 3,
        "lifecycle_status": "must",
        "ttl_ms": 30000,
    }


def test_submit_returns_prewarm_submitted_with_token_count():
    fake_self = _make_fake_self()
    action = _segment_action()
    result = asyncio.run(
        OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, action)
    )
    assert result["prewarm_status"] == "prewarm_submitted"
    assert result["prefill_token_count"] == len(_TOKEN_IDS)


def test_submit_threads_lifecycle_hint_and_segment_id_into_extra_args():
    """The headline assertion: the internal prefill request MUST carry
    ``lifecycle_hint`` (so KVCacheBlock 3-priority eviction respects it)
    and ``workflow_segment_id`` (so KVCacheManager tags the new blocks).
    Without these the V1-native prewarm produces blocks indistinguishable
    from any other request and segment_telemetry can't attribute hits.
    """

    fake_self = _make_fake_self()
    action = _segment_action()
    asyncio.run(
        OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, action)
    )
    assert len(fake_self.engine_client.calls) == 1
    call = fake_self.engine_client.calls[0]
    extra = call["sampling_params"].extra_args
    assert extra is not None
    assert extra["lifecycle_hint"] == "must", (
        f"lifecycle_hint not threaded; got {extra.get('lifecycle_hint')!r} "
        "(KVCacheBlock 3-priority eviction will not honor the hint)"
    )
    assert extra["workflow_segment_id"] == "seg-test-abc", (
        "workflow_segment_id not threaded; KVCacheManager will not call "
        "tag_blocks_with_segment_id and segment_telemetry attribution breaks"
    )
    assert extra["workflow_segment_action_id"] == "segp:test:1"
    assert extra["workflow_segment_family_id"] == "fam-test-1"
    assert extra["workflow_segment_expected_consumers"] == 3
    assert extra["workflow_prefill_only"] is True


def test_submit_uses_prefill_only_sampling():
    fake_self = _make_fake_self()
    asyncio.run(
        OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, _segment_action())
    )
    sp = fake_self.engine_client.calls[0]["sampling_params"]
    # Hidden internal request must NOT decode beyond a single token.
    assert sp.max_tokens == 1
    assert sp.temperature == 0.0


def test_submit_request_id_carries_action_id_for_traceability():
    fake_self = _make_fake_self()
    asyncio.run(
        OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, _segment_action())
    )
    assert (
        fake_self.engine_client.calls[0]["request_id"]
        == "workflow-segment-prewarm-segp:test:1"
    )


def test_submit_rejects_missing_token_ids():
    fake_self = _make_fake_self()
    action = _segment_action()
    action.pop("prompt_token_ids")
    result = asyncio.run(
        OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, action)
    )
    assert result["prewarm_status"] == "prewarm_failed:missing_token_ids"
    assert fake_self.engine_client.calls == []


def test_submit_rejects_missing_action_id():
    fake_self = _make_fake_self()
    action = _segment_action()
    action.pop("action_id")
    result = asyncio.run(
        OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, action)
    )
    assert result["prewarm_status"] == "prewarm_failed:missing_action_id"
    assert fake_self.engine_client.calls == []


def test_submit_rejects_when_engine_errored():
    fake_self = _make_fake_self()
    fake_self.engine_client.errored = True
    result = asyncio.run(
        OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, _segment_action())
    )
    assert result["prewarm_status"] == "prewarm_failed:engine_error"
    assert fake_self.engine_client.calls == []


def test_consume_marks_segment_prefilled_in_registry():
    """End-to-end through the registry: register a segment via
    submit_segment_prepare_action (which puts the entry in the registry
    with prefill_status='prefill_only_unavailable' since fake_handler is
    None), then call submit_workflow_segment_prewarm, drain the consume
    task, and assert the entry's prefill_status flipped to 'prefilled'.

    This is THE test that guards against the Stage D regression: no
    matter what shape the registry has internally, callers querying
    /v1/coopt/segment_telemetry/{segment_id} must see prefill_status
    transition out of the initial 'tokenize_only' / 'prefill_only_unavailable'
    placeholder once the prewarm coroutine completes.
    """

    fake_self = _make_fake_self()
    action = _segment_action()
    # First register the segment via the public route (this puts an
    # entry into the registry — _registry.submit_prepare flow).
    asyncio.run(
        submit_segment_prepare_action(action, chat_handler=None)
    )
    # Sanity: entry exists and is NOT yet prefilled.
    pre_entry = _registry.get_by_action_id(action["action_id"])
    assert pre_entry is not None
    assert pre_entry["prefill_status"] != "prefilled"
    assert pre_entry["prefill_token_count"] == 0  # nothing prefilled yet

    # Now run the V1 prewarm hook through the real method; the consume
    # coroutine launched as an asyncio task must drain and mark the
    # registry entry as prefilled.
    async def _drive() -> None:
        await OpenAIServingChat.submit_workflow_segment_prewarm(fake_self, action)
        # Yield control so the task created via asyncio.create_task runs.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_drive())
    post_entry = _registry.get_by_action_id(action["action_id"])
    assert post_entry is not None
    assert post_entry["prefill_status"] == "prefilled", (
        f"prefill_status did not transition to 'prefilled' after consume "
        f"coroutine drained (got {post_entry['prefill_status']!r}). "
        "Without this transition, segment_telemetry permanently reports "
        "tokenize_only and the WIRES segment-prepare claim is invisible "
        "in evidence JSONLs (Stage D regression)."
    )
    # The prefill input size (action's prompt_token_ids length), NOT
    # the decode side-effect count (always 1 by construction).
    assert post_entry["prefill_token_count"] == len(_TOKEN_IDS), (
        f"prefill_token_count={post_entry['prefill_token_count']} should be "
        f"{len(_TOKEN_IDS)} (the prefill input size). The Stage D verification "
        "showed the count was being mis-reported as 1 (the max_tokens=1 decode "
        "by-product) — segment_telemetry GETs would under-count work done."
    )


def test_request_segment_id_parsed_from_extra_args():
    """Phase E5: Request.segment_id must be set from
    SamplingParams.extra_args["workflow_segment_id"] so the
    KVCacheManager can tag freshly-allocated blocks.
    """

    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    sp = SamplingParams.from_optional(
        temperature=0.0,
        max_tokens=1,
        extra_args={
            "lifecycle_hint": "must",
            "workflow_segment_id": "seg-from-extras",
        },
    )
    req = Request(
        request_id="test-req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
    )
    assert req.segment_id == "seg-from-extras"
    assert req.lifecycle_hint == "must"


def test_request_segment_id_defaults_none_for_normal_request():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    sp = SamplingParams.from_optional(temperature=0.0, max_tokens=10)
    req = Request(
        request_id="normal-req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
    )
    assert req.segment_id is None
    assert req.lifecycle_hint == "may"
