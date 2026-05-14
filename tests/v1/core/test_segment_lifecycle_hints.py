# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M13 — per-segment chat-completion lifecycle hints.

paper/CODE_MISMATCH_NOTES.md M13 entry: chat completion now carries
``segment_lifecycle_hints``, a per-scope list that overrides the legacy
single ``lifecycle_hint`` for newly-allocated blocks whose
``_segment_id`` matches a scope_key. Already-prepared segments are
omitted by the runner; the engine treats absence as "no contribution
from this request" and falls back to the legacy field.

These tests cover three contracts:

1. Request parses ``segment_lifecycle_hints`` from extra_args and
   stores it as a dict.
2. Backward compat: legacy single ``lifecycle_hint`` still applies to
   blocks not covered by per-segment hints.
3. Per-segment override: when a freshly-allocated block carries a
   matching ``_segment_id``, the per-segment hint wins.
"""

from __future__ import annotations

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def _make_request(
    *, extra_args: dict | None = None, request_id: str = "req-m13"
) -> Request:
    sampling_params = SamplingParams(max_tokens=4)
    if extra_args is not None:
        sampling_params.extra_args = extra_args
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=sampling_params,
        pooling_params=None,
    )


# --- Contract 1: Request parses segment_lifecycle_hints --------------------


def test_segment_lifecycle_hints_default_none():
    req = _make_request(extra_args=None)
    assert req.segment_lifecycle_hints is None
    assert req.lifecycle_hint == "may"  # legacy default unchanged


def test_segment_lifecycle_hints_parsed_into_dict():
    req = _make_request(
        extra_args={
            "segment_lifecycle_hints": [
                {"scope_key": "seg-a", "lifecycle_hint": "must"},
                {"scope_key": "seg-b", "lifecycle_hint": "may"},
                {"scope_key": "seg-c", "lifecycle_hint": "no"},
            ]
        }
    )
    assert req.segment_lifecycle_hints == {
        "seg-a": "must",
        "seg-b": "may",
        "seg-c": "no",
    }


def test_segment_lifecycle_hints_invalid_entries_skipped():
    """Malformed entries must NOT abort the request — hint plumbing is
    advisory. Valid entries in the same list are still kept."""

    req = _make_request(
        extra_args={
            "segment_lifecycle_hints": [
                {"scope_key": "seg-a", "lifecycle_hint": "must"},
                "not-a-dict",
                {"scope_key": "", "lifecycle_hint": "must"},  # empty key
                {"scope_key": "seg-b", "lifecycle_hint": "bogus"},
                {"lifecycle_hint": "must"},  # missing scope_key
                {"scope_key": "seg-c"},  # missing hint
                {"scope_key": "seg-d", "lifecycle_hint": "no"},
            ]
        }
    )
    assert req.segment_lifecycle_hints == {"seg-a": "must", "seg-d": "no"}


def test_segment_lifecycle_hints_empty_list_stays_none():
    req = _make_request(extra_args={"segment_lifecycle_hints": []})
    assert req.segment_lifecycle_hints is None


def test_chat_completion_request_accepts_segment_lifecycle_hints_in_xargs():
    """Regression for the M13 schema gap: the prior tests only exercised
    the internal `Request` parser. The OpenAI HTTP layer's
    `ChatCompletionRequest` pydantic schema must also admit the
    list-of-dicts shape inside `vllm_xargs`. Without this, every
    M13-emitting runner request 400s at the FastAPI validator before
    `Request.__init__` ever runs.
    """

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    req = ChatCompletionRequest.model_validate(
        {
            "model": "Qwen/Qwen3-32B",
            "messages": [{"role": "user", "content": "hi"}],
            "vllm_xargs": {
                "segment_lifecycle_hints": [
                    {"scope_key": "seg-a", "lifecycle_hint": "must"},
                    {"scope_key": "seg-b", "lifecycle_hint": "may"},
                ],
                "lifecycle_hint": "may",
            },
        }
    )
    assert req.vllm_xargs["segment_lifecycle_hints"] == [
        {"scope_key": "seg-a", "lifecycle_hint": "must"},
        {"scope_key": "seg-b", "lifecycle_hint": "may"},
    ]
    assert req.vllm_xargs["lifecycle_hint"] == "may"


def test_legacy_lifecycle_hint_still_parsed_alongside_per_segment():
    """Backward compat: the legacy single field MUST still set
    ``Request.lifecycle_hint``; M13 only adds the per-segment list."""

    req = _make_request(
        extra_args={
            "lifecycle_hint": "must",
            "segment_lifecycle_hints": [
                {"scope_key": "seg-a", "lifecycle_hint": "no"},
            ],
        }
    )
    assert req.lifecycle_hint == "must"
    assert req.segment_lifecycle_hints == {"seg-a": "no"}


# --- Contract 2: KVCacheManager honors per-segment override ----------------


class _FakeBlock:
    """Minimal stand-in for KVCacheBlock to exercise the M13 branch in
    KVCacheManager.allocate_slots. The real block carries many more
    fields; we only need ``lifecycle_hint`` + ``_segment_ids`` here.

    M18: schema migrated from single ``_segment_id`` to a
    ``_segment_ids`` tuple so a block can carry multiple overlapping
    segment ids (doc 32 §2.3 "min wins"). The ``segment_id`` kwarg
    on this fake is mapped onto a single-element tuple for legacy
    test cases; pass ``segment_ids`` explicitly for the multi-tag
    cases.
    """

    def __init__(
        self,
        segment_id: str | None = None,
        segment_ids: tuple[str, ...] | None = None,
    ):
        self.lifecycle_hint = "may"
        # source_class Option D, Site 1: the real KVCacheBlock dataclass
        # defaults ``source_class`` to "unstructured" (paper §2.3:
        # access-based promotion is the dominant path). Mirror that
        # here so the allocate-time stamp behavior matches.
        self.source_class = "unstructured"
        if segment_ids is not None:
            self._segment_ids = segment_ids
        elif isinstance(segment_id, str) and segment_id:
            self._segment_ids = (segment_id,)
        else:
            self._segment_ids = ()


def _apply_lifecycle_hints_like_manager(request, new_blocks):
    """Mirror of the M13 branch in
    ``KVCacheManager.allocate_slots`` so we can unit-test the policy
    without spinning up the full manager. Behavior MUST stay in sync
    with ``vllm/v1/core/kv_cache_manager.py``.

    M18: a block may carry multiple overlapping segment ids; per
    doc 32 §2.3 "min wins", when several of those tags match
    scope_keys we pick the MIN-priority hint
    (``no`` < ``may`` < ``must``) to honor the most-conservative
    caller's intent.

    source_class Option D, Site 1: when the resolved hint for a
    block is ``"must"``, stamp ``blk.source_class = "structured"``
    so the eventual ``_stamp_must_promotion`` at append_n picks up
    the 300s structured TTL (vs 60s unstructured).
    """

    _PRIORITY = {"no": 0, "may": 1, "must": 2}
    request_hint = getattr(request, "lifecycle_hint", "may")
    seg_hints = getattr(request, "segment_lifecycle_hints", None)
    if request_hint != "may":
        for group_blocks in new_blocks:
            for blk in group_blocks:
                blk.lifecycle_hint = request_hint
                if request_hint == "must":
                    blk.source_class = "structured"
    if isinstance(seg_hints, dict) and seg_hints:
        for group_blocks in new_blocks:
            for blk in group_blocks:
                blk_seg_ids = getattr(blk, "_segment_ids", ())
                chosen: str | None = None
                for seg in blk_seg_ids:
                    h = seg_hints.get(seg)
                    if h is None:
                        continue
                    if chosen is None or _PRIORITY[h] < _PRIORITY[chosen]:
                        chosen = h
                if chosen is not None:
                    blk.lifecycle_hint = chosen
                    if chosen == "must":
                        blk.source_class = "structured"


def test_legacy_only_applies_uniformly_to_all_blocks():
    """Backward compat: when only the legacy ``lifecycle_hint`` is set,
    every newly-allocated block gets that hint regardless of tagging."""

    req = _make_request(extra_args={"lifecycle_hint": "must"})
    blocks = [[_FakeBlock(), _FakeBlock(segment_id="seg-x"), _FakeBlock()]]
    _apply_lifecycle_hints_like_manager(req, blocks)
    for blk in blocks[0]:
        assert blk.lifecycle_hint == "must"


def test_per_segment_overrides_legacy_for_matching_blocks():
    """Per-segment hint wins for blocks whose ``_segment_id`` matches
    a scope_key; non-matching blocks keep the legacy fallback."""

    req = _make_request(
        extra_args={
            "lifecycle_hint": "must",
            "segment_lifecycle_hints": [
                {"scope_key": "seg-a", "lifecycle_hint": "no"},
                {"scope_key": "seg-b", "lifecycle_hint": "may"},
            ],
        }
    )
    blocks = [
        [
            _FakeBlock(segment_id="seg-a"),
            _FakeBlock(segment_id="seg-b"),
            _FakeBlock(segment_id="seg-untracked"),
            _FakeBlock(segment_id=None),  # never tagged
        ]
    ]
    _apply_lifecycle_hints_like_manager(req, blocks)
    assert blocks[0][0].lifecycle_hint == "no"  # per-segment override
    assert blocks[0][1].lifecycle_hint == "may"  # per-segment override
    assert blocks[0][2].lifecycle_hint == "must"  # legacy fallback
    assert blocks[0][3].lifecycle_hint == "must"  # legacy fallback


def test_per_segment_only_no_legacy_keeps_default_for_others():
    """When ONLY per-segment hints are set (no legacy field), blocks
    that don't match a scope_key keep the default ``may`` — they get
    no contribution from this request."""

    req = _make_request(
        extra_args={
            "segment_lifecycle_hints": [
                {"scope_key": "seg-a", "lifecycle_hint": "must"},
            ],
        }
    )
    blocks = [
        [
            _FakeBlock(segment_id="seg-a"),
            _FakeBlock(segment_id="seg-untracked"),
            _FakeBlock(segment_id=None),
        ]
    ]
    _apply_lifecycle_hints_like_manager(req, blocks)
    assert blocks[0][0].lifecycle_hint == "must"
    assert blocks[0][1].lifecycle_hint == "may"
    assert blocks[0][2].lifecycle_hint == "may"


def test_no_hints_at_all_keeps_default_may():
    """Sanity: a request with no hints whatsoever leaves every block
    at the default ``may``. This is the path baseline lanes
    (stock_vllm, vllm_no_apc) follow."""

    req = _make_request(extra_args=None)
    blocks = [[_FakeBlock(), _FakeBlock(segment_id="seg-x")]]
    _apply_lifecycle_hints_like_manager(req, blocks)
    for blk in blocks[0]:
        assert blk.lifecycle_hint == "may"
