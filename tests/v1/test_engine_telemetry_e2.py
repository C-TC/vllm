# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the wires-fork E2 per-action handler stream (T-45.5).

Self-contained: no GPU. We exercise the E2 emitter directly + drive
``_e2_decide_outcome`` over the full enum domain. The actual handler
wiring (POST /v1/coopt/segment_*) is tested upstream against the
FastAPI router; those tests already pin the response shape, and our
emit_action helper is a thin pass-through that does NOT mutate the
response.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


def _load_module(qualname: str, relpath: str):
    try:
        return importlib.import_module(qualname)
    except Exception:
        module_path = Path(__file__).resolve().parents[2] / "vllm" / relpath
        spec = importlib.util.spec_from_file_location(qualname, module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        return module


wt = _load_module("vllm.v1.wires_telemetry", "v1/wires_telemetry.py")
wet = _load_module("vllm.v1.wires_engine_telemetry", "v1/wires_engine_telemetry.py")


pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_module_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WIRES_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.delenv("WIRES_SWEEP_OUT_DIR", raising=False)
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)
    yield
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)


# ---------------------------------------------------------------------------
# 1. Disabled mode is silent.
# ---------------------------------------------------------------------------


def test_disabled_no_file(tmp_path: Path):
    wt._set_enabled_for_tests(False)
    for i in range(20):
        wet.emit_action(
            endpoint=wet.ENDPOINT_SEGMENT_PREPARE,
            scope_key=f"scope-{i}",
            blocks_touched=10,
            blocks_already_present=0,
            blocks_newly_written=10,
            outcome=wet.ACTION_OUTCOME_PREPARED,
            elapsed_ms=12.3,
        )
    assert not (tmp_path / "actions.jsonl").exists()


def test_disabled_zero_cost():
    wt._set_enabled_for_tests(False)
    t0 = time.perf_counter()
    for _ in range(100_000):
        wet.emit_action(
            endpoint=wet.ENDPOINT_SEGMENT_PREPARE,
            scope_key="scope",
            blocks_touched=0,
            blocks_already_present=0,
            blocks_newly_written=0,
            outcome=wet.ACTION_OUTCOME_PREPARED,
            elapsed_ms=0.0,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"100k disabled E2 emits took {elapsed_ms:.2f} ms"


# ---------------------------------------------------------------------------
# 2. Enabled mode shape: row carries §3.2 keys, all four outcome enums hit.
# ---------------------------------------------------------------------------


def test_all_four_outcomes_round_trip(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    fixtures = [
        (wet.ENDPOINT_SEGMENT_PREPARE, wet.ACTION_OUTCOME_PREPARED, 14, 0, 14),
        (
            wet.ENDPOINT_SEGMENT_REFRESH,
            wet.ACTION_OUTCOME_NOOP_ALREADY_PRESENT,
            14,
            14,
            0,
        ),
        (
            wet.ENDPOINT_SEGMENT_PREPARE,
            wet.ACTION_OUTCOME_REJECTED_DEDUP,
            0,
            0,
            0,
        ),
        (
            wet.ENDPOINT_SEGMENT_REFRESH,
            wet.ACTION_OUTCOME_REJECTED_BAD_INPUT,
            0,
            0,
            0,
        ),
    ]
    for endpoint, outcome, touched, already, newly in fixtures:
        wet.emit_action(
            endpoint=endpoint,
            scope_key="system_prompt:reflexion:v1",
            blocks_touched=touched,
            blocks_already_present=already,
            blocks_newly_written=newly,
            outcome=outcome,
            elapsed_ms=12.3,
        )
    wt.flush_all(timeout_s=5.0)

    rows = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 4
    expected_keys = {
        "ts_epoch",
        "endpoint",
        "scope_key",
        "blocks_touched",
        "blocks_already_present",
        "blocks_newly_written",
        "outcome",
        "elapsed_ms",
    }
    for row, (endpoint, outcome, touched, already, newly) in zip(rows, fixtures):
        assert set(row.keys()) == expected_keys
        assert row["endpoint"] == endpoint
        assert row["outcome"] == outcome
        assert row["blocks_touched"] == touched
        assert row["blocks_already_present"] == already
        assert row["blocks_newly_written"] == newly
        assert row["scope_key"] == "system_prompt:reflexion:v1"
        assert isinstance(row["ts_epoch"], (int, float))
        assert row["elapsed_ms"] == 12.3


# ---------------------------------------------------------------------------
# 3. ts_epoch can be supplied by the caller (handler may already
#    have a wall-clock anchor).
# ---------------------------------------------------------------------------


def test_caller_supplied_ts_epoch(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    wet.emit_action(
        endpoint=wet.ENDPOINT_SEGMENT_PREPARE,
        scope_key="scope-x",
        blocks_touched=1,
        blocks_already_present=0,
        blocks_newly_written=1,
        outcome=wet.ACTION_OUTCOME_PREPARED,
        elapsed_ms=0.7,
        ts_epoch=42.0,
    )
    wt.flush_all(timeout_s=5.0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["ts_epoch"] == 42.0


# ---------------------------------------------------------------------------
# 4. _e2_decide_outcome maps the handler exit states correctly.
# ---------------------------------------------------------------------------


def _load_segment_actions():
    """Import segment_actions, falling back to a direct file-spec load.

    The full module pulls in ``fastapi`` + the chat completion stack;
    keep this guarded so the test still runs in a stripped-down env
    that only has the v1 core. When import fails we skip the
    decide-outcome test rather than fail the whole file.
    """
    try:
        return importlib.import_module(
            "vllm.entrypoints.openai.chat_completion.segment_actions"
        )
    except Exception:
        return None


@pytest.mark.parametrize(
    "rejected,reject_reason,prefill_status,want",
    [
        # rejected -> bad_input or dedup
        (True, "missing_token_hash", "not_attempted", "rejected_bad_input"),
        (True, "duplicate_segment_id", "not_attempted", "rejected_dedup"),
        # accepted -> prepared if prewarm fired
        (False, None, "prewarm_submitted", "prepared"),
        (False, None, "prewarm_completed", "prepared"),
        (False, None, "retouched", "prepared"),
        # accepted -> noop_already_present otherwise
        (False, None, "hint_only", "noop_already_present"),
        (False, None, "not_attempted", "noop_already_present"),
        (False, None, "prefill_only_unavailable", "noop_already_present"),
    ],
)
def test_decide_outcome_table(rejected, reject_reason, prefill_status, want):
    sa = _load_segment_actions()
    if sa is None:
        pytest.skip("segment_actions not importable in this test env")
    assert (
        sa._e2_decide_outcome(
            rejected=rejected,
            reject_reason=reject_reason,
            prefill_status=prefill_status,
        )
        == want
    )
