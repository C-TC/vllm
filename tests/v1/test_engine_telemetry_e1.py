# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the wires-fork E1 per-request lifecycle stream (T-45.4).

Self-contained: no GPU, no model load. We do NOT instantiate the full
scheduler; instead we feed ``emit_request_lifecycle`` directly from
synthetic per-request state so the stream-shape contract gets
exercised without booting EngineCore. The integration with
``_free_request`` is covered by the upstream scheduler tests once
they pick up the new fields; the unit shape contract is what this
file owns.
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
    """Import ``vllm.<qualname>`` with a stdlib-only fallback.

    Mirrors the pattern in ``test_wires_telemetry.py`` so this test
    file can run on a CPU-only image without torch installed.
    """
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
# 1. Disabled-mode produces no file and is effectively free.
# ---------------------------------------------------------------------------


def test_disabled_mode_no_file(tmp_path: Path):
    """ENABLED=0: emit_request_lifecycle should write nothing."""
    wt._set_enabled_for_tests(False)
    for i in range(50):
        wet.emit_request_lifecycle(
            vllm_request_id=f"chatcmpl-{i}",
            ts_arrived=1.0,
            ts_scheduled=1.1,
            ts_first_token=1.5,
            ts_finished=10.0,
            prompt_token_count=100,
            cached_token_count=10,
            output_token_count=20,
            blocks_allocated=4,
            blocks_cache_hit=1,
            evicts_caused=0,
            status="ok",
        )
    assert not (tmp_path / "requests.jsonl").exists()


def test_disabled_mode_zero_cost():
    """100k disabled-mode E1 emits should complete well under 50 ms.

    Mirrors the perf-sanity test in ``test_wires_telemetry``; bounds
    the disabled-mode contract per proposal §5.2 rule 1.
    """
    wt._set_enabled_for_tests(False)
    t0 = time.perf_counter()
    for i in range(100_000):
        wet.emit_request_lifecycle(
            vllm_request_id="chatcmpl-x",
            ts_arrived=1.0,
            ts_scheduled=1.1,
            ts_first_token=1.5,
            ts_finished=10.0,
            prompt_token_count=100,
            cached_token_count=10,
            output_token_count=20,
            blocks_allocated=4,
            blocks_cache_hit=1,
            evicts_caused=0,
            status="ok",
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"100k disabled E1 emits took {elapsed_ms:.2f} ms"


# ---------------------------------------------------------------------------
# 2. Enabled mode emits one row per call with the schema-§3.1 shape.
# ---------------------------------------------------------------------------


def test_enabled_mode_writes_one_row_per_emit(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    rows = [
        {
            "vllm_request_id": f"chatcmpl-{i:04d}",
            "ts_arrived": 1000.0 + i,
            "ts_scheduled": 1000.1 + i,
            "ts_first_token": 1000.5 + i,
            "ts_finished": 1100.0 + i,
            "prompt_token_count": 7613 + i,
            "cached_token_count": 6800,
            "output_token_count": 412,
            "blocks_allocated": 28,
            "blocks_cache_hit": 26,
            "evicts_caused": 1 if i % 3 == 0 else 0,
            "status": "ok",
        }
        for i in range(8)
    ]
    for r in rows:
        wet.emit_request_lifecycle(**r)
    wt.flush_all(timeout_s=5.0)

    out = tmp_path / "requests.jsonl"
    assert out.exists()
    parsed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == len(rows)

    # Each row must have the locked schema-§3.1 keys.
    expected_keys = {
        "ts_epoch",
        "vllm_request_id",
        "ts_arrived",
        "ts_scheduled",
        "ts_first_token",
        "ts_finished",
        "prompt_token_count",
        "cached_token_count",
        "output_token_count",
        "blocks_allocated",
        "blocks_cache_hit",
        "evicts_caused",
        "status",
    }
    for row, want in zip(parsed, rows):
        assert set(row.keys()) == expected_keys, row
        # Spot-check each carried field round-trips verbatim.
        for k, v in want.items():
            assert row[k] == v, (k, row[k], v)
        # ``ts_epoch`` is the canonical timestamp; T-45.4 emits it
        # equal to ``ts_finished`` so the offline aggregator can sort
        # by either.
        assert row["ts_epoch"] == row["ts_finished"]


# ---------------------------------------------------------------------------
# 3. Status enum maps a few interesting RequestStatus values correctly.
# ---------------------------------------------------------------------------


def test_status_field_round_trips(tmp_path: Path):
    """E1 ``status`` is a free-form string at this layer; the
    scheduler is responsible for the enum mapping. Verify the
    transport carries any of the documented values verbatim."""
    wt._set_enabled_for_tests(True)
    for s in ("ok", "error", "aborted", "ignored"):
        wet.emit_request_lifecycle(
            vllm_request_id=f"chatcmpl-{s}",
            ts_arrived=1.0,
            ts_scheduled=1.1,
            ts_first_token=1.5,
            ts_finished=10.0,
            prompt_token_count=10,
            cached_token_count=0,
            output_token_count=1,
            blocks_allocated=1,
            blocks_cache_hit=0,
            evicts_caused=0,
            status=s,
        )
    wt.flush_all(timeout_s=5.0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [r["status"] for r in rows] == ["ok", "error", "aborted", "ignored"]


# ---------------------------------------------------------------------------
# 4. ``ts_scheduled`` / ``ts_first_token`` may be None when the request
#    finished before scheduling (e.g. submission error).
# ---------------------------------------------------------------------------


def test_optional_timestamps_serialise_as_null(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    wet.emit_request_lifecycle(
        vllm_request_id="chatcmpl-aborted",
        ts_arrived=1.0,
        ts_scheduled=None,
        ts_first_token=None,
        ts_finished=2.0,
        prompt_token_count=0,
        cached_token_count=0,
        output_token_count=0,
        blocks_allocated=0,
        blocks_cache_hit=0,
        evicts_caused=0,
        status="aborted",
    )
    wt.flush_all(timeout_s=5.0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["ts_scheduled"] is None
    assert rows[0]["ts_first_token"] is None


# ---------------------------------------------------------------------------
# 5. Synthetic finish loop: simulate 1000 finishes and assert the
#    file ends up with 1000 valid rows. Sanity-checks the lock /
#    drain plumbing under sustained per-finish writes.
# ---------------------------------------------------------------------------


def test_thousand_synthetic_finishes(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    n = 1000
    for i in range(n):
        wet.emit_request_lifecycle(
            vllm_request_id=f"chatcmpl-{i:06d}",
            ts_arrived=float(i),
            ts_scheduled=float(i) + 0.01,
            ts_first_token=float(i) + 0.05,
            ts_finished=float(i) + 1.0,
            prompt_token_count=100,
            cached_token_count=50,
            output_token_count=10,
            blocks_allocated=4,
            blocks_cache_hit=2,
            evicts_caused=i % 5,
            status="ok",
        )
    wt.flush_all(timeout_s=10.0)
    rows = (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == n
    parsed_ids = [json.loads(r)["vllm_request_id"] for r in rows]
    assert parsed_ids == [f"chatcmpl-{i:06d}" for i in range(n)]
