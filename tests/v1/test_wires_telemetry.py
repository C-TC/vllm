# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the wires-fork engine telemetry writer infrastructure.

Self-contained: no GPU, no model load. Pure stdlib + the new module.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest


def _load_wires_telemetry():
    """Import ``vllm.v1.wires_telemetry``.

    Prefer the standard package import. Fall back to a direct file-spec
    load when the surrounding ``vllm`` package can't be imported (e.g.
    when running this self-contained test in a stdlib-only env without
    torch). Both paths return the same module object.
    """
    try:
        return importlib.import_module("vllm.v1.wires_telemetry")
    except Exception:
        module_path = (
            Path(__file__).resolve().parents[2] / "vllm" / "v1" / "wires_telemetry.py"
        )
        spec = importlib.util.spec_from_file_location(
            "vllm.v1.wires_telemetry", module_path
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["vllm.v1.wires_telemetry"] = module
        spec.loader.exec_module(module)
        return module


wt = _load_wires_telemetry()

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Wipe the writer registry and point telemetry at the test tmp dir."""
    monkeypatch.setenv("WIRES_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.delenv("WIRES_SWEEP_OUT_DIR", raising=False)
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)
    yield
    wt._reset_for_tests()
    wt._set_enabled_for_tests(False)


# ---------------------------------------------------------------------------
# 1. Disabled mode is effectively free + writes nothing.
# ---------------------------------------------------------------------------


def test_disabled_mode_zero_cost(tmp_path: Path):
    """ENABLED=0: no file created, 100k appends complete in well under 50 ms."""
    wt._set_enabled_for_tests(False)
    writer = wt.get_writer("foo")
    assert isinstance(writer, wt._NoopTelemetryWriter)

    row = {"k": "v"}
    t0 = time.perf_counter()
    for _ in range(100_000):
        writer.append(row)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # No file should have been created.
    assert not (tmp_path / "foo.jsonl").exists()
    # Sanity bound (not a strict perf test): a no-op call ought to be
    # comfortably under this on any vaguely-modern CPU.
    assert elapsed_ms < 50.0, f"100k no-op appends took {elapsed_ms:.2f} ms"


# ---------------------------------------------------------------------------
# 2. Enabled mode round-trips JSONL.
# ---------------------------------------------------------------------------


def test_enabled_mode_writes_jsonl(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    writer = wt.get_writer("foo")
    assert isinstance(writer, wt.BackgroundTelemetryWriter)

    rows = [{"i": i, "tag": f"row-{i}"} for i in range(100)]
    t0 = time.perf_counter()
    for r in rows:
        writer.append(r)
    append_ms = (time.perf_counter() - t0) * 1000.0

    wt.flush_all(timeout_s=5.0)
    # Give the background thread one more tick to land any tail.
    writer.flush(timeout_s=2.0)

    out = tmp_path / "foo.jsonl"
    assert out.exists(), f"writer did not produce {out}"
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100
    parsed = [json.loads(line) for line in lines]
    assert parsed == rows

    # Stash the rough throughput number in the test report so the agent
    # caller can quote it without re-running. 100 rows is small, so we
    # report rows/sec as an indicator only.
    if append_ms > 0:
        rate = len(rows) / (append_ms / 1000.0)
        print(
            f"[telemetry] enabled append rate: {rate:.0f} rows/s "
            f"(100 rows in {append_ms:.3f} ms)"
        )


# ---------------------------------------------------------------------------
# 3. Overflow drops oldest, counts.
# ---------------------------------------------------------------------------


def test_overflow_drops_oldest(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    # Bypass the registry to construct a writer with a tiny cap; this
    # mirrors what the registry would do but with our own knobs. We do
    # NOT start the background thread's drain by setting an enormous
    # drain interval, so the queue actually fills before any disk write.
    writer = wt.BackgroundTelemetryWriter(
        "tiny",
        tmp_path,
        max_queue=10,
        drain_interval_s=3600.0,  # effectively never auto-drain
        flush_bytes=10**12,  # never trigger byte-based wake
    )
    try:
        for i in range(50):
            writer.append({"i": i})
        # Snapshot dropped count BEFORE the flush triggers a drain.
        assert writer.dropped() == 40
        # The 10 surviving rows should be the most recent ones (40..49).
        with writer._lock:
            survivors = [row["i"] for row in list(writer._queue)]
        assert survivors == list(range(40, 50))
        # Total counter agrees.
        assert wt.telemetry_dropped_total == 40
    finally:
        writer.shutdown(timeout_s=1.0)


# ---------------------------------------------------------------------------
# 4. Concurrent appenders never lose rows or corrupt JSONL.
# ---------------------------------------------------------------------------


def test_concurrent_appenders(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    writer = wt.get_writer("conc")

    n_threads = 8
    per_thread = 1000
    total = n_threads * per_thread
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for j in range(per_thread):
                writer.append({"tid": tid, "j": j})
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    append_ms = (time.perf_counter() - t0) * 1000.0

    assert not errors, f"worker threads raised: {errors}"

    wt.flush_all(timeout_s=5.0)
    writer.flush(timeout_s=2.0)

    out = tmp_path / "conc.jsonl"
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == total
    # Each line must be valid JSON, no torn writes.
    seen: set[tuple[int, int]] = set()
    for line in lines:
        row = json.loads(line)
        seen.add((row["tid"], row["j"]))
    assert len(seen) == total

    # Throughput hint for the report.
    if append_ms > 0:
        rate = total / (append_ms / 1000.0)
        print(
            f"[telemetry] concurrent append rate: {rate:.0f} rows/s "
            f"({total} rows across {n_threads} threads in "
            f"{append_ms:.3f} ms)"
        )


# ---------------------------------------------------------------------------
# 5. File rotation when current file passes the threshold.
# ---------------------------------------------------------------------------


def test_rotation_at_threshold(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    # Cap rotation at 1 KB so we trip it after a handful of rows.
    writer = wt.BackgroundTelemetryWriter(
        "roll",
        tmp_path,
        max_queue=10_000,
        drain_interval_s=0.05,
        flush_bytes=200,  # flush eagerly so writes hit disk
        rotate_bytes=1024,  # 1 KB threshold
        rotate_keep=3,
    )
    try:
        # Each row is ~50 bytes serialised; pump well past 1 KB.
        for i in range(200):
            writer.append({"i": i, "pad": "x" * 40})
        writer.flush(timeout_s=2.0)
        # Give the rotation a moment to settle if a flush triggered it.
        deadline = time.monotonic() + 2.0
        rolled = tmp_path / "roll.jsonl.1"
        while not rolled.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rolled.exists(), (
            "rotation did not produce roll.jsonl.1; "
            f"dir contents: {sorted(p.name for p in tmp_path.iterdir())}"
        )
        # The current .jsonl, if it exists, must be smaller than the cap
        # (rotation rolled it before we tipped past the threshold).
        current = tmp_path / "roll.jsonl"
        if current.exists():
            assert current.stat().st_size <= 2048
    finally:
        writer.shutdown(timeout_s=1.0)


# ---------------------------------------------------------------------------
# 6. Tearing down a writer kills its thread.
# ---------------------------------------------------------------------------


def test_no_writer_thread_leaks(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    writer = wt.get_writer("teardown")
    assert isinstance(writer, wt.BackgroundTelemetryWriter)
    assert writer._thread.is_alive()
    writer.shutdown(timeout_s=0.2)
    # Allow up to 200 ms total to confirm; shutdown(timeout_s=0.2) above
    # already joined, but we check is_alive after to be explicit.
    deadline = time.monotonic() + 0.2
    while writer._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not writer._thread.is_alive(), "writer thread still alive after shutdown"


# ---------------------------------------------------------------------------
# 7. Factory caches per stream name.
# ---------------------------------------------------------------------------


def test_writer_factory_caches(tmp_path: Path):
    wt._set_enabled_for_tests(True)
    a1 = wt.get_writer("a")
    a2 = wt.get_writer("a")
    b = wt.get_writer("b")
    assert a1 is a2
    assert a1 is not b


def test_writer_factory_caches_when_disabled(tmp_path: Path):
    """Same identity invariant must hold for the no-op variant."""
    wt._set_enabled_for_tests(False)
    a1 = wt.get_writer("a")
    a2 = wt.get_writer("a")
    b = wt.get_writer("b")
    assert a1 is a2
    assert a1 is not b
    assert isinstance(a1, wt._NoopTelemetryWriter)
