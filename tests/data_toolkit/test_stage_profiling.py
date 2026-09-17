"""Behavioral tests for opt-in stage profiling."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline.stage_profiling import (
    profile_call,
    profile_stage,
    profiled_stage,
)


def _profile_record(output: str) -> dict[str, float | str]:
    prefix = "[PIXAL3D_PROFILE] "
    assert output.startswith(prefix)
    return json.loads(output.removeprefix(prefix))


def test_profile_stage_emits_nothing_when_profiling_is_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class ForbiddenCuda:
        def synchronize(self, device: str) -> None:
            raise AssertionError(f"disabled profiling synchronized {device}")

    # Given: CUDA would fail if the disabled path used it.
    monkeypatch.delenv("PIXAL3D_PROFILE_STAGES", raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=ForbiddenCuda()))

    # When: a stage is profiled with a CUDA device while disabled.
    with profile_stage("decode", device="cuda:0"):
        pass

    # Then: it neither synchronizes nor emits output.
    assert capsys.readouterr().out == ""


def test_profile_stage_emits_cpu_timings_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: stage profiling is enabled.
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "true")

    # When: a CPU stage completes.
    with profile_stage("decode"):
        sum(range(10_000))

    # Then: the stable JSON record contains CPU timing fields.
    record = _profile_record(capsys.readouterr().out)
    assert set(record) == {"name", "process_cpu_ms", "thread_cpu_ms", "wall_ms"}
    assert record["name"] == "decode"
    assert all(record[field] >= 0 for field in record if field != "name")


def test_profile_stage_emits_cuda_timings_when_device_is_supplied(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.recorded = False

        def record(self) -> None:
            self.recorded = True

        def elapsed_time(self, end: FakeEvent) -> float:
            assert self.recorded and end.recorded
            return 12.5

    class FakeCuda:
        def __init__(self) -> None:
            self.events: list[FakeEvent] = []
            self.synchronized_devices: list[str] = []

        def Event(self, *, enable_timing: bool) -> FakeEvent:
            assert enable_timing
            event = FakeEvent()
            self.events.append(event)
            return event

        def synchronize(self, device: str) -> None:
            self.synchronized_devices.append(device)

    # Given: enabled profiling and a CUDA implementation with observable calls.
    cuda = FakeCuda()
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    # When: a CUDA stage completes.
    with profile_stage("encode", device="cuda:3"):
        pass

    # Then: boundaries synchronize and the record includes CUDA timing fields.
    record = _profile_record(capsys.readouterr().out)
    assert cuda.synchronized_devices == ["cuda:3", "cuda:3"]
    assert len(cuda.events) == 2
    assert set(record) == {
        "cuda_elapsed_ms",
        "host_submit_ms",
        "name",
        "pre_sync_ms",
        "process_cpu_ms",
        "sync_wait_ms",
        "thread_cpu_ms",
        "wall_ms",
    }
    assert record["name"] == "encode"
    assert record["cuda_elapsed_ms"] == 12.5
    assert all(record[field] >= 0 for field in record if field != "name")


def test_profile_stage_propagates_stage_exceptions_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: enabled profiling.
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "yes")

    # When: the profiled stage raises.
    with (
        pytest.raises(RuntimeError, match="stage failed"),
        profile_stage("failing-stage"),
    ):
        raise RuntimeError("stage failed")

    # Then: the stage exception reaches the caller.


def test_profiled_stage_decorator_profiles_cpu_function_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: enabled profiling and a function marked as a stage.
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "on")

    @profiled_stage("save")
    def save(value: int) -> int:
        return value + 1

    # When: the decorated function is called.
    result = save(41)

    # Then: its result and CPU timing record are preserved.
    assert result == 42
    record = _profile_record(capsys.readouterr().out)
    assert record["name"] == "save"
    assert set(record) == {"name", "process_cpu_ms", "thread_cpu_ms", "wall_ms"}


def test_profile_call_preserves_arguments_result_and_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "1")

    result = profile_call("loader.read", pow, 3, 4)

    assert result == 81
    record = _profile_record(capsys.readouterr().out)
    assert record["name"] == "loader.read"
