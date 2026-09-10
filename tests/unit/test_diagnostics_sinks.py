from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from recovar.em.dense_single_volume.diagnostics import (
    NPZ_DIAGNOSTICS,
    NULL_DIAGNOSTICS,
    ConvergenceUpdated,
    DiagnosticsPlan,
    HalfScored,
    IterationFinished,
    IterationStarted,
    MapsUpdated,
    MstepAccumulated,
    ParityDiagnostics,
    TraceKind,
    TraceSpec,
    build_diagnostics_sink,
)


class _MustNotMaterialize:
    def __array__(self, *args, **kwargs):
        raise AssertionError("null diagnostics materialized a payload")

    def block_until_ready(self):
        raise AssertionError("null diagnostics synchronized a payload")


def test_null_diagnostics_has_no_trace_or_payload_side_effects():
    value = _MustNotMaterialize()

    assert NULL_DIAGNOSTICS.trace_spec.is_empty
    assert NULL_DIAGNOSTICS.iteration_started(IterationStarted(1, 4)) is None
    assert NULL_DIAGNOSTICS.half_scored(HalfScored(1, 0, value)) is None
    assert NULL_DIAGNOSTICS.mstep_accumulated(MstepAccumulated(1, 0, value)) is None
    assert NULL_DIAGNOSTICS.maps_updated(MapsUpdated(1, (value,), (value,))) is None
    assert (
        NULL_DIAGNOSTICS.convergence_updated(
            ConvergenceUpdated(1, value, 0.5, False),
        )
        is None
    )
    assert NULL_DIAGNOSTICS.iteration_finished(IterationFinished(1, 4)) is None


def test_trace_spec_is_frozen_and_names_only_extra_kernel_outputs():
    trace = TraceSpec(frozenset({TraceKind.SCORES, TraceKind.OPERANDS}))

    assert trace.requests(TraceKind.SCORES)
    assert trace.requests(TraceKind.OPERANDS)
    assert not trace.requests(TraceKind.POSTERIOR)
    with pytest.raises(FrozenInstanceError):
        trace.outputs = frozenset()


def test_lifecycle_payload_is_frozen_and_does_not_transform_values():
    result = _MustNotMaterialize()
    event = HalfScored(iteration=2, half=1, result=result)

    assert event.result is result
    with pytest.raises(FrozenInstanceError):
        event.half = 0


def test_iteration_start_does_not_require_size_planning():
    event = IterationStarted(iteration=0, relion_iteration=4)

    assert event.current_size is None


def test_npz_diagnostics_writes_explicit_payload_without_schema_changes(tmp_path):
    path = tmp_path / "capture.npz"
    NPZ_DIAGNOSTICS.write_payload(
        path,
        {"count": np.int32(3), "values": np.array([1.0, 2.0], dtype=np.float64)},
        compressed=True,
    )

    with np.load(path, allow_pickle=False) as capture:
        assert capture.files == ["count", "values"]
        assert capture["count"].dtype == np.int32
        assert capture["values"].dtype == np.float64


def test_sink_factory_selects_null_or_parity_from_the_resolved_plan():
    assert build_diagnostics_sink(DiagnosticsPlan.from_environment({})) is NULL_DIAGNOSTICS
    assert isinstance(
        build_diagnostics_sink(
            DiagnosticsPlan.from_environment({"RECOVAR_PARITY_TIMING_DIR": "/tmp/timing"})
        ),
        ParityDiagnostics,
    )
