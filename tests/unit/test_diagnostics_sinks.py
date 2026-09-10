from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume.diagnostics import (
    NULL_DIAGNOSTICS,
    ConvergenceUpdated,
    HalfScored,
    IterationFinished,
    IterationStarted,
    MapsUpdated,
    MstepAccumulated,
    TraceKind,
    TraceSpec,
)


class _MustNotMaterialize:
    def __array__(self, *args, **kwargs):
        raise AssertionError("null diagnostics materialized a payload")

    def block_until_ready(self):
        raise AssertionError("null diagnostics synchronized a payload")


def test_null_diagnostics_has_no_trace_or_payload_side_effects():
    value = _MustNotMaterialize()

    assert NULL_DIAGNOSTICS.trace_spec.is_empty
    assert NULL_DIAGNOSTICS.iteration_started(IterationStarted(1, 4, 64)) is None
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
