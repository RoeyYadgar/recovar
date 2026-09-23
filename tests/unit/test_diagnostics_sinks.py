from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.diagnostics import (
    NPZ_DIAGNOSTICS,
    NULL_DIAGNOSTICS,
    DiagnosticsPlan,
    IterationStarted,
    ParityDiagnostics,
    build_diagnostics_sink,
)


def test_null_diagnostics_has_no_trace_or_payload_side_effects():
    assert NULL_DIAGNOSTICS.iteration_started(IterationStarted(1, 4)) is None


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
        build_diagnostics_sink(DiagnosticsPlan.from_environment({"RECOVAR_PARITY_TIMING_DIR": "/tmp/timing"})),
        ParityDiagnostics,
    )
