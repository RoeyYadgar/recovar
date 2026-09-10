from __future__ import annotations

import inspect

from recovar.em.dense_single_volume import em_engine, iteration_loop, local_em_engine
from recovar.em.dense_single_volume.helpers import significance, sparse_pass2_bucketed


def test_algorithm_modules_do_not_serialize_diagnostic_artifacts_directly():
    modules = (
        iteration_loop,
        em_engine,
        local_em_engine,
        significance,
        sparse_pass2_bucketed,
    )

    for module in modules:
        source = inspect.getsource(module)
        assert "np.save(" not in source, module.__name__
        assert "np.savez(" not in source, module.__name__
        assert "np.savez_compressed(" not in source, module.__name__


def test_numeric_helpers_do_not_define_or_raise_diagnostic_stop_exceptions():
    significance_source = inspect.getsource(significance)
    sparse_source = inspect.getsource(sparse_pass2_bucketed)

    assert "class SignificanceDumpComplete" not in significance_source
    assert "raise SignificanceDumpComplete" not in significance_source
    assert "class Pass2DumpComplete" not in sparse_source
    assert "class BPrefContributionDumpComplete" not in sparse_source
    assert "raise Pass2DumpComplete" not in sparse_source


def test_iteration_lifecycle_uses_one_explicit_sink_and_null_fast_path():
    refine_source = inspect.getsource(iteration_loop.refine_single_volume)
    loop_source = inspect.getsource(iteration_loop._run_relion_iteration_loop)

    assert "diagnostics=build_diagnostics_sink(runtime.diagnostics)" in refine_source
    assert "diagnostics" in inspect.signature(
        iteration_loop._run_relion_iteration_loop
    ).parameters
    assert loop_source.count("if diagnostics is not NULL_DIAGNOSTICS:") == 5
    for method in (
        "iteration_started",
        "half_scored",
        "mstep_accumulated",
        "maps_updated",
        "convergence_updated",
        "iteration_finished",
    ):
        assert f"diagnostics.{method}(" in loop_source


def test_iteration_start_precedes_and_does_not_read_current_size_planning():
    loop_source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    started_at = loop_source.index("diagnostics.iteration_started(")
    planning_at = loop_source.index("# --- Determine current_size")

    assert started_at < planning_at
    assert "current_size=int(cs)" not in loop_source[started_at:planning_at]
    assert iteration_loop._numbered_relion_iteration(3, 0) == 4
