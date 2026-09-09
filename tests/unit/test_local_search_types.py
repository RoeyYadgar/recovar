import inspect
from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume.local_search_iteration import (
    _run_local_search_iteration,
    run_local_search_iteration,
)
from recovar.em.dense_single_volume.local_search_types import (
    LocalSearchIterationCorrections,
    LocalSearchIterationDiagnostics,
    LocalSearchIterationExecution,
    LocalSearchIterationGrid,
    LocalSearchIterationInputs,
    LocalSearchIterationOutputs,
    LocalSearchIterationPosterior,
    LocalSearchIterationProjection,
    LocalSearchIterationReconstruction,
    LocalSearchIterationRequest,
    LocalSearchIterationScoring,
)
from recovar.em.dense_single_volume.runtime_options import ExecutionSettings


@pytest.mark.unit
def test_local_search_iteration_request_is_immutable_and_composed():
    request = LocalSearchIterationRequest(
        inputs=LocalSearchIterationInputs("dataset", "mean", "variance", "noise", "disc"),
        grid=LocalSearchIterationGrid(
            "prior_rotations",
            "grid_rotations",
            "grid_eulers",
            3,
            0.1,
            0.2,
            "translations",
            "prior_translations",
            1.5,
            4.0,
        ),
        execution=LocalSearchIterationExecution(5, 7, 40),
    )

    assert request.scoring == LocalSearchIterationScoring()
    assert request.projection == LocalSearchIterationProjection()
    assert request.corrections == LocalSearchIterationCorrections()
    assert request.posterior == LocalSearchIterationPosterior()
    assert request.reconstruction == LocalSearchIterationReconstruction()
    assert request.outputs == LocalSearchIterationOutputs()
    assert request.diagnostics == LocalSearchIterationDiagnostics()
    with pytest.raises(FrozenInstanceError):
        request.execution.image_batch_size = 9


@pytest.mark.unit
def test_local_search_iteration_adapter_maps_every_legacy_parameter():
    settings = ExecutionSettings()
    request = LocalSearchIterationRequest(
        inputs=LocalSearchIterationInputs("dataset", "mean", "variance", "noise", "disc"),
        grid=LocalSearchIterationGrid(
            "prior_rotations",
            "grid_rotations",
            "grid_eulers",
            3,
            0.1,
            0.2,
            "translations",
            "prior_translations",
            1.5,
            4.0,
            translation_prior_reference_translations="reference_translations",
            translation_prior_centers="translation_centers",
            rotation_log_prior="rotation_prior",
            rotation_grid_random_perturbation=0.3,
            rotation_grid_angular_sampling_deg=1.8,
            local_parent_oversampling_order=2,
            pass2_layout="pass2_layout",
            rotation_grid_mstep_rotations="mstep_rotations",
            generate_relion_mstep_rotations=True,
        ),
        execution=LocalSearchIterationExecution(5, 7, 40, 44, settings),
        scoring=LocalSearchIterationScoring(False, True, True, True, 0.75, 19, False, True, True, True),
        projection=LocalSearchIterationProjection(2, 3, True, True, True, None, True, True, "projector", 71),
        corrections=LocalSearchIterationCorrections("images", "scales", "groups", 11, "data_prior", "shifts"),
        posterior=LocalSearchIterationPosterior("log_z", "log_evidence", "class_priors"),
        reconstruction=LocalSearchIterationReconstruction(True, True, True, True, True),
        outputs=LocalSearchIterationOutputs(True, True, True, True, True, True, True),
        diagnostics=LocalSearchIterationDiagnostics(13, "fine"),
    )
    captured = {}

    def fake_legacy_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "result"

    assert run_local_search_iteration(request, legacy_runner=fake_legacy_runner) == "result"

    legacy_signature = inspect.signature(_run_local_search_iteration)
    bound = legacy_signature.bind(*captured["args"], **captured["kwargs"])
    bound.apply_defaults()
    assert set(bound.arguments) == set(legacy_signature.parameters)
    assert bound.arguments["experiment_dataset"] == "dataset"
    assert bound.arguments["rotation_grid_mstep_rotations"] == "mstep_rotations"
    assert bound.arguments["projection_relion_texture_interp"] is None
    assert bound.arguments["source_faithful_spectrum_norm"] is True
    assert bound.arguments["execution_settings"] is settings
    assert request.execution.settings is settings
