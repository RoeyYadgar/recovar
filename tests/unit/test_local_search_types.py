import inspect
from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume.local_search_iteration import run_local_search_iteration
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
    LocalSearchIterationResult,
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
def test_local_search_iteration_uses_typed_request_and_result_contract():
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
    assert tuple(inspect.signature(run_local_search_iteration).parameters) == ("request",)
    assert request.execution.settings is settings

    result = LocalSearchIterationResult("Ft_y", "Ft_ctf", "assignment", "stats")
    assert result.Ft_y == "Ft_y"
    with pytest.raises(FrozenInstanceError):
        result.Ft_y = "changed"
