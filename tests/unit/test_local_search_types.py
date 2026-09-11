import inspect
from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume.local_em_types import (
    LocalCorrectionInputs,
    LocalEMDiagnostics,
    LocalEMRequestedOutputs,
    LocalProjectionSettings,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)
from recovar.em.dense_single_volume.local_search_iteration import run_local_search_iteration
from recovar.em.dense_single_volume.local_search_types import (
    LocalSearchIterationExecution,
    LocalSearchIterationGrid,
    LocalSearchIterationInputs,
    LocalSearchIterationPosterior,
    LocalSearchIterationRequest,
    LocalSearchIterationResult,
)
from recovar.em.dense_single_volume.runtime_options import ExecutionSettings


@pytest.mark.unit
def test_local_search_iteration_request_is_immutable_and_composed():
    request = LocalSearchIterationRequest(
        inputs=LocalSearchIterationInputs("dataset", "mean", "variance", "noise", "disc"),
        grid=LocalSearchIterationGrid(
            "prior_rotations",
            "grid_rotations",
            3,
            0.1,
            0.2,
            "translations",
            "prior_translations",
            1.5,
        ),
        search=LocalSearchSettings(40),
        execution=LocalSearchIterationExecution(5, 7),
    )

    assert request.scoring == LocalScoringSettings()
    assert request.projection == LocalProjectionSettings()
    assert request.corrections == LocalCorrectionInputs()
    assert request.posterior == LocalSearchIterationPosterior()
    assert request.reconstruction == LocalReconstructionSettings()
    assert request.outputs == LocalEMRequestedOutputs()
    assert request.diagnostics == LocalEMDiagnostics()
    with pytest.raises(FrozenInstanceError):
        request.execution.image_batch_size = 9


@pytest.mark.unit
def test_local_search_iteration_uses_typed_request_and_result_contract():
    settings = ExecutionSettings()
    request = LocalSearchIterationRequest(
        inputs=LocalSearchIterationInputs("dataset", "mean", "variance", "noise", "disc", "projector", 71),
        grid=LocalSearchIterationGrid(
            "prior_rotations",
            "grid_rotations",
            3,
            0.1,
            0.2,
            "translations",
            "prior_translations",
            1.5,
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
        search=LocalSearchSettings(
            current_size=40,
            reconstruction_current_size=44,
            adaptive_fraction=0.75,
            max_significants=19,
            reconstruct_significant_only=False,
            apply_max_significants_to_support=True,
        ),
        execution=LocalSearchIterationExecution(5, 7, settings),
        scoring=LocalScoringSettings(
            score_with_masked_images=False,
            half_spectrum_scoring=True,
            relion_exact_score_translation=True,
            use_float64_scoring=True,
        ),
        projection=LocalProjectionSettings(2, 3, True, None, False, True, True, True),
        corrections=LocalCorrectionInputs("images", "scales", "groups", 11, "data_prior", "shifts"),
        posterior=LocalSearchIterationPosterior("log_z", "log_evidence", "class_priors"),
        reconstruction=LocalReconstructionSettings(
            mstep_subtract_ctf_projection=True,
            mstep_relion_x_half=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            stats_use_reconstruction_probs=True,
            source_faithful_spectrum_norm=True,
            score_only=True,
        ),
        outputs=LocalEMRequestedOutputs(True, True, True, True, True, True, True, True),
        diagnostics=LocalEMDiagnostics(13, "fine"),
    )
    assert tuple(inspect.signature(run_local_search_iteration).parameters) == ("request",)
    assert request.execution.settings is settings

    result = LocalSearchIterationResult("Ft_y", "Ft_ctf", "assignment", "stats")
    assert result.Ft_y == "Ft_y"
    with pytest.raises(FrozenInstanceError):
        result.Ft_y = "changed"
