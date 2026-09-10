from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.local_em_planning import (
    LocalEMInputPlan,
    LocalEMModePlan,
    plan_local_em_inputs,
    plan_local_em_modes,
)
from recovar.em.dense_single_volume.local_em_types import (
    LocalCorrectionInputs,
    LocalEMRequestedOutputs,
    LocalPosteriorInputs,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)


def _layout(n_images=3, n_dims=2):
    return SimpleNamespace(
        n_images=n_images,
        translation_grid=np.zeros((4, n_dims), dtype=np.float32),
    )


def test_local_mode_plan_normalizes_default_execution():
    plan = plan_local_em_modes(
        scoring=LocalScoringSettings(),
        reconstruction=LocalReconstructionSettings(),
        outputs=LocalEMRequestedOutputs(),
    )

    assert plan == LocalEMModePlan(
        score_only=False,
        accumulate_noise=False,
        return_half_volume_accumulators=False,
        return_profile=False,
        mstep_subtract_ctf_projection=False,
        mstep_relion_x_half=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        relion_exact_score_translation=False,
        include_unweighted_norm_high_shell=True,
        source_faithful_spectrum_norm=False,
    )
    with pytest.raises(FrozenInstanceError):
        plan.score_only = True


@pytest.mark.parametrize(
    "outputs",
    [
        LocalEMRequestedOutputs(return_profile=True),
        LocalEMRequestedOutputs(return_reconstruction_probability_values=True),
        LocalEMRequestedOutputs(return_reconstruction_sample_indices=True),
    ],
)
def test_local_mode_plan_resolves_implicit_profile_capture(outputs):
    plan = plan_local_em_modes(
        scoring=LocalScoringSettings(),
        reconstruction=LocalReconstructionSettings(),
        outputs=outputs,
    )

    assert plan.return_profile is True


def test_local_mode_plan_requires_half_spectrum_for_exact_translation():
    with pytest.raises(ValueError, match="exact RELION score translation requires half_spectrum_scoring=True"):
        plan_local_em_modes(
            scoring=LocalScoringSettings(relion_exact_score_translation=True),
            reconstruction=LocalReconstructionSettings(),
            outputs=LocalEMRequestedOutputs(),
        )


@pytest.mark.parametrize(
    ("reconstruction", "outputs", "message"),
    [
        (
            LocalReconstructionSettings(score_only=True),
            LocalEMRequestedOutputs(),
            "requires both adjoints disabled",
        ),
        (
            LocalReconstructionSettings(score_only=True, disable_adjoint_y=True, disable_adjoint_ctf=True),
            LocalEMRequestedOutputs(accumulate_noise=True),
            "does not support noise accumulation",
        ),
        (
            LocalReconstructionSettings(
                score_only=True,
                disable_adjoint_y=True,
                disable_adjoint_ctf=True,
                mstep_subtract_ctf_projection=True,
            ),
            LocalEMRequestedOutputs(),
            "does not support residual M-step subtraction",
        ),
        (
            LocalReconstructionSettings(score_only=True, disable_adjoint_y=True, disable_adjoint_ctf=True),
            LocalEMRequestedOutputs(return_half_volume_accumulators=True),
            "does not return half-volume accumulators",
        ),
    ],
)
def test_local_mode_plan_rejects_invalid_score_only_combinations(reconstruction, outputs, message):
    with pytest.raises(ValueError, match=message):
        plan_local_em_modes(
            scoring=LocalScoringSettings(),
            reconstruction=reconstruction,
            outputs=outputs,
        )


def test_local_mode_plan_preserves_valid_score_only_and_xhalf_modes():
    score_only = plan_local_em_modes(
        scoring=LocalScoringSettings(half_spectrum_scoring=True, relion_exact_score_translation=True),
        reconstruction=LocalReconstructionSettings(
            score_only=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
        ),
        outputs=LocalEMRequestedOutputs(return_best_pose_details=True),
    )
    xhalf = plan_local_em_modes(
        scoring=LocalScoringSettings(),
        reconstruction=LocalReconstructionSettings(
            mstep_relion_x_half=True,
            mstep_subtract_ctf_projection=True,
        ),
        outputs=LocalEMRequestedOutputs(accumulate_noise=True),
    )

    assert score_only.score_only
    assert score_only.relion_exact_score_translation
    assert score_only.disable_adjoint_y and score_only.disable_adjoint_ctf
    assert xhalf.mstep_relion_x_half
    assert xhalf.mstep_subtract_ctf_projection
    assert xhalf.accumulate_noise


def test_local_input_plan_validates_and_preserves_host_inputs():
    group_ids = np.array([0, 2, 1], dtype=np.int32)
    log_z = np.array([1, 2, 3], dtype=np.float32)
    threshold = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    centers = np.array([1.0, -2.0], dtype=np.float32)

    plan = plan_local_em_inputs(
        local_layout=_layout(),
        corrections=LocalCorrectionInputs(
            group_ids=group_ids,
            scale_correction_group_count=5,
        ),
        posterior=LocalPosteriorInputs(
            normalization_log_z=log_z,
            class_log_prior=-1.25,
            translation_prior_centers=centers,
        ),
        search=LocalSearchSettings(
            current_size=40,
            reconstruction_probability_threshold=threshold,
        ),
    )

    assert isinstance(plan, LocalEMInputPlan)
    assert plan.n_images == 3
    assert plan.class_log_prior == -1.25
    assert plan.n_scale_groups == 5
    np.testing.assert_array_equal(plan.group_ids, group_ids)
    assert plan.group_ids.dtype == np.int64
    np.testing.assert_array_equal(plan.normalization_log_z, log_z)
    assert plan.normalization_log_z.dtype == np.float64
    assert plan.normalization_log_evidence is None
    np.testing.assert_allclose(plan.reconstruction_probability_threshold, threshold)
    assert plan.reconstruction_probability_threshold.dtype == np.float64
    assert plan.translation_prior_centers is centers


@pytest.mark.parametrize("value", [-1, 1.5])
def test_local_input_plan_rejects_invalid_explicit_scale_group_count(value):
    with pytest.raises(ValueError, match="scale_correction_group_count must be a non-negative integer"):
        plan_local_em_inputs(
            local_layout=_layout(),
            corrections=LocalCorrectionInputs(scale_correction_group_count=value),
            posterior=LocalPosteriorInputs(),
            search=LocalSearchSettings(current_size=40),
        )


def test_local_input_plan_preserves_infinite_scale_group_conversion_error():
    with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
        plan_local_em_inputs(
            local_layout=_layout(),
            corrections=LocalCorrectionInputs(scale_correction_group_count=np.inf),
            posterior=LocalPosteriorInputs(),
            search=LocalSearchSettings(current_size=40),
        )


@pytest.mark.parametrize(
    ("group_ids", "message"),
    [
        ([0, 1], r"group_ids must have shape \(3,\)"),
        ([0, -1, 1], "group_ids must be non-negative"),
    ],
)
def test_local_input_plan_rejects_invalid_group_ids(group_ids, message):
    with pytest.raises(ValueError, match=message):
        plan_local_em_inputs(
            local_layout=_layout(),
            corrections=LocalCorrectionInputs(group_ids=group_ids),
            posterior=LocalPosteriorInputs(),
            search=LocalSearchSettings(current_size=40),
        )


def test_local_input_plan_rejects_both_external_normalizers():
    with pytest.raises(ValueError, match="Provide only one of normalization_log_z or normalization_log_evidence"):
        plan_local_em_inputs(
            local_layout=_layout(),
            corrections=LocalCorrectionInputs(),
            posterior=LocalPosteriorInputs(
                normalization_log_z=np.zeros(3),
                normalization_log_evidence=np.zeros(3),
            ),
            search=LocalSearchSettings(current_size=40),
        )


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        ([0.1, 0.2], r"must have shape \(3,\)"),
        ([0.1, np.nan, 0.2], "must be finite"),
        ([0.1, -0.1, 0.2], "must be non-negative"),
    ],
)
def test_local_input_plan_rejects_invalid_reconstruction_threshold(threshold, message):
    with pytest.raises(ValueError, match=message):
        plan_local_em_inputs(
            local_layout=_layout(),
            corrections=LocalCorrectionInputs(),
            posterior=LocalPosteriorInputs(),
            search=LocalSearchSettings(
                current_size=40,
                reconstruction_probability_threshold=threshold,
            ),
        )


def test_local_input_plan_preserves_translation_center_dtype_and_shape_validation():
    centers = np.arange(6, dtype=np.float64).reshape(3, 2)
    plan = plan_local_em_inputs(
        local_layout=_layout(),
        corrections=LocalCorrectionInputs(),
        posterior=LocalPosteriorInputs(translation_prior_centers=centers),
        search=LocalSearchSettings(current_size=40),
    )

    assert plan.translation_prior_centers is centers
    with pytest.raises(ValueError, match="when image-specific"):
        plan_local_em_inputs(
            local_layout=_layout(),
            corrections=LocalCorrectionInputs(),
            posterior=LocalPosteriorInputs(translation_prior_centers=np.zeros((2, 2))),
            search=LocalSearchSettings(current_size=40),
        )
