from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume.local_em_planning import (
    LocalEMModePlan,
    plan_local_em_modes,
)
from recovar.em.dense_single_volume.local_em_types import (
    LocalEMRequestedOutputs,
    LocalReconstructionSettings,
    LocalScoringSettings,
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
