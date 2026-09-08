import inspect
import itertools
from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume.dense_em_types import (
    DenseCorrectionInputs,
    DenseEMInputs,
    DenseEMOutputSpec,
    DenseEMRequest,
    DenseEMRequestedOutputs,
    DenseEMResult,
    DenseExecutionSettings,
    DensePosteriorInputs,
    DenseProjectionSettings,
    DenseReconstructionSettings,
    DenseScoringSettings,
    DenseSearchSettings,
)
from recovar.em.dense_single_volume.em_engine import run_dense_em, run_em


@pytest.mark.unit
@pytest.mark.parametrize(
    ("return_stats", "accumulate_noise", "return_profile"),
    itertools.product((False, True), repeat=3),
)
def test_dense_em_result_round_trips_every_legacy_tuple_shape(
    return_stats,
    accumulate_noise,
    return_profile,
):
    output_spec = DenseEMOutputSpec(
        return_stats=return_stats,
        accumulate_noise=accumulate_noise,
        return_profile=return_profile,
    )
    result = DenseEMResult(
        new_mean="new_mean",
        hard_assignment="hard_assignment",
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        relion_stats="relion_stats",
        noise_stats="noise_stats",
        profile_stats="profile_stats",
    )

    legacy_output = result.to_legacy_tuple(output_spec)

    assert len(legacy_output) == output_spec.legacy_tuple_size
    assert DenseEMResult.from_legacy_tuple(legacy_output, output_spec) == DenseEMResult(
        new_mean="new_mean",
        hard_assignment="hard_assignment",
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        relion_stats="relion_stats" if return_stats else None,
        noise_stats="noise_stats" if accumulate_noise else None,
        profile_stats="profile_stats" if return_profile else None,
    )


@pytest.mark.unit
def test_dense_em_result_rejects_legacy_tuple_with_wrong_shape():
    output_spec = DenseEMOutputSpec(return_profile=True)

    with pytest.raises(ValueError, match="expected 5 values, received 4"):
        DenseEMResult.from_legacy_tuple((1, 2, 3, 4), output_spec)


@pytest.mark.unit
def test_dense_em_request_composes_immutable_default_groups():
    request = DenseEMRequest(
        inputs=DenseEMInputs(
            "dataset",
            "mean",
            "mean_variance",
            "noise_variance",
            "rotations",
            "translations",
            "disc_type",
        ),
    )

    assert request.search == DenseSearchSettings()
    assert request.execution == DenseExecutionSettings()
    assert request.scoring == DenseScoringSettings()
    assert request.projection == DenseProjectionSettings()
    assert request.corrections == DenseCorrectionInputs()
    assert request.posterior == DensePosteriorInputs()
    assert request.reconstruction == DenseReconstructionSettings()
    assert request.outputs == DenseEMRequestedOutputs()
    with pytest.raises(FrozenInstanceError):
        request.execution.image_batch_size = 9


@pytest.mark.unit
def test_dense_requested_outputs_define_only_legacy_tuple_suffixes():
    outputs = DenseEMRequestedOutputs(
        return_stats=True,
        accumulate_noise=True,
        return_profile=True,
        return_half_volume_accumulators=True,
    )

    assert outputs.legacy_tuple_spec == DenseEMOutputSpec(
        return_stats=True,
        accumulate_noise=True,
        return_profile=True,
    )
    assert outputs.legacy_tuple_spec.legacy_tuple_size == 7


@pytest.mark.unit
def test_dense_em_adapter_forwards_every_legacy_engine_parameter():
    request = DenseEMRequest(
        inputs=DenseEMInputs(
            "dataset",
            "mean",
            "mean_variance",
            "noise_variance",
            "rotations",
            "translations",
            "disc_type",
        ),
        search=DenseSearchSettings(
            current_size=40,
            rotation_log_prior="rotation_log_prior",
            translation_log_prior="translation_log_prior",
            image_indices="image_indices",
            rotation_translation_mask="rotation_translation_mask",
        ),
        execution=DenseExecutionSettings(
            image_batch_size=5,
            rotation_block_size=7,
            sparse_pass2=False,
        ),
        scoring=DenseScoringSettings(
            score_with_masked_images=True,
            half_spectrum_scoring=True,
            relion_firstiter_score_mode="normalized_cc",
            relion_firstiter_winner_take_all=True,
            use_float64_scoring=True,
        ),
        projection=DenseProjectionSettings(
            projection_padding_factor=2,
            reconstruction_padding_factor=3,
            use_float64_projections=True,
            do_gridding_correction=True,
            square_window=True,
        ),
        corrections=DenseCorrectionInputs(
            image_corrections="image_corrections",
            scale_corrections="scale_corrections",
            image_pre_shifts="image_pre_shifts",
        ),
        posterior=DensePosteriorInputs(
            class_log_prior=-1.25,
            normalization_log_evidence="normalization_log_evidence",
            translation_prior_centers="translation_prior_centers",
        ),
        reconstruction=DenseReconstructionSettings(
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            score_only=True,
            relion_half_volume_mstep=True,
        ),
        outputs=DenseEMRequestedOutputs(
            return_stats=True,
            accumulate_noise=True,
            return_profile=True,
            return_half_volume_accumulators=True,
        ),
    )
    expected_result = DenseEMResult(
        "new_mean",
        "hard_assignment",
        "Ft_y",
        "Ft_ctf",
        relion_stats="relion_stats",
        noise_stats="noise_stats",
        profile_stats="profile_stats",
    )
    captured = {}

    def fake_legacy_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected_result.to_legacy_tuple(request.outputs.legacy_tuple_spec)

    result = run_dense_em(request, legacy_runner=fake_legacy_runner)

    expected_kwargs = {}
    for group in (
        request.search,
        request.execution,
        request.scoring,
        request.projection,
        request.corrections,
        request.posterior,
        request.reconstruction,
        request.outputs,
    ):
        expected_kwargs.update(vars(group))

    assert result == expected_result
    assert captured["args"] == (
        "dataset",
        "mean",
        "mean_variance",
        "noise_variance",
        "rotations",
        "translations",
        "disc_type",
    )
    assert captured["kwargs"] == expected_kwargs
    assert set(expected_kwargs) == set(list(inspect.signature(run_em).parameters)[7:])
