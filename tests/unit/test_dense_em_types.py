import inspect
import itertools
from dataclasses import FrozenInstanceError

import pytest
import recovar.em.dense_single_volume.em_engine as em_engine

from recovar.em.dense_single_volume.dense_em_types import (
    DenseCorrectionInputs,
    DenseEMInputs,
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
from recovar.em.dense_single_volume.em_engine import (
    make_dense_em_request,
    run_em,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("return_stats", "accumulate_noise", "return_profile"),
    itertools.product((False, True), repeat=3),
)
def test_dense_em_result_serializes_every_legacy_tuple_shape(
    return_stats,
    accumulate_noise,
    return_profile,
):
    outputs = DenseEMRequestedOutputs(
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

    legacy_output = result.to_legacy_tuple(outputs)

    expected = ["new_mean", "hard_assignment", "Ft_y", "Ft_ctf"]
    if return_stats:
        expected.append("relion_stats")
    if accumulate_noise:
        expected.append("noise_stats")
    if return_profile:
        expected.append("profile_stats")
    assert legacy_output == tuple(expected)


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
def test_dense_requested_outputs_serialize_only_requested_suffixes():
    outputs = DenseEMRequestedOutputs(
        return_stats=True,
        accumulate_noise=True,
        return_profile=True,
        return_half_volume_accumulators=True,
    )

    result = DenseEMResult(1, 2, 3, 4, 5, 6, 7)
    assert result.to_legacy_tuple(outputs) == (1, 2, 3, 4, 5, 6, 7)


@pytest.mark.unit
def test_dense_em_compatibility_facade_groups_every_legacy_engine_parameter(monkeypatch):
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

    captured = {}

    def fake_run_dense_em(actual_request):
        captured["request"] = actual_request
        return expected_result

    monkeypatch.setattr(em_engine, "run_dense_em", fake_run_dense_em)
    legacy_output = run_em(
        "dataset",
        "mean",
        "mean_variance",
        "noise_variance",
        "rotations",
        "translations",
        "disc_type",
        **expected_kwargs,
    )

    assert legacy_output == expected_result.to_legacy_tuple(request.outputs)
    assert captured["request"] == request
    assert set(expected_kwargs) == set(list(inspect.signature(run_em).parameters)[7:])
    assert make_dense_em_request(request.inputs, expected_kwargs) == request
