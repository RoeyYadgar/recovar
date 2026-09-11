import inspect
import itertools
from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume import local_em_engine
from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact
from recovar.em.dense_single_volume.local_em_types import (
    LocalCorrectionInputs,
    LocalEMDiagnostics,
    LocalEMInputs,
    LocalEMRequest,
    LocalEMRequestedOutputs,
    LocalEMResult,
    LocalExecutionSettings,
    LocalPosteriorInputs,
    LocalProjectionSettings,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)
from recovar.em.dense_single_volume.runtime_options import LocalCacheSettings


@pytest.mark.unit
@pytest.mark.parametrize(
    ("accumulate_noise", "return_profile", "return_best_pose_details", "return_significant_counts"),
    itertools.product((False, True), repeat=4),
)
def test_local_em_result_serializes_every_legacy_tuple_shape(
    accumulate_noise,
    return_profile,
    return_best_pose_details,
    return_significant_counts,
):
    outputs = LocalEMRequestedOutputs(
        accumulate_noise=accumulate_noise,
        return_profile=return_profile,
        return_best_pose_details=return_best_pose_details,
        return_significant_counts=return_significant_counts,
    )
    result = LocalEMResult(
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        hard_assignment="hard_assignment",
        relion_stats="relion_stats",
        best_pose_rotations="best_pose_rotations",
        best_pose_translations="best_pose_translations",
        best_pose_rotation_ids="best_pose_rotation_ids",
        noise_stats="noise_stats",
        profile_summary={"profile": "summary"},
        significant_counts="significant_counts",
    )

    legacy_output = result.to_legacy_tuple(outputs)

    expected = ["Ft_y", "Ft_ctf", "hard_assignment"]
    if return_best_pose_details:
        expected.extend(["best_pose_rotations", "best_pose_translations", "best_pose_rotation_ids"])
    expected.append("relion_stats")
    if accumulate_noise:
        expected.append("noise_stats")
    if return_profile:
        expected.append({"profile": "summary"})
    if return_significant_counts:
        expected.append("significant_counts")
    assert legacy_output == tuple(expected)


@pytest.mark.unit
def test_local_em_request_composes_immutable_default_groups():
    request = LocalEMRequest(
        inputs=LocalEMInputs("dataset", "mean", "mean_variance", "noise_variance", "layout", "disc_type"),
        search=LocalSearchSettings(current_size=40),
        execution=LocalExecutionSettings(image_batch_size=5, rotation_block_size=7),
    )

    assert request.scoring == LocalScoringSettings()
    assert request.projection == LocalProjectionSettings()
    assert request.corrections == LocalCorrectionInputs()
    assert request.posterior == LocalPosteriorInputs()
    assert request.reconstruction == LocalReconstructionSettings()
    assert request.outputs == LocalEMRequestedOutputs()
    assert request.diagnostics == LocalEMDiagnostics()
    with pytest.raises(FrozenInstanceError):
        request.execution.image_batch_size = 9


@pytest.mark.unit
@pytest.mark.parametrize(
    ("probability_values", "sample_indices"),
    [(True, False), (False, True)],
)
def test_local_em_requested_captures_enable_legacy_profile_result(probability_values, sample_indices):
    outputs = LocalEMRequestedOutputs(
        return_reconstruction_probability_values=probability_values,
        return_reconstruction_sample_indices=sample_indices,
    )

    result = LocalEMResult(1, 2, 3, 4, profile_summary={"profile": "summary"})

    assert outputs.return_profile is False
    assert result.to_legacy_tuple(outputs)[-1] == {"profile": "summary"}


@pytest.mark.unit
def test_legacy_local_em_facade_builds_the_canonical_request(monkeypatch):
    request = LocalEMRequest(
        inputs=LocalEMInputs(
            "dataset",
            "mean",
            "mean_variance",
            "noise_variance",
            "local_layout",
            "disc_type",
            relion_projector_half="relion_projector_half",
            relion_projector_r_max=71,
        ),
        search=LocalSearchSettings(
            current_size=40,
            reconstruction_current_size=44,
            reconstruct_significant_only=True,
            adaptive_fraction=0.75,
            max_significants=19,
            reconstruction_probability_threshold="reconstruction_probability_threshold",
        ),
        execution=LocalExecutionSettings(
            image_batch_size=5,
            rotation_block_size=7,
            max_hypotheses_per_microbatch=23,
            unify_local_bucket_sizes=True,
            cache=LocalCacheSettings(
                raw_image_max_gb=2.0,
                processed_half_max_gb=3.0,
                sparse_big_jit_mstep_max_gb=4.0,
            ),
        ),
        scoring=LocalScoringSettings(
            score_with_masked_images=False,
            half_spectrum_scoring=True,
            relion_exact_score_translation=True,
            use_float64_scoring=True,
            use_float64_normalization=False,
        ),
        projection=LocalProjectionSettings(
            projection_padding_factor=2,
            reconstruction_padding_factor=3,
            use_float64_projections=True,
            relion_texture_interp=True,
            relion_acc_double_floorf_quirk=False,
            force_jax=True,
            do_gridding_correction=False,
            square_window=True,
        ),
        corrections=LocalCorrectionInputs(
            image_corrections="image_corrections",
            scale_corrections="scale_corrections",
            group_ids="group_ids",
            scale_correction_group_count=11,
            scale_correction_data_vs_prior="scale_correction_data_vs_prior",
            image_pre_shifts="image_pre_shifts",
        ),
        posterior=LocalPosteriorInputs(
            normalization_log_z="normalization_log_z",
            class_log_prior=-1.25,
            normalization_log_evidence="normalization_log_evidence",
            translation_prior_centers="translation_prior_centers",
        ),
        reconstruction=LocalReconstructionSettings(
            mstep_subtract_ctf_projection=True,
            mstep_relion_x_half=False,
            disable_adjoint_y=True,
            disable_adjoint_ctf=False,
            stats_use_reconstruction_probs=True,
            include_unweighted_norm_high_shell=False,
            source_faithful_spectrum_norm=True,
            score_only=False,
        ),
        outputs=LocalEMRequestedOutputs(
            accumulate_noise=True,
            return_half_volume_accumulators=False,
            return_profile=False,
            return_best_pose_details=True,
            return_reconstruction_probability_values=True,
            return_reconstruction_sample_indices=False,
            return_significant_counts=True,
        ),
        diagnostics=LocalEMDiagnostics(iteration=13, pass_label="fine"),
    )
    expected_result = LocalEMResult(
        "Ft_y",
        "Ft_ctf",
        "hard_assignment",
        "relion_stats",
        best_pose_rotations="best_pose_rotations",
        best_pose_translations="best_pose_translations",
        best_pose_rotation_ids="best_pose_rotation_ids",
        noise_stats="noise_stats",
        profile_summary={"profile": "summary"},
        significant_counts="significant_counts",
    )
    expected_kwargs = {}
    for group in (
        request.search,
        request.scoring,
        request.corrections,
        request.posterior,
        request.reconstruction,
        request.outputs,
    ):
        expected_kwargs.update(vars(group))
    expected_kwargs.update(
        image_batch_size=request.execution.image_batch_size,
        rotation_block_size=request.execution.rotation_block_size,
        max_hypotheses_per_microbatch=request.execution.max_hypotheses_per_microbatch,
        unify_local_bucket_sizes=request.execution.unify_local_bucket_sizes,
        cache_settings=request.execution.cache,
        projection_padding_factor=request.projection.projection_padding_factor,
        reconstruction_padding_factor=request.projection.reconstruction_padding_factor,
        use_float64_projections=request.projection.use_float64_projections,
        projection_relion_texture_interp=request.projection.relion_texture_interp,
        projection_relion_acc_double_floorf_quirk=request.projection.relion_acc_double_floorf_quirk,
        projection_force_jax=request.projection.force_jax,
        do_gridding_correction=request.projection.do_gridding_correction,
        square_window=request.projection.square_window,
        relion_projector_half=request.inputs.relion_projector_half,
        relion_projector_r_max=request.inputs.relion_projector_r_max,
        debug_iteration=request.diagnostics.iteration,
        debug_pass_label=request.diagnostics.pass_label,
    )

    captured = {}

    def fake_typed_runner(actual_request):
        captured["request"] = actual_request
        return expected_result

    monkeypatch.setattr(local_em_engine, "run_local_em", fake_typed_runner)
    legacy_result = run_local_em_exact(
        "dataset",
        "mean",
        "mean_variance",
        "noise_variance",
        "local_layout",
        "disc_type",
        **expected_kwargs,
    )

    assert captured["request"] == request
    assert legacy_result == expected_result.to_legacy_tuple(request.outputs)
    assert set(expected_kwargs) == set(list(inspect.signature(run_local_em_exact).parameters)[6:])
