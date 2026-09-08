import inspect
import itertools
from dataclasses import FrozenInstanceError

import pytest

from recovar.em.dense_single_volume.local_em_engine import run_local_em, run_local_em_exact
from recovar.em.dense_single_volume.local_em_types import (
    ExecutionSettings,
    LocalCorrectionInputs,
    LocalEMDiagnostics,
    LocalEMInputs,
    LocalEMOutputSpec,
    LocalEMRequest,
    LocalEMRequestedOutputs,
    LocalEMResult,
    LocalPosteriorInputs,
    LocalProjectionSettings,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("accumulate_noise", "return_profile", "return_best_pose_details", "return_significant_counts"),
    itertools.product((False, True), repeat=4),
)
def test_local_em_result_round_trips_every_legacy_tuple_shape(
    accumulate_noise,
    return_profile,
    return_best_pose_details,
    return_significant_counts,
):
    output_spec = LocalEMOutputSpec(
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

    legacy_output = result.to_legacy_tuple(output_spec)

    assert len(legacy_output) == output_spec.legacy_tuple_size
    assert LocalEMResult.from_legacy_tuple(legacy_output, output_spec) == LocalEMResult(
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        hard_assignment="hard_assignment",
        relion_stats="relion_stats",
        best_pose_rotations="best_pose_rotations" if return_best_pose_details else None,
        best_pose_translations="best_pose_translations" if return_best_pose_details else None,
        best_pose_rotation_ids="best_pose_rotation_ids" if return_best_pose_details else None,
        noise_stats="noise_stats" if accumulate_noise else None,
        profile_summary={"profile": "summary"} if return_profile else None,
        significant_counts="significant_counts" if return_significant_counts else None,
    )


@pytest.mark.unit
def test_local_em_result_rejects_legacy_tuple_with_wrong_shape():
    output_spec = LocalEMOutputSpec(return_profile=True)

    with pytest.raises(ValueError, match="expected 5 values, received 4"):
        LocalEMResult.from_legacy_tuple((1, 2, 3, 4), output_spec)


@pytest.mark.unit
def test_local_em_request_composes_immutable_default_groups():
    request = LocalEMRequest(
        inputs=LocalEMInputs("dataset", "mean", "mean_variance", "noise_variance", "layout", "disc_type"),
        search=LocalSearchSettings(current_size=40),
        execution=ExecutionSettings(image_batch_size=5, rotation_block_size=7),
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

    assert outputs.return_profile is False
    assert outputs.legacy_tuple_spec.return_profile is True


@pytest.mark.unit
def test_local_em_adapters_round_trip_every_exact_engine_parameter():
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
        execution=ExecutionSettings(
            image_batch_size=5,
            rotation_block_size=7,
            max_hypotheses_per_microbatch=23,
            unify_local_bucket_sizes=True,
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
    captured = {}

    def fake_legacy_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected_result.to_legacy_tuple(request.outputs.legacy_tuple_spec)

    result = run_local_em(request, legacy_runner=fake_legacy_runner)

    expected_kwargs = {}
    for group in (
        request.search,
        request.execution,
        request.scoring,
        request.corrections,
        request.posterior,
        request.reconstruction,
        request.outputs,
    ):
        expected_kwargs.update(vars(group))
    expected_kwargs.update(
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

    assert result == expected_result
    assert captured["args"] == (
        "dataset",
        "mean",
        "mean_variance",
        "noise_variance",
        "local_layout",
        "disc_type",
    )
    assert captured["kwargs"] == expected_kwargs
    assert set(expected_kwargs) == set(list(inspect.signature(run_local_em_exact).parameters)[6:])

    from recovar.em.dense_single_volume.k_class import _local_em_request_from_legacy_kwargs

    round_trip_request = _local_em_request_from_legacy_kwargs(request.inputs, captured["kwargs"])
    round_trip_capture = {}

    def round_trip_runner(*args, **kwargs):
        round_trip_capture["args"] = args
        round_trip_capture["kwargs"] = kwargs
        return expected_result.to_legacy_tuple(round_trip_request.outputs.legacy_tuple_spec)

    assert run_local_em(round_trip_request, legacy_runner=round_trip_runner) == expected_result
    assert round_trip_capture == captured
