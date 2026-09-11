"""Exact local-search iteration orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import numpy as np

from recovar.em.dense_single_volume.helpers.local_search import (
    _local_search_engine_rotation_block_size,
)
from recovar.em.dense_single_volume.k_class import run_local_k_class_em
from recovar.em.dense_single_volume.local_em_engine import run_local_em
from recovar.em.dense_single_volume.local_em_types import (
    LocalEMInputs,
    LocalEMRequest,
    LocalEMRequestedOutputs,
    LocalExecutionSettings,
    LocalPosteriorInputs,
    LocalProjectionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)
from recovar.em.dense_single_volume.local_search_types import LocalSearchIterationRequest, LocalSearchIterationResult
from recovar.em.dense_single_volume.runtime_options import current_environment as _runtime_environment
from recovar.em.sampling import build_local_search_grid_metadata

logger = logging.getLogger(__name__)


# Mirror iteration_loop's constant locally so the helper has a stable home.
EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS = 3_000_000
EXACT_LOCAL_XHALF_BATCH_GUARD_ENV = "RECOVAR_LOCAL_XHALF_BATCH_GUARD"


def _precompute_exact_local_fine_grid_enabled(healpix_order: int) -> bool:
    """Return whether exact local search should materialize the fine grid once."""
    from recovar.em.sampling import rotation_grid_size

    return rotation_grid_size(int(healpix_order)) <= EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS


def run_local_search_iteration(request: LocalSearchIterationRequest) -> LocalSearchIterationResult:
    """Run exact local search over image-specific rotation neighborhoods.

    ``debug_pass_label`` is diagnostic-only and forwarded verbatim to
    ``run_local_em``: pass a distinct label per call site whenever a
    caller invokes this function more than once for the same image at the
    same ``current_size``/``debug_iteration`` (e.g. local search's pass-1
    "parent" probe vs. its pass-2 fine call), or the later call's
    ``RECOVAR_LOCAL_SCORE_DUMP_*`` output silently overwrites the earlier
    one at the same path.
    """
    inputs = request.inputs
    grid = request.grid
    execution = request.execution
    scoring = request.scoring
    projection = request.projection
    corrections = request.corrections
    posterior = request.posterior
    reconstruction = request.reconstruction
    outputs = request.outputs
    diagnostics = request.diagnostics

    experiment_dataset = inputs.experiment_dataset
    mean = inputs.mean
    mean_variance = inputs.mean_variance
    noise_variance = inputs.noise_variance
    disc_type = inputs.disc_type
    prior_rotations = grid.prior_rotations
    rotation_grid_rotations = grid.rotation_grid_rotations
    rotation_grid_eulers = grid.rotation_grid_eulers
    healpix_order = grid.healpix_order
    sigma_rot = grid.sigma_rot
    sigma_psi = grid.sigma_psi
    translations = grid.translations
    prior_translations = grid.prior_translations
    sigma_offset_angstrom = grid.sigma_offset_angstrom
    offset_range_pixels = grid.offset_range_pixels
    translation_prior_reference_translations = grid.translation_prior_reference_translations
    translation_prior_centers = grid.translation_prior_centers
    rotation_log_prior = grid.rotation_log_prior
    rotation_grid_random_perturbation = grid.rotation_grid_random_perturbation
    rotation_grid_angular_sampling_deg = grid.rotation_grid_angular_sampling_deg
    local_parent_oversampling_order = grid.local_parent_oversampling_order
    pass2_layout = grid.pass2_layout
    rotation_grid_mstep_rotations = grid.rotation_grid_mstep_rotations
    generate_relion_mstep_rotations = grid.generate_relion_mstep_rotations
    image_batch_size = execution.image_batch_size
    rotation_block_size = execution.rotation_block_size
    current_size = execution.current_size
    reconstruction_current_size = execution.reconstruction_current_size
    execution_settings = execution.settings
    score_with_masked_images = scoring.score_with_masked_images
    half_spectrum_scoring = scoring.half_spectrum_scoring
    relion_exact_score_translation = scoring.relion_exact_score_translation
    use_float64_scoring = scoring.use_float64_scoring
    adaptive_fraction = scoring.adaptive_fraction
    max_significants = scoring.max_significants
    reconstruct_significant_only = scoring.reconstruct_significant_only
    apply_max_significants_to_support = scoring.apply_max_significants_to_support
    source_faithful_spectrum_norm = reconstruction.source_faithful_spectrum_norm
    projection_padding_factor = projection.projection_padding_factor
    reconstruction_padding_factor = projection.reconstruction_padding_factor
    use_float64_projections = projection.use_float64_projections
    do_gridding_correction = projection.do_gridding_correction
    square_window = projection.square_window
    projection_relion_texture_interp = projection.relion_texture_interp
    projection_relion_acc_double_floorf_quirk = projection.relion_acc_double_floorf_quirk
    projection_force_jax = projection.force_jax
    relion_projector_half = projection.relion_projector_half
    relion_projector_r_max = projection.relion_projector_r_max
    normalization_log_z = posterior.normalization_log_z
    normalization_log_evidence = posterior.normalization_log_evidence
    class_log_priors = posterior.class_log_priors
    mstep_relion_x_half = reconstruction.mstep_relion_x_half
    score_only = reconstruction.score_only
    accumulate_noise = outputs.accumulate_noise
    return_half_volume_accumulators = outputs.return_half_volume_accumulators
    return_profile = outputs.return_profile
    return_best_pose_details = outputs.return_best_pose_details
    return_class_details = outputs.return_class_details
    return_reconstruction_sample_indices = outputs.return_reconstruction_sample_indices
    return_significant_counts = outputs.return_significant_counts

    # Indirection through the iteration_loop module so test monkeypatches that
    # target ``iteration_loop.build_local_hypothesis_layout`` and
    # ``iteration_loop._estimate_relion_em_batch_sizes`` continue to win at the call
    # site even though this function lives in a sibling module.
    from recovar.em.dense_single_volume import iteration_loop as _il

    requested_image_batch_size = int(image_batch_size)
    requested_rotation_block_size = int(rotation_block_size)
    rotation_block_size = _local_search_engine_rotation_block_size(rotation_block_size)
    # Keep the local-search hypothesis grid (rotations/translations/priors)
    # genuinely double precision end to end when either flag requests it;
    # default stays float32 to match RELION's accelerated-GPU precision.
    local_layout_dtype = np.float64 if (use_float64_scoring or use_float64_projections) else np.float32
    prior_rotations = np.asarray(prior_rotations, dtype=local_layout_dtype)
    if prior_rotations.ndim == 3:
        n_prior = prior_rotations.shape[0]
    elif prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        n_prior = prior_rotations.shape[0]
    else:
        raise ValueError(f"prior_rotations must have shape (n,3,3) or (n,3), got {prior_rotations.shape}")
    if prior_translations is None:
        prior_translations = np.zeros(
            (n_prior, np.asarray(translations).shape[1]),
            dtype=local_layout_dtype,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=local_layout_dtype).reshape(
            -1,
            np.asarray(translations).shape[1],
        )

    if pass2_layout is None:
        metadata_t0 = time.time()
        # RELION local priors remain factorized in canonical direction/psi index
        # space even when the scored trial rotations have been perturbed.
        local_grid_metadata = build_local_search_grid_metadata(healpix_order)
        metadata_build_time = time.time() - metadata_t0

        layout_t0 = time.time()
        layout_kwargs = {}
        if int(local_parent_oversampling_order) > 0:
            layout_kwargs["local_parent_oversampling_order"] = int(local_parent_oversampling_order)
        if rotation_grid_mstep_rotations is not None:
            layout_kwargs["rotation_grid_mstep_rotations"] = rotation_grid_mstep_rotations
        if bool(generate_relion_mstep_rotations):
            layout_kwargs["generate_relion_mstep_rotations"] = True
        local_layout = _il.build_local_hypothesis_layout(
            prior_rotations,
            rotation_grid_rotations,
            sigma_rot,
            sigma_psi,
            healpix_order,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            # Match the grouped RELION-mode path: local translation priors use the
            # learned/model sigma, not the older range/3 override.
            None,
            experiment_dataset.voxel_size,
            grid_metadata=local_grid_metadata,
            translation_prior_reference_translations=translation_prior_reference_translations,
            rotation_log_prior=rotation_log_prior,
            rotation_grid_random_perturbation=rotation_grid_random_perturbation,
            rotation_grid_angular_sampling_deg=rotation_grid_angular_sampling_deg,
            dtype=local_layout_dtype,
            **layout_kwargs,
        )
        selector_time = time.time() - layout_t0
    else:
        local_layout = pass2_layout
        metadata_build_time = 0.0
        selector_time = 0.0

    if class_log_priors is not None:
        if source_faithful_spectrum_norm:
            raise ValueError("RELION source-faithful spectrum normalization is fresh K=1-only")
        local_n_classes = int(np.asarray(class_log_priors).size)
    else:
        local_n_classes = 1
    # Exact local K-class invokes the per-class local kernels sequentially
    # (probe pass per class, then M-step per class), so K is not a simultaneous
    # tensor dimension for the shifted-image/projection tiles here.
    local_kernel_classes = 1
    local_rotation_count = (
        int(np.max(np.asarray(local_layout.rotation_counts, dtype=np.int64)))
        if int(np.asarray(local_layout.rotation_counts).size)
        else 1
    )
    # The RELION x-half pass-2 path projects and backprojects only the active
    # Fourier window. Budget it against that window by default; keep the older
    # full-spectrum guard available as an emergency rollback knob for OOM
    # triage on smaller GPUs.
    local_batch_planning_current_size = current_size
    if relion_projector_half is not None and mstep_relion_x_half and not score_only:
        xhalf_guard_mode = _runtime_environment().get(EXACT_LOCAL_XHALF_BATCH_GUARD_ENV, "windowed").strip().lower()
        if xhalf_guard_mode in {"", "full", "full_spectrum", "full-spectrum", "conservative"}:
            local_batch_planning_current_size = None
        elif xhalf_guard_mode in {"window", "windowed", "compact", "current_size", "current-size"}:
            local_batch_planning_current_size = current_size
        else:
            raise ValueError(
                f"{EXACT_LOCAL_XHALF_BATCH_GUARD_ENV} must be 'full' or 'windowed', got {xhalf_guard_mode!r}"
            )
    local_batch_plan = _il._estimate_relion_em_batch_sizes(
        requested_image_batch_size=image_batch_size,
        requested_rotation_block_size=rotation_block_size,
        n_rot=max(1, local_rotation_count),
        n_trans=max(1, int(np.asarray(local_layout.translation_grid).shape[0])),
        image_shape=experiment_dataset.image_shape,
        volume_shape=experiment_dataset.volume_shape,
        padding_factor=max(int(projection_padding_factor), int(reconstruction_padding_factor), 1),
        n_classes=local_kernel_classes,
        current_size=local_batch_planning_current_size,
        settings=(
            None
            if execution_settings is None
            else execution_settings.dense_batch_planning
        ),
    )
    if (
        local_batch_plan.image_batch_size != image_batch_size
        or local_batch_plan.rotation_block_size != rotation_block_size
    ):
        logger.info(
            "Local search memory batch sizing: requested image_batch_size=%d rotation_block_size=%d; "
            "using image_batch_size=%d rotation_block_size=%d "
            "(local_rot_max=%d n_trans=%d K=%d effective_kernel_K=%d, score_pixels=%d, "
            "translation_tile=%.2f/%.2f GB, "
            "projection_tile=%.2f/%.2f GB, persistent_est=%.2f GB, usable_est=%.2f GB, "
            "gpu_used_est=%.2f GB)",
            requested_image_batch_size,
            requested_rotation_block_size,
            local_batch_plan.image_batch_size,
            local_batch_plan.rotation_block_size,
            local_rotation_count,
            int(np.asarray(local_layout.translation_grid).shape[0]),
            local_n_classes,
            local_kernel_classes,
            local_batch_plan.score_pixel_count,
            local_batch_plan.translation_tile_gb,
            local_batch_plan.translation_tile_budget_gb,
            local_batch_plan.projection_block_gb,
            local_batch_plan.projection_budget_gb,
            local_batch_plan.persistent_estimate_gb,
            local_batch_plan.usable_estimate_gb,
            local_batch_plan.gpu_used_estimate_gb,
        )
    image_batch_size = local_batch_plan.image_batch_size
    rotation_block_size = local_batch_plan.rotation_block_size

    if class_log_priors is not None:
        if return_reconstruction_sample_indices:
            raise NotImplementedError("K-class local search does not return reconstruction sample indices")
        if return_significant_counts:
            raise NotImplementedError("K-class local search does not return significant counts")
        if score_only:
            raise NotImplementedError("K-class local search does not support score_only")
        if return_profile:
            raise NotImplementedError("K-class local search does not yet emit local profile summaries")
        if reconstruction.disable_adjoint_y or reconstruction.disable_adjoint_ctf:
            raise NotImplementedError("K-class local search does not support adjoint ablation flags")
        if normalization_log_z is not None:
            raise NotImplementedError("K-class local search requires evidence-space normalization, not pass-2 log_z")
        if normalization_log_evidence is not None:
            raise NotImplementedError("K-class local search does not support external evidence normalization")
        if projection_force_jax:
            raise NotImplementedError("K-class local search does not yet plumb projection_force_jax")
        k_class_result = run_local_k_class_em(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            local_layout,
            disc_type,
            class_log_priors=class_log_priors,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=current_size,
            reconstruction_current_size=reconstruction_current_size,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            score_with_masked_images=score_with_masked_images,
            half_spectrum_scoring=half_spectrum_scoring,
            relion_exact_score_translation=relion_exact_score_translation,
            use_float64_scoring=use_float64_scoring,
            use_float64_normalization=True,
            use_float64_projections=use_float64_projections,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            image_corrections=corrections.image_corrections,
            scale_corrections=corrections.scale_corrections,
            group_ids=corrections.group_ids,
            scale_correction_group_count=corrections.scale_correction_group_count,
            scale_correction_data_vs_prior=corrections.scale_correction_data_vs_prior,
            image_pre_shifts=corrections.image_pre_shifts,
            mstep_relion_x_half=mstep_relion_x_half,
            reconstruct_significant_only=reconstruct_significant_only,
            adaptive_fraction=adaptive_fraction,
            max_significants=-1,
            stats_use_reconstruction_probs=reconstruction.stats_use_reconstruction_probs,
            class_posterior_sums_from_noise=bool(reconstruct_significant_only and accumulate_noise),
            debug_iteration=diagnostics.iteration,
            translation_prior_centers=translation_prior_centers,
            cache_settings=(
                None if execution_settings is None else execution_settings.local_cache
            ),
        )
        use_noise_class_sums = bool(reconstruct_significant_only and accumulate_noise)
        class_mstep_posterior_sums = (
            getattr(k_class_result, "class_mstep_posterior_sums", None) if use_noise_class_sums else None
        )
        if class_mstep_posterior_sums is None:
            class_mstep_posterior_sums = k_class_result.class_posterior_sums
        result = LocalSearchIterationResult(
            Ft_y=k_class_result.Ft_y,
            Ft_ctf=k_class_result.Ft_ctf,
            hard_assignment=np.asarray(k_class_result.pose_assignments, dtype=np.int32),
            relion_stats=k_class_result.stats,
            noise_stats=k_class_result.aggregate_noise_stats if accumulate_noise else None,
            best_pose_rotations=k_class_result.best_pose_rotations if return_best_pose_details else None,
            best_pose_translations=k_class_result.best_pose_translations if return_best_pose_details else None,
            best_pose_rotation_ids=k_class_result.best_pose_rotation_ids if return_best_pose_details else None,
            class_assignments=np.asarray(k_class_result.class_assignments, dtype=np.int32),
            class_posterior_sums=np.asarray(class_mstep_posterior_sums, dtype=np.float64),
            class_full_posterior_sums=np.asarray(k_class_result.class_posterior_sums, dtype=np.float64),
        )
    else:
        engine_result = run_local_em(
            LocalEMRequest(
                inputs=LocalEMInputs(
                    experiment_dataset=experiment_dataset,
                    mean=mean,
                    mean_variance=mean_variance,
                    noise_variance=noise_variance,
                    local_layout=local_layout,
                    disc_type=disc_type,
                    relion_projector_half=relion_projector_half,
                    relion_projector_r_max=relion_projector_r_max,
                ),
                search=LocalSearchSettings(
                    current_size=current_size,
                    reconstruction_current_size=reconstruction_current_size,
                    reconstruct_significant_only=reconstruct_significant_only,
                    adaptive_fraction=adaptive_fraction,
                    # RELION's maximum_significants cap defines the coarse
                    # adaptive support, not the pass-2 reconstruction threshold.
                    max_significants=max_significants if apply_max_significants_to_support else -1,
                ),
                execution=LocalExecutionSettings(
                    image_batch_size=image_batch_size,
                    rotation_block_size=rotation_block_size,
                    cache=(
                        None
                        if execution_settings is None
                        else execution_settings.local_cache
                    ),
                ),
                scoring=LocalScoringSettings(
                    score_with_masked_images=score_with_masked_images,
                    half_spectrum_scoring=half_spectrum_scoring,
                    relion_exact_score_translation=relion_exact_score_translation,
                    use_float64_scoring=use_float64_scoring,
                    # Keep posterior/log-Z reductions in float64 even when
                    # score/projection tensors stay float32 for throughput.
                    use_float64_normalization=True,
                ),
                projection=LocalProjectionSettings(
                    projection_padding_factor=projection_padding_factor,
                    reconstruction_padding_factor=reconstruction_padding_factor,
                    use_float64_projections=use_float64_projections,
                    relion_texture_interp=projection_relion_texture_interp,
                    relion_acc_double_floorf_quirk=projection_relion_acc_double_floorf_quirk,
                    force_jax=projection_force_jax,
                    do_gridding_correction=do_gridding_correction,
                    square_window=square_window,
                ),
                corrections=corrections,
                posterior=LocalPosteriorInputs(
                    normalization_log_z=normalization_log_z,
                    normalization_log_evidence=normalization_log_evidence,
                    translation_prior_centers=translation_prior_centers,
                ),
                reconstruction=reconstruction,
                outputs=LocalEMRequestedOutputs(
                    accumulate_noise=accumulate_noise,
                    return_half_volume_accumulators=return_half_volume_accumulators,
                    return_profile=return_profile,
                    return_best_pose_details=return_best_pose_details,
                    return_reconstruction_sample_indices=return_reconstruction_sample_indices,
                    return_significant_counts=return_significant_counts,
                ),
                diagnostics=diagnostics,
            ),
        )
        result = LocalSearchIterationResult(
            Ft_y=engine_result.Ft_y,
            Ft_ctf=engine_result.Ft_ctf,
            hard_assignment=engine_result.hard_assignment,
            relion_stats=engine_result.relion_stats,
            noise_stats=engine_result.noise_stats,
            profile_summary=engine_result.profile_summary,
            significant_counts=engine_result.significant_counts,
            best_pose_rotations=engine_result.best_pose_rotations,
            best_pose_translations=engine_result.best_pose_translations,
            best_pose_rotation_ids=engine_result.best_pose_rotation_ids,
        )

    if return_profile and result.profile_summary is not None:
        profile_summary = dict(result.profile_summary)
        profile_summary["metadata_build_time_s"] = np.float64(metadata_build_time)
        profile_summary["selector_time_s"] = np.float64(selector_time)
        profile_summary["translation_prior_time_s"] = np.float64(0.0)
        result = replace(result, profile_summary=profile_summary)

    if return_class_details and (
        result.class_assignments is None
        or result.class_posterior_sums is None
        or result.class_full_posterior_sums is None
    ):
        raise ValueError("return_class_details=True requires class_log_priors")
    return result
