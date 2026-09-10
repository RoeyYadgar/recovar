"""Typed dynamic inputs and static policy for the exact-local bucket JIT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


class LocalBigJitImageInputs(NamedTuple):
    """Per-bucket images, CTF parameters, and image-level corrections."""

    images: Any
    ctf_params: Any
    integer_pre_shifts: Any
    fourier_pre_shifts: Any
    image_corrections: Any
    image_only_corrections: Any
    scale_corrections: Any
    translation_sqdist_ang: Any
    group_ids: Any


class LocalBigJitProjectionInputs(NamedTuple):
    """Volume/projector arrays and optional compact projection-cache data."""

    mean: Any
    relion_projector_half: Any
    pixel_indices: Any
    score_take_indices: Any
    reconstruction_take_indices: Any
    cache: Any
    cache_id_map: Any


class LocalBigJitMstepAccumulators(NamedTuple):
    """The two donated M-step buffers carried across bucket calls."""

    y: Any
    ctf: Any


class LocalBigJitNoiseAccumulators(NamedTuple):
    """Noise and scale statistics carried across bucket calls."""

    wsum: Any
    image_power: Any
    a2: Any
    xa: Any
    scale_xa: Any
    scale_aa: Any
    sigma2_offset: Any
    sumw: Any


class LocalBigJitSharedInputs(NamedTuple):
    """Call-wide Fourier, mask, window, and noise arrays."""

    image_mask: Any
    noise_variance_half: Any
    translation_phases_half: Any
    relion_score_translation_angles: Any
    half_weights: Any
    norm_half_weights: Any
    score_window_indices: Any
    reconstruction_window_indices: Any
    mstep_reconstruction_window_indices: Any
    shell_indices_half: Any
    shell_indices_noise: Any
    noise_variance_for_noise: Any
    scale_correction_pixel_mask: Any


class LocalBigJitBucketInputs(NamedTuple):
    """Padded local hypotheses and posterior constraints for one bucket."""

    rotation_ids: Any
    rotations: Any
    mstep_rotations: Any
    rotation_log_prior: Any
    translation_log_prior: Any
    rotation_mask: Any
    sample_mask: Any
    valid_image_mask: Any
    normalization_log_z: Any
    normalization_log_evidence: Any
    reconstruction_probability_threshold: Any


@dataclass(frozen=True)
class LocalBigJitPolicy:
    """Hashable compile-time policy for one exact-local bucket shape/route."""

    # Image preprocessing and scoring.
    mask_mode: str
    score_with_masked_images: bool
    apply_integer_pre_shift: bool
    apply_fourier_pre_shift: bool
    half_spectrum_scoring: bool
    use_float64_scoring: bool
    use_float64_normalization: bool
    use_window: bool

    # Posterior support.
    reconstruct_significant_only: bool
    adaptive_fraction: float
    max_significants: int | None
    has_normalization_log_z: bool
    has_normalization_log_evidence: bool
    has_reconstruction_probability_threshold: bool
    score_only: bool

    # Array geometry and projection implementation.
    image_shape: tuple[int, ...]
    projection_volume_shape: tuple[int, ...]
    reconstruction_volume_shape: tuple[int, ...]
    disc_type: str
    projection_half_volume: bool
    projection_max_r: int | str | None
    mstep_max_r: int | None
    use_compact_relion_projector_projection: bool
    use_relion_projection_cache: bool
    relion_projector_output_size: int
    projection_relion_texture_interp: bool
    projection_force_jax: bool
    use_relion_projector: bool
    relion_projector_r_max: int
    projection_padding_factor: int

    # M-step and statistic routes.
    mstep_subtract_ctf_projection: bool
    mstep_relion_x_half: bool
    disable_adjoint_y: bool
    disable_adjoint_ctf: bool
    accumulate_noise: bool
    accumulate_scale_correction: bool
    return_noise_split: bool
    n_shells: int
    norm_current_size: int | None
    include_unweighted_norm_high_shell: bool
    source_faithful_spectrum_norm: bool

    # Optional compiled outputs and diagnostics.
    return_mstep_tensors: bool
    return_deferred_mstep_inputs: bool
    return_deferred_noise_inputs: bool
    return_debug_arrays: bool
    return_debug_scores: bool
    return_debug_operands: bool


@dataclass(frozen=True)
class LocalBigJitResult:
    """Stable named view of every compiled exact-local bucket result route."""

    mstep: LocalBigJitMstepAccumulators
    noise: LocalBigJitNoiseAccumulators
    bucket_norm_correction: Any
    batch_norm: Any
    log_evidence: Any
    best_log_score: Any
    best_argmax: Any
    max_posterior: Any
    rotation_posterior_sums: Any
    reconstruction_rotation_posterior_sums: Any
    significant_sample_counts: Any
    reconstruction_sample_mask: Any
    reconstruction_rotation_mask: Any
    reconstruction_row_count: Any
    summed: Any | None = None
    ctf_probabilities: Any | None = None
    reconstruction_probabilities: Any | None = None
    shifted_reconstruction: Any | None = None
    ctf2_over_noise_reconstruction: Any | None = None
    shifted_noise: Any | None = None
    processed_score_half: Any | None = None
    debug_scores: Any | None = None
    debug_probabilities: Any | None = None
    debug_shifted_score: Any | None = None
    debug_shifted_reconstruction: Any | None = None
    debug_ctf2_over_noise_score: Any | None = None
    debug_ctf2_over_noise_reconstruction: Any | None = None
    debug_weighted_projection: Any | None = None
    debug_noise_projection: Any | None = None


def unpack_local_big_jit_result(raw_result, policy: LocalBigJitPolicy) -> LocalBigJitResult:
    """Convert the compiled route-dependent tuple into one named host view."""

    values = tuple(raw_result)
    debug_scores = None
    debug_probabilities = None
    debug_shifted_score = None
    debug_shifted_reconstruction = None
    debug_ctf2_over_noise_score = None
    debug_ctf2_over_noise_reconstruction = None
    debug_weighted_projection = None
    debug_noise_projection = None
    if policy.return_debug_arrays:
        if policy.return_debug_operands:
            values, debug_values = values[:-8], values[-8:]
            (
                debug_scores,
                debug_probabilities,
                debug_shifted_score,
                debug_shifted_reconstruction,
                debug_ctf2_over_noise_score,
                debug_ctf2_over_noise_reconstruction,
                debug_weighted_projection,
                debug_noise_projection,
            ) = debug_values
            if policy.score_only:
                debug_shifted_reconstruction = None
                debug_ctf2_over_noise_reconstruction = None
                debug_noise_projection = None
        else:
            values, debug_values = values[:-2], values[-2:]
            debug_scores, debug_probabilities = debug_values

    (
        y,
        ctf,
        noise_wsum,
        noise_image_power,
        noise_a2,
        noise_xa,
        noise_scale_xa,
        noise_scale_aa,
        bucket_norm_correction,
        noise_sigma2_offset,
        noise_sumw,
        batch_norm,
        log_evidence,
        best_log_score,
        best_argmax,
        max_posterior,
        rotation_posterior_sums,
        reconstruction_rotation_posterior_sums,
        significant_sample_counts,
        reconstruction_sample_mask,
        reconstruction_rotation_mask,
        reconstruction_row_count,
        *route_values,
    ) = values

    summed = None
    ctf_probabilities = None
    reconstruction_probabilities = None
    shifted_reconstruction = None
    ctf2_over_noise_reconstruction = None
    shifted_noise = None
    processed_score_half = None
    if policy.return_deferred_mstep_inputs:
        (
            reconstruction_probabilities,
            shifted_reconstruction,
            ctf2_over_noise_reconstruction,
            shifted_noise,
            processed_score_half,
        ) = route_values
    elif policy.return_mstep_tensors:
        summed, ctf_probabilities = route_values
    elif route_values:
        raise ValueError(f"unexpected exact-local big-JIT result tail of length {len(route_values)}")

    return LocalBigJitResult(
        mstep=LocalBigJitMstepAccumulators(y=y, ctf=ctf),
        noise=LocalBigJitNoiseAccumulators(
            wsum=noise_wsum,
            image_power=noise_image_power,
            a2=noise_a2,
            xa=noise_xa,
            scale_xa=noise_scale_xa,
            scale_aa=noise_scale_aa,
            sigma2_offset=noise_sigma2_offset,
            sumw=noise_sumw,
        ),
        bucket_norm_correction=bucket_norm_correction,
        batch_norm=batch_norm,
        log_evidence=log_evidence,
        best_log_score=best_log_score,
        best_argmax=best_argmax,
        max_posterior=max_posterior,
        rotation_posterior_sums=rotation_posterior_sums,
        reconstruction_rotation_posterior_sums=reconstruction_rotation_posterior_sums,
        significant_sample_counts=significant_sample_counts,
        reconstruction_sample_mask=reconstruction_sample_mask,
        reconstruction_rotation_mask=reconstruction_rotation_mask,
        reconstruction_row_count=reconstruction_row_count,
        summed=summed,
        ctf_probabilities=ctf_probabilities,
        reconstruction_probabilities=reconstruction_probabilities,
        shifted_reconstruction=shifted_reconstruction,
        ctf2_over_noise_reconstruction=ctf2_over_noise_reconstruction,
        shifted_noise=shifted_noise,
        processed_score_half=processed_score_half,
        debug_scores=debug_scores,
        debug_probabilities=debug_probabilities,
        debug_shifted_score=debug_shifted_score,
        debug_shifted_reconstruction=debug_shifted_reconstruction,
        debug_ctf2_over_noise_score=debug_ctf2_over_noise_score,
        debug_ctf2_over_noise_reconstruction=debug_ctf2_over_noise_reconstruction,
        debug_weighted_projection=debug_weighted_projection,
        debug_noise_projection=debug_noise_projection,
    )
