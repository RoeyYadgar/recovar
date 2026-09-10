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
