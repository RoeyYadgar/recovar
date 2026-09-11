"""Host-side request groups for one exact local-search iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recovar.em.dense_single_volume.local_em_types import (
    LocalCorrectionInputs,
    LocalEMDiagnostics,
    LocalReconstructionSettings,
)
from recovar.em.dense_single_volume.runtime_options import ExecutionSettings


@dataclass(frozen=True)
class LocalSearchIterationInputs:
    """Dataset and model arrays shared by local-search routes."""

    experiment_dataset: Any
    mean: Any
    mean_variance: Any
    noise_variance: Any
    disc_type: str


@dataclass(frozen=True)
class LocalSearchIterationGrid:
    """Pose grid, priors, and optional prebuilt pass-2 layout."""

    prior_rotations: Any
    rotation_grid_rotations: Any
    rotation_grid_eulers: Any
    healpix_order: int
    sigma_rot: float
    sigma_psi: float
    translations: Any
    prior_translations: Any
    sigma_offset_angstrom: float
    offset_range_pixels: float | None
    translation_prior_reference_translations: Any | None = None
    translation_prior_centers: Any | None = None
    rotation_log_prior: Any | None = None
    rotation_grid_random_perturbation: float = 0.0
    rotation_grid_angular_sampling_deg: float | None = None
    local_parent_oversampling_order: int = 0
    pass2_layout: Any | None = None
    rotation_grid_mstep_rotations: Any | None = None
    generate_relion_mstep_rotations: bool = False


@dataclass(frozen=True)
class LocalSearchIterationExecution:
    """Host batch sizes, Fourier windows, and resolved runtime settings."""

    image_batch_size: int
    rotation_block_size: int
    current_size: int | None
    reconstruction_current_size: int | None = None
    settings: ExecutionSettings | None = None


@dataclass(frozen=True)
class LocalSearchIterationScoring:
    """Score representation and posterior-support policy."""

    score_with_masked_images: bool = True
    half_spectrum_scoring: bool = False
    relion_exact_score_translation: bool = False
    use_float64_scoring: bool = False
    adaptive_fraction: float = 0.999
    max_significants: int = -1
    reconstruct_significant_only: bool = True
    apply_max_significants_to_support: bool = False


@dataclass(frozen=True)
class LocalSearchIterationProjection:
    """Projection geometry, precision, and backend settings."""

    projection_padding_factor: int = 1
    reconstruction_padding_factor: int = 1
    use_float64_projections: bool = False
    do_gridding_correction: bool = False
    square_window: bool = False
    relion_texture_interp: bool | None = False
    relion_acc_double_floorf_quirk: bool = False
    force_jax: bool = False
    relion_projector_half: Any | None = None
    relion_projector_r_max: int | None = None


@dataclass(frozen=True)
class LocalSearchIterationPosterior:
    """External normalization and class-prior inputs."""

    normalization_log_z: Any | None = None
    normalization_log_evidence: Any | None = None
    class_log_priors: Any | None = None


@dataclass(frozen=True)
class LocalSearchIterationOutputs:
    """Optional local-search computation products."""

    accumulate_noise: bool = False
    return_half_volume_accumulators: bool = False
    return_profile: bool = False
    return_best_pose_details: bool = False
    return_class_details: bool = False
    return_reconstruction_sample_indices: bool = False
    return_significant_counts: bool = False


@dataclass(frozen=True)
class LocalSearchIterationRequest:
    """Cohesive host request for one local-search iteration."""

    inputs: LocalSearchIterationInputs
    grid: LocalSearchIterationGrid
    execution: LocalSearchIterationExecution
    scoring: LocalSearchIterationScoring = LocalSearchIterationScoring()
    projection: LocalSearchIterationProjection = LocalSearchIterationProjection()
    corrections: LocalCorrectionInputs = LocalCorrectionInputs()
    posterior: LocalSearchIterationPosterior = LocalSearchIterationPosterior()
    reconstruction: LocalReconstructionSettings = LocalReconstructionSettings()
    outputs: LocalSearchIterationOutputs = LocalSearchIterationOutputs()
    diagnostics: LocalEMDiagnostics = LocalEMDiagnostics()


@dataclass(frozen=True)
class LocalSearchIterationResult:
    """Stable named result from one local-search iteration."""

    Ft_y: Any
    Ft_ctf: Any
    hard_assignment: Any
    relion_stats: Any
    noise_stats: Any | None = None
    profile_summary: dict | None = None
    significant_counts: Any | None = None
    best_pose_rotations: Any | None = None
    best_pose_translations: Any | None = None
    best_pose_rotation_ids: Any | None = None
    class_assignments: Any | None = None
    class_posterior_sums: Any | None = None
    class_full_posterior_sums: Any | None = None
