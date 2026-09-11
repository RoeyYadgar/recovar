"""Host-side request groups for one exact local-search iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recovar.em.dense_single_volume.local_em_types import (
    LocalCorrectionInputs,
    LocalEMDiagnostics,
    LocalEMRequestedOutputs,
    LocalProjectionSettings,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
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
    relion_projector_half: Any | None = None
    relion_projector_r_max: int | None = None


@dataclass(frozen=True)
class LocalSearchIterationGrid:
    """Pose grid, priors, and optional prebuilt pass-2 layout."""

    prior_rotations: Any
    rotation_grid_rotations: Any
    healpix_order: int
    sigma_rot: float
    sigma_psi: float
    translations: Any
    prior_translations: Any
    sigma_offset_angstrom: float
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
    """Host batch sizes and resolved runtime settings."""

    image_batch_size: int
    rotation_block_size: int
    settings: ExecutionSettings | None = None


@dataclass(frozen=True)
class LocalSearchIterationPosterior:
    """External normalization and class-prior inputs."""

    normalization_log_z: Any | None = None
    normalization_log_evidence: Any | None = None
    class_log_priors: Any | None = None


@dataclass(frozen=True)
class LocalSearchIterationRequest:
    """Cohesive host request for one local-search iteration."""

    inputs: LocalSearchIterationInputs
    grid: LocalSearchIterationGrid
    search: LocalSearchSettings
    execution: LocalSearchIterationExecution
    scoring: LocalScoringSettings = LocalScoringSettings()
    projection: LocalProjectionSettings = LocalProjectionSettings()
    corrections: LocalCorrectionInputs = LocalCorrectionInputs()
    posterior: LocalSearchIterationPosterior = LocalSearchIterationPosterior()
    reconstruction: LocalReconstructionSettings = LocalReconstructionSettings()
    outputs: LocalEMRequestedOutputs = LocalEMRequestedOutputs()
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
