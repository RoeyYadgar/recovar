"""Host-side input/output contracts for the exact local EM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from recovar.em.dense_single_volume.runtime_options import LocalCacheSettings


@dataclass(frozen=True)
class LocalEMInputs:
    """Required dataset, model, and hypothesis inputs for one local EM call."""

    experiment_dataset: Any
    mean: Any
    mean_variance: Any
    noise_variance: Any
    local_layout: Any
    disc_type: str
    relion_projector_half: Any | None = None
    relion_projector_r_max: int | None = None


@dataclass(frozen=True)
class LocalSearchSettings:
    """Local support and Fourier-window settings."""

    current_size: int | None
    reconstruction_current_size: int | None = None
    reconstruct_significant_only: bool = False
    adaptive_fraction: float = 0.999
    max_significants: int = -1
    reconstruction_probability_threshold: Any | None = None


@dataclass(frozen=True)
class LocalExecutionSettings:
    """Host batching and bucket-shape controls for exact local execution."""

    image_batch_size: int
    rotation_block_size: int
    max_hypotheses_per_microbatch: int | None = None
    unify_local_bucket_sizes: bool | None = None
    cache: LocalCacheSettings | None = None


# Compatibility alias for external callers written before the runtime-level
# ExecutionSettings snapshot was introduced.
ExecutionSettings = LocalExecutionSettings


@dataclass(frozen=True)
class LocalScoringSettings:
    """Score representation, masking, and normalization policy."""

    score_with_masked_images: bool = True
    half_spectrum_scoring: bool = False
    relion_exact_score_translation: bool = False
    use_float64_scoring: bool = False
    use_float64_normalization: bool = True


@dataclass(frozen=True)
class LocalProjectionSettings:
    """Projection geometry and backend policy."""

    projection_padding_factor: int = 1
    reconstruction_padding_factor: int = 1
    use_float64_projections: bool = False
    relion_texture_interp: bool = False
    relion_acc_double_floorf_quirk: bool = False
    force_jax: bool = False
    do_gridding_correction: bool = False
    square_window: bool = False


@dataclass(frozen=True)
class LocalCorrectionInputs:
    """Optional per-image and per-group correction inputs."""

    image_corrections: Any | None = None
    scale_corrections: Any | None = None
    group_ids: Any | None = None
    scale_correction_group_count: int | None = None
    scale_correction_data_vs_prior: Any | None = None
    image_pre_shifts: Any | None = None


@dataclass(frozen=True)
class LocalPosteriorInputs:
    """Optional external priors and normalizers for posterior evaluation."""

    normalization_log_z: Any | None = None
    class_log_prior: float = 0.0
    normalization_log_evidence: Any | None = None
    translation_prior_centers: Any | None = None


@dataclass(frozen=True)
class LocalReconstructionSettings:
    """M-step route and sufficient-statistic policy."""

    mstep_subtract_ctf_projection: bool = False
    mstep_relion_x_half: bool = False
    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    stats_use_reconstruction_probs: bool = False
    include_unweighted_norm_high_shell: bool = True
    source_faithful_spectrum_norm: bool = False
    score_only: bool = False


@dataclass(frozen=True)
class LocalEMRequestedOutputs:
    """Select optional computation products without changing result positions."""

    accumulate_noise: bool = False
    return_half_volume_accumulators: bool = False
    return_profile: bool = False
    return_best_pose_details: bool = False
    return_reconstruction_probability_values: bool = False
    return_reconstruction_sample_indices: bool = False
    return_significant_counts: bool = False

@dataclass(frozen=True)
class LocalEMDiagnostics:
    """Diagnostic call identity, separate from numerical settings."""

    iteration: int | None = None
    pass_label: str | None = None


@dataclass(frozen=True)
class LocalEMRequest:
    """Composed host-side request for exact local EM."""

    inputs: LocalEMInputs
    search: LocalSearchSettings
    execution: LocalExecutionSettings
    scoring: LocalScoringSettings = LocalScoringSettings()
    projection: LocalProjectionSettings = LocalProjectionSettings()
    corrections: LocalCorrectionInputs = LocalCorrectionInputs()
    posterior: LocalPosteriorInputs = LocalPosteriorInputs()
    reconstruction: LocalReconstructionSettings = LocalReconstructionSettings()
    outputs: LocalEMRequestedOutputs = LocalEMRequestedOutputs()
    diagnostics: LocalEMDiagnostics = LocalEMDiagnostics()


@dataclass(frozen=True)
class LocalEMResult:
    """Stable named result returned by the typed exact-local engine boundary.

    The public ``run_local_em_exact`` compatibility API still returns its
    historical flag-dependent tuple.  This type centralizes that tuple's
    ordering so typed callers can migrate without duplicating cursor logic.
    """

    Ft_y: Any
    Ft_ctf: Any
    hard_assignment: Any
    relion_stats: Any
    best_pose_rotations: Any | None = None
    best_pose_translations: Any | None = None
    best_pose_rotation_ids: Any | None = None
    noise_stats: Any | None = None
    profile_summary: Mapping[str, Any] | None = None
    significant_counts: Any | None = None

    def to_legacy_tuple(self, outputs: LocalEMRequestedOutputs) -> tuple[Any, ...]:
        """Serialize this result using ``run_local_em_exact``'s tuple contract."""

        result = [self.Ft_y, self.Ft_ctf, self.hard_assignment]
        if outputs.return_best_pose_details:
            result.extend(
                [
                    self.best_pose_rotations,
                    self.best_pose_translations,
                    self.best_pose_rotation_ids,
                ]
            )
        result.append(self.relion_stats)
        if outputs.accumulate_noise:
            result.append(self.noise_stats)
        if (
            outputs.return_profile
            or outputs.return_reconstruction_probability_values
            or outputs.return_reconstruction_sample_indices
        ):
            result.append(self.profile_summary)
        if outputs.return_significant_counts:
            result.append(self.significant_counts)
        return tuple(result)
