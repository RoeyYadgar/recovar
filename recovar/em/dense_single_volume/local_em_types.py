"""Host-side input/output contracts for the exact local EM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LocalEMOutputSpec:
    """Describe which optional values are present in the legacy result tuple."""

    accumulate_noise: bool = False
    return_profile: bool = False
    return_best_pose_details: bool = False
    return_significant_counts: bool = False

    @property
    def legacy_tuple_size(self) -> int:
        """Return the exact tuple length selected by this output specification."""

        return (
            4
            + 3 * int(self.return_best_pose_details)
            + int(self.accumulate_noise)
            + int(self.return_profile)
            + int(self.return_significant_counts)
        )


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
class ExecutionSettings:
    """Host batching and bucket-shape controls for exact local execution."""

    image_batch_size: int
    rotation_block_size: int
    max_hypotheses_per_microbatch: int | None = None
    unify_local_bucket_sizes: bool | None = None


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

    @property
    def legacy_tuple_spec(self) -> LocalEMOutputSpec:
        """Return the tuple shape produced by the compatibility engine.

        Probability-value and sample-index capture are stored in the profile,
        so the legacy engine implicitly enables its profile result for either.
        """

        return LocalEMOutputSpec(
            accumulate_noise=self.accumulate_noise,
            return_profile=(
                self.return_profile
                or self.return_reconstruction_probability_values
                or self.return_reconstruction_sample_indices
            ),
            return_best_pose_details=self.return_best_pose_details,
            return_significant_counts=self.return_significant_counts,
        )


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
    execution: ExecutionSettings
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

    def to_legacy_tuple(self, output_spec: LocalEMOutputSpec) -> tuple[Any, ...]:
        """Serialize this result using ``run_local_em_exact``'s tuple contract."""

        result = [self.Ft_y, self.Ft_ctf, self.hard_assignment]
        if output_spec.return_best_pose_details:
            result.extend(
                [
                    self.best_pose_rotations,
                    self.best_pose_translations,
                    self.best_pose_rotation_ids,
                ]
            )
        result.append(self.relion_stats)
        if output_spec.accumulate_noise:
            result.append(self.noise_stats)
        if output_spec.return_profile:
            result.append(self.profile_summary)
        if output_spec.return_significant_counts:
            result.append(self.significant_counts)
        return tuple(result)

    @classmethod
    def from_legacy_tuple(
        cls,
        output: Sequence[Any],
        output_spec: LocalEMOutputSpec,
    ) -> LocalEMResult:
        """Parse and validate a ``run_local_em_exact`` compatibility tuple."""

        expected_size = output_spec.legacy_tuple_size
        if len(output) != expected_size:
            raise ValueError(
                "Local EM output tuple does not match its output specification: "
                f"expected {expected_size} values, received {len(output)}"
            )

        cursor = 0
        Ft_y, Ft_ctf, hard_assignment = output[cursor : cursor + 3]
        cursor += 3

        best_pose_rotations = best_pose_translations = best_pose_rotation_ids = None
        if output_spec.return_best_pose_details:
            best_pose_rotations, best_pose_translations, best_pose_rotation_ids = output[cursor : cursor + 3]
            cursor += 3

        relion_stats = output[cursor]
        cursor += 1
        noise_stats = output[cursor] if output_spec.accumulate_noise else None
        cursor += int(output_spec.accumulate_noise)
        profile_summary = output[cursor] if output_spec.return_profile else None
        cursor += int(output_spec.return_profile)
        significant_counts = output[cursor] if output_spec.return_significant_counts else None

        return cls(
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            hard_assignment=hard_assignment,
            relion_stats=relion_stats,
            best_pose_rotations=best_pose_rotations,
            best_pose_translations=best_pose_translations,
            best_pose_rotation_ids=best_pose_rotation_ids,
            noise_stats=noise_stats,
            profile_summary=profile_summary,
            significant_counts=significant_counts,
        )
