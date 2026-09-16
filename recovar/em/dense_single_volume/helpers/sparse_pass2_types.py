"""Typed host boundary for K=1 sparse pass-2 execution."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Self


def _declared_options(cls, options: Mapping[str, Any], overrides: Mapping[str, Any]):
    """Select only fields declared by ``cls`` from an orchestration option map."""

    field_names = {field.name for field in fields(cls)}
    values = {name: value for name, value in options.items() if name in field_names}
    unknown_overrides = set(overrides) - field_names
    if unknown_overrides:
        unknown = ", ".join(sorted(unknown_overrides))
        raise TypeError(f"unknown {cls.__name__} option(s): {unknown}")
    values.update(overrides)
    return values


@dataclass(frozen=True)
class SparsePass2Data:
    """Dataset, model, priors, and optional per-image sparse-pass inputs."""

    experiment_dataset: Any
    volume: Any
    mean_variance: Any
    noise_variance: Any
    translations: Any
    significant_sample_indices: Any
    rotation_log_prior: Any | None = None
    translation_log_prior: Any | None = None
    image_corrections: Any | None = None
    scale_corrections: Any | None = None
    group_ids: Any | None = None
    scale_correction_group_count: int | None = None
    scale_correction_data_vs_prior: Any | None = None
    image_pre_shifts: Any | None = None
    translation_prior_centers: Any | None = None
    normalization_log_z: Any | None = None
    normalization_other_score_log_z: Any | None = None
    fine_rotations_override: Any | None = None
    fine_mstep_rotations_override: Any | None = None
    fine_rotation_parent_override: Any | None = None
    fine_translations_override: Any | None = None
    fine_translation_parent_override: Any | None = None
    relion_projector_half: Any | None = None

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **overrides: Any) -> Self:
        """Build from a larger orchestration map while rejecting unknown overrides."""

        return cls(**_declared_options(cls, options, overrides))


@dataclass(frozen=True)
class SparsePass2Settings:
    """Search, scoring, reconstruction, output, and execution policy."""

    nside_level: int
    disc_type: str
    oversampling_order: int = 1
    current_size: int | None = None
    reconstruction_current_size: int | None = None
    translation_step: float | None = None
    score_with_masked_images: bool = False
    return_stats: bool = False
    accumulate_noise: bool = False
    half_spectrum_scoring: bool = False
    projection_padding_factor: int = 1
    reconstruction_padding_factor: int = 1
    use_float64_scoring: bool = False
    do_gridding_correction: bool = False
    square_window: bool = False
    random_perturbation: float = 0.0
    normalization_score_mode: str | None = None
    return_score_log_z: bool = False
    return_score_log_z_only: bool = False
    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    rotation_block_size_for_quantization: int = 5000
    relion_half_volume_mstep: bool = False
    relion_x_half_mstep: bool = False
    relion_fine_mstep_prune: bool = False
    relion_firstiter_score_mode: str = "gaussian"
    relion_firstiter_winner_take_all: bool = False
    relion_exact_fine_gaussian: bool = True
    relion_fine_diff2_fused_ffi: bool = False
    relion_f32_fine_posterior: bool = False
    relion_exact_fine_normalized_cc: bool = False
    relion_projector_r_max: int | None = None
    adaptive_fraction: float = 0.999
    bpref_device_signature_active: bool = False
    bpref_class_index: int = 0
    include_unweighted_norm_high_shell: bool = True
    preserve_bpref_particle_order: bool = False
    source_faithful_spectrum_norm: bool = False

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **overrides: Any) -> Self:
        """Build from a larger orchestration map while rejecting unknown overrides."""

        return cls(**_declared_options(cls, options, overrides))


@dataclass(frozen=True)
class SparsePass2Result:
    """Stable sparse pass-2 result independent of requested optional fields."""

    Ft_y: Any | None = None
    Ft_ctf: Any | None = None
    hard_assignment: Any | None = None
    best_rotations: Any | None = None
    best_translations: Any | None = None
    best_rotation_indices: Any | None = None
    relion_stats: Any | None = None
    log_evidence_per_image: Any | None = None
    score_log_z: Any | None = None
    noise_stats: Any | None = None

    def to_legacy_tuple(self, settings: SparsePass2Settings) -> tuple[Any, ...]:
        """Serialize the historical flag-dependent sparse-pass tuple."""

        if settings.return_score_log_z_only:
            return self.log_evidence_per_image, self.score_log_z

        result = [
            self.Ft_y,
            self.Ft_ctf,
            self.hard_assignment,
            self.best_rotations,
            self.best_translations,
            self.best_rotation_indices,
        ]
        if settings.return_stats:
            result.append(self.relion_stats)
            if settings.return_score_log_z:
                result.append(self.score_log_z)
        if settings.accumulate_noise:
            result.append(self.noise_stats)
        return tuple(result)


@dataclass(frozen=True)
class SparseKClassPass2Data:
    """Dataset, class models, priors, and optional per-image fused inputs."""

    experiment_dataset: Any
    volumes: Any
    mean_variance: Any
    noise_variance: Any
    translations: Any
    significant_sample_indices_by_class: Any
    rotation_log_priors_by_class: Any
    translation_log_prior: Any | None = None
    image_corrections: Any | None = None
    scale_corrections: Any | None = None
    group_ids: Any | None = None
    scale_correction_group_count: int | None = None
    scale_correction_data_vs_prior: Any | None = None
    image_pre_shifts: Any | None = None
    translation_prior_centers: Any | None = None
    fine_rotations_override: Any | None = None
    fine_mstep_rotations_override: Any | None = None
    fine_rotation_parent_override: Any | None = None
    fine_translations_override: Any | None = None
    fine_translation_parent_override: Any | None = None
    relion_projector_half: Any | None = None

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **overrides: Any) -> Self:
        """Build fused data from its surrounding K-class option map."""

        return cls(**_declared_options(cls, options, overrides))


@dataclass(frozen=True)
class SparseKClassPass2Settings:
    """Search, scoring, reconstruction, and execution policy for fused K-class pass 2."""

    nside_level: int
    disc_type: str
    oversampling_order: int
    current_size: int | None
    translation_step: float | None = None
    score_with_masked_images: bool = False
    return_stats: bool = True
    accumulate_noise: bool = False
    half_spectrum_scoring: bool = False
    projection_padding_factor: int = 1
    reconstruction_padding_factor: int = 1
    use_float64_scoring: bool = False
    do_gridding_correction: bool = False
    square_window: bool = False
    random_perturbation: float = 0.0
    rotation_block_size_for_quantization: int = 5000
    relion_half_volume_mstep: bool = False
    relion_x_half_mstep: bool = False
    relion_fine_mstep_prune_mode: str | None = None
    relion_firstiter_score_mode: str = "gaussian"
    relion_firstiter_winner_take_all: bool = False
    relion_exact_fine_gaussian: bool = True
    relion_projector_r_max: int | None = None
    adaptive_fraction: float = 0.999
    bpref_device_signature_active: bool = False

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **overrides: Any) -> Self:
        """Build fused settings from its surrounding K-class option map."""

        return cls(**_declared_options(cls, options, overrides))


@dataclass(frozen=True)
class SparseKClassPass2Result:
    """Fused sparse result normalized over the joint class-by-pose grid."""

    class_log_evidence: Any
    class_score_log_z: Any
    Ft_y: tuple[Any, ...]
    Ft_ctf: tuple[Any, ...]
    per_class_hard_assignments: Any
    per_class_stats: tuple[Any, ...]
    noise_stats: tuple[Any, ...] | None
    per_class_best_pose_rotations: tuple[Any, ...] | None
    per_class_best_pose_translations: tuple[Any, ...] | None
    per_class_best_pose_rotation_ids: tuple[Any, ...] | None
    profile_summary: dict[str, Any]
    class_posterior_sums: Any | None = None
