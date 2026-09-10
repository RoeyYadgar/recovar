"""Host-side validation and planning for exact local EM execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from recovar.em.dense_single_volume.helpers.translation_prior import validate_translation_prior_centers
from recovar.em.dense_single_volume.local_em_types import (
    LocalCorrectionInputs,
    LocalEMRequestedOutputs,
    LocalPosteriorInputs,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)


@dataclass(frozen=True)
class LocalEMModePlan:
    """Normalized execution modes resolved before exact-local array work."""

    score_only: bool
    accumulate_noise: bool
    return_half_volume_accumulators: bool
    return_profile: bool
    mstep_subtract_ctf_projection: bool
    mstep_relion_x_half: bool
    disable_adjoint_y: bool
    disable_adjoint_ctf: bool
    relion_exact_score_translation: bool
    include_unweighted_norm_high_shell: bool
    source_faithful_spectrum_norm: bool


@dataclass(frozen=True)
class LocalEMInputPlan:
    """Validated host arrays and dimensions used throughout exact-local EM."""

    n_images: int
    class_log_prior: float
    group_ids: np.ndarray | None
    n_scale_groups: int
    normalization_log_z: np.ndarray | None
    normalization_log_evidence: np.ndarray | None
    reconstruction_probability_threshold: np.ndarray | None
    translation_prior_centers: np.ndarray | None


def plan_local_em_modes(
    *,
    scoring: LocalScoringSettings,
    reconstruction: LocalReconstructionSettings,
    outputs: LocalEMRequestedOutputs,
) -> LocalEMModePlan:
    """Validate mode combinations and return their normalized host plan.

    This function deliberately handles scalar policy only. It must run before
    dataset inspection, diagnostic parsing, or JAX array construction.
    """

    plan = LocalEMModePlan(
        score_only=bool(reconstruction.score_only),
        accumulate_noise=bool(outputs.accumulate_noise),
        return_half_volume_accumulators=bool(outputs.return_half_volume_accumulators),
        return_profile=bool(outputs.legacy_tuple_spec.return_profile),
        mstep_subtract_ctf_projection=bool(reconstruction.mstep_subtract_ctf_projection),
        mstep_relion_x_half=bool(reconstruction.mstep_relion_x_half),
        disable_adjoint_y=bool(reconstruction.disable_adjoint_y),
        disable_adjoint_ctf=bool(reconstruction.disable_adjoint_ctf),
        relion_exact_score_translation=bool(scoring.relion_exact_score_translation),
        include_unweighted_norm_high_shell=bool(reconstruction.include_unweighted_norm_high_shell),
        source_faithful_spectrum_norm=bool(reconstruction.source_faithful_spectrum_norm),
    )

    if plan.relion_exact_score_translation and not scoring.half_spectrum_scoring:
        raise ValueError("exact RELION score translation requires half_spectrum_scoring=True")
    if plan.score_only:
        if not (plan.disable_adjoint_y and plan.disable_adjoint_ctf):
            raise ValueError("score_only exact-local EM requires both adjoints disabled")
        if plan.accumulate_noise:
            raise ValueError("score_only exact-local EM does not support noise accumulation")
        if plan.mstep_subtract_ctf_projection:
            raise ValueError("score_only exact-local EM does not support residual M-step subtraction")
        if plan.return_half_volume_accumulators:
            raise ValueError("score_only exact-local EM does not return half-volume accumulators")

    return plan


def plan_local_em_inputs(
    *,
    local_layout: Any,
    corrections: LocalCorrectionInputs,
    posterior: LocalPosteriorInputs,
    search: LocalSearchSettings,
) -> LocalEMInputPlan:
    """Validate per-image host inputs without constructing device arrays."""

    n_images = int(local_layout.n_images)
    class_log_prior = float(posterior.class_log_prior)

    explicit_scale_group_count = 0
    if corrections.scale_correction_group_count is not None:
        explicit_scale_group_count = int(corrections.scale_correction_group_count)
        if (
            explicit_scale_group_count < 0
            or not np.isfinite(float(corrections.scale_correction_group_count))
            or float(corrections.scale_correction_group_count) != float(explicit_scale_group_count)
        ):
            raise ValueError(
                "scale_correction_group_count must be a non-negative integer, "
                f"got {corrections.scale_correction_group_count!r}"
            )

    group_ids = None
    n_scale_groups = 0
    if corrections.group_ids is not None:
        group_ids = np.asarray(corrections.group_ids, dtype=np.int64).reshape(-1)
        if group_ids.shape != (n_images,):
            raise ValueError(f"group_ids must have shape ({n_images},), got {group_ids.shape}")
        if group_ids.size and int(np.min(group_ids)) < 0:
            raise ValueError("group_ids must be non-negative")
        inferred_scale_group_count = int(np.max(group_ids)) + 1 if group_ids.size else 1
        n_scale_groups = max(explicit_scale_group_count, inferred_scale_group_count)

    normalization_log_z = None
    if posterior.normalization_log_z is not None:
        normalization_log_z = np.asarray(posterior.normalization_log_z, dtype=np.float64)
        if normalization_log_z.shape != (n_images,):
            raise ValueError(
                f"normalization_log_z must have shape ({n_images},), got {normalization_log_z.shape}",
            )

    normalization_log_evidence = None
    if posterior.normalization_log_evidence is not None:
        normalization_log_evidence = np.asarray(posterior.normalization_log_evidence, dtype=np.float64)
        if normalization_log_evidence.shape != (n_images,):
            raise ValueError(
                f"normalization_log_evidence must have shape ({n_images},), got {normalization_log_evidence.shape}",
            )
    if normalization_log_z is not None and normalization_log_evidence is not None:
        raise ValueError("Provide only one of normalization_log_z or normalization_log_evidence")

    reconstruction_probability_threshold = None
    if search.reconstruction_probability_threshold is not None:
        reconstruction_probability_threshold = np.asarray(
            search.reconstruction_probability_threshold,
            dtype=np.float64,
        )
        if reconstruction_probability_threshold.shape != (n_images,):
            raise ValueError(
                "reconstruction_probability_threshold must have shape "
                f"({n_images},), got {reconstruction_probability_threshold.shape}",
            )
        if not np.all(np.isfinite(reconstruction_probability_threshold)):
            raise ValueError("reconstruction_probability_threshold must be finite")
        if np.any(reconstruction_probability_threshold < 0.0):
            raise ValueError("reconstruction_probability_threshold must be non-negative")

    translation_prior_centers = validate_translation_prior_centers(
        posterior.translation_prior_centers,
        n_images=n_images,
        n_dims=local_layout.translation_grid.shape[1],
    )
    return LocalEMInputPlan(
        n_images=n_images,
        class_log_prior=class_log_prior,
        group_ids=group_ids,
        n_scale_groups=n_scale_groups,
        normalization_log_z=normalization_log_z,
        normalization_log_evidence=normalization_log_evidence,
        reconstruction_probability_threshold=reconstruction_probability_threshold,
        translation_prior_centers=translation_prior_centers,
    )
