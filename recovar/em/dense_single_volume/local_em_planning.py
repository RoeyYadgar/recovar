"""Host-side validation and planning for exact local EM execution."""

from __future__ import annotations

from dataclasses import dataclass

from recovar.em.dense_single_volume.local_em_types import (
    LocalEMRequestedOutputs,
    LocalReconstructionSettings,
    LocalScoringSettings,
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
