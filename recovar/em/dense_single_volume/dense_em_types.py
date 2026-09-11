"""Host-side input/output contracts for the dense EM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DenseEMInputs:
    """Required dataset, model, and hypothesis-grid inputs for dense EM."""

    experiment_dataset: Any
    mean: Any
    mean_variance: Any
    noise_variance: Any
    rotations: Any
    translations: Any
    disc_type: str


@dataclass(frozen=True)
class DenseSearchSettings:
    """Dense search window, priors, subset, and candidate constraints."""

    current_size: int | None = None
    rotation_log_prior: Any | None = None
    translation_log_prior: Any | None = None
    image_indices: Any | None = None
    rotation_translation_mask: Any | None = None


@dataclass(frozen=True)
class DenseExecutionSettings:
    """Host batching and pass-2 execution controls for dense EM."""

    image_batch_size: int = 500
    rotation_block_size: int = 5000
    sparse_pass2: bool = True


@dataclass(frozen=True)
class DenseScoringSettings:
    """Score representation, masking, precision, and first-iteration policy."""

    score_with_masked_images: bool = False
    half_spectrum_scoring: bool = False
    relion_firstiter_score_mode: str = "gaussian"
    relion_firstiter_winner_take_all: bool = False
    use_float64_scoring: bool = False


@dataclass(frozen=True)
class DenseProjectionSettings:
    """Projection, reconstruction-grid, precision, and window settings."""

    projection_padding_factor: int = 1
    reconstruction_padding_factor: int = 1
    use_float64_projections: bool = False
    do_gridding_correction: bool = False
    square_window: bool = False


@dataclass(frozen=True)
class DenseCorrectionInputs:
    """Optional per-image correction inputs."""

    image_corrections: Any | None = None
    scale_corrections: Any | None = None
    image_pre_shifts: Any | None = None


@dataclass(frozen=True)
class DensePosteriorInputs:
    """Optional class, normalization, and translation-prior inputs."""

    class_log_prior: float = 0.0
    normalization_log_evidence: Any | None = None
    translation_prior_centers: Any | None = None


@dataclass(frozen=True)
class DenseReconstructionSettings:
    """M-step route and sufficient-statistic policy."""

    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    score_only: bool = False
    relion_half_volume_mstep: bool = False


@dataclass(frozen=True)
class DenseEMRequestedOutputs:
    """Select optional dense computation products without shifting fields."""

    return_stats: bool = False
    accumulate_noise: bool = False
    return_profile: bool = False
    return_half_volume_accumulators: bool = False

@dataclass(frozen=True)
class DenseEMRequest:
    """Composed host-side request for dense EM."""

    inputs: DenseEMInputs
    search: DenseSearchSettings = DenseSearchSettings()
    execution: DenseExecutionSettings = DenseExecutionSettings()
    scoring: DenseScoringSettings = DenseScoringSettings()
    projection: DenseProjectionSettings = DenseProjectionSettings()
    corrections: DenseCorrectionInputs = DenseCorrectionInputs()
    posterior: DensePosteriorInputs = DensePosteriorInputs()
    reconstruction: DenseReconstructionSettings = DenseReconstructionSettings()
    outputs: DenseEMRequestedOutputs = DenseEMRequestedOutputs()


@dataclass(frozen=True)
class DenseEMResult:
    """Stable named result for the dense EM host boundary.

    The public ``run_em`` compatibility API retains its historical
    flag-dependent tuple. This type centralizes that tuple's ordering so typed
    callers can migrate without duplicating optional-output cursor logic.
    """

    new_mean: Any
    hard_assignment: Any
    Ft_y: Any
    Ft_ctf: Any
    relion_stats: Any | None = None
    noise_stats: Any | None = None
    profile_stats: Any | None = None

    def to_legacy_tuple(self, outputs: DenseEMRequestedOutputs) -> tuple[Any, ...]:
        """Serialize this result using ``run_em``'s tuple contract."""

        result = [self.new_mean, self.hard_assignment, self.Ft_y, self.Ft_ctf]
        if outputs.return_stats:
            result.append(self.relion_stats)
        if outputs.accumulate_noise:
            result.append(self.noise_stats)
        if outputs.return_profile:
            result.append(self.profile_stats)
        return tuple(result)
