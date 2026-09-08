"""Host-side input/output contracts for the dense EM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class DenseEMOutputSpec:
    """Describe which optional values are present in the legacy result tuple."""

    return_stats: bool = False
    accumulate_noise: bool = False
    return_profile: bool = False

    @property
    def legacy_tuple_size(self) -> int:
        """Return the exact tuple length selected by this output specification."""

        return 4 + int(self.return_stats) + int(self.accumulate_noise) + int(self.return_profile)


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

    @property
    def legacy_tuple_spec(self) -> DenseEMOutputSpec:
        """Return the tuple shape produced by the compatibility engine."""

        return DenseEMOutputSpec(
            return_stats=self.return_stats,
            accumulate_noise=self.accumulate_noise,
            return_profile=self.return_profile,
        )


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

    def to_legacy_tuple(self, output_spec: DenseEMOutputSpec) -> tuple[Any, ...]:
        """Serialize this result using ``run_em``'s tuple contract."""

        result = [self.new_mean, self.hard_assignment, self.Ft_y, self.Ft_ctf]
        if output_spec.return_stats:
            result.append(self.relion_stats)
        if output_spec.accumulate_noise:
            result.append(self.noise_stats)
        if output_spec.return_profile:
            result.append(self.profile_stats)
        return tuple(result)

    @classmethod
    def from_legacy_tuple(
        cls,
        output: Sequence[Any],
        output_spec: DenseEMOutputSpec,
    ) -> DenseEMResult:
        """Parse and validate a ``run_em`` compatibility tuple."""

        expected_size = output_spec.legacy_tuple_size
        if len(output) != expected_size:
            raise ValueError(
                "Dense EM output tuple does not match its output specification: "
                f"expected {expected_size} values, received {len(output)}"
            )

        cursor = 0
        new_mean, hard_assignment, Ft_y, Ft_ctf = output[cursor : cursor + 4]
        cursor += 4
        relion_stats = output[cursor] if output_spec.return_stats else None
        cursor += int(output_spec.return_stats)
        noise_stats = output[cursor] if output_spec.accumulate_noise else None
        cursor += int(output_spec.accumulate_noise)
        profile_stats = output[cursor] if output_spec.return_profile else None

        return cls(
            new_mean=new_mean,
            hard_assignment=hard_assignment,
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            relion_stats=relion_stats,
            noise_stats=noise_stats,
            profile_stats=profile_stats,
        )
