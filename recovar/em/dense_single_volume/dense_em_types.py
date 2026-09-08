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
