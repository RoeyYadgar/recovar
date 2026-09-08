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
