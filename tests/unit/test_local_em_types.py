import itertools

import pytest

from recovar.em.dense_single_volume.local_em_types import LocalEMOutputSpec, LocalEMResult


@pytest.mark.unit
@pytest.mark.parametrize(
    ("accumulate_noise", "return_profile", "return_best_pose_details", "return_significant_counts"),
    itertools.product((False, True), repeat=4),
)
def test_local_em_result_round_trips_every_legacy_tuple_shape(
    accumulate_noise,
    return_profile,
    return_best_pose_details,
    return_significant_counts,
):
    output_spec = LocalEMOutputSpec(
        accumulate_noise=accumulate_noise,
        return_profile=return_profile,
        return_best_pose_details=return_best_pose_details,
        return_significant_counts=return_significant_counts,
    )
    result = LocalEMResult(
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        hard_assignment="hard_assignment",
        relion_stats="relion_stats",
        best_pose_rotations="best_pose_rotations",
        best_pose_translations="best_pose_translations",
        best_pose_rotation_ids="best_pose_rotation_ids",
        noise_stats="noise_stats",
        profile_summary={"profile": "summary"},
        significant_counts="significant_counts",
    )

    legacy_output = result.to_legacy_tuple(output_spec)

    assert len(legacy_output) == output_spec.legacy_tuple_size
    assert LocalEMResult.from_legacy_tuple(legacy_output, output_spec) == LocalEMResult(
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        hard_assignment="hard_assignment",
        relion_stats="relion_stats",
        best_pose_rotations="best_pose_rotations" if return_best_pose_details else None,
        best_pose_translations="best_pose_translations" if return_best_pose_details else None,
        best_pose_rotation_ids="best_pose_rotation_ids" if return_best_pose_details else None,
        noise_stats="noise_stats" if accumulate_noise else None,
        profile_summary={"profile": "summary"} if return_profile else None,
        significant_counts="significant_counts" if return_significant_counts else None,
    )


@pytest.mark.unit
def test_local_em_result_rejects_legacy_tuple_with_wrong_shape():
    output_spec = LocalEMOutputSpec(return_profile=True)

    with pytest.raises(ValueError, match="expected 5 values, received 4"):
        LocalEMResult.from_legacy_tuple((1, 2, 3, 4), output_spec)
