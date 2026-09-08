import itertools

import pytest

from recovar.em.dense_single_volume.dense_em_types import DenseEMOutputSpec, DenseEMResult


@pytest.mark.unit
@pytest.mark.parametrize(
    ("return_stats", "accumulate_noise", "return_profile"),
    itertools.product((False, True), repeat=3),
)
def test_dense_em_result_round_trips_every_legacy_tuple_shape(
    return_stats,
    accumulate_noise,
    return_profile,
):
    output_spec = DenseEMOutputSpec(
        return_stats=return_stats,
        accumulate_noise=accumulate_noise,
        return_profile=return_profile,
    )
    result = DenseEMResult(
        new_mean="new_mean",
        hard_assignment="hard_assignment",
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        relion_stats="relion_stats",
        noise_stats="noise_stats",
        profile_stats="profile_stats",
    )

    legacy_output = result.to_legacy_tuple(output_spec)

    assert len(legacy_output) == output_spec.legacy_tuple_size
    assert DenseEMResult.from_legacy_tuple(legacy_output, output_spec) == DenseEMResult(
        new_mean="new_mean",
        hard_assignment="hard_assignment",
        Ft_y="Ft_y",
        Ft_ctf="Ft_ctf",
        relion_stats="relion_stats" if return_stats else None,
        noise_stats="noise_stats" if accumulate_noise else None,
        profile_stats="profile_stats" if return_profile else None,
    )


@pytest.mark.unit
def test_dense_em_result_rejects_legacy_tuple_with_wrong_shape():
    output_spec = DenseEMOutputSpec(return_profile=True)

    with pytest.raises(ValueError, match="expected 5 values, received 4"):
        DenseEMResult.from_legacy_tuple((1, 2, 3, 4), output_spec)
