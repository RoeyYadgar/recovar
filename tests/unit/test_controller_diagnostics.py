from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume.debug_dumps import _maybe_dump_noise_update_debug


def test_noise_update_capture_retains_filename_schema_and_dtypes(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_NOISE_DEBUG_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_NOISE_DEBUG_DUMP_ITERATION", "2")
    stats = SimpleNamespace(
        wsum_sigma2_noise=np.array([1.0, 2.0, 3.0]),
        wsum_img_power=np.array([4.0, 5.0, 6.0]),
        sumw=7.0,
        wsum_norm_correction=np.array([8.0]),
        wsum_noise_a2=np.array([9.0, 10.0, 11.0]),
        wsum_noise_xa=np.array([12.0, 13.0, 14.0]),
    )

    _maybe_dump_noise_update_debug(
        iteration=2,
        current_size=4,
        image_shape=(4, 4),
        noise_stats_per_half=(stats, stats),
        previous_noise_radial_per_half=(np.ones(3), np.ones(3)),
        noise_from_res_per_half=(np.full(3, 2.0), np.full(3, 3.0)),
        noise_from_res=np.full(3, 2.5),
    )

    path = tmp_path / "recovar_noise_update_it003.npz"
    assert path.is_file()
    with np.load(path, allow_pickle=False) as capture:
        assert set(capture.files) == {
            "current_size",
            "half1_previous_sigma2_noise",
            "half1_sigma2_noise",
            "half1_sumw",
            "half1_wsum_img_power",
            "half1_wsum_noise_a2",
            "half1_wsum_noise_xa",
            "half1_wsum_norm_correction",
            "half1_wsum_sigma2_noise",
            "half1_wsum_total",
            "half2_previous_sigma2_noise",
            "half2_sigma2_noise",
            "half2_sumw",
            "half2_wsum_img_power",
            "half2_wsum_noise_a2",
            "half2_wsum_noise_xa",
            "half2_wsum_norm_correction",
            "half2_wsum_sigma2_noise",
            "half2_wsum_total",
            "half_shell_counts",
            "half_weighted_shell_counts",
            "image_shape",
            "mean_sigma2_noise",
            "one_based_iteration",
            "relion_half_plane_shell_counts",
            "shell_index_half",
            "zero_based_iteration",
        }
        assert capture["zero_based_iteration"].dtype == np.int32
        assert capture["half1_wsum_total"].dtype == np.float64
        assert capture["shell_index_half"].dtype == np.int32
