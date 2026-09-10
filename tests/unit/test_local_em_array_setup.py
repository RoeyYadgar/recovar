from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_array_setup
from recovar.em.dense_single_volume.local_em_array_setup import (
    LocalEMFourierPlan,
    make_local_em_precision,
    plan_local_em_fourier,
    plan_local_em_reconstruction,
    prepare_local_big_jit_static_inputs,
)
from recovar.em.dense_single_volume.local_em_planning import (
    plan_local_em_geometry,
    plan_local_em_modes,
)
from recovar.em.dense_single_volume.local_em_types import (
    LocalEMRequestedOutputs,
    LocalProjectionSettings,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)


def _plans(
    *,
    image_size=8,
    current_size=8,
    reconstruction_current_size=None,
    reconstruction_padding_factor=1,
    mstep_relion_x_half=False,
    score_only=False,
    **projection_overrides,
):
    search = LocalSearchSettings(
        current_size=current_size,
        reconstruction_current_size=reconstruction_current_size,
    )
    geometry = plan_local_em_geometry(
        experiment_dataset=SimpleNamespace(
            image_shape=(image_size, image_size),
            volume_shape=(image_size, image_size, image_size),
        ),
        local_layout=SimpleNamespace(translation_grid=np.zeros((3, 2), dtype=np.float32)),
        search=search,
    )
    projection = LocalProjectionSettings(
        reconstruction_padding_factor=reconstruction_padding_factor,
        **projection_overrides,
    )
    reconstruction_settings = LocalReconstructionSettings(
        mstep_relion_x_half=mstep_relion_x_half,
        score_only=score_only,
        disable_adjoint_y=score_only,
        disable_adjoint_ctf=score_only,
    )
    mode = plan_local_em_modes(
        scoring=LocalScoringSettings(),
        reconstruction=reconstruction_settings,
        outputs=LocalEMRequestedOutputs(),
    )
    reconstruction = plan_local_em_reconstruction(
        geometry=geometry,
        projection=projection,
        mode=mode,
    )
    return geometry, search, projection, mode, reconstruction


def test_local_em_precision_preserves_grouped_dtype_settings():
    precision = make_local_em_precision(
        scoring=LocalScoringSettings(
            use_float64_scoring=True,
            use_float64_normalization=False,
        ),
        projection=LocalProjectionSettings(use_float64_projections=True),
    )

    assert precision.use_float64_scoring
    assert precision.use_float64_projections
    assert not precision.use_float64_normalization


def test_local_reconstruction_plan_preserves_native_and_relion_shapes():
    *_, native = _plans(reconstruction_padding_factor=2)
    *_, relion = _plans(
        image_size=128,
        current_size=70,
        reconstruction_padding_factor=2,
        mstep_relion_x_half=True,
    )

    assert native.volume_shape == (16, 16, 16)
    assert relion.volume_shape == (143, 143, 143)
    with pytest.raises(FrozenInstanceError):
        native.volume_shape = (8, 8, 8)


def test_local_fourier_plan_preserves_full_box_metadata():
    geometry, search, projection, mode, reconstruction = _plans()
    plan = plan_local_em_fourier(
        geometry=geometry,
        search=search,
        projection=projection,
        mode=mode,
        reconstruction=reconstruction,
    )

    assert isinstance(plan, LocalEMFourierPlan)
    assert plan.reconstruction_accumulator_shape == (8, 8, 5)
    assert plan.reconstruction_accumulator_size == 8 * 8 * 5
    assert plan.accumulator_allocation_size == plan.reconstruction_accumulator_size
    assert not plan.window.use_window
    assert plan.mstep_reconstruction_window_indices is None
    assert plan.mstep_adjoint_max_r is None
    assert plan.projection.kwargs() == {
        "relion_texture_interp": False,
        "relion_acc_double_floorf_quirk": False,
        "force_jax": False,
    }
    assert plan.projection_mode == "full"


def test_local_fourier_plan_preserves_windowed_xhalf_and_score_only_metadata():
    geometry, search, projection, mode, reconstruction = _plans(
        current_size=6,
        reconstruction_padding_factor=2,
        mstep_relion_x_half=True,
        score_only=True,
    )
    plan = plan_local_em_fourier(
        geometry=geometry,
        search=search,
        projection=projection,
        mode=mode,
        reconstruction=reconstruction,
        relion_projector_half=object(),
    )

    assert plan.reconstruction_accumulator_shape == (15, 15, 8)
    assert plan.reconstruction_accumulator_size == 15 * 15 * 8
    assert plan.accumulator_allocation_size == 1
    assert plan.window.use_window
    assert plan.mstep_reconstruction_window_indices.dtype == np.int32
    assert plan.mstep_adjoint_max_r == 3.0
    assert plan.projection.kwargs()["max_r"] == 3.0
    assert plan.projection_mode == "relion_projector"


@pytest.mark.parametrize(
    ("projection_overrides", "indexed_available", "expected_mode"),
    [
        ({"force_jax": True}, True, "windowed_full_jax"),
        ({"relion_texture_interp": True}, True, "windowed_full_texture"),
        ({}, False, "windowed_full_cuda_unavailable"),
        ({}, True, "windowed_indexed_cuda"),
    ],
)
def test_local_fourier_plan_preserves_projection_route_precedence(
    monkeypatch,
    projection_overrides,
    indexed_available,
    expected_mode,
):
    monkeypatch.setattr(
        local_em_array_setup,
        "indexed_projection_available",
        lambda: indexed_available,
    )
    geometry, search, projection, mode, reconstruction = _plans(
        current_size=6,
        **projection_overrides,
    )

    plan = plan_local_em_fourier(
        geometry=geometry,
        search=search,
        projection=projection,
        mode=mode,
        reconstruction=reconstruction,
    )

    assert plan.projection_mode == expected_mode


def test_local_big_jit_static_inputs_preserve_full_window_sentinels():
    geometry, search, projection, mode, reconstruction = _plans()
    fourier = plan_local_em_fourier(
        geometry=geometry,
        search=search,
        projection=projection,
        mode=mode,
        reconstruction=reconstruction,
    )
    precision = make_local_em_precision(
        scoring=LocalScoringSettings(
            score_with_masked_images=False,
            use_float64_scoring=True,
        ),
        projection=projection,
    )

    inputs = prepare_local_big_jit_static_inputs(
        experiment_dataset=SimpleNamespace(),
        geometry=geometry,
        fourier=fourier,
        mode=mode,
        scoring=LocalScoringSettings(score_with_masked_images=False),
        precision=precision,
    )

    assert inputs.image_mask.shape == geometry.image_shape
    assert inputs.image_mask_mode == "none"
    np.testing.assert_array_equal(inputs.score_window_indices, np.arange(geometry.n_half))
    np.testing.assert_array_equal(inputs.reconstruction_window_indices, np.arange(geometry.n_half))
    assert inputs.mstep_reconstruction_window_indices is inputs.reconstruction_window_indices
    assert inputs.disabled_noise_wsum.dtype == np.float64
    assert inputs.disabled_noise_image_power.shape == (1,)
    assert inputs.disabled_noise_a2.shape == (1,)
    assert inputs.disabled_noise_xa.shape == (1,)
    assert inputs.disabled_noise_scale.shape == (1,)
    assert inputs.disabled_group_ids.dtype == np.int32
    assert inputs.disabled_noise_shell_indices.shape == (geometry.n_half,)
