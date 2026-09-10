from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.local_em_array_setup import (
    plan_local_em_fourier,
    plan_local_em_reconstruction,
)
from recovar.em.dense_single_volume.local_em_batch_planning import (
    EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS,
    LocalBucketPlan,
    LocalBucketSummary,
    LocalMicrobatchPlan,
    plan_local_buckets,
    plan_local_microbatch_cap,
    plan_local_microbatch_route,
    summarize_local_buckets,
)
from recovar.em.dense_single_volume.local_em_planning import (
    plan_local_em_geometry,
    plan_local_em_modes,
)
from recovar.em.dense_single_volume.local_em_types import (
    LocalEMRequestedOutputs,
    LocalExecutionSettings,
    LocalProjectionSettings,
    LocalReconstructionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout


def _planning_inputs(
    *,
    image_size=128,
    current_size=70,
    rotation_count=200,
    image_batch_size=2,
    rotation_block_size=128,
    explicit_cap=1000,
    mstep_relion_x_half=True,
    score_only=False,
):
    search = LocalSearchSettings(current_size=current_size)
    rotation_counts = np.asarray([rotation_count], dtype=np.int32)
    layout = LocalHypothesisLayout(
        n_global_rotations=rotation_count,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.asarray([0, rotation_count], dtype=np.int64),
        rotation_ids_flat=np.arange(rotation_count, dtype=np.int32),
        rotations_flat=np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (rotation_count, 3, 3),
        ).copy(),
        rotation_log_priors_flat=np.zeros(rotation_count, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((3, 2), dtype=np.float32),
        translation_log_priors=np.zeros((1, 3), dtype=np.float32),
    )
    geometry = plan_local_em_geometry(
        experiment_dataset=SimpleNamespace(
            image_shape=(image_size, image_size),
            volume_shape=(image_size, image_size, image_size),
        ),
        local_layout=layout,
        search=search,
    )
    projection = LocalProjectionSettings(reconstruction_padding_factor=2)
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
    fourier = plan_local_em_fourier(
        geometry=geometry,
        search=search,
        projection=projection,
        mode=mode,
        reconstruction=reconstruction,
        relion_projector_half=object() if mstep_relion_x_half else None,
    )
    execution = LocalExecutionSettings(
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=explicit_cap,
    )
    return layout, geometry, fourier, execution, mode, reconstruction


def test_local_microbatch_plan_preserves_xhalf_tail_cap():
    layout, geometry, fourier, execution, mode, reconstruction = _planning_inputs()
    route = plan_local_microbatch_route(
        geometry=geometry,
        reconstruction=reconstruction,
        mode=mode,
        relion_projector_half=object(),
    )
    plan = plan_local_microbatch_cap(
        local_layout=layout,
        geometry=geometry,
        fourier=fourier,
        execution=execution,
        mode=mode,
        route=route,
    )

    assert route.xhalf_bpref_mstep
    assert not route.full_bpref
    assert route.auto_boost_factor == 1.0
    assert plan == LocalMicrobatchPlan(
        initial_cap=1000,
        tail_cap=256,
        effective_cap=256,
        projection_target_row_pixels=None,
    )
    with pytest.raises(FrozenInstanceError):
        plan.effective_cap = 1


def test_local_microbatch_plan_preserves_xhalf_projection_cap():
    layout, geometry, fourier, execution, mode, reconstruction = _planning_inputs(
        image_size=256,
        current_size=256,
        rotation_count=128,
        image_batch_size=100,
        explicit_cap=10000,
    )
    route = plan_local_microbatch_route(
        geometry=geometry,
        reconstruction=reconstruction,
        mode=mode,
        relion_projector_half=object(),
    )
    plan = plan_local_microbatch_cap(
        local_layout=layout,
        geometry=geometry,
        fourier=fourier,
        execution=execution,
        mode=mode,
        route=route,
    )

    expected_projection_cap = EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS // fourier.window.n_projection
    assert route.full_bpref
    assert plan.initial_cap == 10000
    assert plan.tail_cap == 10000
    assert plan.effective_cap == expected_projection_cap
    assert plan.projection_target_row_pixels == EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS


def test_local_microbatch_route_disables_xhalf_caps_for_score_only():
    _, geometry, _, _, mode, reconstruction = _planning_inputs(score_only=True)

    route = plan_local_microbatch_route(
        geometry=geometry,
        reconstruction=reconstruction,
        mode=mode,
        relion_projector_half=object(),
    )

    assert not route.xhalf_bpref_mstep
    assert not route.full_bpref
    assert route.auto_boost_factor is None


def test_local_bucket_summary_preserves_shape_frequencies_and_image_counts():
    summary = summarize_local_buckets(
        [
            SimpleNamespace(bucket_rotation_count=4, image_indices=np.zeros(2, dtype=np.int32)),
            SimpleNamespace(bucket_rotation_count=4, image_indices=np.zeros(1, dtype=np.int32)),
            SimpleNamespace(bucket_rotation_count=8, image_indices=np.zeros(3, dtype=np.int32)),
        ]
    )

    assert summary == LocalBucketSummary(
        bucket_count=3,
        image_count=6,
        rotation_size_min=4,
        rotation_size_median=4,
        rotation_size_mean=16 / 3,
        rotation_size_max=8,
        images_per_bucket_median=2,
        images_per_bucket_max=3,
        top_rotation_sizes=((4, 2), (8, 1)),
    )


def test_local_bucket_plan_wraps_existing_builder_in_immutable_topology():
    layout, _, _, execution, _, _ = _planning_inputs()
    microbatch = LocalMicrobatchPlan(
        initial_cap=256,
        tail_cap=256,
        effective_cap=256,
        projection_target_row_pixels=None,
    )

    plan = plan_local_buckets(
        local_layout=layout,
        execution=execution,
        microbatch=microbatch,
    )

    assert isinstance(plan, LocalBucketPlan)
    assert isinstance(plan.buckets, tuple)
    assert plan.total_local_rotations == 200
    assert plan.summary.bucket_count == 1
    assert plan.summary.image_count == 1
    assert plan.summary.rotation_size_max == 256
