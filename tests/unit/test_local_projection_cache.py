from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.local_layout import LocalBucketSpec
from recovar.em.dense_single_volume.local_projection_cache import (
    EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV,
    LocalRelionProjectionCache,
    LocalRelionProjectionCacheStats,
    plan_local_relion_projection_cache,
)


def _bucket(image_index: int, rotation_ids) -> LocalBucketSpec:
    rotation_ids = np.asarray(rotation_ids, dtype=np.int32)
    n_rotations = int(rotation_ids.size)
    return LocalBucketSpec(
        image_indices=np.asarray([image_index], dtype=np.int32),
        bucket_image_count=1,
        bucket_rotation_count=n_rotations,
        actual_rotation_counts=np.asarray([n_rotations], dtype=np.int32),
        local_rotation_ids=rotation_ids[None, :],
        local_rotations=np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (1, n_rotations, 3, 3),
        ).copy(),
        local_rotation_log_prior=np.zeros((1, n_rotations), dtype=np.float32),
        local_rotation_mask=np.ones((1, n_rotations), dtype=bool),
        translation_log_prior=np.zeros((1, 1), dtype=np.float32),
    )


def test_projection_cache_plan_is_disabled_without_route_request():
    buckets = [_bucket(0, [0, 1])]

    plan = plan_local_relion_projection_cache(
        buckets,
        np.asarray([0, 1], dtype=np.int32),
        n_projection_pixels=12,
        enabled=False,
    )

    assert plan.buckets == tuple(buckets)
    assert plan.groups == ()
    assert plan.capacity_rows == 0
    assert plan.cap_gb == 0.0
    assert plan.n_projection_pixels == 0
    assert plan.id_map_rows == 0


def test_projection_cache_plan_groups_and_records_stats(monkeypatch):
    buckets = [_bucket(0, [0, 1]), _bucket(1, [2, 3])]
    bytes_per_row = 12 * np.dtype(np.complex64).itemsize
    monkeypatch.setenv(EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV, str(2 * bytes_per_row / 1e9))

    plan = plan_local_relion_projection_cache(
        buckets,
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        n_projection_pixels=12,
        enabled=True,
    )

    assert plan.groups == ((0, 1, 2), (1, 2, 2))
    assert plan.capacity_rows == 2
    assert plan.n_projection_pixels == 12
    assert plan.id_map_rows == 4

    stats = LocalRelionProjectionCacheStats()
    stats.record(
        LocalRelionProjectionCache(
            projections=np.zeros((2, 12), dtype=np.complex64),
            id_map=np.arange(4, dtype=np.int32),
            enabled=True,
            row_count=2,
            estimated_gb=0.25,
            build_s=1.5,
        )
    )
    assert stats.groups_built == 1
    assert stats.total_build_s == 1.5
    assert stats.max_rows == 2
    assert stats.max_estimated_gb == 0.25


def test_projection_cache_plan_retains_sorted_buckets_when_group_limit_disables(monkeypatch):
    buckets = [_bucket(0, [2, 3, 4]), _bucket(1, [0, 1])]
    bytes_per_row = 12 * np.dtype(np.complex64).itemsize
    monkeypatch.setenv(EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV, str(3 * bytes_per_row / 1e9))
    monkeypatch.setenv(EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV, "1")

    plan = plan_local_relion_projection_cache(
        buckets,
        np.arange(5, dtype=np.int32),
        n_projection_pixels=12,
        enabled=True,
    )

    assert [int(bucket.bucket_rotation_count) for bucket in plan.buckets] == [2, 3]
    assert plan.groups == ()
    assert plan.capacity_rows == 0
    assert plan.id_map_rows == 0
