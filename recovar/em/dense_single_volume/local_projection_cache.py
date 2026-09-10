"""Bounded RELION projection-cache planning and construction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.jax_runtime import block_until_ready as _block_until_ready
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.local_layout import LocalBucketSpec
from recovar.em.dense_single_volume.runtime_options import current_environment as _runtime_environment

logger = logging.getLogger(__name__)

EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB = 0.0
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB"
EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS = 64_000_000
EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV = (
    "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS"
)
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS = 64
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV = "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS"


@dataclass(frozen=True)
class LocalRelionProjectionCache:
    """One materialized projection-cache group."""

    projections: jnp.ndarray
    id_map: jnp.ndarray
    enabled: bool
    row_count: int = 0
    id_map_row_count: int = 0
    n_projection_pixels: int = 0
    estimated_gb: float = 0.0
    build_s: float = 0.0


@dataclass(frozen=True)
class LocalRelionProjectionCachePlan:
    """Immutable group schedule and memory limits for one exact-local call."""

    buckets: tuple[LocalBucketSpec, ...]
    groups: tuple[tuple[int, int, int], ...]
    capacity_rows: int
    cap_gb: float
    n_projection_pixels: int
    id_map_rows: int


@dataclass
class LocalRelionProjectionCacheStats:
    """Mutable observations from materializing scheduled cache groups."""

    groups_built: int = 0
    total_build_s: float = 0.0
    max_rows: int = 0
    max_estimated_gb: float = 0.0

    def record(self, cache: LocalRelionProjectionCache) -> None:
        self.groups_built += int(cache.enabled)
        self.total_build_s += float(cache.build_s)
        self.max_rows = max(self.max_rows, int(cache.row_count))
        self.max_estimated_gb = max(self.max_estimated_gb, float(cache.estimated_gb))


def _optional_nonnegative_float_env(name: str, default: float) -> float:
    raw = _runtime_environment().get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative float, got {raw!r}") from exc
    if value < 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a non-negative finite float, got {raw!r}")
    return value


def _exact_local_relion_projection_cache_chunk_rows(n_projection_pixels: int) -> int:
    target = int(
        _runtime_environment().get(
            EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV,
            EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS,
        )
    )
    if target <= 0:
        raise ValueError(f"{EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV} must be positive")
    return max(1, int(target) // max(1, int(n_projection_pixels)))


def _disabled_relion_projection_cache() -> LocalRelionProjectionCache:
    return LocalRelionProjectionCache(
        projections=jnp.zeros((1, 1), dtype=jnp.complex64),
        id_map=jnp.zeros((1,), dtype=jnp.int32),
        enabled=False,
    )


def _exact_local_relion_projection_cache_capacity_rows(n_projection_pixels: int) -> tuple[int, float]:
    max_gb = _optional_nonnegative_float_env(
        EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV,
        EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB,
    )
    if max_gb <= 0.0 or int(n_projection_pixels) <= 0:
        return 0, float(max_gb)
    bytes_per_row = int(n_projection_pixels) * np.dtype(np.complex64).itemsize
    return int((max_gb * 1e9) // max(1, bytes_per_row)), float(max_gb)


def _bucket_valid_rotation_ids(bucket: LocalBucketSpec) -> np.ndarray:
    ids = np.asarray(bucket.local_rotation_ids, dtype=np.int64)
    mask = np.asarray(bucket.local_rotation_mask, dtype=bool) & (ids >= 0)
    if not np.any(mask):
        return np.zeros(0, dtype=np.int64)
    return np.unique(ids[mask])


def _bucket_rotation_id_center(bucket: LocalBucketSpec) -> int:
    ids = _bucket_valid_rotation_ids(bucket)
    if ids.size == 0:
        return -1
    return int(np.median(ids))


def _sort_buckets_for_relion_projection_cache(bucket_specs: list[LocalBucketSpec]) -> list[LocalBucketSpec]:
    return sorted(
        bucket_specs,
        key=lambda bucket: (
            int(bucket.bucket_rotation_count),
            _bucket_rotation_id_center(bucket),
            int(bucket.image_indices[0]) if int(bucket.image_indices.shape[0]) else -1,
        ),
    )


def _exact_local_relion_projection_cache_max_groups() -> int:
    raw = _runtime_environment().get(
        EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV,
        str(EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS),
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV} must be positive") from exc
    if value <= 0:
        raise ValueError(f"{EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV} must be positive")
    return value


def _plan_exact_local_relion_projection_cache_groups(
    bucket_specs: list[LocalBucketSpec],
    *,
    cache_row_capacity: int,
) -> list[tuple[int, int, int]]:
    """Greedily group consecutive buckets under a compact projection-cache row cap."""

    if cache_row_capacity <= 0 or not bucket_specs:
        return []

    groups: list[tuple[int, int, int]] = []
    start = 0
    active_ids: set[int] = set()
    for bucket_index, bucket in enumerate(bucket_specs):
        bucket_ids = set(int(x) for x in _bucket_valid_rotation_ids(bucket).tolist())
        if len(bucket_ids) > cache_row_capacity:
            logger.info(
                "Exact local RELION projection cache disabled: one bucket needs %d rows, cap is %d",
                len(bucket_ids),
                cache_row_capacity,
            )
            return []
        if active_ids and len(active_ids | bucket_ids) > cache_row_capacity:
            groups.append((start, bucket_index, len(active_ids)))
            start = bucket_index
            active_ids = set(bucket_ids)
        else:
            active_ids |= bucket_ids
    if active_ids or start < len(bucket_specs):
        groups.append((start, len(bucket_specs), len(active_ids)))
    return groups


def plan_local_relion_projection_cache(
    bucket_specs: list[LocalBucketSpec],
    rotation_ids_flat,
    *,
    n_projection_pixels: int,
    enabled: bool,
) -> LocalRelionProjectionCachePlan:
    """Resolve the bounded projection-cache schedule without materializing it."""

    buckets = list(bucket_specs)
    groups: list[tuple[int, int, int]] = []
    capacity_rows = 0
    cap_gb = 0.0
    id_map_rows = 0
    if enabled:
        requested_cache_rows, cap_gb = _exact_local_relion_projection_cache_capacity_rows(n_projection_pixels)
        if requested_cache_rows > 0:
            buckets = _sort_buckets_for_relion_projection_cache(buckets)
        groups = _plan_exact_local_relion_projection_cache_groups(
            buckets,
            cache_row_capacity=int(requested_cache_rows),
        )
        max_cache_groups = _exact_local_relion_projection_cache_max_groups()
        if len(groups) > max_cache_groups:
            logger.info(
                "Exact local RELION projection cache disabled: planned groups=%d exceeds max_groups=%d "
                "(capacity_rows=%d cap=%.2f GB projection_pixels=%d)",
                len(groups),
                max_cache_groups,
                int(requested_cache_rows),
                float(cap_gb),
                int(n_projection_pixels),
            )
            groups = []
        if groups:
            capacity_rows = int(max(row_count for _, _, row_count in groups))
            valid_layout_ids = np.asarray(rotation_ids_flat, dtype=np.int64)
            valid_layout_ids = valid_layout_ids[valid_layout_ids >= 0]
            id_map_rows = int(np.max(valid_layout_ids)) + 1 if valid_layout_ids.size else 1
            logger.info(
                "Exact local RELION projection cache groups enabled: groups=%d capacity_rows=%d "
                "requested_rows=%d cap=%.2f GB projection_pixels=%d id_map_rows=%d",
                len(groups),
                capacity_rows,
                int(requested_cache_rows),
                float(cap_gb),
                int(n_projection_pixels),
                id_map_rows,
            )
    return LocalRelionProjectionCachePlan(
        buckets=tuple(buckets),
        groups=tuple(groups),
        capacity_rows=capacity_rows,
        cap_gb=float(cap_gb),
        n_projection_pixels=int(n_projection_pixels) if groups else 0,
        id_map_rows=id_map_rows,
    )


def _build_exact_local_relion_projection_cache_for_buckets(
    bucket_specs: list[LocalBucketSpec],
    relion_projector_half,
    *,
    image_shape,
    n_projection_pixels: int,
    relion_projector_r_max: int,
    projection_padding_factor: int,
    projection_relion_texture_interp: bool | None,
    projection_pixel_indices,
    projection_relion_acc_double_floorf_quirk: bool = False,
    projector_output_size: int,
    cache_row_capacity: int,
    max_global_rotation_id: int,
    group_index: int,
    n_groups: int,
) -> LocalRelionProjectionCache:
    """Precompute compact RELION projections for one bounded bucket group."""

    if cache_row_capacity <= 0 or not bucket_specs:
        return _disabled_relion_projection_cache()

    ids_parts = []
    rotation_parts = []
    for bucket in bucket_specs:
        ids = np.asarray(bucket.local_rotation_ids, dtype=np.int64)
        mask = np.asarray(bucket.local_rotation_mask, dtype=bool) & (ids >= 0)
        if not np.any(mask):
            continue
        ids_parts.append(ids[mask])
        rotation_parts.append(np.asarray(bucket.local_rotations, dtype=np.float32)[mask])
    if not ids_parts:
        return _disabled_relion_projection_cache()

    valid_ids = np.concatenate(ids_parts, axis=0)
    valid_rotations = np.concatenate(rotation_parts, axis=0)
    unique_ids, first_positions = np.unique(valid_ids, return_index=True)
    row_count = int(unique_ids.size)
    if row_count > int(cache_row_capacity):
        raise RuntimeError(
            "internal projection-cache planner error: group has "
            f"{row_count} rows but capacity is {int(cache_row_capacity)}"
        )
    id_map_row_count = int(max(max_global_rotation_id + 1, int(np.max(valid_ids)) + 1))
    estimated_gb = float(cache_row_capacity * n_projection_pixels * np.dtype(np.complex64).itemsize / 1e9)

    cache_t0 = time.time()
    cache_rotations = valid_rotations[first_positions]
    id_map = np.zeros(id_map_row_count, dtype=np.int32)
    id_map[unique_ids] = np.arange(row_count, dtype=np.int32)

    chunk_rows = _exact_local_relion_projection_cache_chunk_rows(n_projection_pixels)
    host_cache = np.empty((cache_row_capacity, n_projection_pixels), dtype=np.complex64)
    logger.info(
        "Exact local RELION projection cache group %d/%d build: rows=%d capacity=%d "
        "id_map_rows=%d projection_pixels=%d estimated=%.2f GB chunk_rows=%d buckets=%d",
        int(group_index) + 1,
        int(n_groups),
        row_count,
        int(cache_row_capacity),
        id_map_row_count,
        n_projection_pixels,
        estimated_gb,
        int(chunk_rows),
        len(bucket_specs),
    )
    for start in range(0, row_count, chunk_rows):
        stop = min(row_count, start + chunk_rows)
        proj_chunk, _ = _compute_relion_projector_projections_block(
            relion_projector_half,
            jnp.asarray(cache_rotations[start:stop], dtype=jnp.float32),
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            relion_texture_interp=projection_relion_texture_interp,
            relion_acc_double_floorf_quirk=projection_relion_acc_double_floorf_quirk,
            projector_output_size=int(projector_output_size) if int(projector_output_size) > 0 else None,
            pixel_indices=projection_pixel_indices,
        )
        _block_until_ready(proj_chunk)
        host_cache[start:stop] = np.asarray(proj_chunk, dtype=np.complex64)
        del proj_chunk

    projections = jnp.asarray(host_cache)
    id_map_jnp = jnp.asarray(id_map, dtype=jnp.int32)
    _block_until_ready(projections, id_map_jnp)
    build_s = time.time() - cache_t0
    logger.info(
        "Exact local RELION projection cache group %d/%d ready: rows=%d capacity=%d "
        "id_map_rows=%d projection_pixels=%d estimated=%.2f GB build=%.1fs",
        int(group_index) + 1,
        int(n_groups),
        row_count,
        int(cache_row_capacity),
        id_map_row_count,
        n_projection_pixels,
        estimated_gb,
        build_s,
    )
    return LocalRelionProjectionCache(
        projections=projections,
        id_map=id_map_jnp,
        enabled=True,
        row_count=row_count,
        id_map_row_count=id_map_row_count,
        n_projection_pixels=n_projection_pixels,
        estimated_gb=estimated_gb,
        build_s=build_s,
    )
