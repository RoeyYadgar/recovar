"""Memory-aware microbatch and bucket planning for exact-local EM."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

import jax
import numpy as np

from recovar.em.dense_single_volume.local_em_array_setup import (
    LocalEMFourierPlan,
    LocalEMReconstructionPlan,
)
from recovar.em.dense_single_volume.local_em_planning import LocalEMGeometryPlan, LocalEMModePlan
from recovar.em.dense_single_volume.local_em_types import LocalExecutionSettings
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    LocalHypothesisLayout,
    _exact_bucket_rotation_size,
    _exact_local_large_bucket_quantum,
    bucket_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.runtime_options import current_environment as _runtime_environment

logger = logging.getLogger(__name__)


# Keeps common 256^2 local-search buckets at two images without entering the
# three-image working set that previously exceeded memory.
EXACT_LOCAL_TARGET_ROW_PIXELS = 190_000_000
EXACT_LOCAL_TARGET_ROW_PIXELS_ENV = "RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS"
EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB = 4.0
EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB"
EXACT_LOCAL_HIGH_MEMORY_GPU_BYTES = 70 * 1024**3
EXACT_LOCAL_HIGH_MEMORY_TARGET_ROW_PIXELS = 256_000_000
EXACT_LOCAL_HIGH_MEMORY_BIG_JIT_MATMUL_MAX_GB = 8.0
EXACT_LOCAL_AUTO_MICROBATCH_BOOST = 2.0
EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV = "RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST"
EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST = 1.0
EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV = "RECOVAR_EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST"
# The fused RELION-projector M-step has a projection/interpolation temporary
# whose peak follows padded rotation rows times projected pixels. A 384-box
# H100 run completed 37x128x8258 row-pixels but OOMed when the next static
# bucket doubled to 37x256x8258 and requested a 10.12-GiB allocation. Keep
# automatic x-half buckets on the proven side of that boundary. The cap never
# splits one particle's exact rotation neighborhood.
EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS = 40_000_000
EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV = "RECOVAR_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS"
# Score-only big-JIT lowers the score residual to a dense
# (image, rotation, translation, pixel) float32 tile. Limit that one tile to
# a conservative share of memory that is still free at local-search entry;
# the remaining memory is needed by projections, inputs, outputs, and XLA.
EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION = 0.20
EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR = 1.25

_VISIBLE_GPU_MEMORY_BYTES_CACHE: int | None = None


@dataclass(frozen=True)
class LocalMicrobatchRoute:
    """Resolved x-half mode that controls exact-local memory caps."""

    xhalf_bpref_mstep: bool
    full_bpref: bool
    auto_boost_factor: float | None


@dataclass(frozen=True)
class LocalMicrobatchPlan:
    """Effective cap after the generic and x-half-specific stages."""

    initial_cap: int
    tail_cap: int
    effective_cap: int
    projection_target_row_pixels: int | None


@dataclass(frozen=True)
class LocalBucketSummary:
    """Immutable host summary of an exact-local bucket sequence."""

    bucket_count: int
    image_count: int
    rotation_size_min: int
    rotation_size_median: int
    rotation_size_mean: float
    rotation_size_max: int
    images_per_bucket_median: int
    images_per_bucket_max: int
    top_rotation_sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LocalBucketPlan:
    """Production bucket topology before any diagnostic-only filtering."""

    buckets: tuple[LocalBucketSpec, ...]
    total_local_rotations: int
    summary: LocalBucketSummary


def _visible_gpu_memory_bytes() -> int | None:
    """Return visible GPU memory in bytes when nvidia-smi is available."""

    global _VISIBLE_GPU_MEMORY_BYTES_CACHE
    if _VISIBLE_GPU_MEMORY_BYTES_CACHE is not None:
        return _VISIBLE_GPU_MEMORY_BYTES_CACHE
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        _VISIBLE_GPU_MEMORY_BYTES_CACHE = 0
        return None
    if proc.returncode != 0:
        _VISIBLE_GPU_MEMORY_BYTES_CACHE = 0
        return None
    values = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(int(float(stripped.split()[0])))
        except ValueError:
            continue
    if not values:
        _VISIBLE_GPU_MEMORY_BYTES_CACHE = 0
        return None
    _VISIBLE_GPU_MEMORY_BYTES_CACHE = int(max(values)) * 1024**2
    return _VISIBLE_GPU_MEMORY_BYTES_CACHE


def _exact_local_runtime_free_memory_bytes() -> int | None:
    """Return allocator bytes not currently live on the first local GPU."""

    try:
        devices = jax.local_devices()
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    bytes_limit = stats.get("bytes_limit")
    bytes_in_use = stats.get("bytes_in_use")
    if bytes_limit is None or bytes_in_use is None:
        return None
    free_bytes = int(bytes_limit) - int(bytes_in_use)
    return free_bytes if free_bytes > 0 else None


def _exact_local_default_target_row_pixels(*, allow_high_memory_default: bool = True) -> int:
    raw = _runtime_environment().get(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "").strip()
    if raw:
        return int(raw)
    memory_bytes = _visible_gpu_memory_bytes()
    if (
        bool(allow_high_memory_default)
        and memory_bytes is not None
        and int(memory_bytes) >= EXACT_LOCAL_HIGH_MEMORY_GPU_BYTES
    ):
        return int(EXACT_LOCAL_HIGH_MEMORY_TARGET_ROW_PIXELS)
    return int(EXACT_LOCAL_TARGET_ROW_PIXELS)


def _exact_local_default_big_jit_matmul_max_gb(*, allow_high_memory_default: bool = True) -> float:
    raw = _runtime_environment().get(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, "").strip()
    if raw:
        return float(raw)
    memory_bytes = _visible_gpu_memory_bytes()
    if (
        bool(allow_high_memory_default)
        and memory_bytes is not None
        and int(memory_bytes) >= EXACT_LOCAL_HIGH_MEMORY_GPU_BYTES
    ):
        return float(EXACT_LOCAL_HIGH_MEMORY_BIG_JIT_MATMUL_MAX_GB)
    return float(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB)


def _exact_local_max_hypotheses_per_microbatch(
    default: int | None,
    n_windowed: int,
    *,
    n_trans: int = 1,
    n_recon_windowed: int | None = None,
    allow_high_memory_default: bool = True,
) -> int:
    """Return exact-local microbatch cap."""

    if default is not None:
        value = int(default)
        if value <= 0:
            raise ValueError("max_hypotheses_per_microbatch must be positive")
        return value
    target_row_pixels = _exact_local_default_target_row_pixels(allow_high_memory_default=allow_high_memory_default)
    if target_row_pixels <= 0:
        raise ValueError(f"{EXACT_LOCAL_TARGET_ROW_PIXELS_ENV} must be positive")
    value = target_row_pixels // max(1, int(n_windowed))
    max_gb = _exact_local_default_big_jit_matmul_max_gb(allow_high_memory_default=allow_high_memory_default)
    if max_gb > 0.0:
        # The fused local M-step lowers to a matmul whose large outputs are
        # per-rotation image sums, not a literal (rotation, translation, pixel)
        # tensor. Cap the row count by those output rows.
        n_recon = int(n_windowed if n_recon_windowed is None else n_recon_windowed)
        if int(n_trans) <= 1:
            matmul_row_bytes = 4 * max(1, n_recon)
        else:
            matmul_row_bytes = (
                8 * max(1, int(n_windowed)) + 16 * max(1, n_recon) + 4 * max(1, n_recon) + 16 * max(1, int(n_trans))
            )
        matmul_cap = int((max_gb * 1e9) // max(1, matmul_row_bytes))
        value = min(value, matmul_cap)
    return int(max(512, min(65536, value)))


def _exact_local_microbatch_env_overridden() -> bool:
    return bool(
        _runtime_environment().get(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "").strip()
        or _runtime_environment().get(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, "").strip()
    )


def _exact_local_auto_microbatch_boost() -> float:
    raw = _runtime_environment().get(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %.1f",
                EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV,
                raw,
                EXACT_LOCAL_AUTO_MICROBATCH_BOOST,
            )
            return float(EXACT_LOCAL_AUTO_MICROBATCH_BOOST)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"{EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV} must be positive and finite")
        return value
    return float(EXACT_LOCAL_AUTO_MICROBATCH_BOOST)


def _exact_local_xhalf_auto_microbatch_boost() -> float:
    raw = _runtime_environment().get(EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %.2f",
                EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV,
                raw,
                EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST,
            )
            return float(EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"{EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV} must be positive and finite")
        return value
    return float(EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST)


def _exact_local_planned_hypotheses_floor(
    local_layout: LocalHypothesisLayout,
    *,
    image_batch_size: int,
    rotation_block_size: int,
) -> int:
    """Minimum row cap needed to honor the memory planner's image batch."""

    image_batch_size = max(1, int(image_batch_size))
    rotation_counts = np.asarray(local_layout.rotation_counts, dtype=np.int64)
    if rotation_counts.size == 0:
        return image_batch_size
    max_rotation_count = int(np.max(rotation_counts, initial=1))
    large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size)
    bucket_rotation_count = _exact_bucket_rotation_size(
        max_rotation_count,
        rotation_block_size,
        large_bucket_quantum=large_bucket_quantum,
    )
    return int(image_batch_size * max(1, bucket_rotation_count))


def _exact_local_effective_max_hypotheses_per_microbatch(
    default: int | None,
    n_windowed: int,
    *,
    n_trans: int = 1,
    n_recon_windowed: int | None = None,
    local_layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    allow_auto_boost: bool = True,
    auto_boost_factor: float | None = None,
    allow_high_memory_default: bool = True,
    score_only: bool = False,
    runtime_free_memory_bytes: int | None = None,
) -> int:
    cap = _exact_local_max_hypotheses_per_microbatch(
        default,
        n_windowed,
        n_trans=n_trans,
        n_recon_windowed=n_recon_windowed,
        allow_high_memory_default=allow_high_memory_default,
    )
    if default is not None or _exact_local_microbatch_env_overridden():
        return cap
    if not bool(allow_auto_boost):
        return cap
    planned_floor = _exact_local_planned_hypotheses_floor(
        local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
    )
    boost_factor = _exact_local_auto_microbatch_boost() if auto_boost_factor is None else float(auto_boost_factor)
    if boost_factor <= 0.0 or not np.isfinite(boost_factor):
        raise ValueError("auto_boost_factor must be positive and finite")
    boost_cap = int(np.floor(cap * boost_factor))
    effective_cap = int(max(cap, min(65536, planned_floor, boost_cap)))
    if not score_only:
        return effective_cap

    if runtime_free_memory_bytes is None:
        runtime_free_memory_bytes = _exact_local_runtime_free_memory_bytes()
    if runtime_free_memory_bytes is None:
        return cap
    score_tile_bytes_per_hypothesis = (
        max(1, int(n_trans))
        * max(1, int(n_windowed))
        * np.dtype(np.float32).itemsize
        * EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR
    )
    score_tile_cap = int(
        int(runtime_free_memory_bytes) * EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION // score_tile_bytes_per_hypothesis
    )
    return int(max(1, min(effective_cap, score_tile_cap)))


def _exact_local_xhalf_tail_microbatch_cap(
    cap: int,
    local_layout: LocalHypothesisLayout,
    *,
    image_batch_size: int,
    rotation_block_size: int,
) -> int:
    """Respect the outer memory plan for oversized x-half neighborhoods."""

    cap = max(1, int(cap))
    rotation_block_size = max(1, int(rotation_block_size))
    rotation_counts = np.asarray(local_layout.rotation_counts, dtype=np.int64)
    max_rotation_count = int(np.max(rotation_counts, initial=0))
    if max_rotation_count <= rotation_block_size:
        return cap
    planned_row_cap = max(1, int(image_batch_size)) * rotation_block_size
    return min(cap, planned_row_cap)


def _exact_local_xhalf_projection_target_row_pixels() -> int:
    """Resolve the x-half projection row-pixel budget."""

    target_row_pixels = int(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS)
    raw = _runtime_environment().get(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV, "").strip()
    if raw:
        try:
            target_row_pixels = max(1, int(raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %d row-pixels",
                EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV,
                raw,
                EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS,
            )
            target_row_pixels = int(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS)
    return target_row_pixels


def _exact_local_xhalf_projection_microbatch_cap(
    cap: int,
    local_layout: LocalHypothesisLayout,
    *,
    n_projection_pixels: int,
    rotation_block_size: int,
) -> int:
    """Bound fused x-half projection rows without truncating neighborhoods."""

    cap = max(1, int(cap))
    n_projection_pixels = max(1, int(n_projection_pixels))
    rotation_block_size = max(1, int(rotation_block_size))
    target_row_pixels = _exact_local_xhalf_projection_target_row_pixels()

    rotation_counts = np.asarray(local_layout.rotation_counts, dtype=np.int64)
    if rotation_counts.size == 0:
        return cap
    large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size)
    max_bucket_rotation_count = max(
        _exact_bucket_rotation_size(
            int(count),
            rotation_block_size,
            large_bucket_quantum=large_bucket_quantum,
        )
        for count in rotation_counts
    )
    projection_row_cap = max(1, target_row_pixels // n_projection_pixels)
    # Exact neighborhoods are indivisible; permit at least one image from the
    # largest padded bucket even when that exceeds the configured row target.
    safe_cap = max(int(max_bucket_rotation_count), int(projection_row_cap))
    return min(cap, safe_cap)


def plan_local_microbatch_route(
    *,
    geometry: LocalEMGeometryPlan,
    reconstruction: LocalEMReconstructionPlan,
    mode: LocalEMModePlan,
    relion_projector_half=None,
) -> LocalMicrobatchRoute:
    """Resolve the x-half memory route before calculating its caps."""

    xhalf_bpref_mstep = bool(relion_projector_half is not None and mode.mstep_relion_x_half and not mode.score_only)
    auto_boost_factor = _exact_local_xhalf_auto_microbatch_boost() if xhalf_bpref_mstep else None
    full_bpref = bool(
        xhalf_bpref_mstep and int(reconstruction.volume_shape[0]) >= (2 * int(geometry.image_shape[0]) + 1)
    )
    return LocalMicrobatchRoute(
        xhalf_bpref_mstep=xhalf_bpref_mstep,
        full_bpref=full_bpref,
        auto_boost_factor=auto_boost_factor,
    )


def plan_local_microbatch_cap(
    *,
    local_layout: LocalHypothesisLayout,
    geometry: LocalEMGeometryPlan,
    fourier: LocalEMFourierPlan,
    execution: LocalExecutionSettings,
    mode: LocalEMModePlan,
    route: LocalMicrobatchRoute,
) -> LocalMicrobatchPlan:
    """Apply the established generic, tail, and projection caps in order."""

    initial_cap = _exact_local_effective_max_hypotheses_per_microbatch(
        execution.max_hypotheses_per_microbatch,
        fourier.window.n_score,
        n_trans=geometry.n_translations,
        n_recon_windowed=fourier.window.n_recon,
        local_layout=local_layout,
        image_batch_size=execution.image_batch_size,
        rotation_block_size=execution.rotation_block_size,
        allow_auto_boost=True,
        auto_boost_factor=route.auto_boost_factor,
        allow_high_memory_default=not route.xhalf_bpref_mstep,
        score_only=mode.score_only,
    )
    tail_cap = initial_cap
    effective_cap = initial_cap
    projection_target_row_pixels = None
    if route.xhalf_bpref_mstep:
        tail_cap = _exact_local_xhalf_tail_microbatch_cap(
            initial_cap,
            local_layout,
            image_batch_size=execution.image_batch_size,
            rotation_block_size=execution.rotation_block_size,
        )
        effective_cap = _exact_local_xhalf_projection_microbatch_cap(
            tail_cap,
            local_layout,
            n_projection_pixels=fourier.window.n_projection,
            rotation_block_size=execution.rotation_block_size,
        )
        if effective_cap < tail_cap:
            projection_target_row_pixels = _exact_local_xhalf_projection_target_row_pixels()
    return LocalMicrobatchPlan(
        initial_cap=int(initial_cap),
        tail_cap=int(tail_cap),
        effective_cap=int(effective_cap),
        projection_target_row_pixels=projection_target_row_pixels,
    )


def summarize_local_buckets(bucket_specs) -> LocalBucketSummary:
    """Summarize bucket shape classes without changing their order."""

    if not bucket_specs:
        return LocalBucketSummary(
            bucket_count=0,
            image_count=0,
            rotation_size_min=0,
            rotation_size_median=0,
            rotation_size_mean=0.0,
            rotation_size_max=0,
            images_per_bucket_median=0,
            images_per_bucket_max=0,
            top_rotation_sizes=(),
        )
    bucket_rotation_counts = np.asarray(
        [int(bucket.bucket_rotation_count) for bucket in bucket_specs],
        dtype=np.int64,
    )
    bucket_image_counts = np.asarray(
        [int(bucket.image_indices.shape[0]) for bucket in bucket_specs],
        dtype=np.int64,
    )
    unique_bucket_counts, unique_bucket_freq = np.unique(bucket_rotation_counts, return_counts=True)
    top_rotation_sizes = tuple(
        sorted(
            (
                (int(bucket_count), int(frequency))
                for bucket_count, frequency in zip(unique_bucket_counts, unique_bucket_freq)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
    )
    return LocalBucketSummary(
        bucket_count=len(bucket_specs),
        image_count=int(np.sum(bucket_image_counts, dtype=np.int64)),
        rotation_size_min=int(np.min(bucket_rotation_counts)),
        rotation_size_median=int(np.median(bucket_rotation_counts)),
        rotation_size_mean=float(np.mean(bucket_rotation_counts)),
        rotation_size_max=int(np.max(bucket_rotation_counts)),
        images_per_bucket_median=int(np.median(bucket_image_counts)),
        images_per_bucket_max=int(np.max(bucket_image_counts)),
        top_rotation_sizes=top_rotation_sizes,
    )


def plan_local_buckets(
    *,
    local_layout: LocalHypothesisLayout,
    execution: LocalExecutionSettings,
    microbatch: LocalMicrobatchPlan,
) -> LocalBucketPlan:
    """Build the established production bucket sequence and its summary."""

    buckets = tuple(
        bucket_local_hypothesis_layout(
            local_layout,
            image_batch_size=execution.image_batch_size,
            rotation_block_size=execution.rotation_block_size,
            max_hypotheses_per_microbatch=microbatch.effective_cap,
            unify_bucket_sizes=execution.unify_local_bucket_sizes,
        )
    )
    return LocalBucketPlan(
        buckets=buckets,
        total_local_rotations=int(local_layout.total_local_rotations),
        summary=summarize_local_buckets(buckets),
    )
