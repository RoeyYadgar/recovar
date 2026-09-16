"""Shared RELION significance-threshold arithmetic."""

from functools import partial

import jax
import jax.numpy as jnp


def relion_cuda_f32_tail_target(sum_weight, adaptive_fraction: float):
    """Match RELION's parsed adaptive-fraction arithmetic at the CUDA cutoff."""

    parsed_fraction = jnp.asarray(adaptive_fraction, dtype=jnp.float32)
    return jnp.asarray(
        (jnp.float64(1.0) - parsed_fraction.astype(jnp.float64))
        * jnp.asarray(sum_weight, dtype=jnp.float32).astype(jnp.float64),
        dtype=jnp.float32,
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def find_significant_mask_full_sort(
    weights_flat,
    adaptive_fraction=0.999,
    max_significants=500,
    return_cutoff_count=False,
):
    """Find RELION significant samples by a stable full posterior sort.

    The support contains the smallest positive-weight set whose cumulative
    mass strictly exceeds ``adaptive_fraction``. All samples tied at the
    threshold remain included, while ``cutoff_count`` records the pre-tie rank.
    """

    n_images, _ = weights_flat.shape
    sorted_w = jnp.sort(weights_flat, axis=-1)[:, ::-1]
    cumsum = jnp.cumsum(sorted_w, axis=-1)
    total = weights_flat.sum(axis=-1, keepdims=True)
    positive_counts = jnp.sum(weights_flat > 0.0, axis=-1)
    last_positive_idx = jnp.maximum(positive_counts - 1, 0)

    frac = cumsum / jnp.maximum(total, 1e-30)
    crosses = frac > adaptive_fraction
    threshold_idx = jnp.where(
        jnp.any(crosses, axis=-1),
        jnp.argmax(crosses, axis=-1),
        last_positive_idx,
    )
    threshold_idx = jnp.minimum(threshold_idx, last_positive_idx)
    if max_significants is not None and int(max_significants) > 0:
        threshold_idx = jnp.minimum(threshold_idx, int(max_significants) - 1)

    threshold_val = sorted_w[jnp.arange(n_images), threshold_idx]
    mask = (weights_flat > 0.0) & (weights_flat >= threshold_val[:, None])
    n_significant = jnp.sum(mask, axis=-1).astype(jnp.int32)
    cutoff_count = jnp.minimum(threshold_idx + 1, positive_counts).astype(jnp.int32)

    if return_cutoff_count:
        return mask, n_significant, cutoff_count
    return mask, n_significant
