from __future__ import annotations

import inspect

import jax
import numpy as np

from recovar.em.dense_single_volume.local_big_jit import run_local_bucket_big_jit
from recovar.em.dense_single_volume.local_big_jit_types import (
    LocalBigJitBucketInputs,
    LocalBigJitMstepAccumulators,
)


def test_big_jit_boundary_has_grouped_arguments():
    assert tuple(inspect.signature(run_local_bucket_big_jit).parameters) == (
        "image_inputs",
        "projection_inputs",
        "mstep_accumulators",
        "noise_accumulators",
        "shared_inputs",
        "bucket_inputs",
        "config",
        "policy",
    )


def test_donated_mstep_bundle_is_exactly_two_pytree_leaves():
    y = np.zeros(3, dtype=np.complex64)
    ctf = np.zeros(3, dtype=np.float32)

    leaves = jax.tree.leaves(LocalBigJitMstepAccumulators(y=y, ctf=ctf))

    assert len(leaves) == 2
    assert leaves[0] is y
    assert leaves[1] is ctf


def test_bucket_bundle_keeps_optional_sample_mask_as_pytree_structure():
    inputs = LocalBigJitBucketInputs(
        rotation_ids=np.zeros((1, 1), dtype=np.int32),
        rotations=np.zeros((1, 1, 3, 3), dtype=np.float32),
        mstep_rotations=np.zeros((1, 1, 3, 3), dtype=np.float32),
        rotation_log_prior=np.zeros((1, 1), dtype=np.float32),
        translation_log_prior=np.zeros((1, 1), dtype=np.float32),
        rotation_mask=np.ones((1, 1), dtype=bool),
        sample_mask=None,
        valid_image_mask=np.ones(1, dtype=bool),
        normalization_log_z=np.zeros(1, dtype=np.float32),
        normalization_log_evidence=np.zeros(1, dtype=np.float32),
        reconstruction_probability_threshold=np.zeros(1, dtype=np.float64),
    )

    leaves, tree = jax.tree.flatten(inputs)

    assert len(leaves) == 10
    assert tree.unflatten(leaves).sample_mask is None
