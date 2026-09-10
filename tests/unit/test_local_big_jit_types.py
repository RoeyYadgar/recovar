from __future__ import annotations

import inspect
from dataclasses import replace

import jax
import numpy as np

from recovar.em.dense_single_volume.local_big_jit import run_local_bucket_big_jit
from recovar.em.dense_single_volume.local_big_jit_types import (
    LocalBigJitBucketInputs,
    LocalBigJitMstepAccumulators,
    LocalBigJitPolicy,
    unpack_local_big_jit_result,
)


def _policy() -> LocalBigJitPolicy:
    return LocalBigJitPolicy(
        mask_mode="none",
        score_with_masked_images=False,
        apply_integer_pre_shift=False,
        apply_fourier_pre_shift=False,
        half_spectrum_scoring=False,
        use_float64_scoring=False,
        use_float64_normalization=False,
        use_window=False,
        reconstruct_significant_only=False,
        adaptive_fraction=0.999,
        max_significants=None,
        has_normalization_log_z=False,
        has_normalization_log_evidence=False,
        has_reconstruction_probability_threshold=False,
        score_only=False,
        image_shape=(4, 4),
        projection_volume_shape=(4, 4, 4),
        reconstruction_volume_shape=(4, 4, 4),
        disc_type="linear_interp",
        projection_half_volume=False,
        projection_max_r=None,
        mstep_max_r=None,
        use_compact_relion_projector_projection=False,
        use_relion_projection_cache=False,
        relion_projector_output_size=0,
        projection_relion_texture_interp=False,
        projection_force_jax=False,
        use_relion_projector=False,
        relion_projector_r_max=0,
        projection_padding_factor=1,
        mstep_subtract_ctf_projection=False,
        mstep_relion_x_half=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        accumulate_noise=False,
        accumulate_scale_correction=False,
        return_noise_split=False,
        n_shells=1,
        norm_current_size=None,
        include_unweighted_norm_high_shell=True,
        source_faithful_spectrum_norm=False,
        return_mstep_tensors=False,
        return_deferred_mstep_inputs=False,
        return_deferred_noise_inputs=False,
        return_debug_arrays=False,
        return_debug_scores=False,
        return_debug_operands=False,
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


def test_big_jit_result_unpacker_names_base_and_mstep_tensor_routes():
    base = tuple(range(22))

    result = unpack_local_big_jit_result(base + ("summed", "ctf_probs"), replace(_policy(), return_mstep_tensors=True))

    assert result.mstep == LocalBigJitMstepAccumulators(y=0, ctf=1)
    assert result.noise.sigma2_offset == 9
    assert result.log_evidence == 12
    assert result.reconstruction_row_count == 21
    assert result.summed == "summed"
    assert result.ctf_probabilities == "ctf_probs"
    assert result.reconstruction_probabilities is None


def test_big_jit_result_unpacker_names_deferred_and_debug_routes():
    base = tuple(range(22))
    deferred = ("probs", "shifted_recon", "ctf2", "shifted_noise", "processed")
    debug = tuple(f"debug_{index}" for index in range(8))
    policy = replace(
        _policy(),
        return_deferred_mstep_inputs=True,
        return_debug_arrays=True,
        return_debug_scores=True,
        return_debug_operands=True,
    )

    result = unpack_local_big_jit_result(base + deferred + debug, policy)

    assert result.reconstruction_probabilities == "probs"
    assert result.shifted_reconstruction == "shifted_recon"
    assert result.processed_score_half == "processed"
    assert result.debug_scores == "debug_0"
    assert result.debug_noise_projection == "debug_7"
