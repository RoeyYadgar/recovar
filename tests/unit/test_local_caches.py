from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume.local_caches import (
    EXACT_LOCAL_DISABLE_BIG_JIT_ENV,
    LocalCacheRouteConstraints,
    plan_local_cache_route,
)
from recovar.em.dense_single_volume.local_em_types import (
    LocalCacheSettings,
    LocalCorrectionInputs,
    LocalEMInputs,
    LocalEMRequest,
    LocalEMRequestedOutputs,
    LocalExecutionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)


def _request(
    *,
    cache=None,
    image_pre_shifts=None,
    reconstruct_significant_only=True,
    return_reconstruction_probability_values=False,
    accumulate_noise=False,
    relion_projector_half=None,
):
    layout = SimpleNamespace(rotation_counts=np.asarray([5, 5, 5, 5], dtype=np.int32))
    return LocalEMRequest(
        inputs=LocalEMInputs(
            experiment_dataset=SimpleNamespace(),
            mean=None,
            mean_variance=None,
            noise_variance=None,
            local_layout=layout,
            disc_type="linear_interp",
            relion_projector_half=relion_projector_half,
        ),
        search=LocalSearchSettings(
            current_size=8,
            reconstruct_significant_only=reconstruct_significant_only,
        ),
        execution=LocalExecutionSettings(
            image_batch_size=2,
            rotation_block_size=4,
            cache=cache,
        ),
        scoring=LocalScoringSettings(score_with_masked_images=True),
        corrections=LocalCorrectionInputs(image_pre_shifts=image_pre_shifts),
        outputs=LocalEMRequestedOutputs(
            accumulate_noise=accumulate_noise,
            return_reconstruction_probability_values=return_reconstruction_probability_values,
        ),
    )


def _plan(request, *, use_window=False, capture=False, debug_noise=False):
    return plan_local_cache_route(
        request=request,
        inputs=SimpleNamespace(n_images=4),
        geometry=SimpleNamespace(n_half=40),
        fourier=SimpleNamespace(window=SimpleNamespace(use_window=use_window)),
        mode=SimpleNamespace(score_only=False),
        constraints=LocalCacheRouteConstraints(
            bpref_contribution_capture_active=capture,
            debug_noise_dump_requested=debug_noise,
        ),
    )


def test_local_cache_route_prefers_big_jit_without_processed_cache():
    route = _plan(
        _request(
            cache=LocalCacheSettings(
                raw_image_max_gb=0.0,
                processed_half_max_gb=0.0,
                sparse_big_jit_mstep_max_gb=0.0,
            )
        )
    )

    assert route.local_support_rows == 20
    assert route.significant_backprojection_candidate
    assert not route.use_relion_projector
    assert not route.processed_half_cache_preferred
    assert route.use_big_jit_buckets
    assert not route.use_processed_half_cache


def test_local_cache_route_prefers_processed_cache_for_integral_shifts():
    route = _plan(
        _request(
            cache=LocalCacheSettings(
                raw_image_max_gb=0.0,
                processed_half_max_gb=1.0,
                sparse_big_jit_mstep_max_gb=0.0,
            ),
            image_pre_shifts=np.zeros((4, 2), dtype=np.float32),
        )
    )

    assert route.processed_half_cache_preferred
    assert not route.use_big_jit_buckets
    assert route.use_processed_half_cache


def test_local_cache_route_keeps_fractional_shifts_off_processed_cache():
    shifts = np.zeros((4, 2), dtype=np.float32)
    shifts[0, 0] = 0.25
    route = _plan(
        _request(
            cache=LocalCacheSettings(
                raw_image_max_gb=0.0,
                processed_half_max_gb=1.0,
                sparse_big_jit_mstep_max_gb=0.0,
            ),
            image_pre_shifts=shifts,
        )
    )

    assert not route.processed_half_cache_preferred
    assert route.use_big_jit_buckets


def test_local_cache_route_keeps_diagnostic_disablers_explicit(monkeypatch):
    request = _request(accumulate_noise=True)
    assert not _plan(request, debug_noise=True).use_big_jit_buckets
    assert not _plan(request, capture=True).use_big_jit_buckets

    monkeypatch.setenv(EXACT_LOCAL_DISABLE_BIG_JIT_ENV, "true")
    route = _plan(request)
    assert route.big_jit_disabled
    assert not route.use_big_jit_buckets
