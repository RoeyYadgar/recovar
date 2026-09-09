from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.batch_planning import (
    _estimate_relion_em_batch_sizes,
    _maybe_cache_raw_image_loaders,
)
from recovar.em.dense_single_volume.firstiter_cc import _safe_firstiter_cc_image_batch_size
from recovar.em.dense_single_volume.runtime_options import (
    EM_RAW_IMAGE_CACHE_ENV,
    EM_RAW_IMAGE_CACHE_MAX_GB_ENV,
    RELION_EM_BATCH_PROJECTION_FRACTION,
    RELION_EM_BATCH_PROJECTION_FRACTION_ENV,
    RELION_FIRSTITER_RECON_COMPLEX_BUDGET,
    RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV,
    DenseBatchPlanningSettings,
    FirstIterationBatchSettings,
    RawImageCacheSettings,
    load_dense_batch_planning_settings,
    load_first_iteration_batch_settings,
    load_raw_image_cache_settings,
)


@pytest.mark.unit
def test_first_iteration_batch_settings_use_relion_compatible_default():
    assert load_first_iteration_batch_settings({}) == FirstIterationBatchSettings(
        reconstruction_complex_budget=RELION_FIRSTITER_RECON_COMPLEX_BUDGET
    )


@pytest.mark.unit
def test_first_iteration_batch_settings_parse_compatibility_environment():
    settings = load_first_iteration_batch_settings({RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV: "805306368"})

    assert settings.reconstruction_complex_budget == 805_306_368


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["0", "-1", "invalid"])
def test_first_iteration_batch_settings_reject_invalid_environment(raw):
    with pytest.raises(ValueError, match=RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV):
        load_first_iteration_batch_settings({RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV: raw})


@pytest.mark.unit
def test_firstiter_batch_cap_accepts_resolved_settings(monkeypatch):
    monkeypatch.setenv(RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV, "invalid")
    settings = FirstIterationBatchSettings(reconstruction_complex_budget=3 * 268_435_456)

    assert _safe_firstiter_cc_image_batch_size(116, (256, 256), settings=settings) >= 187


@pytest.mark.unit
def test_raw_image_cache_settings_preserve_compatibility_parsing():
    settings = load_raw_image_cache_settings(
        {
            EM_RAW_IMAGE_CACHE_ENV: " FORCE ",
            EM_RAW_IMAGE_CACHE_MAX_GB_ENV: "2.5",
        }
    )

    assert settings == RawImageCacheSettings(mode="force", max_gb=2.5)


@pytest.mark.unit
def test_raw_image_cache_accepts_resolved_settings_without_reading_environment(monkeypatch):
    class FakeLoader:
        num_images = 2
        image_size = 4
        _dtype = np.dtype(np.float32)
        _cached = None
        load_count = 0

        def load_all(self):
            self.load_count += 1

    loader = FakeLoader()
    dataset = SimpleNamespace(image_source=SimpleNamespace(backend=SimpleNamespace(source=loader)))
    monkeypatch.setenv(EM_RAW_IMAGE_CACHE_ENV, "force")
    monkeypatch.setenv(EM_RAW_IMAGE_CACHE_MAX_GB_ENV, "invalid")

    _maybe_cache_raw_image_loaders(
        [dataset],
        settings=RawImageCacheSettings(mode="off", max_gb=1.0),
    )

    assert loader.load_count == 0


@pytest.mark.unit
def test_raw_image_cache_compatibility_path_preserves_lazy_max_gb_parsing(monkeypatch):
    monkeypatch.setenv(EM_RAW_IMAGE_CACHE_ENV, "off")
    monkeypatch.setenv(EM_RAW_IMAGE_CACHE_MAX_GB_ENV, "invalid")

    _maybe_cache_raw_image_loaders([SimpleNamespace(image_source=None)])


@pytest.mark.unit
def test_dense_batch_planning_settings_use_existing_default():
    assert load_dense_batch_planning_settings({}) == DenseBatchPlanningSettings(
        projection_fraction=RELION_EM_BATCH_PROJECTION_FRACTION
    )


@pytest.mark.unit
def test_dense_batch_planning_settings_parse_compatibility_environment():
    settings = load_dense_batch_planning_settings({RELION_EM_BATCH_PROJECTION_FRACTION_ENV: "0.40"})

    assert settings == DenseBatchPlanningSettings(projection_fraction=0.40)


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "invalid"])
def test_dense_batch_planning_settings_reject_invalid_environment(raw):
    with pytest.raises(ValueError, match=RELION_EM_BATCH_PROJECTION_FRACTION_ENV):
        load_dense_batch_planning_settings({RELION_EM_BATCH_PROJECTION_FRACTION_ENV: raw})


@pytest.mark.unit
def test_dense_batch_planner_accepts_resolved_settings_without_reading_environment(monkeypatch):
    common = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
        current_size=100,
    )
    monkeypatch.setenv(RELION_EM_BATCH_PROJECTION_FRACTION_ENV, "invalid")

    plan = _estimate_relion_em_batch_sizes(
        **common,
        settings=DenseBatchPlanningSettings(projection_fraction=0.40),
    )

    assert plan.rotation_block_size == 4339
    assert plan.projection_budget_gb == 10.0
