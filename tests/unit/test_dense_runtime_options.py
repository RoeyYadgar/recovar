from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.batch_planning import (
    _estimate_relion_em_batch_sizes,
    _maybe_cache_raw_image_loaders,
)
from recovar.em.dense_single_volume.diagnostics.config import (
    DiagnosticsPlan,
    EnvironmentVariableClass,
    classify_environment_name,
    diagnostic_environment_overrides,
    diagnostics_environment,
)
from recovar.em.dense_single_volume.firstiter_cc import _safe_firstiter_cc_image_batch_size
from recovar.em.dense_single_volume.local_caches import (
    _local_processed_half_cache_enabled,
    _local_raw_cache_enabled,
    _sparse_big_jit_mstep_tensors_memory_gb,
)
from recovar.em.dense_single_volume.runtime_options import (
    EM_RAW_IMAGE_CACHE_ENV,
    EM_RAW_IMAGE_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB,
    EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_RAW_CACHE_MAX_GB,
    EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB,
    EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV,
    RELION_EM_BATCH_PROJECTION_FRACTION,
    RELION_EM_BATCH_PROJECTION_FRACTION_ENV,
    RELION_FIRSTITER_RECON_COMPLEX_BUDGET,
    RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV,
    DenseBatchPlanningSettings,
    EnvironmentSnapshot,
    ExecutionSettings,
    FirstIterationBatchSettings,
    LocalCacheSettings,
    RawImageCacheSettings,
    capture_environment,
    environment_scope,
    load_dense_batch_planning_settings,
    load_execution_settings,
    load_first_iteration_batch_settings,
    load_local_cache_settings,
    load_raw_image_cache_settings,
)


@pytest.mark.unit
def test_environment_snapshot_is_immutable_during_host_scope(monkeypatch):
    monkeypatch.setenv(RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV, "123")
    snapshot = capture_environment()

    monkeypatch.setenv(RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV, "456")
    with environment_scope(snapshot):
        assert load_first_iteration_batch_settings().reconstruction_complex_budget == 123

    assert load_first_iteration_batch_settings().reconstruction_complex_budget == 456
    with pytest.raises(TypeError):
        snapshot._mapping[RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV] = "789"


@pytest.mark.unit
def test_diagnostics_plan_separates_passive_and_invasive_settings():
    plan = DiagnosticsPlan.from_environment(
        {
            "RECOVAR_PASS2_DUMP_DIR": "/tmp/pass2",
            "RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET": "1",
            "RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES": "1000",
            "IGNORED": "value",
        }
    )

    assert plan.passive["RECOVAR_PASS2_DUMP_DIR"] == "/tmp/pass2"
    assert plan.invasive["RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET"] == "1"
    assert "RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES" not in plan.passive
    assert classify_environment_name("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES") is EnvironmentVariableClass.TUNING
    assert classify_environment_name("IGNORED") is None


@pytest.mark.unit
def test_diagnostic_overrides_do_not_mutate_process_environment(monkeypatch):
    name = "RECOVAR_LOCAL_SCORE_DUMP_LABEL"
    monkeypatch.setenv(name, "outer")
    snapshot = EnvironmentSnapshot(((name, "captured"),))

    with environment_scope(snapshot):
        with diagnostic_environment_overrides(**{name: "inner"}):
            assert diagnostics_environment()[name] == "inner"
        assert diagnostics_environment()[name] == "captured"
    assert diagnostics_environment()[name] == "outer"


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


@pytest.mark.unit
def test_local_cache_settings_use_existing_defaults():
    assert load_local_cache_settings({}) == LocalCacheSettings(
        raw_image_max_gb=EXACT_LOCAL_RAW_CACHE_MAX_GB,
        processed_half_max_gb=EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB,
        sparse_big_jit_mstep_max_gb=EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB,
    )


@pytest.mark.unit
def test_local_cache_settings_parse_compatibility_environment():
    settings = load_local_cache_settings(
        {
            EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV: "20.0",
            EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV: "3.5",
            EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV: "8.25",
        }
    )

    assert settings == LocalCacheSettings(
        raw_image_max_gb=20.0,
        processed_half_max_gb=3.5,
        sparse_big_jit_mstep_max_gb=8.25,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "env_name",
    [
        EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV,
        EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV,
        EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV,
    ],
)
def test_local_cache_settings_preserve_invalid_float_errors(env_name):
    with pytest.raises(ValueError):
        load_local_cache_settings({env_name: "invalid"})


@pytest.mark.unit
def test_local_cache_helpers_accept_resolved_settings_without_reading_environment(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV, "invalid")
    monkeypatch.setenv(EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV, "invalid")
    monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "invalid")
    settings = LocalCacheSettings(
        raw_image_max_gb=20.0,
        processed_half_max_gb=1.0,
        sparse_big_jit_mstep_max_gb=7.5,
    )

    assert _local_raw_cache_enabled(50_000, (256, 256), np.float32, settings=settings)
    assert _local_processed_half_cache_enabled(
        50_000,
        100,
        np.complex64,
        store_recon_half=False,
        settings=settings,
    )
    estimated_gb, max_gb = _sparse_big_jit_mstep_tensors_memory_gb(
        image_count=10,
        rotation_count=20,
        n_recon_windowed=30,
        use_float64_scoring=False,
        settings=settings,
    )
    assert estimated_gb == 72_000 / 1e9
    assert max_gb == 7.5


@pytest.mark.unit
def test_local_cache_compatibility_helpers_preserve_lazy_field_parsing(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV, "20.0")
    monkeypatch.setenv(EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV, "invalid")
    monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "invalid")

    assert _local_raw_cache_enabled(50_000, (256, 256), np.float32)


@pytest.mark.unit
def test_execution_settings_compose_all_resolved_leaf_settings():
    settings = load_execution_settings(
        {
            RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV: "805306368",
            EM_RAW_IMAGE_CACHE_ENV: "off",
            EM_RAW_IMAGE_CACHE_MAX_GB_ENV: "2.5",
            RELION_EM_BATCH_PROJECTION_FRACTION_ENV: "0.4",
            EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV: "20",
            EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV: "3.5",
            EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV: "8.25",
        }
    )

    assert settings == ExecutionSettings(
        first_iteration=FirstIterationBatchSettings(reconstruction_complex_budget=805_306_368),
        raw_image_cache=RawImageCacheSettings(mode="off", max_gb=2.5),
        dense_batch_planning=DenseBatchPlanningSettings(projection_fraction=0.4),
        local_cache=LocalCacheSettings(
            raw_image_max_gb=20.0,
            processed_half_max_gb=3.5,
            sparse_big_jit_mstep_max_gb=8.25,
        ),
    )
