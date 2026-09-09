import pytest

from recovar.em.dense_single_volume.firstiter_cc import _safe_firstiter_cc_image_batch_size
from recovar.em.dense_single_volume.runtime_options import (
    RELION_FIRSTITER_RECON_COMPLEX_BUDGET,
    RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV,
    FirstIterationBatchSettings,
    load_first_iteration_batch_settings,
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
