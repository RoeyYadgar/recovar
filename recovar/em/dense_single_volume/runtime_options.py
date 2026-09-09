"""Environment compatibility parsing for host-side EM runtime settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# Historical defaults remain stable while environment variables are adapters.
RELION_FIRSTITER_RECON_COMPLEX_BUDGET = 268_435_456
RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV = "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET"


@dataclass(frozen=True)
class FirstIterationBatchSettings:
    """Resolved memory budget for first-iteration dense reconstruction."""

    reconstruction_complex_budget: int = RELION_FIRSTITER_RECON_COMPLEX_BUDGET

    def __post_init__(self) -> None:
        if int(self.reconstruction_complex_budget) <= 0:
            raise ValueError("reconstruction_complex_budget must be a positive integer")


def load_first_iteration_batch_settings(
    environ: Mapping[str, str] | None = None,
) -> FirstIterationBatchSettings:
    """Resolve first-iteration batch settings from compatibility variables."""

    env = os.environ if environ is None else environ
    raw = env.get(RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV)
    if raw is None or raw.strip() == "":
        return FirstIterationBatchSettings()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV} must be a positive integer")
    return FirstIterationBatchSettings(reconstruction_complex_budget=value)
