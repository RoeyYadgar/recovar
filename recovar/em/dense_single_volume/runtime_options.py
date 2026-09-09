"""Environment compatibility parsing for host-side EM runtime settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# Historical defaults remain stable while environment variables are adapters.
RELION_FIRSTITER_RECON_COMPLEX_BUDGET = 268_435_456
RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV = "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET"
EM_RAW_IMAGE_CACHE_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE"
EM_RAW_IMAGE_CACHE_MAX_GB_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB"
EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB = 16.0


@dataclass(frozen=True)
class FirstIterationBatchSettings:
    """Resolved memory budget for first-iteration dense reconstruction."""

    reconstruction_complex_budget: int = RELION_FIRSTITER_RECON_COMPLEX_BUDGET

    def __post_init__(self) -> None:
        if int(self.reconstruction_complex_budget) <= 0:
            raise ValueError("reconstruction_complex_budget must be a positive integer")


@dataclass(frozen=True)
class RawImageCacheSettings:
    """Resolved host cache mode and memory ceiling for raw particle images."""

    mode: str = "auto"
    max_gb: float = EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB


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


def load_raw_image_cache_settings(
    environ: Mapping[str, str] | None = None,
) -> RawImageCacheSettings:
    """Resolve raw-image cache settings from compatibility variables."""

    return RawImageCacheSettings(
        mode=load_raw_image_cache_mode(environ),
        max_gb=load_raw_image_cache_max_gb(environ),
    )


def load_raw_image_cache_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve only the cache mode for legacy lazy-validation paths."""

    env = os.environ if environ is None else environ
    return env.get(EM_RAW_IMAGE_CACHE_ENV, "auto").strip().lower()


def load_raw_image_cache_max_gb(environ: Mapping[str, str] | None = None) -> float:
    """Resolve only the cache ceiling for legacy lazy-validation paths."""

    env = os.environ if environ is None else environ
    return float(env.get(EM_RAW_IMAGE_CACHE_MAX_GB_ENV, EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB))
