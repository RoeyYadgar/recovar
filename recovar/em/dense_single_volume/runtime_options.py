"""Environment compatibility parsing for host-side EM runtime settings."""

from __future__ import annotations

import math
import os
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from recovar.em.dense_single_volume.diagnostics.config import DiagnosticsPlan

logger = logging.getLogger(__name__)

# Historical defaults remain stable while environment variables are adapters.
RELION_FIRSTITER_RECON_COMPLEX_BUDGET = 268_435_456
RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV = "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET"
EM_RAW_IMAGE_CACHE_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE"
EM_RAW_IMAGE_CACHE_MAX_GB_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB"
EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB = 16.0
RELION_EM_BATCH_PROJECTION_FRACTION = 0.20
RELION_EM_BATCH_PROJECTION_FRACTION_ENV = "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION"
EXACT_LOCAL_RAW_CACHE_MAX_GB = 16.0
EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_RAW_CACHE_MAX_GB"
EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB = 0.0
EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB"
EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB = 12.0
EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB"
USE_FLOAT64_SCORING_ENV = "RECOVAR_USE_FLOAT64_SCORING"
USE_FLOAT64_PROJECTIONS_ENV = "RECOVAR_USE_FLOAT64_PROJECTIONS"
DISABLE_RELION_EXACT_FINE_GAUSSIAN_ENV = "RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN"
RELION_ACC_DOUBLE_FLOORF_QUIRK_ENV = "RECOVAR_RELION_ACC_DOUBLE_FLOORF_QUIRK"
K1_RELION_EXACT_TRANSLATION_GRID_ENV = "RECOVAR_K1_RELION_EXACT_TRANSLATION_GRID"
FINAL_ALL_DATA_GRID_CORRECT_ENV = "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class EnvironmentSnapshot(Mapping[str, str]):
    """Immutable process-environment view captured at a host boundary."""

    values: tuple[tuple[str, str], ...]
    _mapping: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = tuple(sorted((str(name), str(value)) for name, value in self.values))
        object.__setattr__(self, "values", normalized)
        object.__setattr__(self, "_mapping", MappingProxyType(dict(normalized)))

    def __getitem__(self, key: str) -> str:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def with_overrides(
        self,
        overrides: Mapping[str, str | None],
    ) -> EnvironmentSnapshot:
        """Return a snapshot with explicit replacements and removals."""

        values = dict(self._mapping)
        for name, value in overrides.items():
            if value is None:
                values.pop(name, None)
            else:
                values[name] = str(value)
        return EnvironmentSnapshot(tuple(values.items()))


_ACTIVE_ENVIRONMENT: ContextVar[EnvironmentSnapshot | None] = ContextVar(
    "dense_single_volume_environment",
    default=None,
)
_ACTIVE_ALGORITHM_SETTINGS: ContextVar[AlgorithmSettings | None] = ContextVar(
    "dense_single_volume_algorithm_settings",
    default=None,
)


def capture_environment(
    environ: Mapping[str, str] | None = None,
) -> EnvironmentSnapshot:
    """Capture an immutable environment without retaining a mutable mapping."""

    source = os.environ if environ is None else environ
    return EnvironmentSnapshot(tuple(source.items()))


def current_environment() -> Mapping[str, str]:
    """Return the active snapshot or the live mapping for legacy direct calls."""

    snapshot = _ACTIVE_ENVIRONMENT.get()
    return os.environ if snapshot is None else snapshot


@contextmanager
def environment_scope(snapshot: EnvironmentSnapshot):
    """Use one immutable environment snapshot for all nested host work."""

    token = _ACTIVE_ENVIRONMENT.set(snapshot)
    try:
        yield snapshot
    finally:
        _ACTIVE_ENVIRONMENT.reset(token)


@dataclass(frozen=True)
class AlgorithmSettings:
    """Resolved numerical and RELION-policy choices for one refinement run."""

    use_float64_scoring: bool = False
    use_float64_projections: bool = False
    relion_exact_fine_gaussian: bool = True
    relion_acc_double_floorf_quirk: bool = False
    k1_relion_exact_translation_grid: bool = True
    final_all_data_grid_correct: bool = False


def _compatibility_bool(
    environ: Mapping[str, str],
    name: str,
    *,
    default: bool,
    strict: bool = False,
) -> bool:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    if strict:
        raise ValueError(f"{name} must be a boolean value, got {normalized!r}")
    logger.warning("Ignoring invalid %s=%r; using default %s", name, raw, str(default).lower())
    return default


def load_algorithm_settings(
    environ: Mapping[str, str] | None = None,
) -> AlgorithmSettings:
    """Resolve major numerical policy from legacy environment aliases."""

    env = current_environment() if environ is None else environ
    return AlgorithmSettings(
        use_float64_scoring=_compatibility_bool(
            env,
            USE_FLOAT64_SCORING_ENV,
            default=False,
        ),
        use_float64_projections=_compatibility_bool(
            env,
            USE_FLOAT64_PROJECTIONS_ENV,
            default=False,
        ),
        relion_exact_fine_gaussian=not _compatibility_bool(
            env,
            DISABLE_RELION_EXACT_FINE_GAUSSIAN_ENV,
            default=False,
        ),
        relion_acc_double_floorf_quirk=_compatibility_bool(
            env,
            RELION_ACC_DOUBLE_FLOORF_QUIRK_ENV,
            default=False,
            strict=True,
        ),
        k1_relion_exact_translation_grid=_compatibility_bool(
            env,
            K1_RELION_EXACT_TRANSLATION_GRID_ENV,
            default=True,
            strict=True,
        ),
        final_all_data_grid_correct=_compatibility_bool(
            env,
            FINAL_ALL_DATA_GRID_CORRECT_ENV,
            default=False,
        ),
    )


def current_algorithm_settings() -> AlgorithmSettings:
    """Return run-scoped settings or resolve a legacy direct-call snapshot."""

    settings = _ACTIVE_ALGORITHM_SETTINGS.get()
    return load_algorithm_settings() if settings is None else settings


@contextmanager
def algorithm_settings_scope(settings: AlgorithmSettings):
    """Make one resolved algorithm policy available to nested host helpers."""

    token = _ACTIVE_ALGORITHM_SETTINGS.set(settings)
    try:
        yield settings
    finally:
        _ACTIVE_ALGORITHM_SETTINGS.reset(token)


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


@dataclass(frozen=True)
class DenseBatchPlanningSettings:
    """Resolved memory fractions used by the dense EM batch planner."""

    projection_fraction: float = RELION_EM_BATCH_PROJECTION_FRACTION

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.projection_fraction)) or float(self.projection_fraction) <= 0:
            raise ValueError("projection_fraction must be a positive finite float")


@dataclass(frozen=True)
class LocalCacheSettings:
    """Resolved host-memory ceilings for exact-local execution caches."""

    raw_image_max_gb: float = EXACT_LOCAL_RAW_CACHE_MAX_GB
    processed_half_max_gb: float = EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB
    sparse_big_jit_mstep_max_gb: float = EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB


@dataclass(frozen=True)
class ExecutionSettings:
    """Resolved host-only performance settings for one refinement run."""

    first_iteration: FirstIterationBatchSettings = field(default_factory=FirstIterationBatchSettings)
    raw_image_cache: RawImageCacheSettings = field(default_factory=RawImageCacheSettings)
    dense_batch_planning: DenseBatchPlanningSettings = field(default_factory=DenseBatchPlanningSettings)
    local_cache: LocalCacheSettings = field(default_factory=LocalCacheSettings)


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Immutable algorithm, execution, diagnostics, and environment snapshot."""

    environment: EnvironmentSnapshot
    algorithm: AlgorithmSettings
    execution: ExecutionSettings
    diagnostics: DiagnosticsPlan


_ACTIVE_RUNTIME_CONFIGURATION: ContextVar[RuntimeConfiguration | None] = ContextVar(
    "dense_single_volume_runtime_configuration",
    default=None,
)


@contextmanager
def runtime_configuration_scope(configuration: RuntimeConfiguration):
    """Activate every resolved host setting for one refinement invocation."""

    token = _ACTIVE_RUNTIME_CONFIGURATION.set(configuration)
    try:
        with environment_scope(configuration.environment), algorithm_settings_scope(configuration.algorithm):
            yield configuration
    finally:
        _ACTIVE_RUNTIME_CONFIGURATION.reset(token)


def current_runtime_configuration() -> RuntimeConfiguration | None:
    """Return the active run configuration, if called inside refinement."""

    return _ACTIVE_RUNTIME_CONFIGURATION.get()


def load_first_iteration_batch_settings(
    environ: Mapping[str, str] | None = None,
) -> FirstIterationBatchSettings:
    """Resolve first-iteration batch settings from compatibility variables."""

    env = current_environment() if environ is None else environ
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

    env = current_environment() if environ is None else environ
    return env.get(EM_RAW_IMAGE_CACHE_ENV, "auto").strip().lower()


def load_raw_image_cache_max_gb(environ: Mapping[str, str] | None = None) -> float:
    """Resolve only the cache ceiling for legacy lazy-validation paths."""

    env = current_environment() if environ is None else environ
    return float(env.get(EM_RAW_IMAGE_CACHE_MAX_GB_ENV, EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB))


def load_dense_batch_planning_settings(
    environ: Mapping[str, str] | None = None,
) -> DenseBatchPlanningSettings:
    """Resolve dense batch-planning settings from compatibility variables."""

    env = current_environment() if environ is None else environ
    raw = env.get(RELION_EM_BATCH_PROJECTION_FRACTION_ENV)
    if raw is None or raw.strip() == "":
        return DenseBatchPlanningSettings()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{RELION_EM_BATCH_PROJECTION_FRACTION_ENV} must be a positive finite float, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{RELION_EM_BATCH_PROJECTION_FRACTION_ENV} must be a positive finite float, got {raw!r}")
    return DenseBatchPlanningSettings(projection_fraction=value)


def load_local_raw_cache_max_gb(environ: Mapping[str, str] | None = None) -> float:
    """Resolve only the raw-image cache ceiling for lazy compatibility paths."""

    env = current_environment() if environ is None else environ
    return float(env.get(EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV, EXACT_LOCAL_RAW_CACHE_MAX_GB))


def load_local_processed_half_cache_max_gb(environ: Mapping[str, str] | None = None) -> float:
    """Resolve only the processed-half cache ceiling for lazy compatibility paths."""

    env = current_environment() if environ is None else environ
    return float(
        env.get(
            EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV,
            EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB,
        )
    )


def load_local_sparse_big_jit_mstep_max_gb(environ: Mapping[str, str] | None = None) -> float:
    """Resolve only the sparse big-JIT M-step ceiling for compatibility paths."""

    env = current_environment() if environ is None else environ
    return float(
        env.get(
            EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV,
            EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB,
        )
    )


def load_local_cache_settings(environ: Mapping[str, str] | None = None) -> LocalCacheSettings:
    """Resolve all exact-local cache ceilings for explicit settings injection."""

    return LocalCacheSettings(
        raw_image_max_gb=load_local_raw_cache_max_gb(environ),
        processed_half_max_gb=load_local_processed_half_cache_max_gb(environ),
        sparse_big_jit_mstep_max_gb=load_local_sparse_big_jit_mstep_max_gb(environ),
    )


def load_execution_settings(environ: Mapping[str, str] | None = None) -> ExecutionSettings:
    """Resolve a complete host execution snapshot from compatibility variables."""

    return ExecutionSettings(
        first_iteration=load_first_iteration_batch_settings(environ),
        raw_image_cache=load_raw_image_cache_settings(environ),
        dense_batch_planning=load_dense_batch_planning_settings(environ),
        local_cache=load_local_cache_settings(environ),
    )


def load_runtime_configuration(
    environ: Mapping[str, str] | None = None,
    *,
    algorithm: AlgorithmSettings | None = None,
    execution: ExecutionSettings | None = None,
    diagnostics: DiagnosticsPlan | None = None,
) -> RuntimeConfiguration:
    """Resolve one complete host configuration from one immutable snapshot."""

    from recovar.em.dense_single_volume.diagnostics.config import DiagnosticsPlan

    snapshot = capture_environment(environ)
    return RuntimeConfiguration(
        environment=snapshot,
        algorithm=algorithm if algorithm is not None else load_algorithm_settings(snapshot),
        execution=execution if execution is not None else load_execution_settings(snapshot),
        diagnostics=(diagnostics if diagnostics is not None else DiagnosticsPlan.from_environment(snapshot)),
    )
