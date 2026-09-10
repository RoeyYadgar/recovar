"""Environment compatibility boundary for diagnostic and experimental modes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from recovar.em.dense_single_volume.runtime_options import (
    EnvironmentSnapshot,
    capture_environment,
    current_environment,
    environment_scope,
)

from .events import DiagnosticEffect, TraceKind, TraceSpec


class EnvironmentVariableClass(str, Enum):
    """Effect class for a supported dense-EM environment variable."""

    ALGORITHM = "algorithm"
    TUNING = "tuning"
    PASSIVE_DIAGNOSTIC = "passive_diagnostic"
    INVASIVE_EXPERIMENT = "invasive_experiment"


_PASSIVE_TOKENS = (
    "_CAPTURE",
    "_DEBUG",
    "_DUMP",
    "_DTYPE_DEBUG",
    "_PROGRESS",
    "_SIGNATURE",
    "_SUMMARY",
    "_TIMING",
)
_INVASIVE_TOKENS = (
    "_STOP_AFTER_",
    "_TARGET_ONLY",
    "_FORCE_SPLIT",
    "_REPLAY_RELION_REFERENCES",
    "_REVERSE_PHYSICAL_ORDER",
    "_NORM_RESIDUAL_ONLY",
    "_CONSERVATIVE_EXECUTION",
    "_HIGH_PRECISION_OPERAND_BUNDLE",
    "_DIAGNOSTIC_",
)
_TUNING_TOKENS = (
    "_CACHE",
    "_MAX_BYTES",
    "_MAX_GB",
    "_MAX_IMAGES",
    "_MAX_HYPOTHESES",
    "_TARGET_ROW_PIXELS",
    "_BUCKET",
    "_MICROBATCH",
    "_CHUNK",
    "_TILE",
    "_MEMORY_GB",
    "_BUDGET",
    "_FRACTION",
    "_QUANTUM",
    "_FUSED",
    "_PREMATMUL",
    "_ACTIVE_ROWS",
    "_PACKED_MSTEP",
)
_EXTERNAL_TUNING_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "OMPI_COMM_WORLD_RANK",
        "SLURM_PROCID",
    }
)

# These captures request extra arithmetic or otherwise materialize values that
# the normal path does not need. They remain observational, but are not
# performance-neutral PASSIVE captures.
_SHADOW_PREFIXES = (
    "RECOVAR_BPREF_DEVICE_SIGNATURE_",
    "RECOVAR_DEBUG_CC_COMPONENT_",
    "RECOVAR_DENSE_NOISE_COMPONENT_",
    "RECOVAR_LOCAL_FUSED_POSTERIOR_",
    "RECOVAR_LOCAL_NOISE_COMPONENT_",
    "RECOVAR_LOCAL_SCORE_DUMP_",
    "RECOVAR_PASS2_DUMP_",
    "RECOVAR_SIGNIFICANCE_DUMP_",
)


def _trace_kind_for_name(name: str) -> frozenset[TraceKind]:
    kinds: set[TraceKind] = set()
    if any(token in name for token in ("SCORE", "SIGNIFICANCE", "PASS2_DUMP")):
        kinds.add(TraceKind.SCORES)
    if "POSTERIOR" in name:
        kinds.add(TraceKind.POSTERIOR)
    if any(token in name for token in ("OPERAND", "NORM_RESIDUAL", "SIGNIFICANCE")):
        kinds.add(TraceKind.OPERANDS)
    if "MEMBERSHIP" in name:
        kinds.add(TraceKind.MEMBERSHIP)
    if "PROJECTOR" in name:
        kinds.add(TraceKind.PROJECTOR)
    if "BPREF" in name:
        kinds.add(TraceKind.BPREF)
    return frozenset(kinds)


@dataclass(frozen=True)
class DiagnosticRoutes:
    """Resolved diagnostics partitioned by behavioral effect."""

    passive: EnvironmentSnapshot
    shadow: EnvironmentSnapshot
    invasive: EnvironmentSnapshot
    trace_spec: TraceSpec

    @property
    def production_authoritative(self) -> bool:
        return not bool(self.invasive)

    def effect_for(self, name: str) -> DiagnosticEffect:
        if name in self.invasive:
            return DiagnosticEffect.INVASIVE
        if name in self.shadow:
            return DiagnosticEffect.SHADOW
        if name in self.passive:
            return DiagnosticEffect.PASSIVE
        raise KeyError(name)


def classify_environment_name(name: str) -> EnvironmentVariableClass | None:
    """Classify a supported environment name by its observable effect."""

    if name in _EXTERNAL_TUNING_NAMES:
        return EnvironmentVariableClass.TUNING
    if not name.startswith("RECOVAR_"):
        return None
    if any(token in name for token in _INVASIVE_TOKENS):
        return EnvironmentVariableClass.INVASIVE_EXPERIMENT
    if any(token in name for token in _PASSIVE_TOKENS):
        return EnvironmentVariableClass.PASSIVE_DIAGNOSTIC
    if any(token in name for token in _TUNING_TOKENS):
        return EnvironmentVariableClass.TUNING
    return EnvironmentVariableClass.ALGORITHM


def _filtered_snapshot(
    snapshot: EnvironmentSnapshot,
    effect: EnvironmentVariableClass,
) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        tuple((name, value) for name, value in snapshot.items() if classify_environment_name(name) is effect)
    )


@dataclass(frozen=True)
class DiagnosticsPlan:
    """Resolved passive captures and visibly invasive experiment switches."""

    passive: EnvironmentSnapshot
    invasive: EnvironmentSnapshot

    def __post_init__(self) -> None:
        values = {**dict(self.passive), **dict(self.invasive)}
        mutually_exclusive = (
            "RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF",
            "RECOVAR_PASS2_DUMP_TARGET_HALF",
        )
        if all(str(values.get(name, "")).strip() for name in mutually_exclusive):
            raise ValueError(f"{mutually_exclusive[0]} and {mutually_exclusive[1]} are mutually exclusive")

    @property
    def routes(self) -> DiagnosticRoutes:
        """Partition compatibility settings without changing their storage."""

        shadow_names = {name for name in self.passive if any(name.startswith(prefix) for prefix in _SHADOW_PREFIXES)}
        passive = EnvironmentSnapshot(
            tuple((name, self.passive[name]) for name in self.passive if name not in shadow_names)
        )
        shadow = EnvironmentSnapshot(tuple((name, self.passive[name]) for name in self.passive if name in shadow_names))
        trace_kinds: set[TraceKind] = set()
        for name in (*passive, *shadow, *self.invasive):
            trace_kinds.update(_trace_kind_for_name(name))
        return DiagnosticRoutes(
            passive=passive,
            shadow=shadow,
            invasive=self.invasive,
            trace_spec=TraceSpec(frozenset(trace_kinds)),
        )

    @classmethod
    def from_environment(
        cls,
        environ=None,
    ) -> DiagnosticsPlan:
        snapshot = capture_environment(environ)
        return cls(
            passive=_filtered_snapshot(snapshot, EnvironmentVariableClass.PASSIVE_DIAGNOSTIC),
            invasive=_filtered_snapshot(snapshot, EnvironmentVariableClass.INVASIVE_EXPERIMENT),
        )


def diagnostics_environment():
    """Return the active immutable environment view for compatibility helpers."""

    return current_environment()


@contextmanager
def diagnostic_environment_overrides(**overrides: str | None):
    """Apply diagnostic-only overrides without mutating ``os.environ``."""

    snapshot = capture_environment(current_environment()).with_overrides(overrides)
    with environment_scope(snapshot):
        yield snapshot
