"""Typed host-side events and trace requests for dense EM diagnostics.

The payloads in this module are deliberately shallow.  They retain references
to values already produced by the algorithm; sinks decide whether and how to
materialize them.  In particular, constructing an event must never transfer a
JAX array to the host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiagnosticEffect(str, Enum):
    """How a diagnostic request may affect an algorithm run."""

    PASSIVE = "passive"
    SHADOW = "shadow"
    INVASIVE = "invasive"


class TraceKind(str, Enum):
    """Additional values that a numerical kernel may be asked to return."""

    SCORES = "scores"
    POSTERIOR = "posterior"
    OPERANDS = "operands"
    MEMBERSHIP = "membership"
    PROJECTOR = "projector"
    BPREF = "bpref"


@dataclass(frozen=True)
class TraceSpec:
    """Static identity of extra kernel outputs requested by diagnostics."""

    outputs: frozenset[TraceKind] = field(default_factory=frozenset)

    @classmethod
    def none(cls) -> TraceSpec:
        return cls()

    def requests(self, kind: TraceKind) -> bool:
        return kind in self.outputs

    @property
    def is_empty(self) -> bool:
        return not self.outputs


@dataclass(frozen=True)
class IterationStarted:
    iteration: int
    relion_iteration: int
    # Current-size planning is part of the iteration and has not run when this
    # event is emitted.  Keep the field optional for compatibility with sinks
    # that may attach a size when emitting the event at a later boundary.
    current_size: int | None = None


@dataclass(frozen=True)
class HalfScored:
    iteration: int
    half: int
    result: Any


@dataclass(frozen=True)
class MstepAccumulated:
    iteration: int
    half: int
    accumulators: Any


@dataclass(frozen=True)
class MapsUpdated:
    iteration: int
    means: tuple[Any, ...]
    unregularized_means: tuple[Any, ...]


@dataclass(frozen=True)
class ConvergenceUpdated:
    iteration: int
    fsc: Any
    average_max_posterior: float
    converged: bool


@dataclass(frozen=True)
class IterationFinished:
    iteration: int
    relion_iteration: int
    wall_time_s: float | None = None
