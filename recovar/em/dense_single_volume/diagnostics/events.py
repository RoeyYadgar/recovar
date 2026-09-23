"""Typed host-side events and trace requests for dense EM diagnostics.

The payloads in this module are deliberately shallow.  They retain references
to values already produced by the algorithm; sinks decide whether and how to
materialize them.  In particular, constructing an event must never transfer a
JAX array to the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticEffect(str, Enum):
    """How a diagnostic request may affect an algorithm run."""

    PASSIVE = "passive"
    SHADOW = "shadow"
    INVASIVE = "invasive"


@dataclass(frozen=True)
class IterationStarted:
    iteration: int
    relion_iteration: int
    # Current-size planning is part of the iteration and has not run when this
    # event is emitted.  Keep the field optional for compatibility with sinks
    # that may attach a size when emitting the event at a later boundary.
    current_size: int | None = None
