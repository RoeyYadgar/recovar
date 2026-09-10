"""Small, explicit sink protocol for host-side dense EM diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Protocol

import numpy as np

from .events import (
    ConvergenceUpdated,
    HalfScored,
    IterationFinished,
    IterationStarted,
    MapsUpdated,
    MstepAccumulated,
    TraceSpec,
)


class DiagnosticsSink(Protocol):
    """Observer invoked only at existing host orchestration boundaries.

    Sink methods intentionally return ``None``: passive diagnostics can observe
    production values but cannot replace or select algorithm outputs.
    """

    @property
    def trace_spec(self) -> TraceSpec: ...

    def iteration_started(self, event: IterationStarted) -> None: ...

    def half_scored(self, event: HalfScored) -> None: ...

    def mstep_accumulated(self, event: MstepAccumulated) -> None: ...

    def maps_updated(self, event: MapsUpdated) -> None: ...

    def convergence_updated(self, event: ConvergenceUpdated) -> None: ...

    def iteration_finished(self, event: IterationFinished) -> None: ...


class NullDiagnostics:
    """Production sink that performs no inspection, conversion, or I/O."""

    __slots__ = ()

    @property
    def trace_spec(self) -> TraceSpec:
        return NO_TRACE

    def iteration_started(self, event: IterationStarted) -> None:
        pass

    def half_scored(self, event: HalfScored) -> None:
        pass

    def mstep_accumulated(self, event: MstepAccumulated) -> None:
        pass

    def maps_updated(self, event: MapsUpdated) -> None:
        pass

    def convergence_updated(self, event: ConvergenceUpdated) -> None:
        pass

    def iteration_finished(self, event: IterationFinished) -> None:
        pass


class NpzDiagnostics:
    """Host serializer shared by schema-specific diagnostic adapters."""

    __slots__ = ()

    def write_fields(
        self,
        path: str | Path,
        /,
        *,
        compressed: bool,
        **payload: Any,
    ) -> None:
        writer = np.savez_compressed if compressed else np.savez
        writer(path, **payload)

    def write_payload(
        self,
        path: str | Path,
        payload: dict[str, Any],
        /,
        *,
        compressed: bool,
    ) -> None:
        self.write_fields(path, compressed=compressed, **payload)


NO_TRACE = TraceSpec.none()
NULL_DIAGNOSTICS = NullDiagnostics()
NPZ_DIAGNOSTICS = NpzDiagnostics()
