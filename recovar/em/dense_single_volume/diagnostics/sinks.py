"""Small, explicit sink protocol for host-side dense EM diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .events import IterationStarted

if TYPE_CHECKING:
    from .config import DiagnosticsPlan


class DiagnosticsSink(Protocol):
    """Observer invoked only at existing host orchestration boundaries.

    Sink methods intentionally return ``None``: passive diagnostics can observe
    production values but cannot replace or select algorithm outputs.
    """

    def iteration_started(self, event: IterationStarted) -> None: ...


class NullDiagnostics:
    """Production sink that performs no inspection, conversion, or I/O."""

    __slots__ = ()

    def iteration_started(self, event: IterationStarted) -> None:
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


NULL_DIAGNOSTICS = NullDiagnostics()
NPZ_DIAGNOSTICS = NpzDiagnostics()


def build_diagnostics_sink(plan: DiagnosticsPlan) -> DiagnosticsSink:
    """Build the run-wide lifecycle sink from a resolved diagnostics plan."""

    names = {*plan.passive, *plan.invasive}
    if names.intersection({"RECOVAR_PARITY_DUMP_DIR", "RECOVAR_PARITY_TIMING_DIR"}):
        from .parity import PARITY_DIAGNOSTICS

        return PARITY_DIAGNOSTICS
    return NULL_DIAGNOSTICS
