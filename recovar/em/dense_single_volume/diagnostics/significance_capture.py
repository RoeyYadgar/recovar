"""Serialization and invasive completion policy for significance captures."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import diagnostics_environment as _runtime_environment

logger = logging.getLogger(__name__)

_STOP_AFTER_TARGET_ENV = "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET"


class SignificanceDumpComplete(RuntimeError):
    """Raised after an explicitly targeted significance dump is durable."""

    def __init__(self, *, dump_path: str):
        self.dump_path = str(dump_path)
        super().__init__(f"requested RECOVAR coarse-significance target was written (dump_path={self.dump_path})")


@dataclass(frozen=True)
class SignificanceTarget:
    """Invasive completion boundary for a targeted significance capture."""

    dump_dir: str
    original_indices: frozenset[int]
    current_size: int | None
    debug_iteration: int | None


def stop_after_significance_dump(dump_path: str, target: SignificanceTarget) -> None:
    """Stop an explicit diagnostic only after its complete target set exists."""

    if _runtime_environment().get(_STOP_AFTER_TARGET_ENV) != "1":
        return
    if not os.path.isfile(dump_path):
        raise RuntimeError(f"RECOVAR significance stop target is missing its dump file: {dump_path}")
    target_iteration = _runtime_environment().get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")
    iteration_suffix = "" if not target_iteration else f"_it{int(target.debug_iteration):03d}"
    current_size_label = -1 if target.current_size is None else int(target.current_size)
    expected_paths = [
        os.path.join(
            target.dump_dir,
            f"significance_orig{original_index:06d}{iteration_suffix}_cs{current_size_label:03d}.npz",
        )
        for original_index in sorted(target.original_indices)
    ]
    missing_paths = [path for path in expected_paths if not os.path.isfile(path)]
    if missing_paths:
        logger.info(
            "RECOVAR coarse-significance stop target progress: %d/%d files written",
            len(expected_paths) - len(missing_paths),
            len(expected_paths),
        )
        return
    raise SignificanceDumpComplete(dump_path=dump_path)


def write_tree_rescore(path: str | Path, **payload: Any) -> None:
    """Write the stable bounded tree-rescore schema."""

    np.savez_compressed(path, **payload)


def write_single_class_significance(path: str | Path, **payload: Any) -> None:
    """Write the stable single-class significance schema."""

    np.savez_compressed(path, **payload)


def write_kclass_significance(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    target: SignificanceTarget,
) -> None:
    """Write a K-class schema and apply its explicit invasive stop policy."""

    np.savez_compressed(path, **payload)
    stop_after_significance_dump(str(path), target)
