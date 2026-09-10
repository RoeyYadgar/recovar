"""Exact-local diagnostic routing and host-side capture session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recovar.em.dense_single_volume.diagnostics.local_capture import (
    current_size_matches_request,
    iteration_matches_request,
    maybe_write_debug_fused_posterior_dump,
    maybe_write_debug_noise_component_dump,
    maybe_write_debug_score_dump,
    parse_debug_fused_posterior_dump_request,
    parse_debug_noise_component_dump_request,
    parse_debug_score_dump_request,
)
from recovar.em.dense_single_volume.local_em_batch_planning import summarize_local_buckets
from recovar.em.dense_single_volume.local_layout import LocalBucketSpec
from recovar.em.dense_single_volume.runtime_options import current_environment as _runtime_environment

logger = logging.getLogger(__name__)

LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV = "RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT"
LOCAL_SCORE_DUMP_OPERANDS_ENV = "RECOVAR_LOCAL_SCORE_DUMP_OPERANDS"
LOCAL_SCORE_DUMP_TARGET_ONLY_ENV = "RECOVAR_LOCAL_SCORE_DUMP_TARGET_ONLY"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return _runtime_environment().get(name, "").strip().lower() in _TRUE_ENV_VALUES


@dataclass
class LocalDiagnosticRequest:
    """One parsed, mutable one-shot diagnostic target set."""

    dump_dir: Path | None
    pending_targets: set[int]
    requested_current_sizes: set[int] | None
    requested_iterations: set[int] | None
    enabled_for_call: bool

    @property
    def configured(self) -> bool:
        return self.dump_dir is not None


@dataclass
class LocalDiagnosticsSession:
    """Resolved exact-local capture routes for one engine call."""

    current_size: int | None
    iteration: int | None
    pass_label: str | None
    score: LocalDiagnosticRequest
    fused_posterior: LocalDiagnosticRequest
    noise: LocalDiagnosticRequest
    score_operands: bool
    score_force_split: bool
    score_big_jit: bool

    @classmethod
    def from_environment(
        cls,
        *,
        current_size: int | None,
        iteration: int | None,
        pass_label: str | None,
    ) -> LocalDiagnosticsSession:
        """Parse all exact-local capture routes once, in legacy order."""

        score = cls._request(*parse_debug_score_dump_request(), current_size=current_size, iteration=iteration)
        fused = cls._request(
            *parse_debug_fused_posterior_dump_request(),
            current_size=current_size,
            iteration=iteration,
        )
        noise = cls._request(
            *parse_debug_noise_component_dump_request(), current_size=current_size, iteration=iteration
        )
        score_operands = bool(score.enabled_for_call and _env_flag(LOCAL_SCORE_DUMP_OPERANDS_ENV))
        score_force_split = bool(score.enabled_for_call and _env_flag(LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV))
        return cls(
            current_size=current_size,
            iteration=iteration,
            pass_label=pass_label,
            score=score,
            fused_posterior=fused,
            noise=noise,
            score_operands=score_operands,
            score_force_split=score_force_split,
            score_big_jit=bool(score.enabled_for_call and not score_force_split),
        )

    @staticmethod
    def _request(
        dump_dir,
        pending_targets,
        requested_current_sizes,
        requested_iterations,
        *,
        current_size,
        iteration,
    ) -> LocalDiagnosticRequest:
        return LocalDiagnosticRequest(
            dump_dir=dump_dir,
            pending_targets=pending_targets,
            requested_current_sizes=requested_current_sizes,
            requested_iterations=requested_iterations,
            enabled_for_call=bool(
                dump_dir is not None
                and current_size_matches_request(requested_current_sizes, current_size)
                and iteration_matches_request(requested_iterations, iteration)
            ),
        )

    @property
    def target_only_targets(self) -> set[int]:
        targets: set[int] = set()
        if self.score.enabled_for_call:
            targets.update(self.score.pending_targets)
        if self.fused_posterior.enabled_for_call:
            targets.update(self.fused_posterior.pending_targets)
        return targets

    def filter_target_only_buckets(
        self,
        experiment_dataset,
        bucket_specs: list[LocalBucketSpec],
        bucket_summary,
        total_local_rotations: int,
        *,
        score_only: bool,
    ):
        """Apply the explicit invasive target-only diagnostic view."""

        targets = self.target_only_targets
        if not self.target_only_enabled(score_only=score_only):
            return bucket_specs, bucket_summary, total_local_rotations

        filtered = [
            bucket for bucket in bucket_specs if self._bucket_matches(experiment_dataset, bucket.image_indices, targets)
        ]
        filtered_total_rotations = int(
            sum(int(np.sum(bucket.actual_rotation_counts, dtype=np.int64)) for bucket in filtered)
        )
        filtered_summary = summarize_local_buckets(filtered)
        logger.info(
            "Exact local debug target-only: keeping %d/%d buckets and %d/%d images "
            "for requested original ids %s; unset %s to retain the full score-only computation",
            len(filtered),
            bucket_summary.bucket_count,
            filtered_summary.image_count,
            bucket_summary.image_count,
            sorted(int(target) for target in targets),
            LOCAL_SCORE_DUMP_TARGET_ONLY_ENV,
        )
        return filtered, filtered_summary, filtered_total_rotations

    def target_only_enabled(self, *, score_only: bool) -> bool:
        """Return whether the invasive target-only bucket view was requested."""

        return bool(score_only and self.target_only_targets and _env_flag(LOCAL_SCORE_DUMP_TARGET_ONLY_ENV))

    @staticmethod
    def _bucket_matches(experiment_dataset, image_indices, pending_targets: set[int] | None) -> bool:
        if not pending_targets:
            return False
        original_indices = np.asarray(
            experiment_dataset.original_image_indices_from_local(image_indices),
            dtype=np.int64,
        )
        return any(int(original_index) in pending_targets for original_index in original_indices.tolist())

    def score_bucket_matches(self, experiment_dataset, image_indices) -> bool:
        return bool(
            self.score.enabled_for_call
            and self._bucket_matches(experiment_dataset, image_indices, self.score.pending_targets)
        )

    def fused_bucket_matches(self, experiment_dataset, image_indices) -> bool:
        return bool(
            self.fused_posterior.enabled_for_call
            and self._bucket_matches(experiment_dataset, image_indices, self.fused_posterior.pending_targets)
        )

    def emit_fused_posterior(self, **payload: Any) -> None:
        self.fused_posterior.pending_targets = maybe_write_debug_fused_posterior_dump(
            **payload,
            current_size=self.current_size,
            debug_iteration=self.iteration,
            dump_dir=self.fused_posterior.dump_dir,
            pending_targets=self.fused_posterior.pending_targets,
            requested_current_sizes=self.fused_posterior.requested_current_sizes,
            requested_iterations=self.fused_posterior.requested_iterations,
        )

    def emit_score(self, **payload: Any) -> None:
        self.score.pending_targets = maybe_write_debug_score_dump(
            **payload,
            current_size=self.current_size,
            debug_iteration=self.iteration,
            debug_pass_label=self.pass_label,
            dump_dir=self.score.dump_dir,
            pending_targets=self.score.pending_targets,
            requested_current_sizes=self.score.requested_current_sizes,
            requested_iterations=self.score.requested_iterations,
        )

    def emit_noise(self, **payload: Any) -> None:
        self.noise.pending_targets = maybe_write_debug_noise_component_dump(
            **payload,
            current_size=self.current_size,
            debug_iteration=self.iteration,
            dump_dir=self.noise.dump_dir,
            pending_targets=self.noise.pending_targets,
            requested_current_sizes=self.noise.requested_current_sizes,
            requested_iterations=self.noise.requested_iterations,
        )

    def warn_for_unobserved_targets(self) -> None:
        if self.score.enabled_for_call and self.score.pending_targets and self.score.requested_iterations is None:
            logger.warning(
                "Requested local score dump indices were not observed in this dataset view: %s",
                sorted(self.score.pending_targets),
            )
        if (
            self.fused_posterior.enabled_for_call
            and self.fused_posterior.pending_targets
            and self.fused_posterior.requested_iterations is None
        ):
            logger.warning(
                "Requested fused posterior dump indices were not observed in this dataset view: %s",
                sorted(self.fused_posterior.pending_targets),
            )
