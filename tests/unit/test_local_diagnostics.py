from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume import local_diagnostics
from recovar.em.dense_single_volume.local_diagnostics import (
    LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV,
    LOCAL_SCORE_DUMP_OPERANDS_ENV,
    LOCAL_SCORE_DUMP_TARGET_ONLY_ENV,
    LocalDiagnosticRequest,
    LocalDiagnosticsSession,
)
from recovar.em.dense_single_volume.local_layout import LocalBucketSpec


def _request(*, targets=(), enabled=True):
    return LocalDiagnosticRequest(
        dump_dir=Path("debug") if enabled else None,
        pending_targets=set(targets),
        requested_current_sizes=None,
        requested_iterations=None,
        enabled_for_call=enabled,
    )


def _bucket(image_index: int) -> LocalBucketSpec:
    return LocalBucketSpec(
        image_indices=np.asarray([image_index], dtype=np.int32),
        bucket_image_count=1,
        bucket_rotation_count=1,
        actual_rotation_counts=np.ones(1, dtype=np.int32),
        local_rotation_ids=np.asarray([[image_index]], dtype=np.int32),
        local_rotations=np.eye(3, dtype=np.float32)[None, None],
        local_rotation_log_prior=np.zeros((1, 1), dtype=np.float32),
        local_rotation_mask=np.ones((1, 1), dtype=bool),
        translation_log_prior=np.zeros((1, 1), dtype=np.float32),
    )


def test_local_diagnostics_session_resolves_score_flags_once(monkeypatch):
    monkeypatch.setattr(
        local_diagnostics,
        "parse_debug_score_dump_request",
        lambda: (Path("score"), {7}, {8}, {3}),
    )
    monkeypatch.setattr(
        local_diagnostics,
        "parse_debug_fused_posterior_dump_request",
        lambda: (None, set(), None, None),
    )
    monkeypatch.setattr(
        local_diagnostics,
        "parse_debug_noise_component_dump_request",
        lambda: (Path("noise"), {9}, None, None),
    )
    monkeypatch.setenv(LOCAL_SCORE_DUMP_OPERANDS_ENV, "1")
    monkeypatch.setenv(LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV, "true")

    session = LocalDiagnosticsSession.from_environment(current_size=8, iteration=3, pass_label="fine")

    assert session.score.enabled_for_call
    assert session.noise.enabled_for_call
    assert session.score_operands
    assert session.score_force_split
    assert not session.score_big_jit


def test_local_diagnostics_target_only_filter_is_explicit(monkeypatch):
    session = LocalDiagnosticsSession(
        current_size=8,
        iteration=3,
        pass_label=None,
        score=_request(targets={11}),
        fused_posterior=_request(enabled=False),
        noise=_request(enabled=False),
        score_operands=False,
        score_force_split=False,
        score_big_jit=True,
    )
    monkeypatch.setenv(LOCAL_SCORE_DUMP_TARGET_ONLY_ENV, "1")
    buckets = [_bucket(0), _bucket(1)]
    summary = SimpleNamespace(bucket_count=2, image_count=2)
    dataset = SimpleNamespace(
        original_image_indices_from_local=lambda indices: np.asarray(indices, dtype=np.int64) + 10
    )

    filtered, filtered_summary, rotation_count = session.filter_target_only_buckets(
        dataset,
        buckets,
        summary,
        2,
        score_only=True,
    )

    assert [int(bucket.image_indices[0]) for bucket in filtered] == [1]
    assert filtered_summary.bucket_count == 1
    assert rotation_count == 1


def test_local_diagnostics_emit_injects_call_identity_and_updates_targets(monkeypatch):
    session = LocalDiagnosticsSession(
        current_size=8,
        iteration=3,
        pass_label="fine",
        score=_request(targets={11}),
        fused_posterior=_request(enabled=False),
        noise=_request(enabled=False),
        score_operands=False,
        score_force_split=False,
        score_big_jit=True,
    )
    captured = {}

    def fake_writer(**kwargs):
        captured.update(kwargs)
        return set()

    monkeypatch.setattr(local_diagnostics, "maybe_write_debug_score_dump", fake_writer)

    session.emit_score(scores="scores")

    assert captured["scores"] == "scores"
    assert captured["current_size"] == 8
    assert captured["debug_iteration"] == 3
    assert captured["debug_pass_label"] == "fine"
    assert session.score.pending_targets == set()
