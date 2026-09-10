from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume.diagnostics.local_capture import (
    DenseCcComponentCapture,
    DenseNoiseComponentCapture,
    maybe_write_dense_cc_components,
    write_dense_noise_components,
)
from recovar.em.dense_single_volume import em_engine


class _IdentityWindow:
    @staticmethod
    def score_values(values):
        return values


class _MustNotMaterialize:
    def __array__(self, *args, **kwargs):
        raise AssertionError("disabled passive capture materialized an array")


def _cc_capture(value) -> DenseCcComponentCapture:
    return DenseCcComponentCapture(
        indices=np.array([3]),
        original_indices=np.array([42]),
        shifted_windowed=value,
        ctf2_over_noise_windowed=value,
        batch_norm=value,
        projector_half=value,
        projector_abs2_half=value,
        window_spec=_IdentityWindow(),
        block=SimpleNamespace(index=2, r0=0, r1=1),
        batch_size=1,
        translation_count=1,
        score_mode="normalized_cc",
    )


def test_disabled_dense_cc_capture_does_not_materialize(monkeypatch):
    monkeypatch.delenv("RECOVAR_DEBUG_CC_COMPONENT_DUMP_DIR", raising=False)
    maybe_write_dense_cc_components(_cc_capture(_MustNotMaterialize()))


def test_dense_cc_capture_retains_filename_keys_and_values(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_DEBUG_CC_COMPONENT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_DEBUG_CC_COMPONENT_DUMP_TARGET", "42")
    monkeypatch.setenv("RECOVAR_DEBUG_CC_COMPONENT_DUMP_TARGET_IS_ORIGINAL", "1")
    capture = _cc_capture(np.ones((1, 1, 2), dtype=np.float32))
    capture = DenseCcComponentCapture(
        **{
            **capture.__dict__,
            "ctf2_over_noise_windowed": np.ones((1, 2), dtype=np.float32),
            "batch_norm": np.array([[5.0]], dtype=np.float32),
            "projector_half": np.array([[2.0, 3.0]], dtype=np.float32),
            "projector_abs2_half": np.array([[4.0, 9.0]], dtype=np.float32),
        }
    )

    maybe_write_dense_cc_components(capture)

    path = tmp_path / "cc_components_target000042_block0002.npz"
    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "active_rotations",
            "batch_norm",
            "cross_tr",
            "local_index",
            "n_score",
            "n_trans",
            "norms_r",
            "original_index",
            "r0",
            "r1",
            "score_mode",
            "stored_rotations",
        }
        np.testing.assert_array_equal(artifact["cross_tr"], np.array([[-10.0]]))
        np.testing.assert_array_equal(artifact["norms_r"], np.array([13.0]))
        assert artifact["score_mode"] == "normalized_cc"


def test_dense_noise_capture_retains_schema(tmp_path):
    write_dense_noise_components(
        DenseNoiseComponentCapture(
            dump_dir=tmp_path,
            accumulators={
                42: {
                    "local_idx": 3,
                    "p_img_shells": np.array([1.0, 2.0]),
                    "a2_shells": np.array([3.0, 4.0]),
                    "xa_shells": np.array([0.5, 1.0]),
                }
            },
            current_size=8,
            rotation_count=5,
            translation_count=2,
            shell_indices_half=np.array([0, 1]),
            shell_indices_noise=np.array([0, 1]),
        )
    )

    path = tmp_path / "dense_noise_components_cs008_image_42.npz"
    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "a2_shells",
            "current_size",
            "n_rot",
            "n_trans",
            "p_img_shells",
            "selected_global_image_indices",
            "selected_local_image_indices",
            "shell_indices_half",
            "shell_indices_noise",
            "total_shells",
            "xa_shells",
        }
        np.testing.assert_array_equal(artifact["total_shells"], np.array([3.0, 4.0]))
        assert artifact["selected_global_image_indices"].dtype == np.int64
        assert artifact["shell_indices_half"].dtype == np.int32


def test_dense_engine_has_no_direct_artifact_serialization():
    assert "np.save" not in inspect.getsource(em_engine)
