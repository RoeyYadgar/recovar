"""JAX-aware precision, reconstruction, and Fourier setup for exact-local EM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import (
    FourierWindowSpec,
    centered_half_indices_to_fftw_half_indices,
    make_fourier_window_spec,
)
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    half_volume_accumulator_shape,
    relion_backprojector_volume_shape,
)
from recovar.em.dense_single_volume.helpers.projection import indexed_projection_available
from recovar.em.dense_single_volume.local_em_planning import LocalEMGeometryPlan, LocalEMModePlan
from recovar.em.dense_single_volume.local_em_types import (
    LocalProjectionSettings,
    LocalScoringSettings,
    LocalSearchSettings,
)


@dataclass(frozen=True)
class LocalEMReconstructionPlan:
    """Resolved reconstruction volume geometry."""

    volume_shape: tuple[int, ...]


@dataclass(frozen=True)
class LocalProjectionPlan:
    """Immutable projection options resolved from the Fourier window."""

    use_window: bool
    max_r: float | None
    relion_texture_interp: bool
    relion_acc_double_floorf_quirk: bool
    force_jax: bool

    def kwargs(self) -> dict[str, Any]:
        """Materialize the legacy projection keyword mapping."""

        kwargs: dict[str, Any] = {}
        if self.use_window:
            kwargs["max_r"] = self.max_r
        kwargs["relion_texture_interp"] = self.relion_texture_interp
        kwargs["relion_acc_double_floorf_quirk"] = self.relion_acc_double_floorf_quirk
        kwargs["force_jax"] = self.force_jax
        return kwargs


@dataclass(frozen=True)
class LocalEMFourierPlan:
    """Array metadata built once before exact-local bucket execution."""

    reconstruction_accumulator_shape: tuple[int, ...]
    reconstruction_accumulator_size: int
    accumulator_allocation_size: int
    window: FourierWindowSpec
    mstep_reconstruction_window_indices: Any
    mstep_adjoint_max_r: float | None
    projection: LocalProjectionPlan
    projection_mode: str


def make_local_em_precision(
    *,
    scoring: LocalScoringSettings,
    projection: LocalProjectionSettings,
) -> DensePrecisionPolicy:
    """Build the existing dense precision policy from cohesive local settings."""

    return DensePrecisionPolicy(
        use_float64_scoring=scoring.use_float64_scoring,
        use_float64_projections=projection.use_float64_projections,
        use_float64_normalization=scoring.use_float64_normalization,
    )


def plan_local_em_reconstruction(
    *,
    geometry: LocalEMGeometryPlan,
    projection: LocalProjectionSettings,
    mode: LocalEMModePlan,
) -> LocalEMReconstructionPlan:
    """Resolve reconstruction volume geometry without allocating arrays."""

    volume_shape = geometry.volume_shape
    if mode.mstep_relion_x_half:
        # RELION BPref::initZeros(current_size) sizes the accumulator from the
        # iteration r_max. The reconstruction boundary crops back to the
        # original volume shape.
        volume_shape = relion_backprojector_volume_shape(
            volume_shape,
            projection.reconstruction_padding_factor,
            current_size=geometry.mstep_current_size,
        )
    elif projection.reconstruction_padding_factor > 1:
        volume_shape = tuple(dimension * projection.reconstruction_padding_factor for dimension in volume_shape)
    return LocalEMReconstructionPlan(volume_shape=volume_shape)


def local_mstep_adjoint_window(
    image_shape,
    n_half: int,
    current_size: int | None,
    *,
    use_window: bool,
    recon_window_indices,
    mstep_relion_x_half: bool,
):
    """Return coordinate indices/max radius for exact-local M-step adjoints."""

    mstep_reconstruction_window_indices = recon_window_indices
    if mstep_relion_x_half:
        if mstep_reconstruction_window_indices is None:
            mstep_reconstruction_window_indices = jnp.arange(int(n_half), dtype=jnp.int32)
        mstep_reconstruction_window_indices = centered_half_indices_to_fftw_half_indices(
            image_shape,
            mstep_reconstruction_window_indices,
        )
    mstep_adjoint_max_r = None
    if use_window or mstep_relion_x_half:
        mstep_current_size = int(current_size) if current_size is not None else int(image_shape[0])
        mstep_adjoint_max_r = float(mstep_current_size // 2)
    return mstep_reconstruction_window_indices, mstep_adjoint_max_r


def _projection_mode(
    window: FourierWindowSpec,
    projection: LocalProjectionPlan,
    relion_projector_half=None,
) -> str:
    if relion_projector_half is not None:
        return "relion_projector"
    if not window.use_window:
        return "full"
    if projection.force_jax:
        return "windowed_full_jax"
    if projection.relion_texture_interp:
        return "windowed_full_texture"
    if not indexed_projection_available():
        return "windowed_full_cuda_unavailable"
    return "windowed_indexed_cuda"


def plan_local_em_fourier(
    *,
    geometry: LocalEMGeometryPlan,
    search: LocalSearchSettings,
    projection: LocalProjectionSettings,
    mode: LocalEMModePlan,
    reconstruction: LocalEMReconstructionPlan,
    relion_projector_half=None,
) -> LocalEMFourierPlan:
    """Build existing accumulator and Fourier-window metadata in original order."""

    accumulator_shape = half_volume_accumulator_shape(reconstruction.volume_shape)
    accumulator_size = int(np.prod(accumulator_shape))
    window = make_fourier_window_spec(
        geometry.image_shape,
        search.current_size,
        geometry.n_half,
        reconstruction_current_size=geometry.mstep_current_size,
        square=projection.square_window,
        include_recon_window=True,
    )
    mstep_reconstruction_window_indices, mstep_adjoint_max_r = local_mstep_adjoint_window(
        geometry.image_shape,
        geometry.n_half,
        geometry.mstep_current_size,
        use_window=window.use_window,
        recon_window_indices=window.recon_indices,
        mstep_relion_x_half=mode.mstep_relion_x_half,
    )
    projection_plan = LocalProjectionPlan(
        use_window=window.use_window,
        max_r=window.projection_max_r,
        relion_texture_interp=projection.relion_texture_interp,
        relion_acc_double_floorf_quirk=projection.relion_acc_double_floorf_quirk,
        force_jax=bool(projection.force_jax),
    )
    return LocalEMFourierPlan(
        reconstruction_accumulator_shape=accumulator_shape,
        reconstruction_accumulator_size=accumulator_size,
        accumulator_allocation_size=1 if mode.score_only else accumulator_size,
        window=window,
        mstep_reconstruction_window_indices=mstep_reconstruction_window_indices,
        mstep_adjoint_max_r=mstep_adjoint_max_r,
        projection=projection_plan,
        projection_mode=_projection_mode(window, projection_plan, relion_projector_half),
    )
