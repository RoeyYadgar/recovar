"""Compatibility facade for the host-side RELION parity diagnostics sink.

New orchestration code should depend on ``diagnostics.parity``.  These aliases
keep existing imports, monkeypatches, and artifact behavior stable while C3
migrates callers one boundary at a time.
"""

from recovar.em.dense_single_volume.diagnostics.parity import _E_STEP as _E_STEP
from recovar.em.dense_single_volume.diagnostics.parity import _ITER_TIMERS as _ITER_TIMERS
from recovar.em.dense_single_volume.diagnostics.parity import collect_e_step as collect_e_step
from recovar.em.dense_single_volume.diagnostics.parity import dump_dir as dump_dir
from recovar.em.dense_single_volume.diagnostics.parity import dump_iteration as dump_iteration
from recovar.em.dense_single_volume.diagnostics.parity import dump_timing_iteration as dump_timing_iteration
from recovar.em.dense_single_volume.diagnostics.parity import get_iteration_timing as get_iteration_timing
from recovar.em.dense_single_volume.diagnostics.parity import is_active as is_active
from recovar.em.dense_single_volume.diagnostics.parity import mark_stage as mark_stage
from recovar.em.dense_single_volume.diagnostics.parity import reset_iteration as reset_iteration
from recovar.em.dense_single_volume.diagnostics.parity import reset_iteration_timer as reset_iteration_timer
from recovar.em.dense_single_volume.diagnostics.parity import start_iteration as start_iteration
from recovar.em.dense_single_volume.diagnostics.parity import timing_dir as timing_dir
from recovar.em.dense_single_volume.diagnostics.parity import timing_is_active as timing_is_active
