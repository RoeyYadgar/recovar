"""Compatibility alias for compact sparse-pass diagnostic capture."""

import sys

from ..diagnostics import sparse_capture as _sparse_capture

# Legacy users mutate diagnostic limits and counters on this module. Replacing
# the module entry keeps those writes attached to the implementation.
sys.modules[__name__] = _sparse_capture
