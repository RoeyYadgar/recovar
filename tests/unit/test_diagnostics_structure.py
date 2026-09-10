from __future__ import annotations

import inspect

from recovar.em.dense_single_volume import em_engine, iteration_loop, local_em_engine
from recovar.em.dense_single_volume.helpers import significance, sparse_pass2_bucketed


def test_algorithm_modules_do_not_serialize_diagnostic_artifacts_directly():
    modules = (
        iteration_loop,
        em_engine,
        local_em_engine,
        significance,
        sparse_pass2_bucketed,
    )

    for module in modules:
        source = inspect.getsource(module)
        assert "np.save(" not in source, module.__name__
        assert "np.savez(" not in source, module.__name__
        assert "np.savez_compressed(" not in source, module.__name__


def test_numeric_helpers_do_not_define_or_raise_diagnostic_stop_exceptions():
    significance_source = inspect.getsource(significance)
    sparse_source = inspect.getsource(sparse_pass2_bucketed)

    assert "class SignificanceDumpComplete" not in significance_source
    assert "raise SignificanceDumpComplete" not in significance_source
    assert "class Pass2DumpComplete" not in sparse_source
    assert "class BPrefContributionDumpComplete" not in sparse_source
    assert "raise Pass2DumpComplete" not in sparse_source
