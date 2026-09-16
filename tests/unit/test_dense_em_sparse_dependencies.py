"""Dependency-direction guards for dense-EM sparse search."""

import ast
from pathlib import Path

HELPERS = Path(__file__).parents[2] / "recovar" / "em" / "dense_single_volume" / "helpers"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_significance_does_not_import_sparse_pass2_implementation():
    imports = _imported_modules(HELPERS / "significance.py")

    assert "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed" not in imports
