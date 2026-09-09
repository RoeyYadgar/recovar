"""Structural ratchets for the dense-EM runtime/environment boundary."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from recovar.em.dense_single_volume.diagnostics.config import (
    classify_environment_name,
)

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(__file__).parents[2] / "recovar" / "em" / "dense_single_volume"
PROCESS_ENVIRONMENT_BOUNDARIES = {
    PACKAGE_ROOT / "runtime_options.py",
    PACKAGE_ROOT / "diagnostics" / "config.py",
}
ENVIRONMENT_ACCESSORS = {
    "_runtime_environment",
    "current_environment",
    "diagnostics_environment",
}
ENVIRONMENT_NAME = re.compile(r"^RECOVAR_[A-Z0-9_]+$")


def _python_sources():
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _is_jitted(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any("jax.jit" in ast.unparse(decorator) for decorator in function.decorator_list)


def test_only_configuration_boundaries_read_the_process_environment():
    violations = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        os_aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "os"
        }
        imported_accessors = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "os"
            for alias in node.names
            if alias.name in {"environ", "getenv"}
        }
        for node in ast.walk(tree):
            is_os_environment = (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in os_aliases
                and node.attr == "environ"
            )
            is_os_getenv = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in os_aliases
                and node.func.attr == "getenv"
            )
            is_imported_accessor = isinstance(node, ast.Name) and node.id in imported_accessors
            if (
                is_os_environment or is_os_getenv or is_imported_accessor
            ) and path not in PROCESS_ENVIRONMENT_BOUNDARIES:
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")

    assert violations == []


def test_jitted_kernels_do_not_consult_runtime_environment_accessors():
    violations = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_jitted(node)
        ):
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ENVIRONMENT_ACCESSORS
                ):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{function.lineno}:{function.name}")

    assert violations == []


def test_every_named_dense_em_environment_setting_has_an_effect_class():
    names = set()
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        names.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and ENVIRONMENT_NAME.fullmatch(node.value)
        )

    assert names
    assert {name for name in names if classify_environment_name(name) is None} == set()
