"""Environment flag parsing shared by dense and local EM engines."""

from __future__ import annotations


from recovar.em.dense_single_volume.runtime_options import current_environment as _runtime_environment


def parse_int_set(value: str | None) -> set[int] | None:
    """Parse comma/semicolon/whitespace separated integer sets."""

    if not value:
        return None
    parsed = {int(token) for token in value.replace(",", " ").replace(";", " ").split()}
    return parsed or None


def parse_env_int_set(name: str) -> set[int] | None:
    """Parse an integer-set environment variable."""

    return parse_int_set(_runtime_environment().get(name))
