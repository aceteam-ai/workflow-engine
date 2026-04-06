# workflow_engine/utils/mappings.py

from collections.abc import Mapping

from functools import reduce
from operator import eq
from typing import Callable, TypeVar

from .iter import same

K = TypeVar("K")
V = TypeVar("V")


def mapping_intersection(
    *mappings: Mapping[K, V],
    compare_fn: Callable[[V, V], bool] = eq,
) -> Mapping[K, V]:
    """
    Computes the intersection of the given mappings, which consists of the keys
    in common to all mappings.

    For each key in the intersection, the associated value must be the same
    across all mappings.

    A custom compare function can be provided to compare the values when the
    equality operator is not sufficient.
    """
    if len(mappings) == 0:
        return {}
    if len(mappings) == 1:
        return mappings[0]

    keys = reduce(
        lambda acc, mapping: acc & set(mapping.keys()),
        mappings[1:],
        set(mappings[0].keys()),
    )
    return {
        key: same(
            (mapping[key] for mapping in mappings),
            compare_fn=compare_fn,
        )
        for key in keys
    }


__all__ = [
    "mapping_intersection",
]
