# workflow_engine/utils/iter.py
from collections.abc import Iterable
from operator import eq
from typing import Callable, TypeVar

T = TypeVar("T")


def only(it: Iterable[T]) -> T:
    (x,) = iter(it)
    return x


def same(
    it: Iterable[T],
    *,
    compare_fn: Callable[[T, T], bool] = eq,
) -> T:
    """
    Returns the only distinct element in the iterable if it is the same as all
    other elements, otherwise raises a ValueError if the values are not the same
    or StopIteration if the iterable is empty.

    A custom compare function can be provided to compare the values when the
    equality operator is not sufficient.
    """
    it = iter(it)
    x = next(it)
    for y in it:
        if not compare_fn(x, y):
            raise ValueError("Values are not the same")
    return x


__all__ = [
    "only",
    "same",
]
