import asyncio
from collections.abc import Iterable, Sequence, Sized
from typing import Awaitable, Callable, TypeVar

from .iter import only

T = TypeVar("T")


def is_coroutine(fn: Callable):
    # NOTE (PR 92): asyncio.iscoroutine is faster than inspect.iscoroutine
    return asyncio.iscoroutine(fn)


async def gather(tasks: Iterable[Awaitable[T]]) -> Sequence[T]:
    """
    A faster version of asyncio.gather for homogeneous iterables of tasks.
    """
    # NOTE (PR 92): Gather has overhead that can be avoided for empty and
    # singleton sequences.
    if not isinstance(tasks, Sized):
        tasks = tuple(tasks)
    num_tasks = len(tasks)
    if num_tasks == 0:
        return ()
    elif num_tasks == 1:
        return (await only(tasks),)
    else:
        return await asyncio.gather(*tasks)
