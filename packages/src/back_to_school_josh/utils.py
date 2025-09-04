__all__ = ["unwrap", "T", "flatten"]

from collections.abc import Iterable
from typing import (
    TypeVar,
)
from collections.abc import Iterator

T = TypeVar("T")


def unwrap(container: Iterable[T], msg: str = "") -> T:
    """
    Unwrap an iterable only if it has a single element; raise ValueError otherwise
    """
    if msg:
        msg += ": "

    iterator = iter(container)

    try:
        value = next(iterator)
    except StopIteration:
        raise ValueError(msg + "container has no elements")

    try:
        next(iterator)
    except StopIteration:
        return value

    raise ValueError(msg + "container has multiple elements")


def flatten(container: Iterable[Iterable[T]]) -> Iterator[T]:
    for inner in container:
        yield from inner
