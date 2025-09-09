__all__ = ["unwrap", "T", "flatten", "sibpath"]

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import (
    TypeVar,
)

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


def sibpath(path: str | Path) -> Path:
    """
    Make a sibling path to the current script relative to the working directory.

    If there is no current script (``sys.modules["__main__"] is None``), return
    the path unmodified. In other words, interpret it as a path relative to the
    working directory instead of relative to the parent directory of the script.
    """
    script_path = sys.modules["__main__"].__file__
    if script_path is None:
        return Path(path)
    return Path.relative_to(
        Path(script_path).parent.resolve() / path,
        Path(".").resolve(),
    )


def filter_none(iterable: Iterable[T | None]) -> Iterator[T]:
    for elem in iterable:
        if elem is not None:
            yield elem
