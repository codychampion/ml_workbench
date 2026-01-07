"""No-op decorators (Prefect removed). Backward compatibility only."""

from typing import Callable, TypeVar

F = TypeVar('F', bound=Callable)


def flow(*args, **kwargs) -> Callable[[F], F]:
    """No-op flow decorator. Returns function unchanged."""
    if args and callable(args[0]):
        return args[0]  # @flow without parens
    return lambda fn: fn  # @flow(...) with parens


def task(*args, **kwargs) -> Callable[[F], F]:
    """No-op task decorator. Returns function unchanged."""
    if args and callable(args[0]):
        return args[0]  # @task without parens
    return lambda fn: fn  # @task(...) with parens
