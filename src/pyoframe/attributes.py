"""
Contains the base classes to support .params and .attr containers for user-defined parameters and attributes.
"""

from typing import Any


class Container:
    """
    A container for user-defined attributes or parameters.

    Parameters:
        preprocess : Callable[str, Any], optional
            A function to preprocess user-defined values before adding them to the container.

    Examples:
        >>> params = Container()
        >>> params.a = 1
        >>> params.b = 2
        >>> params.a
        1
        >>> params.b
        2
        >>> for k, v in params:
        ...     print(k, v)
        a 1
        b 2
    """

    def __init__(self, setter, getter):
        self._setter = setter
        self._getter = getter

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            return super().__setattr__(name, value)
        self._setter(name, value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._getter(name)
