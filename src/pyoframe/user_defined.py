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

    def __init__(self, preprocess=None):
        self._preprocess = preprocess
        self._attributes = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            return super().__setattr__(name, value)
        if self._preprocess is not None:
            value = self._preprocess(name, value)
        self._attributes[name] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._attributes[name]

    def __iter__(self):
        return iter(self._attributes.items())


class AttrContainerMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attr = Container(preprocess=self._preprocess_attr)

    def _preprocess_attr(self, name: str, value: Any) -> Any:
        """
        Preprocesses user-defined values before adding them to the Params container.
        By default this function does nothing but subclasses can override it.
        """
        return value
