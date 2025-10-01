"""Custom Griffe extensions so the documentation displays properly.

See details at https://mkdocstrings.github.io/griffe/guide/users/how-to/support-decorators/
"""

from typing import Any

import griffe


class ReturnNew(griffe.Extension):
    """Changes the return type of functions decorated with @return_new appropriately."""

    #  TODO: don't make Set an Operable to simplify this and ensure func.returns is never None
    def on_function_instance(self, *, func: griffe.Function, **kwargs: Any) -> None:
        for decorator in func.decorators:
            if decorator.callable_path == "pyoframe._utils.return_new":
                if ".Expression." in func.path:
                    func.returns = "Expression"
                elif ".Set." in func.path:
                    func.returns = "Set"
                elif ".Variable." in func.path:
                    func.returns = "Expression"
                else:
                    func.returns = None  # for decorated functions of the parent class (BaseOperableBlock)
