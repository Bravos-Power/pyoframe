"""Contains the base classes used for benchmarking.

Note that this contributes to every benchmark so we try to keep the imports mostly clear.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseBenchmark(ABC):
    def __init__(
        self,
        solver="gurobi",
        block_solver=False,
        input_dir=None,
        results_dir=None,
        size=None,
        emit_benchmarking_logs=False,
        **kwargs,
    ):
        self.solver = solver
        self.block_solver = block_solver
        self.input_dir = Path(input_dir) if input_dir else None
        self.results_dir = Path(results_dir) if results_dir else None
        self.size = size
        self.emit_benchmarking_logs = emit_benchmarking_logs
        self.kwargs = kwargs

    @abstractmethod
    def build(self, **kwargs) -> Any: ...

    @abstractmethod
    def set_timeout_to_zero(self, model) -> None: ...

    @abstractmethod
    def solve(self, model) -> None: ...

    def write_results(self, model, **kwargs) -> None: ...

    def get_objective(self) -> float:
        assert not self.block_solver, (
            "Cannot get objective value when block_solver is True."
        )
        return self._get_objective(self.model)

    @abstractmethod
    def _get_objective(self, model) -> float: ...

    def run(self):
        if self.emit_benchmarking_logs:
            print("PF_BENCHMARK: 1_START", flush=True)
        with (
            contextlib.chdir(self.input_dir)
            if self.input_dir
            else contextlib.nullcontext()
        ):
            self.model = self.build(**self.kwargs)

        if self.block_solver:
            self.set_timeout_to_zero(self.model)

        if self.emit_benchmarking_logs:
            print("PF_BENCHMARK: 2_SOLVE", flush=True)
        self.solve(self.model)
        if self.emit_benchmarking_logs:
            print("PF_BENCHMARK: 5_SOLVE_RETURNED", flush=True)

        if self.results_dir is not None and not self.block_solver:
            with contextlib.chdir(self.results_dir):
                self.write_results(self.model, **self.kwargs)
        if self.emit_benchmarking_logs:
            print("PF_BENCHMARK: 6_DONE", flush=True)
        return self.model
