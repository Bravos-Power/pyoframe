"""Script to run the benchmarks specified in config.yaml.

Saves results to results/benchmark_results.csv.
"""

import argparse
import itertools
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import psutil
import tomllib
import yaml
from polars.testing import assert_frame_equal

POLL_MIN_S, POLL_MAX_S, POLL_TRANSITION_S = 0.01, 1, 30

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

CWD = Path(__file__).parent


@dataclass
class Benchmark:
    problem: str
    solver: str
    library: str
    size: int | None
    construct_only: bool = False
    julia_trace_compile: bool = False


def run_all_benchmarks(config, ignore_past_results=False):
    base_dir = CWD / "results" / config["name"]
    base_dir.mkdir(parents=True, exist_ok=True)

    past_results = PastResults(base_dir, ignore_past_results=ignore_past_results)

    timeout = config.get("timeout", None)

    for problem, problem_config in config["problems"].items():
        prepare_benchmark_problem(problem, problem_config)

        num_repeats = problem_config.get("repeat", config.get("repeat", 1))
        for size in sorted(problem_config.get("size", [None])):
            with get_base_results_dir(
                config, base_dir, problem, size
            ) as base_results_dir:
                for solver, library in itertools.product(
                    config["solvers"], config["libraries"]
                ):
                    benchmark = Benchmark(
                        problem=problem,
                        solver=solver,
                        library=library,
                        size=size,
                        construct_only=problem_config.get("construct_only", False),
                        julia_trace_compile=config.get("julia_trace_compile", False),
                    )
                    if not get_benchmark_code(benchmark).exists():
                        print(f"{problem}: Skipping {library} as no benchmark found.")
                        continue

                    if not should_run_benchmark(
                        benchmark, past_results, timeout, num_repeats
                    ):
                        print(
                            f"{problem} (n={size}): Skipping {library}, already benchmarked or timed out."
                        )
                        continue

                    input_dir = (
                        CWD / "src" / problem / "model_data"
                        if "inputs" in problem_config
                        else None
                    )
                    try:
                        for i in range(num_repeats):
                            print(
                                f"{problem} (n={size}): Running with {library} and {solver} ({i + 1}/{num_repeats})..."
                            )

                            run_benchmark(
                                benchmark,
                                past_results,
                                timeout=timeout,
                                input_dir=input_dir,
                                results_dir=get_results_dir(
                                    base_results_dir, library, solver
                                ),
                            )
                    except BenchmarkError as e:
                        print(f"{problem}: {e}")

                check_results_output_match(
                    problem,
                    base_results_dir,
                    past_results.read(date=TIMESTAMP, size=size, problem=problem),
                )

                check_results_csv_align(past_results, problem=problem, size=size)


def check_results_csv_align(past_results, problem, size):
    df = past_results.read(problem=problem, size=size).filter(pl.col("error").is_null())

    # Check objective values
    df = df.with_columns(pl.col("objective_value").round_sig_figs(6))

    num_objectives = df.group_by("objective_value").agg(
        pl.col("solver", "library").first()
    )
    if num_objectives.height <= 1:
        print(f"{problem}: Objective values match across all runs.")
    else:
        raise ValueError(
            f"{problem}: Objective values do not match for size {size}, see .csv results:\n{num_objectives}"
        )

    # Check number of variables
    df = df.with_columns(
        pl.when(library="pyoframe")
        .then(pl.col("num_variables") - 1)
        .otherwise("num_variables")
    )

    num_variables = df.group_by("num_variables").agg(
        pl.col("solver", "library").unique()
    )
    if num_variables.height <= 1:
        print(f"{problem}: Number of variables match across all runs.")
    else:
        raise ValueError(
            f"{problem}: Number of variables do not match for size {size}, see .csv results:\n{num_variables}"
        )


def should_run_benchmark(benchmark: Benchmark, past_results, timeout, num_repeats):
    past_results_df = past_results.read(
        problem=benchmark.problem, library=benchmark.library, solver=benchmark.solver
    )

    # Previously errored on this run, don't try again.
    if past_results_df.filter(pl.col("error").is_not_null(), date=TIMESTAMP).height > 0:
        return False

    if benchmark.size is not None:
        past_results_df = past_results_df.filter(size=benchmark.size)

    # Check if already completed
    if past_results_df.filter(pl.col("error").is_null()).height >= num_repeats:
        return False

    # Previously timed out at this size, don't try again.
    prior_timeouts = past_results_df.filter(error="TIMEOUT")
    if (
        timeout is not None
        and prior_timeouts.height > 0
        and prior_timeouts["total_time_s"].max() >= timeout
    ):
        return False

    return True


def prepare_benchmark_problem(problem: str, problem_config):
    if "inputs" not in problem_config:
        return

    print(f"{problem}: Generating required input files...")

    cmd = ["snakemake", "--cores", "all"]
    if problem_config["inputs"] != "*":
        input_files = [
            f"./model_data/{input_file}" for input_file in problem_config["inputs"]
        ]
        cmd.extend(input_files)

    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, cwd=CWD / "src" / problem, check=True
    )


def precompile_julia_benchmarks(
    problem: str, *, bench_file: Path, image_path: Path, trace_compile_path: Path
):
    if image_path.exists():
        return

    print(f"{problem}: Creating system image for Julia benchmarks...")

    if not trace_compile_path.exists():
        raise FileNotFoundError(
            f"Precompile statements file not found at {trace_compile_path}. Try running python test.py to generate it."
        )

    with open(CWD / "Project.toml", "rb") as f:
        project_toml = tomllib.load(f)
    dependencies = list(project_toml.get("deps", {}).keys())
    dependencies_str = ", ".join(f'"{dep}"' for dep in dependencies)

    cmd = [
        "julia",
        f"--project={CWD}",
        "-e",
        f'''using PackageCompiler; create_sysimage(
            [{dependencies_str}],
            sysimage_path="{image_path}",
            precompile_statements_file="{trace_compile_path}",
        )''',
    ]

    subprocess.run(cmd, check=True)


def run_benchmark(
    benchmark: Benchmark,
    past_results: "PastResults",
    timeout: int | None = None,
    input_dir=None,
    results_dir: Path | None = None,
):
    def save_result(
        total_time: float | None = None,
        monitor_result: MonitorResult | None = None,
        error=None,
    ):
        if monitor_result is None:
            monitor_result = MonitorResult()

        past_results.append(
            {
                "date": TIMESTAMP,
                "solver": benchmark.solver,
                "problem": benchmark.problem,
                "size": benchmark.size,
                "library": benchmark.library,
                "num_variables": monitor_result.num_variables,
                "total_time_s": safe_round(total_time, 3),
                "solve_time_s": safe_round(monitor_result.solve_time, 3),
                "max_memory_uss_mb": safe_round(monitor_result.max_memory_uss_mb, 3),
                "objective_value": monitor_result.objective_value,
                "error": error,
            }
        )

    using_julia = benchmark.library == "jump"

    if not using_julia:
        args = dict(solver=f"'{benchmark.solver}'", emit_benchmarking_logs="True")
        if benchmark.size is not None:
            args["size"] = str(benchmark.size)
        if benchmark.construct_only:
            args["block_solver"] = "True"
        if input_dir is not None:
            args["input_dir"] = f"'{input_dir}'"
            args["results_dir"] = f"'{results_dir}'"

        args = ", ".join(f"{k}={v}" for k, v in args.items())

        cmd = [
            "python",
            "-c",
            f"from {benchmark.problem}.bm_{benchmark.library} import Bench; Bench({args}).run()",
        ]
    else:
        problem_dir = CWD / "src" / benchmark.problem
        image_path = problem_dir / "julia_sysimage.so"
        benchmark_path = problem_dir / "bm_jump.jl"
        trace_compile_path = problem_dir / "julia_precompile_statements.jl"

        cmd = ["julia", f"--project={CWD}"]

        if benchmark.julia_trace_compile:
            cmd += ["--trace-compile", str(trace_compile_path)]
        else:
            precompile_julia_benchmarks(
                benchmark.problem,
                bench_file=benchmark_path,
                image_path=image_path,
                trace_compile_path=trace_compile_path,
            )
            cmd += ["--sysimage", str(image_path)]

        cmd += [
            str(benchmark_path),
            benchmark.solver,
            str(benchmark.size),
            str(results_dir),
            "true" if benchmark.construct_only else "false",
        ]

    max_memory_queue = queue.Queue()

    mem_log_dir = past_results.base_dir / benchmark.problem / "mem_log"
    mem_log_dir.mkdir(parents=True, exist_ok=True)

    # See paper for explanation
    env = os.environ.copy()
    env["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:1000"

    start_time = time.time()

    with subprocess.Popen(
        cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, text=True, bufsize=1, env=env
    ) as benchmark_proc:
        memory_thread = threading.Thread(
            target=monitor_benchmark,
            args=(
                benchmark_proc,
                max_memory_queue,
                mem_log_dir
                / f"{TIMESTAMP}_{benchmark.library}_{benchmark.solver}_{benchmark.size}.parquet",
            ),
        )
        memory_thread.start()

        try:
            return_code = benchmark_proc.wait(timeout=timeout)
            total_time = time.time() - start_time
        except subprocess.TimeoutExpired:
            kill_process(benchmark_proc, using_julia)
            save_result(total_time=timeout, error="TIMEOUT")
            raise BenchmarkError("Benchmark timed out")
        except KeyboardInterrupt as e:
            kill_process(benchmark_proc, using_julia)
            raise e

        if return_code != 0:
            save_result(total_time=total_time, error="ERROR")
            raise BenchmarkError("Benchmark failed")

        result = max_memory_queue.get(timeout=10)
        memory_thread.join(timeout=10)

    if benchmark.construct_only:
        result.solve_time = 0
    else:
        if result.objective_value is None:
            save_result(total_time=total_time, error="ERROR")
            raise BenchmarkError("No objective value found in benchmark output")

    save_result(
        total_time=total_time,
        monitor_result=result,
    )


@dataclass
class MonitorResult:
    num_variables: int | None = None
    solve_time: float | None = None
    max_memory_uss_mb: float | None = None
    objective_value: float | None = None


def monitor_benchmark(proc, result_queue, output_file):
    start_time = time.time()
    pid = proc.pid
    ps_proc = psutil.Process(pid)

    memory_data = []
    process_names = {pid: "main"}
    stdout = proc.stdout

    result = MonitorResult()

    keep_checking = True

    os.set_blocking(stdout.fileno(), False)  # Requires Python 3.12 for windows

    while keep_checking:
        elapsed_time = time.time() - start_time

        marker = None

        # This setup allows us to get memory one last time after process ends
        if not ps_proc.is_running():
            keep_checking = False

        try:
            for line in iter(stdout.readline, ""):
                # print(f"OUT: '{line.strip()}'")
                if line.startswith("PF_BENCHMARK:"):
                    marker = line.removeprefix("PF_BENCHMARK:").strip()
                elif line.startswith("Optimize a model with "):
                    marker = "3_GUROBI_START"
                    result.num_variables = int(
                        re.search(r"(\d+) columns", line).group(1)
                    )
                elif line.startswith("Solved in "):
                    assert result.solve_time is None, "Multiple solve times found"
                    result.solve_time = float(
                        re.search(r"([\d.]+) seconds", line).group(1)
                    )
                    marker = "4_GUROBI_END"
                elif line.startswith("Optimal objective "):
                    assert result.objective_value is None, (
                        "Multiple objective values found"
                    )
                    result.objective_value = float(line.strip().rpartition(" ")[2])
                elif line.startswith("Best objective "):
                    assert result.objective_value is None, (
                        "Multiple objective values found"
                    )
                    result.objective_value = float(line.split(" ")[2].rstrip(","))
        except ValueError:
            pass

        # We use USS (Unique Set Size) to measure memory usage because it works
        # across OSes and represents the memory freed if the process were to end
        # in my opinion is a good metric.
        # https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_full_info
        try:
            memory_info = ps_proc.memory_full_info()
            num_threads = len(ps_proc.threads())
        except psutil.NoSuchProcess:
            assert ps_proc.is_running() is not None, "Process disappeared unexpectedly"
            break

        memory_data.append(
            (
                elapsed_time,
                pid,
                memory_info.uss,
                memory_info.rss,
                memory_info.vms,
                num_threads,
                marker,
            )
        )

        for child in ps_proc.children(recursive=True):
            try:
                if child.pid not in process_names:
                    process_names[child.pid] = child.name()
                memory_info = child.memory_full_info()
                num_threads = len(child.threads())
            except psutil.NoSuchProcess:
                continue
            memory_data.append(
                (
                    elapsed_time,
                    child.pid,
                    memory_info.uss,
                    memory_info.rss,
                    memory_info.vms,
                    num_threads,
                    None,
                )
            )

        delay = POLL_MAX_S - (POLL_MAX_S - POLL_MIN_S) * math.exp(
            -elapsed_time / POLL_TRANSITION_S
        )
        time.sleep(delay)

    df = pl.DataFrame(
        memory_data,
        schema={
            "time_s": pl.Float64,
            "pid": pl.Int32,
            "uss_MiB": pl.Float64,
            "rss_MiB": pl.Float64,
            "vms_MiB": pl.Float64,
            "num_threads": pl.UInt16,
            "marker": pl.Utf8,
        },
        orient="row",
    )

    if df.height != 0:
        df = df.with_columns(pl.col("uss_MiB", "rss_MiB", "vms_MiB") / (1024 * 1024))

        result.max_memory_uss_mb = (
            df.group_by("time_s")
            .agg(pl.col("uss_MiB").sum())
            .get_column("uss_MiB")
            .max()
        )  # type: ignore

        df = df.with_columns(
            pl.col("pid").replace_strict(process_names, return_dtype=pl.Utf8)
        ).rename({"pid": "process_name"})

        df.write_parquet(output_file)

    result_queue.put(result)


def check_results_output_match(
    problem: str, base_results_dir: Path | str, results: pl.DataFrame
):
    reference = None

    libs_compared = set()

    for (library, solver), group in results.group_by(["library", "solver"]):
        results_dir = get_results_dir(base_results_dir, library, solver)
        files = list(f.name for f in results_dir.glob("*"))

        if reference is None:
            reference = (library, solver, results_dir, files)
            continue

        ref_lib, ref_solver, ref_dir, files_in_ref = reference

        if set(files) != set(files_in_ref):
            missing_in_ref = set(files) - set(files_in_ref)
            missing_in_dir = set(files_in_ref) - set(files)
            assert len(missing_in_ref) > 0 or len(missing_in_dir) > 0
            if len(missing_in_dir) > 0:
                raise BenchmarkError(
                    f"{problem}: Benchmark ({library}, {solver}) is missing files: {', '.join(missing_in_dir)} compared to {(ref_lib, ref_solver)}."
                )
            if len(missing_in_ref) > 0:
                raise BenchmarkError(
                    f"{problem}: Benchmark ({ref_lib}, {ref_solver}) has extra files: {', '.join(missing_in_ref)} compared to {(library, solver)}."
                )

        if len(files_in_ref) == 0:
            continue

        for filename in files_in_ref:
            ref = pl.read_parquet(ref_dir / filename)
            diff = pl.read_parquet(results_dir / filename)
            try:
                assert_frame_equal(
                    ref,
                    diff,
                    check_dtypes=False,
                    check_row_order=False,
                    check_column_order=False,
                )
            except AssertionError as e:
                raise BenchmarkError(
                    f"Benchmarks produced different results between {(ref_lib, ref_solver)} and {(library, solver)}."
                ) from e

        libs_compared.add(ref_lib)
        libs_compared.add(library)

    if len(libs_compared) > 1:
        print(f"{problem}: Outputs match across {', '.join(libs_compared)}")


def kill_process(proc, using_julia, timeout=2):
    if using_julia:
        proc.kill()
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def read_config(name="config.yaml") -> dict:
    with open(CWD / name) as f:
        return yaml.safe_load(f)


def get_benchmark_code(benchmark: Benchmark) -> Path:
    ext = "jl" if benchmark.library == "jump" else "py"
    return CWD / "src" / benchmark.problem / f"bm_{benchmark.library}.{ext}"


def get_base_results_dir(config, base_dir: Path, problem: str, size: int | None):
    if config.get("save_outputs", False):
        p: Path = (
            base_dir
            / problem
            / "outputs"
            / (str(size) if size is not None else "default")
        )
        p.mkdir(parents=True, exist_ok=True)
        return nullcontext(p)
    else:
        return TemporaryDirectory()


def get_results_dir(base_results_dir: Path | str, library: str, solver: str):
    results_dir = Path(base_results_dir) / solver / library
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


class PastResults:
    BENCHMARK_RESULTS_SCHEMA: dict = {
        "date": pl.Utf8,
        "problem": pl.Utf8,
        "library": pl.Utf8,
        "solver": pl.Utf8,
        "size": pl.Int64,
        "num_variables": pl.Int64,
        "total_time_s": pl.Float64,
        "solve_time_s": pl.Float64,
        "max_memory_uss_mb": pl.Float64,
        "objective_value": pl.Float64,
        "error": pl.Utf8,
    }

    FILE_NAME = "benchmark_results.csv"

    def __init__(self, base_dir: Path, ignore_past_results: bool = False) -> None:
        self.base_dir = base_dir
        self._path = base_dir / self.FILE_NAME

        if self._path.exists():
            self._data = pl.read_csv(self._path).cast(self.BENCHMARK_RESULTS_SCHEMA)
        else:
            self._path.parent.mkdir(exist_ok=True)
            self._data = pl.DataFrame(schema=self.BENCHMARK_RESULTS_SCHEMA)
            self._data.write_csv(self._path)

        if ignore_past_results:
            self._data = self._data.filter(date=TIMESTAMP)

    def read(
        self, *, size=None, date=None, library=None, solver=None, problem=None
    ) -> pl.DataFrame:
        df = self._data
        if size is not None:
            df = df.filter(size=size)
        if date is not None:
            df = df.filter(date=date)
        if library is not None:
            df = df.filter(library=library)
        if solver is not None:
            df = df.filter(solver=solver)
        if problem is not None:
            df = df.filter(problem=problem)
        return df

    def append(self, row: dict):
        self._data = self._data.vstack(
            pl.DataFrame(row, schema=self.BENCHMARK_RESULTS_SCHEMA)
        )

        self._data.write_csv(self._path)


def safe_round(value, ndigits):
    if value is None:
        return None
    return round(value, ndigits)


class BenchmarkError(Exception):
    pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ignore-cache", action="store_true", help="Reset benchmark results file."
    )
    argparser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML file."
    )
    args = argparser.parse_args()
    config = read_config(name=args.config)
    run_all_benchmarks(config, ignore_past_results=args.ignore_cache)
