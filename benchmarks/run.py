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
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import psutil
import yaml
from polars.testing import assert_frame_equal

POLL_MIN_S, POLL_MAX_S, POLL_TRANSITION_S = 0.01, 1, 30

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

CWD = Path(__file__).parent

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


def run_all_benchmarks(config, ignore_past_results=False):
    base_dir = CWD / "results" / config["name"]
    base_dir.mkdir(parents=True, exist_ok=True)
    past_results_path = base_dir / "benchmark_results.csv"

    past_results = read_past_results(past_results_path)

    if ignore_past_results:
        past_results = past_results.filter(date=TIMESTAMP)

    for problem, problem_config in config["problems"].items():
        prepare_benchmark_problem(problem, problem_config)

        if config.get("save_outputs", False):
            p: Path = base_dir / problem / "outputs"
            p.mkdir(parents=True, exist_ok=True)
            context_manager = nullcontext(p)
        else:
            context_manager = TemporaryDirectory()

        with context_manager as results_dir:
            results_dir = Path(results_dir)
            all_completed_benchmarks = []
            for solver, library in itertools.product(
                config["solvers"], config["libraries"]
            ):
                result_dir = results_dir / solver / library
                result_dir.mkdir(parents=True, exist_ok=True)
                try:
                    completed_benchmarks = run_library_benchmarks(
                        solver,
                        library,
                        problem,
                        problem_config,
                        config,
                        past_results.filter(
                            problem=problem, library=library, solver=solver
                        ),
                        result_dir,
                        past_results_path,
                    )
                    all_completed_benchmarks.extend(completed_benchmarks)
                except BenchmarkError as e:
                    print(f"{problem}: {e}")

            check_results_match(problem, all_completed_benchmarks)

        df = read_past_results(past_results_path)
        df = df.filter(date=TIMESTAMP, problem=problem)
        if df["objective_value"].n_unique() <= 1:
            print(f"{problem}: Objective values match across all runs.")
        else:
            raise ValueError(
                f"{problem}: Objective values do not match, see .csv results."
            )


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


def run_library_benchmarks(
    solver,
    library,
    problem: str,
    problem_config,
    config,
    past_results,
    base_results_dir: Path,
    past_results_path: Path,
) -> list[Path]:
    problem_dir = CWD / "src" / problem

    ext = "jl" if library == "jump" else "py"
    if not (problem_dir / f"bm_{library}.{ext}").exists():
        print(f"{problem}: Skipping {library} as no benchmark found.")
        return []

    num_repeats = problem_config.get("repeat", config.get("repeat", 1))
    timeout = config.get("timeout", None)

    sizes = sorted(problem_config.get("size", [None]))

    completed_benchmarks = []
    for size in sorted(sizes):
        if size is not None:
            past_result = past_results.filter(size=size)
        else:
            past_result = past_results

        # Past non-error result, no need to repeat
        if past_result.filter(pl.col("error").is_null()).height >= num_repeats:
            continue

        # Previously timed out at this size, don't try again.
        prior_timeouts = past_result.filter(error="TIMEOUT")
        if (
            timeout is not None
            and prior_timeouts.height > 0
            and prior_timeouts["total_time_s"].max() >= timeout
        ):
            break

        results_dir = (
            base_results_dir / str(size)
            if size is not None
            else base_results_dir / "default"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        input_dir = problem_dir / "model_data" if "inputs" in problem_config else None

        for i in range(num_repeats):
            print(
                f"{problem} (n={size}): Running with {library} and {solver} ({i + 1}/{num_repeats})..."
            )

            run_benchmark(
                problem,
                library,
                solver,
                past_results_path,
                size,
                timeout=timeout,
                input_dir=input_dir,
                results_dir=results_dir,
            )
            completed_benchmarks.append(results_dir)

    if not completed_benchmarks:
        print(f"{problem}: All sizes already benchmarked for {library}.")

    return completed_benchmarks


def run_benchmark(
    problem,
    library,
    solver,
    past_results_path: Path,
    size: int | None = None,
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
        new_result = pl.DataFrame(
            {
                "date": TIMESTAMP,
                "problem": problem,
                "library": library,
                "solver": solver,
                "size": size,
                "num_variables": monitor_result.num_variables,
                "total_time_s": safe_round(total_time, 3),
                "solve_time_s": safe_round(monitor_result.solve_time, 3),
                "max_memory_uss_mb": safe_round(monitor_result.max_memory_uss_mb, 3),
                "objective_value": monitor_result.objective_value,
                "error": error,
            },
            schema=BENCHMARK_RESULTS_SCHEMA,
        )

        read_past_results(past_results_path).vstack(new_result).write_csv(
            past_results_path
        )

    using_julia = library == "jump"

    if not using_julia:
        args = dict(solver=f"'{solver}'", emit_benchmarking_logs="True")
        if size is not None:
            args["size"] = str(size)
        if input_dir is not None:
            args["input_dir"] = f"'{input_dir}'"
            args["results_dir"] = f"'{results_dir}'"

        args = ", ".join(f"{k}={v}" for k, v in args.items())

        cmd = [
            "python",
            "-c",
            f"from {problem}.bm_{library} import Bench; Bench({args}).run()",
        ]
    else:
        cmd = [
            "julia",
            f"--project={CWD}",
            CWD / f"src/{problem}/bm_jump.jl",
            solver,
            str(size),
            str(results_dir),
        ]

    max_memory_queue = queue.Queue()

    mem_log_dir = past_results_path.parent / problem / "mem_log"
    mem_log_dir.mkdir(parents=True, exist_ok=True)

    # See paper for explanation
    env = os.environ.copy()
    env["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:1000"

    start_time = time.time()

    with subprocess.Popen(
        cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, text=True, bufsize=1, env=env
    ) as benchmark:
        memory_thread = threading.Thread(
            target=monitor_benchmark,
            args=(
                benchmark,
                max_memory_queue,
                mem_log_dir / f"{TIMESTAMP}_{library}_{solver}_{size}.parquet",
            ),
        )
        memory_thread.start()

        try:
            return_code = benchmark.wait(timeout=timeout)
            total_time = time.time() - start_time
        except subprocess.TimeoutExpired:
            kill_process(benchmark, using_julia)
            save_result(total_time=timeout, error="TIMEOUT")
            raise BenchmarkError("Benchmark timed out")
        except KeyboardInterrupt as e:
            kill_process(benchmark, using_julia)
            raise e

        if return_code != 0:
            save_result(total_time=total_time, error="ERROR")
            raise BenchmarkError("Benchmark failed")

        result = max_memory_queue.get(timeout=10)
        memory_thread.join(timeout=10)

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
                    result.objective_value = float(line.strip().rpartition(" ")[2])
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


def check_results_match(problem: str, completed_benchmarks: list[Path]):
    print(len(completed_benchmarks))
    dirs_to_compare = defaultdict(list)
    for dir in completed_benchmarks:
        size = dir.parts[-1]
        library = dir.parts[-2]
        solver = dir.parts[-3]
        dirs_to_compare[size].append((library, solver, dir))

    comparisons_completed = set()
    for size, dirs in dirs_to_compare.items():
        ref_lib, ref_solver, ref_dir = dirs[0]

        files_in_ref = list(f.name for f in ref_dir.glob("*"))

        for library, solver, dir in dirs[1:]:
            files = (f.name for f in dir.glob("*"))

            if set(files) != set(files_in_ref):
                missing_in_ref = set(files) - set(files_in_ref)
                missing_in_dir = set(files_in_ref) - set(files)
                assert len(missing_in_ref) > 0 or len(missing_in_dir) > 0
                if len(missing_in_dir) > 0:
                    raise BenchmarkError(
                        f"{problem}: For size {size}, benchmark ({library}, {solver}) is missing files: {', '.join(missing_in_dir)} compared to {(ref_lib, ref_solver)}."
                    )
                if len(missing_in_ref) > 0:
                    raise BenchmarkError(
                        f"{problem}: For size {size}, benchmark ({ref_lib}, {ref_solver}) has extra files: {', '.join(missing_in_ref)} compared to {(library, solver)}."
                    )

            for filename in files_in_ref:
                ref = pl.read_parquet(ref_dir / filename)
                diff = pl.read_parquet(dir / filename)
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
                        f"Benchmarks produced different results between {(ref_lib, ref_solver)} and {(library, solver)} for size {size}."
                    ) from e

                comparisons_completed.add(ref_lib)
                comparisons_completed.add(library)

    if len(comparisons_completed) > 1:
        print(f"{problem}: Outputs match across {', '.join(comparisons_completed)}")


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


def read_past_results(path) -> pl.DataFrame:
    if path.exists():
        return pl.read_csv(path).cast(BENCHMARK_RESULTS_SCHEMA)

    path.parent.mkdir(exist_ok=True)
    df = pl.DataFrame(schema=BENCHMARK_RESULTS_SCHEMA)
    df.write_csv(path)
    return df


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
