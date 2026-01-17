"""Script to run the benchmarks specified in config.yaml.

Saves results to results/benchmark_results.csv.
"""

import argparse
import itertools
import math
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import psutil
import yaml

POLL_MIN_S, POLL_MAX_S, POLL_TRANSITION_S = 0.01, 1, 30

MAX_POLL_RATE = 0.1  # seconds
MIN_POLL_RATE = 1  # seconds
MIN_TO_MAX_TRANSITION = 30


DATE = time.strftime("%Y/%m/%d__%H:%M:%S")
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

CWD = Path(__file__).parent
BENCHMARK_RESULTS = CWD / "results/benchmark_results.csv"


def run_all_benchmarks(ignore_cache=False):
    with open(CWD / "config.yaml") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(BENCHMARK_RESULTS):
        if not os.path.exists(BENCHMARK_RESULTS.parent):
            os.makedirs(BENCHMARK_RESULTS.parent)
        with open(BENCHMARK_RESULTS, "w") as f:
            f.write("date,problem,library,solver,size,time_s,memory_uss_mb,error\n")

    df = pl.read_csv(BENCHMARK_RESULTS).cast({"size": pl.Int64, "time_s": pl.Float64})
    if ignore_cache:
        df = df.filter(date=DATE)

    problems = config["problems"]

    for problem, problem_config in problems.items():
        problem_dir = CWD / "src" / problem
        if "inputs" in problem_config:
            print(f"{problem}: Generating required input files...")

            cmd = ["snakemake", "--cores", "all"]
            if problem_config["inputs"] != "*":
                cmd.extend(
                    f"./model_data/{input_file}"
                    for input_file in problem_config["inputs"]
                )

            subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=problem_dir, check=True)

        num_repeats = problem_config.get("repeat", config["repeat"])

        for solver, library in itertools.product(
            config["solvers"], config["libraries"]
        ):
            ext = "jl" if library == "jump" else "py"
            if not os.path.exists(problem_dir / f"bm_{library}.{ext}"):
                print(f"{problem}: Skipping {library} as no benchmark found.")
                continue
            ran_one = False
            if "size" not in problem_config:
                problem_config["size"] = [None]
            for size in sorted(problem_config["size"]):
                prior_results = df.filter(
                    problem=problem, library=library, solver=solver
                )
                if size is not None:
                    prior_results = prior_results.filter(size=size)

                # Past non-error result, no need to repeat
                if (
                    prior_results.filter(pl.col("error").is_null()).height
                    >= num_repeats
                ):
                    continue

                # Previously timed out at this size, don't try again.
                prior_timeouts = prior_results.filter(error="TIMEOUT")
                if (
                    prior_timeouts.height > 0
                    and prior_timeouts["time_s"].max() >= config["timeout"]
                ):
                    break

                error = False
                for i in range(num_repeats):
                    ran_one = True
                    error = run_benchmark(
                        f"{i + 1}/{num_repeats}",
                        problem,
                        library,
                        solver,
                        size,
                        timeout=config["timeout"],
                        input_dir=problem_dir / "model_data"
                        if "inputs" in problem_config
                        else None,
                    )
                    if error:
                        break
                if error:
                    break
            if not ran_one:
                print(f"{problem}: All sizes already benchmarked for {library}.")


def run_benchmark(
    run_number: str,
    problem,
    library,
    solver,
    size: int | None = None,
    timeout: int | None = None,
    input_dir=None,
):
    def append_result(time, memory, error=""):
        time = round(time, 3)
        _size = size if size is not None else ""
        with open(BENCHMARK_RESULTS, "a") as f:
            f.write(
                f"{DATE},{problem},{library},{solver},{_size},{time},{memory},{error}\n"
            )

    using_julia = library == "jump"

    with TemporaryDirectory() as temp_dir:
        if not using_julia:
            code = f"from {problem}.bm_{library} import Bench; Bench(solver='{solver}'"
            if size is not None:
                code += f", size={size}"
            if input_dir is not None:
                code += f", input_dir='{input_dir}'"
                code += f", results_dir='{temp_dir}'"
            code += ").run()"

            cmd = [
                "python",
                "-c",
                code,
            ]
        else:
            cmd = [
                "julia",
                f"--project={CWD}",
                CWD / f"src/{problem}/bm_jump.jl",
                solver,
                str(size),
            ]

        t1 = time.time()

        max_memory_queue = queue.Queue()

        print(
            f"{problem} (n={size}): Running with {library} and {solver} ({run_number})..."
        )

        mem_log_dir = CWD / "results" / problem / "mem_log"
        mem_log_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:1000"

        with subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        ) as proc:
            memory_thread = threading.Thread(
                target=monitor_memory,
                args=(
                    proc,
                    max_memory_queue,
                    mem_log_dir / f"{TIMESTAMP}_{library}_{solver}_{size}.parquet",
                ),
            )
            memory_thread.start()

            try:
                return_code = proc.wait(timeout=timeout)
                t2 = time.time()

            except subprocess.TimeoutExpired:
                kill_process(proc, using_julia)
                append_result(timeout, "", error="TIMEOUT")
                return True
            except KeyboardInterrupt as e:
                kill_process(proc, using_julia)
                raise e

            max_memory_used = max_memory_queue.get(timeout=10)
            memory_thread.join(timeout=10)

    if return_code != 0:
        append_result(t2 - t1, "", error="ERROR")
        return True

    if max_memory_used is None:
        append_result(t2 - t1, "")
    else:
        append_result(t2 - t1, round(max_memory_used, 3))

    return False


def monitor_memory(proc, result_queue, output_file):
    start_time = time.time()
    pid = proc.pid
    ps_proc = psutil.Process(pid)

    memory_data = []
    process_names = {pid: "main"}
    stdout = proc.stdout

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
                elif line.startswith("Solved in "):
                    marker = "4_GUROBI_END"
        except BlockingIOError:
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
            if child.pid not in process_names:
                process_names[child.pid] = child.name()
            try:
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

    if df.height == 0:
        result_queue.put(None)
        return

    df = df.with_columns(pl.col("uss_MiB", "rss_MiB", "vms_MiB") / (1024 * 1024))

    max_uss: float = df.group_by("time_s").agg(pl.col("uss_MiB").sum())["uss_MiB"].max()
    result_queue.put(max_uss)

    df = df.with_columns(
        pl.col("pid").replace_strict(process_names, return_dtype=pl.Utf8)
    ).rename({"pid": "process_name"})

    df.write_parquet(output_file)


def kill_process(proc, using_julia, timeout=2):
    if using_julia:
        proc.kill()
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ignore-cache", action="store_true", help="Reset benchmark results file."
    )
    args = argparser.parse_args()

    run_all_benchmarks(ignore_cache=args.ignore_cache)
