"""Script to run the benchmarks specified in config.yaml.

Saves results to results/benchmark_results.csv.
"""

import argparse
import itertools
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import polars as pl
import psutil
import yaml

MAX_POLL_RATE = 0.1  # seconds
MIN_POLL_RATE = 1  # seconds
MIN_TO_MAX_TRANSITION = 30


DATE = time.strftime("%Y/%m/%d__%H:%M:%S")

CWD = Path(__file__).parent
BENCHMARK_RESULTS = CWD / "results/benchmark_results.csv"


def run_all_benchmarks():
    with open(CWD / "config.yaml") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(BENCHMARK_RESULTS):
        if not os.path.exists(BENCHMARK_RESULTS.parent):
            os.makedirs(BENCHMARK_RESULTS.parent)
        with open(BENCHMARK_RESULTS, "w") as f:
            f.write("date,problem,library,solver,size,time_s,memory_uss_mb,error\n")

    df = pl.read_csv(BENCHMARK_RESULTS).cast({"size": pl.Int64, "time_s": pl.Float64})

    problems = config["problems"]

    for problem in problems:
        problem_dir = CWD / "src" / problem
        requires_inputs = config["problems"][problem].get("requires_inputs", False)
        if requires_inputs:
            # Run snakemake in the directory using subprocess
            subprocess.run(
                ["snakemake", "--cores", "all"],
                stdout=subprocess.DEVNULL,
                cwd=problem_dir,
            )

        for solver, library in itertools.product(
            config["solvers"], config["libraries"]
        ):
            ext = "jl" if library == "jump" else "py"
            if not os.path.exists(problem_dir / f"bm_{library}.{ext}"):
                print(f"{problem}: Skipping {library} as no benchmark found.")
                continue
            ran_one = False
            for size in sorted(problems[problem]["size"]):
                prior_results = df.filter(
                    (pl.col("problem") == problem)
                    & (pl.col("library") == library)
                    & (pl.col("solver") == solver)
                    & (pl.col("size") == size)
                )
                if prior_results.height > 0:
                    if prior_results.filter(pl.col("error").is_null()).height > 0:
                        continue

                    prior_timeouts = prior_results.filter(pl.col("error") == "TIMEOUT")
                    if (
                        prior_timeouts.height > 0
                        and prior_timeouts["time_s"].max() >= config["timeout"]
                    ):
                        break

                error = False
                for _ in range(config["repeat"]):
                    ran_one = True
                    error = run_benchmark(
                        problem,
                        library,
                        solver,
                        size,
                        timeout=config["timeout"],
                        input_dir=problem_dir / "model_data"
                        if requires_inputs
                        else None,
                    )
                    if error:
                        break
                if error:
                    break
            if not ran_one:
                print(f"{problem}: All sizes already benchmarked for {library}.")


def run_benchmark(
    problem, library, solver, size: int, timeout: int | None = None, input_dir=None
):
    def append_result(time, memory, error=""):
        time = round(time, 3)
        with open(BENCHMARK_RESULTS, "a") as f:
            f.write(
                f"{DATE},{problem},{library},{solver},{size},{time},{memory},{error}\n"
            )

    if input_dir is not None:
        input_dir = f"'{input_dir}'"

    using_julia = library == "jump"

    if not using_julia:
        cmd = [
            "python",
            "-c",
            f"from {problem}.bm_{library} import Bench; Bench(solver='{solver}', size={size}, input_dir={input_dir}).run()",
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

    print(f"{problem} (n={size}): Running with {library} and {solver}...")

    with subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE) as proc:
        memory_thread = threading.Thread(
            target=monitor_memory, args=(proc.pid, max_memory_queue)
        )
        memory_thread.daemon = True
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

    if return_code != 0:
        append_result(t2 - t1, "", error="ERROR")
        return True

    if max_memory_used is None:
        append_result(t2 - t1, "")
    else:
        append_result(t2 - t1, round(max_memory_used, 3))

    return False


def monitor_memory(pid, result_queue):
    ps_proc = psutil.Process(pid)

    max_memory_used = 0
    scans = 0

    keep_checking = True

    while keep_checking:
        # This setup allows us to get memory one last time after process ends
        if not ps_proc.is_running():
            keep_checking = False

        # We use USS (Unique Set Size) to measure memory usage because it works
        # across OSes and represents the memory freed if the process were to end
        # in my opinion is a good metric.
        # https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_full_info
        try:
            memory_used = ps_proc.memory_full_info().uss
        except psutil.NoSuchProcess:
            assert ps_proc.is_running() is not None, "Process disappeared unexpectedly"
            break

        for child in ps_proc.children(recursive=True):
            try:
                memory_used += child.memory_full_info().uss
            except psutil.NoSuchProcess:
                continue

        max_memory_used = max(max_memory_used, memory_used)
        scans += 1

        if scans * MAX_POLL_RATE >= MIN_TO_MAX_TRANSITION:
            time.sleep(MIN_POLL_RATE)
        else:
            time.sleep(MAX_POLL_RATE)

    result_queue.put(max_memory_used / (1024 * 1024) if scans > 0 else None)


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
        "--delete-cache", action="store_true", help="Reset benchmark results file."
    )
    args = argparser.parse_args()

    if args.delete_cache and os.path.exists(BENCHMARK_RESULTS):
        os.remove(BENCHMARK_RESULTS)

    run_all_benchmarks()
