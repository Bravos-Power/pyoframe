configfile: "benchmarks/config.yaml"

def generate_all_runs():
    import itertools

    problem_size_pairs = []
    for problem, problem_data in config["problems"].items():
        if problem_data is None or "size" not in problem_data:
            problem_size_pairs.append((problem, "0"))
        else:
            for size in problem_data["size"]:
                problem_size_pairs.append((problem, size))
    
    runs = [
        (problem, size, library, solver)
        for (problem, size), library, solver in itertools.product(
            problem_size_pairs,
            config["libraries"],
            config["solvers"]
        )
    ]
    return runs

rule all:
    input:
        "benchmarks/results.csv"

rule collect_benchmarks:
    input:
        [f"benchmarks/{problem}/results/{library}_{solver}_{size}.tsv"
         for problem, size, library, solver in generate_all_runs()]
    output:
        "benchmarks/results.csv"
    script:
        "benchmarks/collect_benchmarks.py"

rule run_benchmark:
    benchmark:
        repeat("benchmarks/{problem}/results/{library}_{solver}_{size}.tsv", config["repeat"])
    shell:
        "python benchmarks/run_benchmark.py {wildcards.problem} --library {wildcards.library} --solver {wildcards.solver} --size {wildcards.size}"
