"""Copy benchmark code into a staging directory and measure its lines of code."""

from __future__ import annotations

import ast
import io
import shutil
import subprocess
import tokenize
from math import log
from pathlib import Path

import great_tables as gt
import matplotlib as mpl
import polars as pl
import yaml


def main() -> None:
    cwd = Path(__file__).parent
    source_root = cwd.parent
    working_root = cwd / "working"

    with open(cwd / "config.yaml") as file:
        config = yaml.safe_load(file)

    benchmark_files = prepare_working_directory(
        source_root, working_root, config["benchmarks"]
    )

    records: list[dict[str, object]] = []
    for benchmark_name, files in benchmark_files.items():
        benchmark_label = config["benchmarks"][benchmark_name]["label"]
        for file_path in files:
            records.append(
                {
                    "library": file_path.stem.removeprefix("bm_"),
                    "benchmark": benchmark_label,
                    "total": measure_lines_of_code(file_path),
                }
            )

    # unpivoted dataframe (library, benchmark, total)
    df_unpivoted = pl.DataFrame(records)

    # Create PNG table using the unpivoted DataFrame; create_png_table will
    # prepare cell contents and pivot as its last step.
    create_output_table(df_unpivoted, cwd)


def prepare_working_directory(
    source_root: Path,
    working_root: Path,
    benchmarks: dict[str, dict[str, str]],
) -> dict[str, list[Path]]:
    if working_root.exists():
        shutil.rmtree(working_root)
    working_root.mkdir()

    benchmark_files: dict[str, list[Path]] = {}
    for benchmark_name, benchmark_config in benchmarks.items():
        target_dir = working_root / benchmark_name
        target_dir.mkdir(parents=True, exist_ok=True)

        copied_files: list[Path] = []
        source_path = source_root / benchmark_config["path"]
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark source directory: {source_path}"
            )

        for source_file in source_path.glob("bm_*.*"):
            target_file = target_dir / source_file.name
            shutil.copy2(source_file, target_file)
            copied_files.append(target_file)

        if not copied_files:
            raise FileNotFoundError(f"No bm_*.* files found in {source_path}")

        benchmark_files[benchmark_name] = copied_files

    for python_file in working_root.rglob("bm_*.py"):
        strip_main_guard(python_file)

    return benchmark_files


def strip_main_guard(file_path: Path) -> None:
    lines = file_path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == 'if __name__ == "__main__":':
            file_path.write_text("".join(lines[:index]))
            return

    raise ValueError(f"Main guard not found in {file_path}")


def measure_lines_of_code(file_path: Path) -> int:
    cleaned_file = file_path.with_suffix(f".stripped{file_path.suffix}")

    if file_path.suffix == ".py":
        # Use AST to locate docstrings and tokenize to remove comments and those
        # docstring string tokens. Write the cleaned source to `cleaned_file`.

        src = file_path.read_text()

        # find docstring positions (start tuples) for module, functions, and classes
        docstring_starts: set[tuple[int, int]] = set()
        try:
            tree = ast.parse(src)
        except Exception:
            # If parsing fails, fall back to writing original file
            cleaned_file.write_text(src)
        else:
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    body = getattr(node, "body", None)
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(getattr(body[0], "value", None), ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        expr = body[0]
                        # record the starting (lineno, col_offset) of the docstring expression
                        start = (expr.lineno, expr.col_offset)
                        docstring_starts.add(start)

            # tokenize and filter out comments and docstring STRING tokens
            toks = []
            reader = io.StringIO(src).readline
            for tok in tokenize.generate_tokens(reader):
                ttype = tok.type
                tstring = tok.string
                tstart = tok.start
                # drop comments
                if ttype == tokenize.COMMENT:
                    continue
                # drop string tokens that start at a docstring location
                if ttype == tokenize.STRING and tstart in docstring_starts:
                    continue
                toks.append((ttype, tstring))

            cleaned = tokenize.untokenize(toks)
            cleaned_file.write_text(cleaned)

            # Run ruff format on file
            cmd = ["black", "--skip-magic-trailing-comma", "--quiet", str(cleaned_file)]
            print(f"{' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ruff format failed with return code {result.returncode}: {result.stderr}"
                )

            # Remove blank lines from file
            cleaned_file.write_text(
                "\n".join(
                    line
                    for line in cleaned_file.read_text().splitlines()
                    if line.strip() != ""
                )
            )

    else:
        cmd = ["cloc", "--strip-comments=stripped", "--original-dir", str(file_path)]

        print(f"{' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"cloc failed with return code {result.returncode}: {result.stderr}"
            )

        file_path.with_suffix(f"{file_path.suffix}.stripped").rename(cleaned_file)

    with open(cleaned_file) as f:
        num_lines = len(f.readlines())

    return num_lines


def create_output_table(results: pl.DataFrame, cwd: Path) -> None:
    pf_row = results.filter(library="pyoframe").drop("library")
    if pf_row.is_empty():
        raise RuntimeError(
            "pyoframe baseline not found in results; cannot compute factors"
        )

    results = results.join(
        pf_row, on="benchmark", how="left", suffix="_pf", validate="m:1"
    )

    results = results.with_columns(factor=pl.col("total") / pl.col("total_pf"))

    vmin, vmax = 1 / 3, 3
    color_norm = mpl.colors.Normalize(vmin=log(vmin), vmax=log(vmax), clip=True)
    anchors = [(vmin, "#A5D6A7"), (1, "white"), (vmax, "#EF9A9A")]
    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "green_red",
        [
            ((log(v) - log(vmin)) / (log(vmax) - log(vmin)), color)
            for v, color in anchors
        ],
    )

    results = results.with_columns(
        color=pl.col("factor").map_elements(
            lambda v: mpl.colors.to_hex(color_map(color_norm(log(v)))),
            pl.String,
        )
    )

    results = results.with_columns(
        label=pl.concat_str(
            pl.lit("\cellcolor[HTML]{"),
            pl.col("color").str.slice(1),  # drop '#' from hex color
            pl.lit("} "),
            pl.col("total").map_elements(lambda v: f"{int(v)}", pl.String),
            pl.lit(" ("),
            pl.col("factor").map_elements(lambda v: f"{v:.1f}x", pl.String),
            pl.lit(")"),
        )
    )

    library_order = (
        ["pyoframe"]
        + results.filter(benchmark="Electrical Grid Problem")
        .sort("factor")["library"]
        .to_list()
        + ["cvxpy", "pulp"]
    )

    results = results.pivot(
        index="library",
        on="benchmark",
        values="label",
    ).fill_null("—")

    results = results.sort(
        pl.col("library").map_elements(lambda lib: library_order.index(lib), pl.Int64)
    )

    results = results.rename({"library": "Modeling Interface"})

    print(results)

    gt_table = gt.GT(results)

    with open(cwd / "results_readability.tex", "w") as f:
        latex_code = gt_table.as_latex()
        latex_code = (
            latex_code.replace(r"\\cellcolor", r"\cellcolor")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
        )
        latex_subset = []
        began = False
        for line in latex_code.splitlines():
            if not began and line.startswith(r"\begin{tabular"):
                began = True

            if began:
                latex_subset.append(line)

            if line.startswith(r"\end{tabular"):
                break
        f.write("\n".join(latex_subset))

    out_png = cwd / "results_readability.png"
    gt_table.save(out_png, web_driver="edge", scale=2)
    print(f"Wrote table PNG: {out_png}")


if __name__ == "__main__":
    main()
