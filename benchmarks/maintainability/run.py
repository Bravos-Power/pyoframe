"""Script to download Github repos and measure their lines of code using cloc."""

import json
import subprocess
from pathlib import Path

import polars as pl
import yaml


def main():
    cwd = Path(__file__).parent

    with open(cwd / "config.yaml") as file:
        config = yaml.safe_load(file)

    results = pl.DataFrame()

    for modeling_interface, mi_config in config["modeling_interfaces"].items():
        github = mi_config["github"].lower()
        results = pl.concat(
            [
                results,
                measure_lines_of_code(
                    github,
                    cwd / "downloads" / github.partition("/")[2],
                    include_paths=mi_config["include_paths"],
                    exclude_paths=mi_config.get("exclude_paths", []),
                    include_exts=config["valid_extensions"],
                    exclude_dirs=config["always_exclude_dirs"],
                    branch=mi_config.get("branch", None),
                ).with_columns(modeling_interface=pl.lit(modeling_interface)),
            ],
            how="diagonal",
        )

    results = results.with_columns(
        (
            pl.col("C++").fill_null(0)
            + pl.col("C/C++ Header").fill_null(0)
            + pl.col("C").fill_null(0)
        ).alias("C/C++")
    ).drop("C/C++ Header", "C", "C++")
    results = results.select(
        "modeling_interface",
        *[col for col in results.columns if col not in ("modeling_interface", "SUM")],
        total="SUM",
    ).sort("total")
    results = results.with_columns(
        factor=(
            pl.col("total")
            / results.filter(modeling_interface="pyoframe")["total"].item()
        ).round(1)
    )

    results = results.sort("factor")

    print(results)

    results.write_csv(cwd / "results.csv")


def measure_lines_of_code(
    github: str,
    downloads_dir: Path,
    *,
    include_paths: list[str],
    exclude_paths: list[str],
    exclude_dirs: list[str],
    include_exts: list[str],
    branch: str | None = None,
) -> pl.DataFrame:
    if not downloads_dir.exists() or not any(downloads_dir.iterdir()):
        url = "https://github.com/" + github
        subprocess.run(["git", "clone", url, downloads_dir], check=True)

    if branch is not None:
        subprocess.run(["git", "checkout", branch], check=True, cwd=downloads_dir)
    else:
        # Checkout the default branch (e.g., main or master)
        default_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "origin/HEAD"],
                cwd=downloads_dir,
                text=True,
            )
            .strip()
            .replace("origin/", "")
        )
        subprocess.run(
            ["git", "switch", default_branch],
            check=True,
            cwd=downloads_dir,
        )

    cmd = ["cloc"]
    cmd += include_paths
    if exclude_paths:
        cmd += ["--fullpath", "--not-match-d=(" + "|".join(exclude_paths) + ")"]

    cmd += [
        "--include-ext=" + ",".join(include_exts),
        "--exclude-dir=" + ",".join(exclude_dirs),
        "--json",
    ]
    print(f"{github}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=downloads_dir)

    if result.returncode != 0:
        raise RuntimeError(
            f"cloc failed with return code {result.returncode}: {result.stderr}"
        )

    try:
        cloc_output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse cloc output as JSON: {result.stdout}"
        ) from e

    df = pl.DataFrame(cloc_output)
    df = df.select(
        pl.col(c).struct.field("code").alias(c) for c in df.columns if c != "header"
    )

    return df


if __name__ == "__main__":
    main()
