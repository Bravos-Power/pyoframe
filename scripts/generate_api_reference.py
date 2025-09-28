"""Script to auto-generate the reference API documentation pages."""

from pathlib import Path

import mkdocs_gen_files

import pyoframe as pf

root = Path(__file__).parent.parent
src = root / "src" / "pyoframe"

objects_to_gen = [obj for obj in pf.__all__ if obj not in ("sum", "sum_by")]

non_pyoframe_objects = [
    "pandas.DataFrame.to_expr",
    "pandas.Series.to_expr",
    "polars.DataFrame.to_expr",
]

for object_name in objects_to_gen:
    full_doc_path = Path("reference", f"pyoframe.{object_name}.md")

    if object_name == "Config":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# pyoframe.{object_name} \n\n::: pyoframe.{object_name}")

with mkdocs_gen_files.open(Path("reference", "index.md"), "a") as index_file:
    for entry in objects_to_gen:
        index_file.write(f"- [pf.{entry}](pyoframe.{entry}.md)" + "\n")

    index_file.write(
        "\nAdditionally, importing Pyoframe patches Pandas and Polars such that the following methods are available:\n\n"
    )

    for entry in non_pyoframe_objects:
        index_file.write(f"- [{entry}]({entry}.md)\n")

with mkdocs_gen_files.open(Path("reference", ".nav.yml"), "a") as nav_file:
    nav_file.write("  - index.md\n")
    for entry in objects_to_gen:
        nav_file.write(f"  - pyoframe.{entry}.md\n")
    for entry in non_pyoframe_objects:
        nav_file.write(f"  - {entry}.md\n")
