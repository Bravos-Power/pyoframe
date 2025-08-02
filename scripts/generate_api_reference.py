"""Script to auto-generate the reference API documentation pages."""

from pathlib import Path

import mkdocs_gen_files

import pyoframe as pf

root = Path(__file__).parent.parent
src = root / "src" / "pyoframe"


for object_name in pf.__all__:
    full_doc_path = Path("reference", f"pyoframe.{object_name}.md")

    if object_name == "Config":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# pyoframe.{object_name} \n\n::: pyoframe.{object_name}")

with mkdocs_gen_files.open(Path("reference", "index.md"), "a") as index_file:
    for entry in pf.__all__:
        index_file.write(f"- [{entry}](pyoframe.{entry}.md)" + "\n")


with mkdocs_gen_files.open(Path("reference", ".nav.yml"), "a") as nav_file:
    nav_file.write("  - index.md\n")
    for entry in pf.__all__:
        nav_file.write(f"  - pyoframe.{entry}.md\n")
    nav_file.write("  - polars.DataFrame.to_expr.md\n")
    nav_file.write("  - pandas.DataFrame.to_expr.md\n")
