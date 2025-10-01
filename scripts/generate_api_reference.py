"""Script to auto-generate the reference API documentation pages."""

from pathlib import Path

import mkdocs_gen_files

import pyoframe as pf

root = Path(__file__).parent.parent
src = root / "src" / "pyoframe"

objects_to_gen = [obj for obj in pf.__all__ if obj not in ("sum", "sum_by")]

for object_name in objects_to_gen:
    full_doc_path = Path("reference", "public", f"{object_name}.md")

    if object_name == "Config":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# {object_name} \n\n::: pyoframe.{object_name}")

with mkdocs_gen_files.open(Path("reference", "public", ".nav.yml"), "a") as nav_file:
    for entry in objects_to_gen:
        nav_file.write(f"  - {entry}.md\n")
