"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

import pyoframe as pf

root = Path(__file__).parent.parent
src = root / "src" / "pyoframe"

index_content = []

for object_name in pf.__all__:
    full_doc_path = Path("reference", f"pyoframe.{object_name}.md")

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# pyoframe.{object_name} \n\n::: pyoframe.{object_name}")
    index_content.append(f"- [{object_name}](pyoframe.{object_name}.md)")

with mkdocs_gen_files.open(Path("reference", "index.md"), "a") as index_file:
    index_file.write("\n".join(index_content))
