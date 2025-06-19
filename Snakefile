from shutil import move
import gdown
from urllib.request import urlretrieve
from pathlib import Path

CATS_GITHUB_URL = "https://raw.githubusercontent.com/WISPO-POP/CATS-CaliforniaTestSystem/f260d8bd89e68997bf12d24e767475b2f2b88a77/GIS/"

ENERGY_BENCHMARKS = Path("benchmarks/energy_model")
PREPROCESS_DIR = ENERGY_BENCHMARKS / "data/preprocess"
POSTPROCESS_DIR = ENERGY_BENCHMARKS / "data/postprocess"
SCRIPTS_DIR = ENERGY_BENCHMARKS / "scripts"


rule fetch_load_data:
    """Downloads the load data from the Google Drive folder hosted by the CATS project (https://drive.google.com/drive/folders/1Zo6ZeZ1OSjHCOWZybbTd6PgO4DQFs8_K)"""
    output:
        PREPROCESS_DIR / "CATS_loads.csv",
    run:
        gdown.download(id="1Sz8st7g4Us6oijy1UYMPUvkA1XeZlIr8", output=output[0])


rule fetch_generation_data:
    """Downloads the generation data from the Google Drive folder hosted by the CATS project (https://drive.google.com/drive/folders/1Zo6ZeZ1OSjHCOWZybbTd6PgO4DQFs8_K)"""
    output:
        PREPROCESS_DIR / "CATS_generation.csv",
    run:
        gdown.download(id="1CxLlcwAEUy-JvJQdAfVydJ1p9Ecot-4d", output=output[0])


rule fetch_line_data:
    output:
        PREPROCESS_DIR / "CATS_lines.json",
    run:
        urlretrieve(CATS_GITHUB_URL + "CATS_lines.json", output[0])


rule fetch_generator_data:
    output:
        PREPROCESS_DIR / "CATS_generators.csv",
    run:
        urlretrieve(CATS_GITHUB_URL + "CATS_gens.csv", output[0])


rule process_load_data:
    """Convert the load data to narrow format and keep only the active loads."""
    input:
        PREPROCESS_DIR / "CATS_loads.csv",
    output:
        POSTPROCESS_DIR / "loads.parquet",
    notebook:
        SCRIPTS_DIR / "process_load_data.ipynb"


rule process_line_data:
    """Convert from .json to .parquet and keep only relevant columns."""
    input:
        PREPROCESS_DIR / "CATS_lines.json",
    output:
        POSTPROCESS_DIR / "lines.parquet",
    notebook:
        str(SCRIPTS_DIR / "process_lines_json.ipynb")


rule process_generator_data:
    """Group the generators by type and bus."""
    input:
        PREPROCESS_DIR / "CATS_generators.csv",
    output:
        POSTPROCESS_DIR / "generators.parquet",
    notebook:
        str(SCRIPTS_DIR / "process_generator_data.ipynb")

rule compute_capacity_factors:
    """Use the hourly generation data to create capacity factors by fuel type."""
    input:
        gen_capacity=POSTPROCESS_DIR / "generators.parquet",
        gen_dispatch=PREPROCESS_DIR / "CATS_generation.csv",
    output:
        yearly_limit=POSTPROCESS_DIR / "yearly_limits.parquet",
        vcf=POSTPROCESS_DIR / "variable_capacity_factors.parquet",
    notebook:
        str(SCRIPTS_DIR / "compute_capacity_factors.ipynb")