from __future__ import annotations

from benchmarks.util import mock_snakemake

if __name__ == "__main__":
    # Example usage
    snakemake = mock_snakemake("process_load_data")
    print(snakemake.input)
    print(snakemake.output)
