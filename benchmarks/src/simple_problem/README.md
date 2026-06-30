# Dummy benchmark

This benchmark requires reading a parquet file containing $N$ rows with a row ID and a cost parameter $c_{id}$.

For each row, a variable $0 \leq X_{id} \leq 1$ is created. The objective is to minimize the total cost $\sum c_{id}X_{id}$. To make the problem not completely trivial, we add the constraint: $\sum X_{id} \leq 0.5 N$.

Results for each variable must be written back to a parquet file.

## Running the benchmark

To generate the input parquet files, run, from this directory, `snakemake --cores "all" model_data/input_N.parquet` where `N` is replaced with the number of desired rows. Next, run the benchmark file.