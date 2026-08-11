# Coding challenge: Build an electrical dispatch model

In this coding challenge, you will build a simplified electrical dispatch model. Dispatch models are used by electrical grid operators to determine which power plants should run when.

This coding challenge was developed for the PowerUp 2026 conference in Boulder, Colorado and uses data from the California Test System[^1].

[^1]: Taylor, S. et al. California Test System (CATS): A Geographically Accurate Test System Based on the California Grid. Policy and Regulation IEEE Transactions on Energy Markets 2, 107–118 (2024).

!!! tip "Use agentic AI judiciously"

    Agentic AI tools like Claude Code can certainly "solve" this coding challenge for you. But, you might learn more (and have more fun!) if you avoid such tools for this challenge. The concepts taught in this challenge continue to be useful to serious modelers that wish to better understand how modeling frameworks work, how to effectively guide AI agents, and how to improve the performance of their code. Remember, the point is to learn, not to solve my made-up and meaningless coding challenge!
    
## A. Set up project

1. Run the following command to install Pyoframe, HiGHS (a free solver), and Altair (a plotting library).

    ```bash
    pip install pyoframe[highs] altair
    ```

    If you prefer using another solver like Gurobi, refer to our [installation instructions](../get-started/installation.md).

2. Download and unzip the data and starter code for this challenge.

    [:material-folder-download: Download data and starter code](#){.md-button}

3. Run `main.py`. It shouldn't produce any errors and should print:

    ```bash
    No results to plot.
    ```

## B. Get familiar with the data

The `input_data` folder contains several files. For now, you only need to inspect the following two files:

1. **`generators.parquet`**: a table containing one row per power generator. For now, you will only need the following columns:

    - `gen_id`, a unique ID for each generator
    
    - `Pmax`, the maximum power the generator can output (in MW)
    
    - `cost_per_MWh_linear`, the (linearized) cost of producing one MWh of energy 
    
2. **`loads.parquet`**: a table for the electrical demand at every hour and electrical bus

!!! tip "Parquet files"

    Parquet files are a modern, machine-readable alternative to .csv files. You can inspect them using the [Data Wrangler extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.datawrangler) in VSCode, or by simply adding a line in `main.py` to print the corresponding DataFrame. 

## C. Build a copper-plate model

Copper-plate models are simplistic models that assume unlimited, lossless transmission between regions, as if all power generators and loads were located in the same region (or on the same copper plate).

Build a copper-plate model in `main.py` using the two previously mentioned data files. You can breakdown the tasks to do so as follows.

1. Create a Pyoframe [`Model` object](../develop/create-a-model.md) onto which you will build your model (1 line of code).

2. Add a [`Dispatch` variable](../develop/create-variables.md) to the model by indexing over the `gen_id` column and using the `Pmax` column to set an upper bound.

3. [Define the objective](../develop/define-objective.md) to minimize the cost (i.e., the sum of the product of `Dispatch` with `cost_per_MWh_linear`)

4. [Run `.optimize()`](../develop/run-and-configure-the-solver.md) and [retrieve the solution for `Dispatch`](../develop/read-results.md). Modify the `main` function in `main.py` to return this solution so that it can be plotted.

