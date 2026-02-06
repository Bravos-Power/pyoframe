include("../benchmark_utils/jump.jl")

using JuMP
using ..Benchmark
using Parquet
using DataFrames

function main(model::JuMP.Model, size::Int)
    # Load input data
    data = DataFrame(read_parquet("input_$(size).parquet"))
    
    @variable(model, 0 <= x[data.id] <= 1)
    data.x = Array(x)
    @objective(model, Min, data.cost' * data.x)
    @constraint(model, sum(data.x) >= size / 2)

    Benchmark.optimize!(model)

    table = Containers.rowtable(value, x; header = [:id, :solution])
    solution = DataFrames.DataFrame(table)

    # Write via Polars again
    write_parquet("output_$(size).parquet", solution)
end

Benchmark.run(@__DIR__, main)

# To test run from the benchmarks directory:
# julia --project=. src/simple_problem/bm_jump.jl gurobi 1000 src/simple_problem/model_results
