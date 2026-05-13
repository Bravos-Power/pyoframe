# Inspired from:
# Copyright (c) 2022: Miles Lubin and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.
# See https://github.com/jump-dev/JuMPPaperBenchmarks


include("../benchmark_utils/jump.jl")

using JuMP
using ..Benchmark
using Parquet
using DataFrames

function main(model::JuMP.Model, size::Int)
    F = G = size

    @variables(model, begin
        0 <= y[1:F, 1:2] <= 1
        s[0:G, 0:G, 1:F] >= 0
        z[0:G, 0:G, 1:F], Bin
        r[0:G, 0:G, 1:F, 1:2]
        d
    end)
    @objective(model, Min, d)
    @constraint(model, [i in 0:G, j in 0:G], sum(z[i,j,f] for f in 1:F) == 1)
    # d is the maximum of distances between customers
    # and their facilities. The original constraint is # d >= ||x - y|| - M(1-z)
    # where M = 1 for our data. Because Gurobi/CPLEX can't take SOCPs directly,
    # we need to rewite as a set of constraints and auxiliary variables:
    # s = d + M(1 - z) >= 0
    # r = x - y
    # r'r <= s^2
    M = 2 * sqrt(2)
    for i in 0:G, j in 0:G, f in 1:F
        @constraints(model, begin
            s[i, j, f] == d + M * (1 - z[i, j, f])
            r[i, j, f, 1] == i / G - y[f, 1]
            r[i, j, f, 2] == j / G - y[f, 2]
            sum(r[i, j, f, k]^2 for k in 1:2) <= s[i, j, f]^2
        end)
    end


    Benchmark.optimize!(model)
end

Benchmark.run(@__DIR__, main)


