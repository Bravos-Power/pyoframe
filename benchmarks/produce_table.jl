# Copyright (c) 2022: Miles Lubin and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import DelimitedFiles
df = DelimitedFiles.readdlm(joinpath(@__DIR__, "benchmarks.csv"))
models = Dict{String,Dict{String,Int}}()
for i in 1:size(df, 1)
    type, model, variables, time = df[i, :]
    inner = get!(models, model, Dict{String,Int}())
    inner[type] = time
    if type == "direct"
        inner["variables"] = variables
    end
end

for key in [
    "fac-25",
    "fac-50",
    "fac-75",
    "fac-100",
    # "lqcp-500",
    # "lqcp-1000",
    # "lqcp-1500",
    # "lqcp-2000",
]
    inner = models[key]
    print(rpad(key, 15))
    for type in ["variables", "direct", "default", "pyomo", "gurobi"]
        print(" & ", lpad(inner[type], 2))
    end
    println(" \\\\")
end
