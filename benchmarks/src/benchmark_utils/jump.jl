module Benchmark

export run

using JuMP

# Read args from command line
args_dict = Dict()
for arg in ARGS
    k, v = split(arg, "=", limit=2)
    if v == "true"
        v = true
    elseif v == "false"
        v = false
    end
    args_dict[Symbol(k)] = v
end
solver = pop!(args_dict, :solver, "gurobi")
solver_args = pop!(args_dict, :solver_args, nothing)
block_solver = pop!(args_dict, :block_solver, false)
output_dir = pop!(args_dict, :results_dir, nothing)
problem_size = parse(Int, pop!(args_dict, :problem_size))
if output_dir !== nothing
    output_dir = abspath(output_dir)
end

# Create model
if solver == "gurobi"
    import Gurobi

    # direct_model is faster and more memory efficient (according to tests on July 13, 2025).
    # We use it to make a fair comparison with the other models.
    base_model = direct_model(Gurobi.Optimizer())
elseif solver == "highs"
    import HiGHS
    base_model = Model(HiGHS.Optimizer)
elseif solver == "ipopt"
    import Ipopt
    base_model = Model(Ipopt.Optimizer)
else
    error("Unsupported solver: $(solver)")
end


function optimize!(model::JuMP.Model)
    println("PF_BENCHMARK: 2_SOLVE")
    flush(stdout)

    if block_solver
        set_time_limit_sec(model, 0.0)
        set_optimizer_attribute(model, "Presolve", 0)
    end

    if solver_args !== nothing
        for arg in split(solver_args, ",")
            k, v = split(arg, "=", limit=2)
            v = parse(Int, v)
            set_optimizer_attribute(model, k, v)
        end
    end

    JuMP.optimize!(model)

    println("PF_BENCHMARK: 5_SOLVE_RETURNED")
    flush(stdout)

    if output_dir !== nothing
        cd(output_dir)
    end
end

function run(base_directory::String, main::Function)
    println("PF_BENCHMARK: 1_START")
    flush(stdout)

    model_data_dir = joinpath(base_directory, "model_data")
    if isdir(model_data_dir)
        cd(model_data_dir)
    end

    main(base_model, problem_size; args_dict...)

    println("PF_BENCHMARK: 6_DONE")
    flush(stdout)
end

end

