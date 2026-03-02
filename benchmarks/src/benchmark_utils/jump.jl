module Benchmark

export run

using JuMP

if isempty(ARGS) || ARGS[1] == "gurobi"
    import Gurobi

    # direct_model is faster and more memory efficient (according to tests on July 13, 2025).
    # We use it to make a fair comparison with the other models.
    base_model = direct_model(Gurobi.Optimizer())
elseif ARGS[1] == "highs"
    import HiGHS
    base_model = Model(HiGHS.Optimizer)
elseif ARGS[1] == "ipopt"
    import Ipopt
    base_model = Model(Ipopt.Optimizer)
else
    error("Unsupported solver: $(ARGS[1])")
end


function optimize!(model::JuMP.Model)
    println("PF_BENCHMARK: 2_SOLVE")
    flush(stdout)

    if length(ARGS) >= 4 && ARGS[4] == "true"
        set_time_limit_sec(model, 0.0)
        set_optimizer_attribute(model, "Presolve", 0)
    end

    JuMP.optimize!(model)

    println("PF_BENCHMARK: 5_SOLVE_RETURNED")
    flush(stdout)

    if length(ARGS) >= 3
        cd(ARGS[3])
    end
end

function run(base_directory::String, main::Function)
    println("PF_BENCHMARK: 1_START")
    flush(stdout)

    # Convert to full paths
    if length(ARGS) >= 3
        ARGS[3] = abspath(ARGS[3])
    end

    problem_size = parse(Int, ARGS[2])

    model_data_dir = joinpath(base_directory, "model_data")
    if isdir(model_data_dir)
        cd(model_data_dir)
    end

    main(base_model, problem_size)

    println("PF_BENCHMARK: 6_DONE")
    flush(stdout)
end

end

