using JuMP

if ARGS[1] == "gurobi"
    import Gurobi

    # direct_model is faster and more memory efficient (according to tests on July 13, 2025).
    # We use it to make a fair comparison with the other models.
    model = direct_model(Gurobi.Optimizer())
elseif ARGS[1] == "highs"
    import HiGHS
    model = Model(HiGHS.Optimizer)
elseif ARGS[1] == "ipopt"
    import Ipopt
    model = Model(Ipopt.Optimizer)
end

N = parse(Int, ARGS[2])

set_time_limit_sec(model, 0.0)
set_optimizer_attribute(model, "Presolve", 0)

@variables(model, begin
    x[1:N]
end)
@objective(model, Min, sum(x))
optimize!(model)
