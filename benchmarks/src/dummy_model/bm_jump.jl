using JuMP
import Gurobi

function solve(solver, N)
    # direct_model is faster and more memory efficient (according to tests on July 13, 2025).
    model = direct_model(Gurobi.Optimizer())

    set_time_limit_sec(model, 0.0)
    set_optimizer_attribute(model, "Presolve", 0)

    @variables(model, begin
        x[1:N, 1:N]
        y[1:N, 1:N]
    end)
    @objective(model, Min, sum(2 * x) + sum(y))
    @constraint(model, [i in 1:N, j in 1:N], x[i, j] - y[i, j] >= i)
    @constraint(model, [i in 1:N, j in 1:N], x[i, j] + y[i, j] >= 0)
    optimize!(model)
    return model
end

size = parse(Int, ARGS[2])
m = solve(ARGS[1], size)
# println(objective_value(m))