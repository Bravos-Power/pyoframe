include("../benchmark_utils/jump.jl")

using JuMP
using ..Benchmark
using DataFrames
using CSV
using Parquet2
using Parquet2: Dataset

function main(
	model::JuMP.Model,
	problem_size::Int;
	capacity_expansion::Bool = true,
	security_constrained::Bool = true,
	yearly_limits::Bool = true,
	variable_capacity_factors::Bool = true,
)

	# -------------------------
	# LOAD DATA
	# -------------------------
	gens = DataFrame(Dataset("generators.parquet"); copycols=false)
	lines = DataFrame(Dataset("lines_simplified.parquet"); copycols=false)
	loads_df = DataFrame(Dataset("loads.parquet"); copycols=false)
	capex = CSV.read("capex_costs.csv", DataFrame)
	cost_params = CSV.read("cost_parameters.csv", DataFrame)

	BASE_MW = 100.0
	COST_UNSERVED_LOAD = cost_params[cost_params.name .== "load_unserved_MWh", :cost][1]
	SLACK_BUS = 1

	hours = sort(unique(loads_df.datetime))
	if problem_size !== nothing
		hours = hours[1:problem_size]
		loads_df = loads_df[in.(loads_df.datetime, Ref(hours)), :]
	end

	# -------------------------
	# SETS
	# -------------------------
	G = gens.gen_id
	L = lines.line_id
	T = hours
	B = sort(unique(vcat(lines.from_bus, lines.to_bus)))


	# -------------------------
	# INDEX MAPS (para acceso rápido)
	# -------------------------
	gen_bus = Dict(zip(gens.gen_id, gens.bus))
	gen_type = Dict(zip(gens.gen_id, gens.type))
	line_from = Dict(zip(lines.line_id, lines.from_bus))
	line_to = Dict(zip(lines.line_id, lines.to_bus))

	# -------------------------
	# PARAMETERS como contenedores
	# -------------------------
	gen_pmax = JuMP.Containers.DenseAxisArray(gens.Pmax, G)
	gen_cost = JuMP.Containers.DenseAxisArray(gens.cost_per_MWh_linear, G)
	gen_overhead = JuMP.Containers.DenseAxisArray(gens.hourly_overhead_per_MW_capacity, G)

	line_rating = JuMP.Containers.DenseAxisArray(lines.line_rating_MW, L)
	susceptance = JuMP.Containers.DenseAxisArray(1.0 ./ lines.reactance, L)

	gens_at_bus = Dict(b => [g for g in G if gen_bus[g] == b] for b in B)
	lines_to_bus = Dict(b => [l for l in L if line_to[l] == b] for b in B)
	lines_from_bus = Dict(b => [l for l in L if line_from[l] == b] for b in B)

	# carga como contenedor denso (default 0)
	load = JuMP.Containers.DenseAxisArray(
		zeros(length(B), length(T)), B, T,
	)
	for r in eachrow(loads_df)
		load[r.bus, r.datetime] = r.active_load
	end

	# capex
	capex_dict = Dict(zip(capex.type, capex.yearly_capex_cost_per_KW))

	# -------------------------
	# VARIABLES (contenedores JuMP)
	# -------------------------
	if capacity_expansion
		@variable(model, 0 <= Build_Out[g in G] <= gen_pmax[g])
		@variable(model, Dispatch[g in G, t in T] >= 0)

		@constraint(model, [g in G, t in T], Dispatch[g, t] <= Build_Out[g])
	else
		@variable(model, 0 <= Dispatch[g in G, t in T] <= gen_pmax[g])
	end

	@variable(model, Voltage_Angle[b in B, t in T])
	@variable(model, Power_Flow[l in L, t in T])

	@constraint(model, Power_Flow_lower_b[l in L, t in T], -line_rating[l] <= Power_Flow[l, t])
	@constraint(model, Power_Flow_upper_b[l in L, t in T], Power_Flow[l, t] <= line_rating[l])

	@variable(model, Load_Unserved[b in B, t in T; load[b, t] > 0] >= 0)
	@constraint(
		model,
		[b in B, t in T; load[b, t] > 0],
		Load_Unserved[b, t] <= load[b, t]
	)

	# -------------------------
	# CONSTRAINTS
	# -------------------------
	@constraint(model, [t in T], Voltage_Angle[SLACK_BUS, t] == 0)

	@constraint(
		model,
		[l in L, t in T],
		(
			Power_Flow[l, t] ==
			BASE_MW * susceptance[l] *
			(Voltage_Angle[line_to[l], t] - Voltage_Angle[line_from[l], t])
		)
	)

	@constraint(
		model,
		Con_Power_Balance[b in B, t in T],
		(
			0 ==
			-load[b, t] +
			sum(Dispatch[g, t] for g in gens_at_bus[b]) +
			sum(Power_Flow[l, t] for l in lines_to_bus[b]) -
			sum(Power_Flow[l, t] for l in lines_from_bus[b]) +
			((b, t) in Load_Unserved ? Load_Unserved[b, t] : 0.0)
		)
	)

	# -------------------------
	# OBJECTIVE
	# -------------------------
	@objective(
		model,
		Min,
		(
			sum(
				COST_UNSERVED_LOAD * Load_Unserved[b, t]
				for b in B, t in T if (b, t) in Load_Unserved
			) +
			sum(gen_cost[g] * Dispatch[g, t] for g in G, t in T) +
			(capacity_expansion ?
			 (
				sum(
					capex_dict[gen_type[g]] * Build_Out[g]
					for g in G if haskey(capex_dict, gen_type[g])
				) + sum(
					gen_overhead[g] * Build_Out[g] * length(T) for g in G
				)
			) : 0.0
			)
		)
	)

	# -------------------------
	# SECURITY CONSTRAINTS
	# -------------------------
	if security_constrained
		bodf = DataFrame(Dataset("branch_outage_dist_facts.parquet"); copycols=false)
		OUT = bodf.outage_line_id
		AFF = bodf.affected_line_id
		FAC = bodf.factor

		@constraint(model, [i in eachindex(OUT), t in T],
			Power_Flow[AFF[i], t] + FAC[i]*Power_Flow[OUT[i], t]
			<=
			line_rating[AFF[i]])

		@constraint(model, [i in eachindex(OUT), t in T],
			Power_Flow[AFF[i], t] + FAC[i]*Power_Flow[OUT[i], t]
			>=
			-line_rating[AFF[i]])
	end

	# -------------------------
	# VARIABLE CAPACITY FACTORS
	# -------------------------
	if variable_capacity_factors
		vcf_df = DataFrame(Dataset("variable_capacity_factors.parquet"); copycols=false)
		map_df = CSV.read("map_type_to_vcf_type.csv", DataFrame)

		type_to_vcf = Dict(zip(map_df.type, map_df.vcf_type))

		vcf = Dict((r.vcf_type, r.datetime) => r.capacity_factor
				   for r in eachrow(vcf_df) if r.datetime in T)

		@constraint(model, [g in G, t in T;
				haskey(type_to_vcf, gen_type[g])],
			Dispatch[g, t] <=
			vcf[(type_to_vcf[gen_type[g]], t)] * gen_pmax[g])
	end

	# -------------------------
	# YEARLY LIMITS
	# -------------------------
	if yearly_limits
		yl = DataFrame(Dataset("yearly_limits.parquet"); copycols=false)
		yearly_limit = Dict(zip(yl.type, yl.limit))

		@constraint(model, [cat in keys(yearly_limit)],
			sum(Dispatch[g, t] for g in G, t in T if gen_type[g] == cat)
			<=
			yearly_limit[cat] * length(T) / (24*365))
	end

	Benchmark.optimize!(model)

	# WRITE RESULTS
	write_container_parquet(value.(Dispatch); filename = "dispatch.parquet", header = [:gen_id, :datetime, :solution])
	write_container_parquet(value.(Load_Unserved); filename = "load_unserved.parquet", header = [:bus, :datetime, :solution])
	write_container_parquet(dual.(Con_Power_Balance); filename = "power_balance_duals.parquet", header = [:bus, :datetime, :dual])

    
    pf_df = DataFrame(Containers.rowtable(
        value.(Power_Flow);
        header = [:line_id, :datetime, :solution],
    ))
    lb_df = DataFrame(Containers.rowtable(
        dual.(Power_Flow_lower_b);
        header = [:line_id, :datetime, :lb_dual],
    ))

    ub_df = DataFrame(Containers.rowtable(
        dual.(Power_Flow_upper_b);
        header = [:line_id, :datetime, :ub_dual],
    ))


    pf_df = leftjoin(pf_df, ub_df, on=[:line_id, :datetime])
    pf_df = leftjoin(pf_df, lb_df, on=[:line_id, :datetime])

    Parquet2.writefile("power_flow.parquet", pf_df)
end

function write_container_parquet(
	container;
	filename::AbstractString,
	header::Vector{Symbol},
)
	df = DataFrame(Containers.rowtable(container; header = header))
	Parquet2.writefile(filename, df)
end


Benchmark.run(@__DIR__, main)

# To test run from the benchmarks directory:
# julia --project=. src/simple_problem/bm_jump.jl gurobi 1000 src/simple_problem/model_results