
using DataDrivenDiffEq, ModelingToolkit, OrdinaryDiffEq, DataDrivenSparse
using LinearAlgebra
using NCDatasets
using CairoMakie
using Statistics
using StatsAPI
using DataInterpolations
using Dates 
using JSON  
using MLUtils

#I run an univariate SINDy experiment using only ["I", "C", "H"].
#I presented the results of this experiment in the 15-04-2026 .pdf presentation in /notes.

"""
    run_experiment(data_path, output_path, log_file)
 
Top-level orchestrator. Runs the full SINDy discovery pipeline:
load → preprocess → discover → simulate → plot →  log.
 
 
# Arguments
- `data_path`: Path to the source `.nc` file.
- `output_path`: Directory for output figures.
- `log_file`: Path to the `.jsonl` experiment registry.
"""
function run_experiment(data_path::String, output_path::String, log_file::String)
 
    exp_key = "EXP_" * Dates.format(now(), "yyyymmdd_HHMMSS")
    println("==========================================")
    println(" STARTING EXPERIMENT: $exp_key")
    println("==========================================\n")
 
    # ------------------------------------------------------------------
    # Configuration — edit here to change variables or candidate library.
    # Symbolic variables are passed in order: [state_vars..., input_vars...]
    # ------------------------------------------------------------------
    state_vars = ["H"]
    input_vars = ["I", "C"]
 
    my_library = (h, i, c) -> [
        1;
        h;
        i; c;
        h * i; h * c
    ]
 
    # --- 1. Load ---
    t_raw, data = MyProject.load_raw_data(
        data_path, -800e3, 0e3,
        vcat(state_vars, input_vars)
    )
 
    # --- 2. Preprocess ---
    X, U, t_norm_final, scaling = MyProject.preprocess_data(
        t_raw, data;
        state_vars      = state_vars,
        input_vars      = input_vars,
        upsample_factor = 1,
        do_upsample     = false
    )
 
    # Named dicts — no hardcoded row indices downstream
    state_data = Dict(v => X[i, :] for (i, v) in enumerate(state_vars))
    input_data = Dict(v => U[i, :] for (i, v) in enumerate(input_vars))
 
    # --- 3. Discover equations ---
    sampler = DataProcessing(split = 0.8, shuffle = false, batchsize = 50,
                             rng = MLUtils.Xoshiro(42))
 
    sol, ddprob = MyProject.solve_sindy(
        X, U, t_norm_final,
        state_vars, input_vars, my_library;
        sampler = sampler,
        λs      = exp10.(-2.0:0.1:0.0),
        options = DataDrivenCommonOptions(
                      data_processing = sampler,
                      digits          = 6,
                      maxiters        = 50000)
    )
 
    println(get_basis(sol))
    println("r²: $(r2(sol))\n")
 
    # --- 4. Simulate ---
    sim_sol, _, p_opt = MyProject.simulate_ode(
        sol, state_data, input_data, state_vars, input_vars, t_norm_final
    )
 
    # --- 5. Plot ---
    plot_file = MyProject.plot_results(
        exp_key, output_path, sol, sim_sol, ddprob,
        X, U, t_norm_final, t_raw,
        state_data, state_vars, scaling
    )
 
    # --- 6. Log ---
    println("\nWould you like to log the metadata of this experiment? (y/[n])")
    if lowercase(readline()) == "y"
        MyProject.save_experiment_log(log_file, exp_key, data_path, t_raw,
                            state_vars, input_vars, sol, p_opt)
        println("✅ Plot saved to: $plot_file")
    else
        rm(plot_file, force = true)
        println("Metadata and results not saved.")
    end
end
 
 
# ==============================================================================
# EXECUTION
# ==============================================================================
 
ROOT        = normpath(joinpath(@__DIR__, "..", ".."))
data_path   = joinpath(ROOT, "analysis", "experiment-A1", "input", "BASE.nc")
output_path = joinpath(ROOT, "analysis", "experiment-A1", "output")
log_file    = joinpath(output_path, "experiments_log.jsonl")
mkpath(output_path)
 
run_experiment(data_path, output_path, log_file)
 