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
using ..MyProject

"""
    run_experiment(data_path, output_path, log_file)
 
Top-level orchestrator. Runs the full SINDy discovery pipeline:
load → preprocess → discover → simulate → plot → (optionally) log.
 
Edit `state_vars`, `input_vars`, and `my_library` in the configuration block
to change which variables and candidate terms are used — no other function
needs to change.
 
# Arguments
- `data_path`: Path to the source `.nc` file.
- `output_path`: Directory for output figures.
- `log_file`: Path to the `.jsonl` experiment registry.
"""
function run_experiment(data_path::String, exp_key::String, exp_dir::String, log_file::String) 
    println("==========================================")
    println(" STARTING EXPERIMENT: $exp_key")
    println("==========================================\n")
 
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    state_vars = ["H", "C", "B", "Hsed", "T"]

    input_vars = ["I"]
 
    my_library = (h, i, c, b, t, hsed) -> [
        1;
        h;
        hsed;
        c;
        t;
        i;
        h^3;
        b;
        h^4;
        h^2;
        h*t;
    ]
 
    # --- 1. Load ---
    t_raw, data = MyProject.load_raw_data(
        data_path, -800e3, -0e3,
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
 
    state_data = Dict(v => X[i, :] for (i, v) in enumerate(state_vars))
    input_data = Dict(v => U[i, :] for (i, v) in enumerate(input_vars))
 
    # --- 3. Discover equations ---
    sampler = DataProcessing(split = 0.99, shuffle = false, batchsize = 50,
                             rng = MLUtils.Xoshiro(42))
    λs      = exp10.(-0.95:0.0005:-0.85) 

    sol, ddprob = MyProject.solve_sindy(
        X, U, t_norm_final,
        state_vars, input_vars, my_library;
        sampler = sampler,
        opt     = SR3(λs), 
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
 
    # --- 5. Plot Standard Diagnostics ---
    plot_files = MyProject.plot_results(
        exp_key, exp_dir, sol, sim_sol, ddprob,
        X, U, t_norm_final, t_raw,
        state_data, state_vars, scaling, state_vars
    )
    
    # --- 5.5 Plot Hysteresis ---
    println("\n📊 Plotting hysteresis curves...")
    hysteresis_files = MyProject.plot_hysteresis(
        exp_key, exp_dir, sol, X, t_norm_final, 
        state_vars, input_vars, state_data, input_data, scaling
    )
    append!(plot_files, hysteresis_files) # Add hysteresis files to the log tracker
    
    # --- 6. Log ---
    println("\nWould you like to log the metadata of this experiment? (y/[n])")
    if lowercase(readline()) == "y"
        MyProject.save_experiment_log(log_file, exp_key, data_path, t_raw,
                            state_vars, input_vars, sol, p_opt)
        println("✅ Experiment data saved to directory: $exp_dir")
        for f in plot_files
            println("  - $(basename(f))")
        end
    else
        rm(exp_dir, recursive = true, force = true)
        println("Metadata and results not saved. Cleaned up temporary folder.")
    end
end
 
 
# ==============================================================================
# EXECUTION
# ==============================================================================
 
exp_key = "EXP_" * Dates.format(now(), "yyyymmdd_HHMMSS")

ROOT        = normpath(joinpath(@__DIR__, "..", ".."))
data_path   = joinpath(ROOT, "experiments", "experiment-A2", "input", "BASE.nc")
output_path = joinpath(ROOT, "experiments", "experiment-A2", "output")
exp_dir = joinpath(output_path, exp_key)
log_file    = joinpath(output_path, "experiments_log.jsonl")
mkdir(exp_dir)

run_experiment(data_path, exp_key, exp_dir, log_file) 