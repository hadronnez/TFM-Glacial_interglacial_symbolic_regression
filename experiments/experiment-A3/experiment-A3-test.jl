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
using FFTW
using StatsBase
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
 
    state_vars = ["H", "C", "B", "T", "Hsed"]
    input_vars = ["I"]
 
    my_library = (h, i, c, b, t, hsed) -> [
        1;
        h; hsed; c; t; i; b;
        h^2; h^3; h^4; h^5;
        t^2; t^3; t^4; t^5;
        h*t; h^2*t;
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
 
# --- 1. Definir opciones de procesamiento limpio para Sistemas Dinámicos ---
# Eliminamos batchsize para procesar la trayectoria completa y damos un split sensato (80/20 o 90/10)
    sindy_options = DataDrivenCommonOptions(
    data_processing = DataProcessing(
        split   = 0.80,          # 80% entrenamiento, 20% validación (estándar seguro)
        shuffle = false,         # ¡CRÍTICO! Mantener false para preservar el orden del tiempo
        rng     = MLUtils.Xoshiro(42)
    ),
    digits   = 6,
    maxiters = 50000
)

    λs = exp10.(0)

# --- 2. Descubrir ecuaciones (Ensemble-SINDy) ---
    sol, ddprob, p_ensemble = MyProject.solve_ensemble_sindy(
    X, U, t_norm_final,
    state_vars, input_vars, my_library;
    n_ensemble          = 10,
    subsample_fraction  = 0.8,
    inclusion_threshold = 0.8,
    opt                 = SR3(λs, HardThreshold()),
    options             = sindy_options # Pasamos la configuración unificada aquí
)
    println(get_basis(sol))
    println("r²: $(r2(sol))\n")
    println("Ensemble parameters: ", round.(p_ensemble, digits = 6))

    # --- 4. Simulate with ensemble parameters ---
    sim_sol, _, p_opt = MyProject.simulate_ode(
        sol, state_data, input_data, state_vars, input_vars, t_norm_final, scaling;
        p_override = p_ensemble
    )
 
    # --- 5. Plot ---
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
    append!(plot_files, hysteresis_files)
    
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
data_path   = joinpath(ROOT, "experiments", "experiment-A3", "input", "BASE.nc")
output_path = joinpath(ROOT, "experiments", "experiment-A3", "output")
exp_dir = joinpath(output_path, exp_key)
log_file    = joinpath(output_path, "experiments_log.jsonl")
mkdir(exp_dir)

run_experiment(data_path, exp_key, exp_dir, log_file) 