"""
    run_sindy(data::NormalizedTrajectoryData, experiment::ExperimentConfig) -> SINDyResult

Builds common solver options from `experiment.sindy` and dispatches to the
appropriate identification routine based on the configured method.
"""
function run_sindy(data::NormalizedTrajectoryData, experiment::ExperimentConfig)::SINDyResult
    cfg = experiment.sindy

    options = DataDrivenCommonOptions(
        digits   = cfg.digits,
        maxiters = cfg.maxiters,
        data_processing = DataProcessing(
            split   = cfg.split,
            shuffle = cfg.shuffle
        )
    )

    if cfg.method isa StandardSindy
        println("Running Standard SINDy...")
        return solve_standard_sindy(data, experiment; options = options)

    elseif cfg.method isa EnsembleSindy
        println("Running Ensemble SINDy...")
        return solve_ensemble_sindy(data, experiment; options = options)

    else
        error("Unknown SINDy method: $(typeof(cfg.method))")
    end
end


function _plot_dispatch(
    experiment::ExperimentConfig,
    base_result::StandardSINDyResult,
    sim_denorm::DenormalizedTrajectoryData,
    normalized_data::NormalizedTrajectoryData,
    raw_data::TrajectoryData
)::Vector{String}

    cfg = experiment.plots
    
    any_var_plot = cfg.generate_timeseries || 
                   cfg.generate_derivatives ||
                   cfg.generate_phase_portraits || 
                   cfg.generate_psd ||
                   cfg.generate_hysteresis # Agregado aquí

    state_data = Dict{String, Vector{Float64}}(
        v => raw_data.X[i, :] for (i, v) in enumerate(experiment.state_vars)
    )
    
    input_data = Dict{String, Vector{Float64}}(
        v => raw_data.U[i, :] for (i, v) in enumerate(experiment.input_vars)
    )

    plot_files = String[]

    if any_var_plot
        # Ahora llamamos a plot_results pasando tanto state_data/vars como input_data/vars
        files = plot_results(
            experiment.key,
            experiment.exp_dir,
            base_result.sol,
            sim_denorm,
            base_result.ddprob, 
            normalized_data,
            raw_data.t,
            state_data,
            input_data,
            experiment.state_vars,
            experiment.input_vars,
            experiment 
        )
        append!(plot_files, files)
    end

    return plot_files
end


function plot_from_experiment(
experiment::ExperimentConfig,
sindy_result::StandardSINDyResult,
sim_denorm::DenormalizedTrajectoryData,
normalized_data::NormalizedTrajectoryData,
raw_data::TrajectoryData
)::Vector{String}
return _plot_dispatch(experiment, sindy_result, sim_denorm, normalized_data, raw_data)
end

function plot_from_experiment(
experiment::ExperimentConfig,
sindy_result::EnsembleSINDyResult,
sim_denorm::DenormalizedTrajectoryData,
normalized_data::NormalizedTrajectoryData,
raw_data::TrajectoryData
)::Vector{String}

plot_files = _plot_dispatch(
    experiment, sindy_result.base_result, sim_denorm, normalized_data, raw_data
)

ensemble_files = plot_ensemble_statistics(
    experiment.key,
    experiment.exp_dir,
    sindy_result.inclusion_rates,
    sindy_result.n_ensemble,
    sindy_result.n_success,
    experiment.state_vars
)
append!(plot_files, ensemble_files)

return plot_files
end


"""
    display_discovered_equations(sindy_result::SINDyResult)

Displays the discovered equations from the SINDy result.
"""
function display_discovered_equations(sindy_result::SINDyResult)
    println("=" ^ 70)
    println(" DISCOVERED EQUATIONS")
    println("=" ^ 70)
    
    if sindy_result isa StandardSINDyResult
        println(sindy_result.basis)
    elseif sindy_result isa EnsembleSINDyResult
        println(sindy_result.base_result.basis)
    end
    
    println("=" ^ 70 * "\n")
end



"""
    handle_experiment_logging(experiment, sindy_result, plot_files)

Prompts the user to save or discard the experiment results.
"""
function handle_experiment_logging(experiment::ExperimentConfig, sindy_result::SINDyResult, plot_files::Vector{String})
    println("Would you like to log this experiment? (y/[n])")
    if lowercase(readline()) == "y"
        save_experiment_log(experiment, sindy_result, plot_files)
    else
        rm(experiment.exp_dir, recursive = true, force = true)
        println("Experiment data not saved")
    end
end



"""
    run_experiment(experiment::ExperimentConfig)

Orchestrates the full SINDy pipeline: load data, preprocess, identify equations,
simulate, plot, and optionally persist results.
"""
function run_experiment(experiment::ExperimentConfig)
    println("=" ^ 70)
    println(" STARTING EXPERIMENT: $(experiment.key)")
    println("=" ^ 70 * "\n")

    try
        raw_data = load_experiment_data(experiment)
        println("Data loaded\n")

        normalized_data = preprocess_data(raw_data, experiment)
        println("Data preprocessed\n")

        sindy_result = run_sindy(normalized_data, experiment)
        println("SINDy completed\n")

        sim_denorm = simulate(sindy_result, normalized_data, experiment)
        println("ODE simulation completed\n")

        plot_files = plot_from_experiment(experiment, sindy_result, sim_denorm, normalized_data, raw_data)
        println("Plots generated\n")

        display_discovered_equations(sindy_result)
        handle_experiment_logging(experiment, sindy_result, plot_files)

    catch e
        @error "Experiment failed with error: $e"
        try
            rm(experiment.exp_dir, recursive = true, force = true)
            println("\nExperiment directory cleaned up due to error.")
        catch
            @warn "Could not delete experiment directory: $(experiment.exp_dir)"
        end
        rethrow(e)  
    end
end