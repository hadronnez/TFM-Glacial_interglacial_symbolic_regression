
using JSON
using Dates
using DataDrivenDiffEq


# ==============================================================================
# LOG FOR STANDARD SINDY RESULT
# ==============================================================================
function save_experiment_log(
    experiment::ExperimentConfig,
    sindy_result::StandardSINDyResult,
    plot_files::Vector{String}
)
    sol = sindy_result.sol
    basis_obj = get_basis(sol)
    parameters = sindy_result.parameters
    
    equations_strings = map(string, equations(basis_obj))
    
    metadata = Dict(
        "experiment_key" => experiment.key,
        "timestamp" => string(Dates.now()),
        "data_path" => experiment.data_path,
        "tspan" => (experiment.t_start, experiment.t_end),
        "state_vars" => experiment.state_vars,
        "input_vars" => experiment.input_vars,
        "sindy_method" => "standard",
        "basis_functions" => equations_strings, 
        "parameters" => parameters,
        "dof" => dof(sol),
        "r2" => r2(sol),
        "algorithm" => string(get_algorithm(sol)),
        "plot_files" => plot_files,
        "save_plots" => !isempty(plot_files)
    )
    
    open(experiment.log_file, "a") do io
        JSON.print(io, metadata)
        println(io)
    end
    
    println("Experiment log saved to: $(experiment.log_file)")
end

# ==============================================================================
# LOG FOR ENSEMBLE SINDY RESULT
# ==============================================================================
function save_experiment_log(
    experiment::ExperimentConfig,
    sindy_result::EnsembleSINDyResult,
    plot_files::Vector{String}
)
    base_result = sindy_result.base_result
    sol = base_result.sol
    basis_obj = get_basis(sol)
    
    equations_strings = map(string, equations(basis_obj))
    
    metadata = Dict(
        "experiment_key" => experiment.key,
        "timestamp" => string(Dates.now()),
        "data_path" => experiment.data_path,
        "tspan" => (experiment.t_start, experiment.t_end),
        "state_vars" => experiment.state_vars,
        "input_vars" => experiment.input_vars,
        "sindy_method" => "ensemble",
        "basis_functions" => equations_strings, 
        "parameters_ensemble" => sindy_result.Ξ_ensemble,
        "inclusion_rates" => sindy_result.inclusion_rates,
        "n_ensemble" => sindy_result.n_ensemble,
        "n_success" => sindy_result.n_success,
        "subsample_fraction" => sindy_result.subsample_fraction,
        "dof" => dof(sol),
        "r2" => r2(sol),
        "algorithm" => string(get_algorithm(sol)),
        "plot_files" => plot_files,
        "save_plots" => !isempty(plot_files)
    )
    
    open(experiment.log_file, "a") do io
        JSON.print(io, metadata)
        println(io)
    end
    
    println("Experiment log saved to: $(experiment.log_file)")
end