
using JSON
using Dates


function save_experiment_log(
    experiment::ExperimentConfig,
    sindy_result::StandardSINDyResult,
    plot_files::Vector{String}
)
    
    basis = sindy_result.basis
    sol = sindy_result.sol
    parameters = sindy_result.parameters
    
    metadata = Dict(
        "experiment_key" => experiment.key,
        "timestamp" => string(Dates.now()),
        "data_path" => experiment.data_path,
        "tspan" => (experiment.t_start, experiment.t_end),
        "state_vars" => experiment.state_vars,
        "input_vars" => experiment.input_vars,
        "sindy_method" => "standard",
        "basis_functions" => string(basis),
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
 
function save_experiment_log(
    experiment::ExperimentConfig,
    sindy_result::EnsembleSINDyResult,
    plot_files::Vector{String}
)
    
    base_result = sindy_result.base_result
    basis = base_result.basis
    sol = base_result.sol
    
    metadata = Dict(
        "experiment_key" => experiment.key,
        "timestamp" => string(Dates.now()),
        "data_path" => experiment.data_path,
        "tspan" => (experiment.t_start, experiment.t_end),
        "state_vars" => experiment.state_vars,
        "input_vars" => experiment.input_vars,
        "sindy_method" => "ensemble",
        "basis_functions" => string(basis),
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
 
