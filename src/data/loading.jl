using NCDatasets

"""
    load_experiment_data(experiment::ExperimentConfig) -> TrajectoryData{Float64}

Extracts a designated time window from the NetCDF file defined in the configuration.
Populates state and input matrices in a space-major orientation ([variables × time_steps]).
NCDataset variables may contain Missing values; all arrays are explicitly converted to
Float64 before returning.
"""
function load_experiment_data(experiment::ExperimentConfig)::TrajectoryData{Float64}
    println("Loading NetCDF data from: $(basename(experiment.data_path))")

    ds = NCDataset(experiment.data_path)

    time_vec  = Float64.(vec(ds["time"][:]))
    tspan_idx = find_times(experiment.t_start, experiment.t_end, time_vec)
    start_idx = tspan_idx[1]
    end_idx   = tspan_idx[2]
    n_steps   = (end_idx - start_idx) + 1

    t_raw = time_vec[start_idx:end_idx]

    X_raw = Matrix{Float64}(undef, length(experiment.state_vars), n_steps)
    U_raw = Matrix{Float64}(undef, length(experiment.input_vars),  n_steps)

    for (j, v) in enumerate(experiment.state_vars)
        X_raw[j, :] .= Float64.(vec(ds[v][start_idx:end_idx]))
    end

    for (j, v) in enumerate(experiment.input_vars)
        U_raw[j, :] .= Float64.(vec(ds[v][start_idx:end_idx]))
    end

    close(ds)

    return TrajectoryData{Float64}(X_raw, U_raw, t_raw)
end
