"""
    preprocess_data(raw_data::TrajectoryData, experiment::ExperimentConfig)
        -> NormalizedTrajectoryData{Float64}

Executes MinMax scaling on time, state, and input variables.
Maintains space-major array orientations ([variables × time_steps]) throughout execution.
"""
function preprocess_data(
    raw_data::TrajectoryData,
    experiment::ExperimentConfig
)::NormalizedTrajectoryData{Float64}

    println("Preprocessing data.")

    t_norm, t_min, t_range = minmax_normalization(raw_data.t)

    n_steps  = length(t_norm)
    n_states = size(raw_data.X, 1)
    n_inputs = size(raw_data.U, 1)

    X_norm = zeros(Float64, n_states, n_steps)
    U_norm = zeros(Float64, n_inputs, n_steps)

    X_scaling = Dict{String, Tuple{Float64, Float64}}()
    U_scaling = Dict{String, Tuple{Float64, Float64}}()

    for (j, v) in enumerate(experiment.state_vars)
        x_norm, x_min, x_range = minmax_normalization(raw_data.X[j, :])
        X_scaling[v] = (Float64(x_min), Float64(x_range))
        X_norm[j, :] .= x_norm
    end

    for (j, v) in enumerate(experiment.input_vars)
        u_norm, u_min, u_range = minmax_normalization(raw_data.U[j, :])
        U_scaling[v] = (Float64(u_min), Float64(u_range))
        U_norm[j, :] .= u_norm
    end

    scaling = ScalingParams((
        state = X_scaling,
        input = U_scaling,
        time  = (Float64(t_min), Float64(t_range))
    ))

    return NormalizedTrajectoryData{Float64}(X_norm, U_norm, t_norm, scaling)
end
