"""
    preprocess_data(raw_data, experiment) -> NormalizedTrajectoryData

IQR-based normalization for state, input, and time variables.
Replaces MinMax to avoid coefficient explosion when variables have very
different ranges (e.g. H: 0–2000 vs Hsed: 5.0–5.5).
"""
function preprocess_data(
    raw_data::TrajectoryData,
    experiment::ExperimentConfig
)::NormalizedTrajectoryData{Float64}
    println("Preprocessing data (Min-Max normalization).")

    t_norm, t_center, t_scale = minmax_normalization(raw_data.t)

    n_steps  = length(t_norm)
    n_states = size(raw_data.X, 1)
    n_inputs = size(raw_data.U, 1)

    X_norm    = zeros(Float64, n_states, n_steps)
    U_norm    = zeros(Float64, n_inputs, n_steps)
    X_scaling = Dict{String, Tuple{Float64, Float64}}()
    U_scaling = Dict{String, Tuple{Float64, Float64}}()

    for (j, v) in enumerate(experiment.state_vars)
        x_norm, x_center, x_scale = minmax_normalization(raw_data.X[j, :])
        X_scaling[v]  = (Float64(x_center), Float64(x_scale))
        X_norm[j, :] .= x_norm
    end

    for (j, v) in enumerate(experiment.input_vars)
        u_norm, u_center, u_scale = minmax_normalization(raw_data.U[j, :])
        U_scaling[v]  = (Float64(u_center), Float64(u_scale))
        U_norm[j, :] .= u_norm
    end

    scaling = ScalingParams((
        state = X_scaling,
        input = U_scaling,
        time  = (Float64(t_center), Float64(t_scale))
    ))

    return NormalizedTrajectoryData{Float64}(X_norm, U_norm, t_norm, scaling)
end
