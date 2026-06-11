
using OrdinaryDiffEq
using DiffEqCallbacks
using DataInterpolations

abstract type IntegrationScheme end

struct ClosedLoop    <: IntegrationScheme end
struct ContinuousODE <: IntegrationScheme end

const CLOSED_LOOP    = ClosedLoop()
const CONTINUOUS_ODE = ContinuousODE()

function simulate(
    result::StandardSINDyResult,
    normalized_data::NormalizedTrajectoryData,
    experiment::ExperimentConfig;
    scheme::IntegrationScheme = ClosedLoop()
)::DenormalizedTrajectoryData
    println("Reconstructing trajectory (Standard SINDy) — scheme: $(typeof(scheme))")
    return _dispatch(result.basis, result.parameters, normalized_data, experiment, scheme)
end

function simulate(
    result::EnsembleSINDyResult,
    normalized_data::NormalizedTrajectoryData,
    experiment::ExperimentConfig;
    scheme::IntegrationScheme = ClosedLoop()
)::DenormalizedTrajectoryData
    println("Reconstructing trajectory (Ensemble SINDy) — scheme: $(typeof(scheme))")
    return _dispatch(result.base_result.basis, result.base_result.parameters,
                     normalized_data, experiment, scheme)
end


_dispatch(basis, p_opt, normalized_data, experiment, ::ClosedLoop) =
    _integrate_closedloop(basis, p_opt, normalized_data, experiment)

_dispatch(basis, p_opt, normalized_data, experiment, ::ContinuousODE) =
    _integrate_ode(basis, p_opt, normalized_data, experiment)


function _denormalize(X_sim, U, t, state_vars, input_vars, scaling)
    n_t          = length(t)
    X_sim_denorm = zeros(Float64, length(state_vars), n_t)
    for (i, v) in enumerate(state_vars)
        μ, r = scaling.state[v]
        X_sim_denorm[i, :] .= X_sim[i, :] .* r .+ μ
    end
    U_denorm = zeros(Float64, length(input_vars), n_t)
    for (i, v) in enumerate(input_vars)
        μ, r = scaling.input[v]
        U_denorm[i, :] .= U[i, :] .* r .+ μ
    end
    return DenormalizedTrajectoryData{Float64}(X_sim_denorm, U_denorm, t, state_vars, input_vars)
end


function _integrate_closedloop(
    basis,
    p_opt::Vector{Float64},
    normalized_data::NormalizedTrajectoryData,
    experiment::ExperimentConfig
)::DenormalizedTrajectoryData
    println("  → Closed-Loop trapezoidal (One-Step-Ahead)")
    X, U, t = normalized_data.X, normalized_data.U, normalized_data.t
    n_states, n_t = size(X)
    interp = ConstantInterpolation(U, t)

    X_sim        = zeros(Float64, n_states, n_t)
    X_sim[:, 1] .= X[:, 1]

    @views for k in 2:n_t
        dt     = t[k] - t[k-1]
        u_prev = interp(t[k-1])
        u_curr = interp(t[k])

        dX_prev = basis(X[:, k-1], p_opt, t[k-1], u_prev)
        dX_curr = basis(X[:, k],   p_opt, t[k],   u_curr)

        X_sim[:, k] .= X_sim[:, k-1] .+ 0.5 .* dt .* (dX_prev .+ dX_curr)
    end

    println("  Denormalizing...")
    return _denormalize(X_sim, U, t,
                        experiment.state_vars, experiment.input_vars,
                        normalized_data.scaling)
end



function _integrate_ode(basis, p_opt, data, exp)
    interp   = [ConstantInterpolation(data.U[i, :], data.t) for i in 1:size(data.U, 1)]
    n_states = size(data.X, 1)
    n_inputs = size(data.U, 1)
    prob = ODEProblem(
        function (du, u, p, t_val)
            u_mat = reshape(u, n_states, 1)          
            U_mat = reshape([f(t_val) for f in interp], n_inputs, 1)  
            result = basis(u_mat, p, [t_val], U_mat)  
            du .= result[:, 1]                        
        end,
        Vector(data.X[:, 1]),
        (data.t[1], data.t[end]),
        p_opt
    )
    sol = solve(prob, Tsit5(), saveat=data.t, maxiters=exp.ode.maxiters)

    t_sol   = sol.t
    X_sol   = Array(sol)
    n_valid = length(t_sol)
    U_partial = data.U[:, 1:n_valid]

    return _denormalize(X_sol, U_partial, t_sol, exp.state_vars, exp.input_vars, data.scaling)
end