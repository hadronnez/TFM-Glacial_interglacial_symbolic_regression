using OrdinaryDiffEq
using DiffEqCallbacks
using DataInterpolations

"""
    simulate(result::SINDyResult, normalized_data::NormalizedTrajectoryData,
             experiment::ExperimentConfig) -> DenormalizedTrajectoryData

Integrates the discovered ODE forward in time using the normalized initial
conditions and control interpolants derived from `normalized_data`, then
denormalizes the resulting trajectory.

Dispatches on `result` to extract the appropriate basis and parameter vector:
- `StandardSINDyResult`: uses `result.parameters`
- `EnsembleSINDyResult`: uses `result.Ξ_ensemble`
"""
function simulate(
    result::StandardSINDyResult,
    normalized_data::NormalizedTrajectoryData,
    experiment::ExperimentConfig
)::DenormalizedTrajectoryData
    println("Integrating ODE (Standard SINDy)")
    return _integrate(result.basis, result.parameters, normalized_data, experiment)
end

function simulate(
    result::EnsembleSINDyResult,
    normalized_data::NormalizedTrajectoryData,
    experiment::ExperimentConfig
)::DenormalizedTrajectoryData
    println("Integrating ODE (Ensemble SINDy)")
    return _integrate(result.base_result.basis, result.base_result.parameters, normalized_data, experiment)
end


function _integrate(
    basis,
    p_opt::Vector{Float64},
    normalized_data::NormalizedTrajectoryData,
    experiment::ExperimentConfig
)::DenormalizedTrajectoryData
    cfg             = experiment.ode
    scaling         = normalized_data.scaling
    state_vars      = experiment.state_vars
    input_vars      = experiment.input_vars
    variable_bounds = cfg.variable_bounds
    X = normalized_data.X
    U = normalized_data.U
    t = normalized_data.t

    
    interps = [CubicSpline(U[i, :], t) for i in 1:size(U, 1)]
    t_min   = t[1]
    t_max   = t[end]
    
    bounds = map(state_vars) do v
        if haskey(variable_bounds, v)
            μ, r   = scaling.state[v]
            lo, hi = variable_bounds[v]
            ((lo - μ) / r, (hi - μ) / r)
        else
            (-Inf, Inf)
        end
    end
    lower = [b[1] for b in bounds]
    upper = [b[2] for b in bounds]

    clamp_counts = zeros(Int, length(state_vars))

    function discovered_ode!(du, u, p, t_val)
        safe_t    = clamp(t_val, t_min, t_max)
        u_control = [interp(safe_t) for interp in interps]
        du .= basis(u, p, safe_t, u_control)
    end

    clamp_affect! = integrator -> begin
        for i in eachindex(integrator.u)
            val     = integrator.u[i]
            clamped = clamp(val, lower[i], upper[i])
            if clamped != val
                clamp_counts[i] += 1
            end
            integrator.u[i] = clamped
        end
    end

    bounds_callback = DiscreteCallback(
        (u, t_cb, integrator) -> true,
        clamp_affect!;
        save_positions = (false, false)
    )

    u0       = Vector{Float64}(X[:, 1])
    ode_prob = ODEProblem(discovered_ode!, u0, (t_min, t_max), p_opt)

    println("Solving ODE...")
    sim_sol_norm = solve(
        ode_prob, cfg.algorithm;
        reltol   = cfg.reltol,
        abstol   = cfg.abstol,
        saveat   = t,
        callback = bounds_callback,
        maxiters = cfg.maxiters
    )

    println("Denormalizing trajectory...")

    n_t          = length(t)
    X_sim_denorm = zeros(Float64, length(state_vars), n_t)
    for (i, v) in enumerate(state_vars)
        μ, r = scaling.state[v]
        X_sim_denorm[i, :] .= sim_sol_norm[i, :] .* r .+ μ
    end

    U_denorm = zeros(Float64, length(input_vars), n_t)
    for (i, v) in enumerate(input_vars)
        μ, r = scaling.input[v]
        U_denorm[i, :] .= U[i, :] .* r .+ μ
    end

    return DenormalizedTrajectoryData{Float64}(X_sim_denorm, U_denorm, t, state_vars, input_vars)
end
