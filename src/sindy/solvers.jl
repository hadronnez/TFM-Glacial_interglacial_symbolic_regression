using DataDrivenDiffEq


"""
    solve_standard_sindy(data, experiment; options) -> StandardSINDyResult
Solves the continuous SINDy identification problem defined by `data` and
`experiment.sindy`, returning a `StandardSINDyResult`.
"""
function solve_standard_sindy(
    data::NormalizedTrajectoryData,
    experiment::ExperimentConfig;
    options::DataDrivenCommonOptions
)::StandardSINDyResult
    cfg = experiment.sindy
    println("Solving with SINDy (Standard)")
    ddprob    = ContinuousDataDrivenProblem(data.X, data.t; U = data.U)
    basis     = build_sindy_basis(experiment.state_vars, experiment.input_vars, cfg.basis_library)
    
    if cfg.optimizer_type == SR3
        optimizer = cfg.optimizer_type(cfg.threshold, cfg.threshold_function())
    elseif cfg.optimizer_type == STLSQ
        optimizer = cfg.optimizer_type(cfg.threshold)
    elseif cfg.optimizer_type == ADMM
        optimizer = cfg.optimizer_type(cfg.threshold)
    else
        error("Unknown optimizer type: $(cfg.optimizer_type)")
    end
    
    sol       = solve(ddprob, basis, optimizer; options = options)
    p_opt     = DataDrivenDiffEq.get_parameter_values(sol.basis)
    return StandardSINDyResult(sol.basis, sol, ddprob, p_opt)
end


"""
    solve_ensemble_sindy(data, experiment; options) -> EnsembleSINDyResult

Runs the ensemble SINDy routine: fits `m.n_ensemble` subsampled models and
aggregates their coefficient vectors, zeroing terms whose inclusion rate falls
below `m.inclusion_threshold`.
"""
function solve_ensemble_sindy(
    data::NormalizedTrajectoryData,
    experiment::ExperimentConfig;
    options::DataDrivenCommonOptions
)::EnsembleSINDyResult

    cfg = experiment.sindy
    m   = cfg.method::EnsembleSindy

    println("Running Ensemble SINDy (n=$(m.n_ensemble), threshold=$(m.inclusion_threshold * 100)%)")

    n_points = size(data.X, 2)
    n_sub    = round(Int, m.subsample_fraction * n_points)

    res_full      = solve_standard_sindy(data, experiment; options = options)
    ref_params    = DataDrivenDiffEq.parameters(res_full.basis)
    n_total_terms = length(ref_params)
    all_coefs     = zeros(Float64, m.n_ensemble, n_total_terms)

    n_success = 0
    for k in 1:m.n_ensemble
        print("\r  Run $k / $(m.n_ensemble)...")
        idx      = sort(sample(1:n_points, n_sub; replace = false))
        data_sub = NormalizedTrajectoryData(
            data.X[:, idx], data.U[:, idx], data.t[idx], data.scaling
        )
        try
            res_k    = solve_standard_sindy(data_sub, experiment; options = options)
            params_k = DataDrivenDiffEq.parameters(res_k.basis)
            values_k = DataDrivenDiffEq.get_parameter_values(res_k.basis)
            param_map = Dict(string(p) => v for (p, v) in zip(params_k, values_k))

            for (j, p_ref) in enumerate(ref_params)
                all_coefs[k, j] = get(param_map, string(p_ref), 0.0)
            end
            n_success += 1
        catch e
            @warn "Run $k failed: $e"
        end
    end

    inclusion_rate = vec(mean(all_coefs .!= 0.0; dims = 1))
    p_ensemble     = vec(mean(all_coefs; dims = 1))
    mask           = inclusion_rate .>= m.inclusion_threshold

    println("\n  $n_success / $(m.n_ensemble) runs succeeded")
    println("  Terms pruned by threshold: $(sum(.!mask)) / $n_total_terms")

    basis_filtered = _prune_basis(res_full.basis, mask, p_ensemble)

    base_result = StandardSINDyResult(
        basis_filtered,
        res_full.sol,        
        res_full.ddprob,
        p_ensemble[mask]     
    )

    return EnsembleSINDyResult(
        base_result,
        p_ensemble,
        inclusion_rate,
        m.n_ensemble,
        n_success,
        m.subsample_fraction
    )
end

function _prune_basis(basis::Basis, mask::BitVector, coef_values::Vector{Float64})
    eqs_all    = equations(basis)        
    states_b   = states(basis)
    controls_b = controls(basis)
    iv_b       = get_iv(basis)
    old_params = DataDrivenDiffEq.parameters(basis)  

    sub_map = Dict{Any, Any}()
    new_params   = []
    new_defaults = Dict()
    j_new = 0

    for (i, p_old) in enumerate(old_params)
        if mask[i]
            j_new += 1
            p_new = only(@parameters $(Symbol("p_", j_new)))
            sub_map[p_old]         = p_new
            new_defaults[p_new]    = coef_values[i]
            push!(new_params, p_new)
        else
            sub_map[p_old] = 0.0
        end
    end

    eqs_substituted = map(eqs_all) do eq
        lhs_new = substitute(eq.lhs, sub_map)
        rhs_new = substitute(eq.rhs, sub_map)
        lhs_new ~ rhs_new
    end

    return Basis(
        eqs_substituted,
        states_b;
        parameters = new_params,
        controls   = controls_b,
        iv         = iv_b,
        defaults   = new_defaults
    )
end