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


"""
    preprocess_data(t_raw, data; state_vars, input_vars, upsample_factor,
                    t_scale, do_upsample) -> (X, U, t_out, scaling)
 
Normalise and optionally upsample state and input time series.
 
State variables are normalised with `MyProject.normalization(..., "max")` and
their scale parameters `(μ, σ)` are stored for later denormalisation. Input
variables are normalised but their scale parameters are discarded (inputs are
treated as known forcings, not predicted quantities).
 
# Arguments
- `t_raw`: Raw time vector from `load_raw_data`.
- `data`: Dict of raw time series (keys are variable names).
- `state_vars`: Names of state variables (rows of `X`).
- `input_vars`: Names of input/control variables (rows of `U`).
- `upsample_factor`: Upsampling multiplier applied to the number of time points.
- `t_scale`: Divisor used to normalise time (default `1e5`).
- `do_upsample`: If `true`, interpolate onto a finer time grid.
 
# Returns
- `X`: Normalised state matrix `(n_state × n_time)`.
- `U`: Normalised input matrix `(n_input × n_time)`.
- `t_out`: Normalised (and possibly upsampled) time vector.
- `scaling`: NamedTuple `(state, t_scale)` where `state` maps each state
  variable name to its `(μ, σ)` normalisation parameters.
"""
function preprocess_data(
    t_raw,
    data::Dict{String, Vector{Float64}};
    state_vars::Vector{String},
    input_vars::Vector{String} = String[],
    upsample_factor::Int       = 5,
    t_scale::Float64           = 1e5,
    do_upsample::Bool          = false
)
    println("⚙️  Preprocessing...")
 
    # --- Time normalisation ---
    t_norm, _, _ = MyProject.normalization(t_raw, "max", t_scale)
    t_out = t_norm
 
    if do_upsample
        n_points = length(t_norm) * upsample_factor
        t_out    = collect(range(t_norm[1], t_norm[end], length = n_points))
    end
 
    # --- State matrix ---
    X         = zeros(length(state_vars), length(t_out))
    X_scaling = Dict{String, Tuple{Float64, Float64}}()
 
    for (i, v) in enumerate(state_vars)
        x_norm, μ, σ = MyProject.normalization(data[v], "max")
        X_scaling[v] = (μ, σ)
 
        X[i, :] = if do_upsample
            CubicSpline(x_norm, t_norm).(t_out)
        else
            x_norm
        end
    end
 
    # --- Input matrix ---
    U = zeros(length(input_vars), length(t_out))
 
    for (i, v) in enumerate(input_vars)
        u_norm, _, _ = MyProject.normalization(data[v], "max")
 
        U[i, :] = if do_upsample
            LinearInterpolation(u_norm, t_norm).(t_out)
        else
            u_norm
        end
    end
 
    scaling = (state = X_scaling, t_scale = t_scale)
    return X, U, t_out, scaling
end




"""
    build_sindy_basis(state_vars, input_vars, function_library) -> Basis
 
Construct a `DataDrivenDiffEq.Basis` from symbolic variables and a
user-supplied library of candidate terms.
 
Symbolic variables are created in the order `[state_vars..., input_vars...]`
and passed positionally to `function_library`, so the callable must accept
arguments in that same order.
 
# Arguments
- `state_vars`: Names of state variables (become the `u_vars` of the basis).
- `input_vars`: Names of input/control variables (become the `controls`).
- `function_library`: A callable `(sym_vars...) -> Vector` returning the
  candidate library terms built from the symbolic variables.
 
# Example
```julia
my_library = (h, i, c) -> [1; h; i; c; h*i; h*c]
basis = build_sindy_basis(["H"], ["I", "C"], my_library)
```
"""
function build_sindy_basis(
    state_vars::Vector{String},
    input_vars::Vector{String},
    function_library   # callable: (sym_vars...) -> Vector of symbolic terms
)
    @parameters τ
 
    all_names = vcat(state_vars, input_vars)
    sym_vars  = [only(@variables $(Symbol(v))(τ)) for v in all_names]
 
    n_state = length(state_vars)
    u_vars  = sym_vars[1:n_state]
    c_vars  = sym_vars[n_state+1:end]
 
    return Basis(function_library(sym_vars...), u_vars, controls = c_vars, iv = τ)
end
 



"""
    solve_sindy(X, U, t_norm_final, state_vars, input_vars, function_library;
                sampler, λs, opt, options) -> (sol, ddprob)
 
Identify governing equations from data using the SINDy framework
(`DataDrivenDiffEq`).
 
Builds a `ContinuousDataDrivenProblem`, constructs the candidate basis via
`build_sindy_basis`, then solves with the chosen optimiser.
 
All solver hyperparameters have sensible defaults and can be overridden
individually. If a fully constructed `options` object is supplied it takes
precedence; in that case embed the desired sampler inside `options` directly.
 
# Arguments
- `X`: State matrix `(n_state × n_time)`.
- `U`: Input matrix `(n_input × n_time)`.
- `t_norm_final`: Normalised time vector.
- `state_vars`, `input_vars`: Variable name vectors (forwarded to `build_sindy_basis`).
- `function_library`: Callable producing the symbolic library (see `build_sindy_basis`).
 
# Keyword Arguments
- `sampler`: Data processing / train-test split strategy. Used only when
  `options` is not supplied explicitly.
- `λs`: Range of sparsity thresholds for the default `SR3` optimiser
  (ignored when `opt` is supplied).
- `opt`: Optimiser instance. Defaults to `SR3(λs)`.
- `options`: Full `DataDrivenCommonOptions`. When provided, `sampler` and `λs`
  are ignored.
 
# Returns
- `sol`: `DataDrivenSolution` containing discovered equations and parameters.
- `ddprob`: The `ContinuousDataDrivenProblem` (useful for extracting `DX`).
"""
function solve_sindy(
    X, U, t_norm_final,
    state_vars::Vector{String},
    input_vars::Vector{String},
    function_library;
    sampler = DataProcessing(split = 0.8, shuffle = false, batchsize = 50,
                             rng = MLUtils.Xoshiro(42)),
    λs      = exp10.(-2.0:0.1:1.0),
    opt     = nothing,
    options = nothing
)
    println("\n🚀 Solving with SINDy")
 
    ddprob   = ContinuousDataDrivenProblem(X, t_norm_final, U = U)
    basis    = build_sindy_basis(state_vars, input_vars, function_library)
    _opt     = isnothing(opt)     ? SR3(λs) : opt
    _options = isnothing(options) ? DataDrivenCommonOptions(
                                        data_processing = sampler,
                                        digits          = 4,
                                        maxiters        = 25000) : options
 
    sol = solve(ddprob, basis, _opt, options = _options)
    return sol, ddprob
end




"""
    simulate_ode(sol, state_data, input_data, state_vars, input_vars,
                 t_norm_final) -> (sim_sol, discovered_sys, p_opt)
 
Integrate the discovered ODE system forward in time using optimal parameters
from a SINDy solution.
 
Input variables are interpolated with `CubicSpline` and clamped at the time
boundaries to prevent extrapolation blow-ups. Integration halts early if any
state component exceeds an absolute value of 5 (normalised units).
 
# Arguments
- `sol`: `DataDrivenSolution` returned by `solve_sindy`.
- `state_data`: Dict mapping state variable names to their normalised time series.
- `input_data`: Dict mapping input variable names to their normalised time series.
- `state_vars`, `input_vars`: Ordered variable name vectors.
- `t_norm_final`: Normalised time vector (used as `saveat` grid and for
  interpolant bounds).
 
# Returns
- `sim_sol`: `ODESolution` from `OrdinaryDiffEq`.
- `discovered_sys`: The extracted symbolic basis system.
- `p_opt`: Optimal parameter vector.
"""
function simulate_ode(
    sol,
    state_data::Dict{String, Vector{Float64}},
    input_data::Dict{String, Vector{Float64}},
    state_vars::Vector{String},
    input_vars::Vector{String},
    t_norm_final::Vector{Float64}
)
    discovered_sys = get_basis(sol)
    p_opt          = get_parameter_values(discovered_sys)
 
    interps = [CubicSpline(input_data[v], t_norm_final) for v in input_vars]
    t_min   = t_norm_final[1]
    t_max   = t_norm_final[end]
 
    function discovered_ode!(du, u, p, t_val)
        safe_t    = clamp(t_val, t_min, t_max)
        u_control = [interp(safe_t) for interp in interps]
        du .= discovered_sys(u, p, safe_t, u_control)
    end
 
    u0_norm  = [state_data[v][1] for v in state_vars]
    ode_prob = ODEProblem(discovered_ode!, u0_norm, (t_min, t_max), p_opt)
 
    domain_guard = (u, p, t) -> any(x -> abs(x) > 5.0, u)
 
    sim_sol = solve(ode_prob, Rosenbrock23(),
                    reltol        = 1e-4,
                    abstol        = 1e-4,
                    saveat        = t_norm_final,
                    isoutofdomain = domain_guard,
                    maxiters      = 10_000)
 
    return sim_sol, discovered_sys, p_opt
end





"""
    save_experiment_log(log_file, exp_key, data_path, t_raw,
                        state_vars, input_vars, sol, p_opt)
 
Append a JSON record of experiment metadata to a `.jsonl` log file.
 
Each line is a self-contained JSON object, making the file suitable for
line-by-line parsing with `JSON.parse`.
 
# Arguments
- `log_file`: Path to the append-mode `.jsonl` registry file.
- `exp_key`: Unique experiment identifier.
- `data_path`: Path to the source NetCDF file.
- `t_raw`: Raw time vector (`tspan` stored as first/last values).
- `state_vars`, `input_vars`: Variable name vectors (stored verbatim).
- `sol`: `DataDrivenSolution` (provides basis string, DOF, R², algorithm).
- `p_opt`: Optimal parameter vector.
"""
function save_experiment_log(
    log_file, exp_key, data_path, t_raw,
    state_vars::Vector{String},
    input_vars::Vector{String},
    sol, p_opt
)
    metadata = Dict(
        "experiment_key"  => exp_key,
        "data_path"       => data_path,
        "tspan"           => (t_raw[1], t_raw[end]),
        "state_vars"      => state_vars,
        "input_vars"      => input_vars,
        "basis_functions" => string(get_basis(sol)),
        "parameters"      => p_opt,
        "dof"             => dof(sol),
        "r2"              => r2(sol),
        "solver"          => string(get_algorithm(sol)),
    )
 
    open(log_file, "a") do io
        JSON.print(io, metadata)
        println(io)
    end
    println("✅ Metadata appended to registry: $log_file")
end
