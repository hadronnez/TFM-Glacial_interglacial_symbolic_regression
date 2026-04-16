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

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

"""
    find_times(t0, tend, time_vector)

Finds indices closest to t0 and tend using a non-allocating search.
"""
function find_times(t0::T, tend::T, time_vector::AbstractVector{T}) where T <: Real
     t0_index = argmin(abs(t - t0) for t in time_vector)
     tend_index = argmin(abs(t - tend) for t in time_vector)
     return t0_index, tend_index
end

"""
    normalization(vec, p)
    normalization(vec, p, scale)
"""
function normalization(vec::AbstractVector{T}, p::Union{String, Char}) where T <: Number
    method = string(p)
    if method == "z"
        m, s = mean(vec), std(vec)
        return (vec .- m) ./ s, m, s
    elseif method == "max" || method == "m"
        s = maximum(abs, vec)
        return vec ./ s, zero(eltype(vec ./ s)), s
    else
        error("Normalization type 'p' must be 'z', 'm', or 'max'")
    end
end

function normalization(vec::AbstractVector{T}, p::Union{String, Char}, scale::Number) where T <: Number
    method = string(p)
    if method == "max" || method == "m"
        return vec ./ scale, zero(eltype(vec ./ scale)), scale
    else
        error("Manual scale only supported for 'max' or 'm'")
    end
end

# ==============================================================================
# PIPELINE STAGES
# ==============================================================================

function load_raw_data(data_path::String, t_start::Real, t_end::Real)
    println("📂 Loading NetCDF data...")
    ds = NCDataset(data_path)

    tspan_idx = find_times(t_start, t_end, vec(ds["time"][:]))

    t_raw = vec(ds["time"][tspan_idx[1]:tspan_idx[2]]) 
    I_raw = vec(ds["I"][tspan_idx[1]:tspan_idx[2]])
    C_raw = vec(ds["C"][tspan_idx[1]:tspan_idx[2]])
    H_raw = vec(ds["H"][tspan_idx[1]:tspan_idx[2]])

    close(ds)
    return t_raw, I_raw, C_raw, H_raw
end

function preprocess_and_upsample(t_raw, I_raw, C_raw, H_raw; upsample_factor::Int=5, t_scale::Float64=1e5)
    # Normalization
    H_norm, H_mean, H_std = normalization(H_raw, "max")
    I_norm, _, _          = normalization(I_raw, "max")
    C_norm, _, _          = normalization(C_raw, "max")
    t_norm, _, _          = normalization(t_raw, "max", t_scale)

    # Interpolators
    H_interp_func = CubicSpline(H_norm, t_norm)
    I_interp_func = LinearInterpolation(I_norm, t_norm)
    C_interp_func = LinearInterpolation(C_norm, t_norm)

    # Grids
    n_points = length(t_norm) * upsample_factor
    t_norm_final = collect(range(t_norm[1], t_norm[end], length = n_points))
    t_fine_real = collect(range(t_raw[1], t_raw[end], length = n_points))

    # Evaluate
    H_fine = H_interp_func.(t_norm_final)
    I_fine = I_interp_func.(t_norm_final)
    C_fine = C_interp_func.(t_norm_final)

    # Matrices
    X = reshape(H_fine, 1, :) 
    U = reshape(vcat(I_fine', C_fine'), 2, :) 
    
    # Bundle scaling parameters for later use
    scaling = (H_mean=H_mean, H_std=H_std, t_scale=t_scale)

    return X, U, t_norm_final, t_fine_real, H_fine, I_fine, C_fine, scaling
end

function build_sindy_basis()
    @parameters τ
    @variables i(τ) h(τ) c(τ)
    
    u_vars = [h]            
    c_vars = [i, c]         
    vars = [i, h, c]        
    

    function_library = [
            # 1;
            # h; h^2; h^3;        # Allowed back in to create the "tipping point" fold
            # i; c;               # Standard linear forcings
            # i^2; c^2;           # Standard nonlinear forcings
            # h * i; h * c;       # Linear interactions
            # h^2 * i; h^2 * c;   # THE TRIGGERS: High ice (h^2) + forcing = rapid collapse
            # h * i^2; h * c^2    # High forcing interacting with state
            polynomial_basis(vars, 2);
    ]

    return Basis(function_library, u_vars, controls=c_vars, iv = τ)
end

function solve_sindy(X, U, t_norm_final)
    println("\n🚀 Solving with STLSQ")
    ddprob = ContinuousDataDrivenProblem(X, t_norm_final, U=U)
    basis = build_sindy_basis()
    
    sampler = DataProcessing(split = 0.8, shuffle = false, batchsize = 50)
    
    # Provide a SEARCH SPACE for the threshold, not just a hardcoded value.
    # SINDy will test these and pick the best trade-off between accuracy and sparsity.
    λs = exp10.(-3.0:0.1:1.0)    
    opt = STLSQ(λs)
    options = DataDrivenCommonOptions(data_processing = sampler, digits = 4, maxiters = 25000)
    
    sol = solve(ddprob, basis, opt, options=options)
    return sol, ddprob
end

function simulate_ode(sol, H_fine, I_fine, C_fine, t_norm_final)
    discovered_sys = get_basis(sol)
    p_opt = get_parameter_values(discovered_sys)

    I_interp_ode = CubicSpline(I_fine, t_norm_final)
    C_interp_ode = CubicSpline(C_fine, t_norm_final)

    # Pre-calculate bounds for extreme safety
    t_min = t_norm_final[1]
    t_max = t_norm_final[end]

    function discovered_ode!(du, u, p, t_val)
        # CLAMP time to prevent spline extrapolation explosions
        safe_t = clamp(t_val, t_min, t_max)
        u_control = [I_interp_ode(safe_t), C_interp_ode(safe_t)]
        du .= discovered_sys(u, p, safe_t, u_control)
    end

    u0_norm = [H_fine[1]]
    tspan_norm = (t_norm_final[1], t_norm_final[end])
    ode_prob = ODEProblem(discovered_ode!, u0_norm, tspan_norm, p_opt)
    
    # SAFETY GUARD: Stop if normalized state > 5.0
    domain_guard = (u, p, t) -> any(x -> abs(x) > 5.0, u)

    sim_sol = solve(ode_prob, Rosenbrock23(), 
                    reltol=1e-4, abstol=1e-4, # Slightly loosened tolerances for stability
                    saveat=t_norm_final,
                    isoutofdomain=domain_guard,
                    maxiters=10_000)
    
    return sim_sol, discovered_sys, p_opt
end

function plot_results(exp_key, output_path, sol, sim_sol, ddprob, X, U, t_norm_final, t_fine_real, H_fine, scaling)
    println("\n📊 Plotting results...")

    discovered_sys = get_basis(sol)
    p_opt = get_parameter_values(discovered_sys)

    dX_predicted_norm = discovered_sys(X, p_opt, t_norm_final, U)
    dX_true_norm = ddprob.DX 
    
    # Denormalization
    deriv_scale = scaling.H_std / scaling.t_scale
    dX_true_real = dX_true_norm .* deriv_scale
    dX_predicted_real = dX_predicted_norm .* deriv_scale
    H_fine_real = H_fine .* scaling.H_std .+ scaling.H_mean

    # Label setup
    target_times = range(ceil(t_fine_real[1]/50000)*50000, floor(t_fine_real[end]/50000)*50000, step=50000)
    label_idx = [argmin(abs.(t_fine_real .- tt)) for tt in target_times]
    time_labels = [string(round(Int, t_fine_real[i] / 1000), "k") for i in label_idx]

    fig = Figure(size = (1300, 700))

    # Ax1: Time vs H
    ax1 = Axis(fig[1, 1], title = "State Variable: H(t)", xlabel = "Time", ylabel = "H")
    scatterlines!(ax1, t_fine_real, H_fine_real, label = "Upsampled Data", color = (:blue, 0.3), markersize=2)
    
    if sim_sol.retcode == ReturnCode.Success
        sim_H_real = sim_sol[1, :] .* scaling.H_std .+ scaling.H_mean
        lines!(ax1, t_fine_real, sim_H_real, label = "Simulated ODE", color = :red, linewidth=2)
    else
        sim_H_real_failed = sim_sol[1,:] .* scaling.H_std .+ scaling.H_mean
        lines!(ax1, sim_sol.t .* scaling.t_scale, sim_H_real_failed, label = "Failed ODE (Exploded)", color = :orange, linestyle=:dot)
        println("\n⚠️ WARNING: ODE exploded or stalled, plotting valid points up to failure.")
    end
    axislegend(ax1)

    # Ax2: Time vs dH/dt
    ax2 = Axis(fig[2, 1], title = "Derivatives: dH/dt", xlabel = "Time", ylabel = "dH/dt")
    lines!(ax2, t_fine_real, vec(dX_true_real[1, :]), label = "True (Upsampled)", color = :blue)
    lines!(ax2, t_fine_real, vec(dX_predicted_real[1, :]), label = "Predicted", color = :red)
    axislegend(ax2)

    # Ax3: Phase Portrait
    ax3 = Axis(fig[1:2, 2], title = "Phase Portrait: H vs dH/dt", xlabel = "H", ylabel = "dH/dt")
    lines!(ax3, H_fine_real, vec(dX_true_real[1, :]), color = (:blue, 0.4), label="True")
    lines!(ax3, H_fine_real, vec(dX_predicted_real[1, :]), color = (:red, 0.8), label="Model")
    
    scatter!(ax3, H_fine_real[label_idx], vec(dX_true_real[1, :])[label_idx], color = :darkblue)
    text!(ax3, H_fine_real[label_idx], vec(dX_true_real[1, :])[label_idx], text = time_labels, color = :darkblue, offset=(5,5))

    plot_file = joinpath(output_path, "$(exp_key).png")
    save(plot_file, fig)
    display(fig)
    
    return plot_file
end

function save_experiment_log(log_file, exp_key, data_path, t_raw, sol, p_opt)
    metadata = Dict(
        "experiment_key" => exp_key,                   
        "data_path" => data_path,
        "tspan" => (t_raw[1], t_raw[end]),
        "variables" => ["time", "I", "C", "H"],
        "basis_functions" => string(get_basis(sol)),
        "parameters" => p_opt,
        "dof" => dof(sol),
        "r2" => r2(sol),
        "solver" => string(get_algorithm(sol)),
    )
    
    open(log_file, "a") do io
        JSON.print(io, metadata)
        println(io)
    end
    println("✅ Metadata appended to registry: $log_file")
end

# ==============================================================================
# MAIN EXPERIMENT FUNCTION
# ==============================================================================

function run_experiment(data_path::String, output_path::String, log_file::String)
    exp_key = "EXP_" * Dates.format(now(), "yyyymmdd_HHMMSS")
    println("==========================================")
    println(" STARTING EXPERIMENT: $exp_key")
    println("==========================================\n")

    # 1. Load Data
    t_raw, I_raw, C_raw, H_raw = load_raw_data(data_path, -714e3, -00e3)
    
    # 2. Preprocess & Upsample
    X, U, t_norm_final, t_fine_real, H_fine, I_fine, C_fine, scaling = preprocess_and_upsample(
        t_raw, I_raw, C_raw, H_raw; upsample_factor=1
    )
    println("  → Upsampled Data matrix shape: $(size(X)) (vars × time)")

    # 3. Discover Equations
    sol, ddprob = solve_sindy(X, U, t_norm_final)
    println(get_basis(sol))
    println("r^2: $(r2(sol))\n")

    # 4. Simulate ODE
    sim_sol, discovered_sys, p_opt = simulate_ode(sol, H_fine, I_fine, C_fine, t_norm_final)

    # 5. Plot Results
    plot_file = plot_results(
        exp_key, output_path, sol, sim_sol, ddprob, 
        X, U, t_norm_final, t_fine_real, H_fine, scaling
    )

    # 6. Logging
    println("\nWould you like to log the metadata (and the results) of this experiment? (y/[n])")
    if lowercase(readline()) == "y"
        save_experiment_log(log_file, exp_key, data_path, t_raw, sol, p_opt)
        println("✅ Plot saved to: $plot_file")
    else
        rm(plot_file, force=true)
        println("Metadata and results are not saved.")
    end
end

# ==============================================================================
# EXECUTION
# ==============================================================================

ROOT = normpath(joinpath(@__DIR__, "..", ".."))
data_path = joinpath(ROOT, "analysis", "experiment-A", "input", "BASE.nc")
output_path = joinpath(ROOT, "analysis", "experiment-A", "output")
log_file = joinpath(output_path, "experiments_log.jsonl")
mkpath(output_path)

# Pass the paths explicitly to avoid global variable pollution
run_experiment(data_path, output_path, log_file)