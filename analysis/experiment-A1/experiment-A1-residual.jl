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

function find_times(t0::T, tend::T, time_vector::AbstractVector{T}) where T <: Real
     t0_index = argmin(abs(t - t0) for t in time_vector)
     tend_index = argmin(abs(t - tend) for t in time_vector)
     return t0_index, tend_index
end

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
    H_norm, H_mean, H_std = normalization(H_raw, "max")
    I_norm, _, _          = normalization(I_raw, "max")
    C_norm, _, _          = normalization(C_raw, "max")
    t_norm, _, _          = normalization(t_raw, "max", t_scale)

    H_interp_func = CubicSpline(H_norm, t_norm)
    I_interp_func = CubicSpline(I_norm, t_norm) 
    C_interp_func = CubicSpline(C_norm, t_norm)

    n_points = length(t_norm) * upsample_factor
    t_norm_final = collect(range(t_norm[1], t_norm[end], length = n_points))
    t_fine_real = collect(range(t_raw[1], t_raw[end], length = n_points))

    H_fine = H_interp_func.(t_norm_final)
    I_fine = I_interp_func.(t_norm_final)
    C_fine = C_interp_func.(t_norm_final)

    X = reshape(H_fine, 1, :) 
    U = reshape(vcat(I_fine', C_fine'), 2, :) 
    
    scaling = (H_mean=H_mean, H_std=H_std, t_scale=t_scale)

    return X, U, t_norm_final, t_fine_real, H_fine, I_fine, C_fine, scaling
end

function build_sindy_basis()
    @parameters τ
    @variables h(τ) i(τ) c(τ)
    
    u_vars = [h]            
    c_vars = [i, c]
    
    # 🛑 PURE FORCING BASIS: Notice 'h' is completely absent from the library.
    # SINDy is only allowed to combine the bounded external forcings.
    function_library = [
        1, i, c, 
        i^2, c^2, i * c,
        i^3, c^3, i^2 * c, i * c^2
    ]

    return Basis(function_library, u_vars, controls=c_vars, iv = τ)
end

function solve_sindy(X, U, t_norm_final)
    println("\n🚀 Solving with Externally Forced Attractor Architecture")
    
    dummy_prob = ContinuousDataDrivenProblem(X, t_norm_final, U=U)
    dX_true = dummy_prob.DX 
    
    # 1. Find the natural baseline to get a starting point for H
    A_lin = vcat(X, U, ones(1, size(U, 2)))'
    lin_coeffs_raw = A_lin \ dX_true' 
    
    # 2. Force an Unconditionally Stable Anchor
    # Even if the data suggests positive feedback, we force a minimum anchor of -0.5
    alpha_H = min(lin_coeffs_raw[1], -0.5) 
    
    # 3. Calculate the new residual (True Derivative - Anchor Pull)
    # Now we are asking SINDy: "What forcing overcomes this anchor to create the true climate?"
    dX_residual = dX_true .- (alpha_H .* X)
    
    # 4. Train SINDy exclusively to build the Forcing Function
    ddprob = ContinuousDataDrivenProblem(X, t_norm_final, dX_residual, U=U)
    basis = build_sindy_basis()
    
    sampler = DataProcessing(split = 0.8, shuffle = false, batchsize = 50, rng=MLUtils.Xoshiro(42))
    
    # 🔄 Switched to SR3 optimizer for better handling of derivative spikes
    λs = exp10.(-3.0:0.1:1.0)    
    opt = SR3(λs) 
    options = DataDrivenCommonOptions(data_processing = sampler, digits = 4, maxiters = 25000)
    
    sol = solve(ddprob, basis, opt, options=options)
    
    return sol, dummy_prob, alpha_H
end

function simulate_ode(sol, H_fine, I_fine, C_fine, t_norm_final, alpha_H)
    discovered_sys = get_basis(sol)
    p_opt = get_parameter_values(discovered_sys)

    I_interp_ode = LinearInterpolation(I_fine, t_norm_final)
    C_interp_ode = LinearInterpolation(C_fine, t_norm_final)

    t_min = t_norm_final[1]
    t_max = t_norm_final[end]

    function discovered_ode!(du, u, p, t_val)
        safe_t = clamp(t_val, t_min, t_max)
        i_val = I_interp_ode(safe_t)
        c_val = C_interp_ode(safe_t)
        u_control = [i_val, c_val]
        
        # SINDy Forcing Function F(I, C)
        forcing_part = discovered_sys(u, p, safe_t, u_control)
        
        # Hardcoded Stable Anchor: -alpha * H
        anchor_part = alpha_H * u[1] 
        
        # Total ODE: Guaranteed bounded response
        du .= anchor_part .+ forcing_part
    end

    u0_norm = [H_fine[1]]
    tspan_norm = (t_norm_final[1], t_norm_final[end])
    ode_prob = ODEProblem(discovered_ode!, u0_norm, tspan_norm, p_opt)
    
    domain_guard = (u, p, t) -> any(x -> abs(x) > 10.0, u)

    sim_sol = solve(ode_prob, Rosenbrock23(), 
                    reltol=1e-4, abstol=1e-4, 
                    saveat=t_norm_final,
                    isoutofdomain=domain_guard,
                    maxiters=10_000)
    
    return sim_sol, discovered_sys, p_opt
end

function plot_results(exp_key, output_path, sol, sim_sol, ddprob, X, U, t_norm_final, t_fine_real, H_fine, scaling, alpha_H)
    println("\n📊 Plotting results...")

    discovered_sys = get_basis(sol)
    p_opt = get_parameter_values(discovered_sys)

    # Reconstruct Full Prediction
    dX_sindy_forcing = discovered_sys(X, p_opt, t_norm_final, U)
    dX_anchor = alpha_H .* X 
    
    dX_predicted_norm = dX_anchor .+ dX_sindy_forcing
    dX_true_norm = ddprob.DX 
    
    deriv_scale = scaling.H_std / scaling.t_scale
    dX_true_real = dX_true_norm .* deriv_scale
    dX_predicted_real = dX_predicted_norm .* deriv_scale
    H_fine_real = H_fine .* scaling.H_std .+ scaling.H_mean

    target_times = range(ceil(t_fine_real[1]/50000)*50000, floor(t_fine_real[end]/50000)*50000, step=50000)
    label_idx = [argmin(abs.(t_fine_real .- tt)) for tt in target_times]
    time_labels = [string(round(Int, t_fine_real[i] / 1000), "k") for i in label_idx]

    fig = Figure(size = (1300, 700))

    ax1 = Axis(fig[1, 1], title = "State Variable: H(t)", xlabel = "Time", ylabel = "H")
    scatterlines!(ax1, t_fine_real, H_fine_real, label = "Upsampled Data", color = (:blue, 0.3), markersize=2)
    
    if sim_sol.retcode == ReturnCode.Success
        sim_H_real = sim_sol[1, :] .* scaling.H_std .+ scaling.H_mean
        lines!(ax1, t_fine_real, sim_H_real, label = "Simulated ODE", color = :red, linewidth=2)
    else
        sim_H_real_failed = sim_sol[1,:] .* scaling.H_std .+ scaling.H_mean
        lines!(ax1, sim_sol.t .* scaling.t_scale, sim_H_real_failed, label = "Failed ODE", color = :orange, linestyle=:dot)
    end
    axislegend(ax1)

    ax2 = Axis(fig[2, 1], title = "Derivatives: dH/dt", xlabel = "Time", ylabel = "dH/dt")
    lines!(ax2, t_fine_real, vec(dX_true_real[1, :]), label = "True (Upsampled)", color = :blue)
    lines!(ax2, t_fine_real, vec(dX_predicted_real[1, :]), label = "Predicted", color = :red)
    axislegend(ax2)

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

function save_experiment_log(log_file, exp_key, data_path, t_raw, sol, p_opt, alpha_H)
    metadata = Dict(
        "experiment_key" => exp_key,                   
        "data_path" => data_path,
        "tspan" => (t_raw[1], t_raw[end]),
        "stable_anchor_alpha" => alpha_H,
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

    t_raw, I_raw, C_raw, H_raw = load_raw_data(data_path, -714e3, -00e3)
    
    X, U, t_norm_final, t_fine_real, H_fine, I_fine, C_fine, scaling = preprocess_and_upsample(
        t_raw, I_raw, C_raw, H_raw; upsample_factor=5
    )
    println("  → Upsampled Data matrix shape: $(size(X)) (vars × time)")

    sol, ddprob, alpha_H = solve_sindy(X, U, t_norm_final)
    
    println("\n⚓ Stable Anchor Discovered:")
    println("dH/dt_anchor = $(round(alpha_H, digits=3)) * h")
    println("\n🧩 SINDy Forcing Function F(I, C):")
    println(get_basis(sol))
    println("r^2: $(r2(sol))\n")

    sim_sol, discovered_sys, p_opt = simulate_ode(sol, H_fine, I_fine, C_fine, t_norm_final, alpha_H)

    plot_file = plot_results(
        exp_key, output_path, sol, sim_sol, ddprob, 
        X, U, t_norm_final, t_fine_real, H_fine, scaling, alpha_H
    )

    println("\nWould you like to log the metadata (and the results) of this experiment? (y/[n])")
    if lowercase(readline()) == "y"
        save_experiment_log(log_file, exp_key, data_path, t_raw, sol, p_opt, alpha_H)
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

run_experiment(data_path, output_path, log_file)