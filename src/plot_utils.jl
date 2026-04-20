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
    plot_results(exp_key, output_path, sol, sim_sol, ddprob, X, U,
                 t_norm_final, t_raw, state_data, state_vars, scaling) -> plot_file
 
Generate and save a three-panel diagnostic figure for the first state variable:
 
1. **Time series** — observed data vs. simulated ODE trajectory.
2. **Derivative fit** — true vs. predicted `dX/dt`.
3. **Phase portrait** — `X` vs. `dX/dt` with 50-kyr time annotations.
 
# Arguments
- `exp_key`: Unique experiment identifier (used as the filename stem).
- `output_path`: Directory where the figure is saved.
- `sol`: `DataDrivenSolution` from `solve_sindy`.
- `sim_sol`: `ODESolution` from `simulate_ode`.
- `ddprob`: `ContinuousDataDrivenProblem` (provides true derivatives `DX`).
- `X`, `U`: Normalised state and input matrices.
- `t_norm_final`: Normalised time vector.
- `t_raw`: Original (unnormalised) time vector, used for axis labels.
- `state_data`: Dict of normalised state time series.
- `state_vars`: Ordered state variable names.
- `scaling`: NamedTuple from `preprocess_data` containing `state` and `t_scale`.
 
# Returns
- `plot_file`: Full path to the saved `.png` figure.
"""
function plot_results(
    exp_key, output_path, sol, sim_sol, ddprob,
    X, U, t_norm_final, t_raw,
    state_data::Dict{String, Vector{Float64}},
    state_vars::Vector{String},
    scaling
)
    println("\n📊 Plotting results...")
 
    discovered_sys    = get_basis(sol)
    p_opt             = get_parameter_values(discovered_sys)
 
    dX_predicted_norm = discovered_sys(X, p_opt, t_norm_final, U)
    dX_true_norm      = ddprob.DX
 
    # Denormalise using the first state variable
    primary_var       = state_vars[1]
    μ, σ              = scaling.state[primary_var]
    deriv_scale       = σ / scaling.t_scale
 
    dX_true_real      = dX_true_norm      .* deriv_scale
    dX_predicted_real = dX_predicted_norm .* deriv_scale
    X_primary_real    = state_data[primary_var] .* σ .+ μ
 
    # Time axis annotations every 50 kyr
    target_times = range(
        ceil(t_raw[1]    / 50000) * 50000,
        floor(t_raw[end] / 50000) * 50000,
        step = 50000
    )
    label_idx   = [argmin(abs.(t_raw .- tt)) for tt in target_times]
    time_labels = [string(round(Int, t_raw[i] / 1000), "k") for i in label_idx]
 
    fig = Figure(size = (1300, 700))
 
    # --- Panel 1: Time series ---
    ax1 = Axis(fig[1, 1],
               title  = "State Variable: $(primary_var)(t)",
               xlabel = "Time",
               ylabel = primary_var)
 
    scatterlines!(ax1, t_raw, X_primary_real,
                  label = "Data", color = (:blue, 0.3), markersize = 2)
 
    if sim_sol.retcode == ReturnCode.Success
        lines!(ax1, t_raw, sim_sol[1, :] .* σ .+ μ,
               label = "Simulated ODE", color = :red, linewidth = 2)
    else
        lines!(ax1, sim_sol.t .* scaling.t_scale, sim_sol[1, :] .* σ .+ μ,
               label = "Failed ODE (exploded)", color = :orange, linestyle = :dot)
        println("\n⚠️  WARNING: ODE exploded or stalled — plotting partial trajectory.")
    end
    axislegend(ax1)
 
    # --- Panel 2: Derivatives ---
    ax2 = Axis(fig[2, 1],
               title  = "Derivatives: d$(primary_var)/dt",
               xlabel = "Time",
               ylabel = "d$(primary_var)/dt")
 
    lines!(ax2, t_raw, vec(dX_true_real[1, :]),      label = "True",      color = :blue)
    lines!(ax2, t_raw, vec(dX_predicted_real[1, :]), label = "Predicted", color = :red)
    axislegend(ax2)
 
    # --- Panel 3: Phase portrait ---
    ax3 = Axis(fig[1:2, 2],
               title  = "Phase Portrait: $(primary_var) vs d$(primary_var)/dt",
               xlabel = primary_var,
               ylabel = "d$(primary_var)/dt")
 
    lines!(ax3, X_primary_real, vec(dX_true_real[1, :]),
           color = (:blue, 0.4), label = "True")
    lines!(ax3, X_primary_real, vec(dX_predicted_real[1, :]),
           color = (:red, 0.8), label = "Model")
    scatter!(ax3, X_primary_real[label_idx], vec(dX_true_real[1, :])[label_idx],
             color = :darkblue)
    text!(ax3, X_primary_real[label_idx], vec(dX_true_real[1, :])[label_idx],
          text = time_labels, color = :darkblue, offset = (5, 5))
    axislegend(ax3)
 
    plot_file = joinpath(output_path, "$(exp_key).png")
    save(plot_file, fig)
    display(fig)
    return plot_file
end



