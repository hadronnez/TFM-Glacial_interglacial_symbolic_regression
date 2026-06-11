using CairoMakie
using DataDrivenDiffEq
using NonlinearSolve
using LinearAlgebra
using SciMLBase
using Statistics
import FFTW

# ==============================================================================
#                               PLOTTING UTILS
# ==============================================================================


function compute_psd(data::Vector{Float64}, t_kyr::Vector{<:AbstractFloat})
    dt_kyr = (t_kyr[end] - t_kyr[1]) / (length(t_kyr) - 1)
    
    demeaned_data = data .- mean(data)
    
    n_padded = 2 ^ ceil(Int, log2(length(demeaned_data) * 4))
    
    padded_data = zeros(Float64, n_padded)
    padded_data[1:length(demeaned_data)] = demeaned_data
    
    power = abs.(rfft(padded_data)) .^ 2
    freqs = (0:length(power)-1) ./ (n_padded * dt_kyr)
    
    mask = freqs .> 0
    
    return freqs[mask], power[mask]
end

function compute_hysteresis_sweeps(
    sol::DataDrivenSolution,
    normalized_data::NormalizedTrajectoryData,
    input_vars::Vector{String},
    input_data::Dict{String, Vector{Float64}},
    state_vars::Vector{String};
    n_points::Int = 100
)
    scaling        = normalized_data.scaling
    target_input   = input_vars[1]
    μ_i, r_i       = scaling.input[target_input]
    U_real         = input_data[target_input]
    discovered_sys = get_basis(sol)
    p_opt          = get_parameter_values(discovered_sys)
    n_vars         = length(state_vars)

    I_min, I_max = minimum(U_real), maximum(U_real)
    I_pad        = (I_max - I_min) * 0.05

    I_forward  = range(I_min - I_pad, I_max + I_pad; length = n_points)
    I_backward = range(I_max + I_pad, I_min - I_pad; length = n_points)
    I_forward_norm  = collect(I_forward  .- μ_i) ./ r_i
    I_backward_norm = collect(I_backward .- μ_i) ./ r_i

    function get_u_input(i_val_norm::Float64)::Vector{Float64}
        u_in = zeros(Float64, length(input_vars))
        for (j, iv) in enumerate(input_vars)
            μ_j, r_j = scaling.input[iv]
            u_in[j]  = (iv == target_input) ? i_val_norm : (mean(input_data[iv]) - μ_j) / r_j
        end
        return u_in
    end

    function nl_sys!(du, u, p)
        p_sys, u_in = p
        du .= discovered_sys(u, p_sys, 0.0, u_in)
    end

    # Inicialización desde el estado observado en mínima y máxima insolación
    idx_min = argmin(U_real)
    idx_max = argmax(U_real)
    u_at_I_min = normalized_data.X[:, idx_min]
    u_at_I_max = normalized_data.X[:, idx_max]

    function sweep(i_vals_norm::Vector{Float64}, u_seed::Vector{Float64})::Matrix{Float64}
        eq     = zeros(Float64, n_vars, n_points)
        u_curr = copy(u_seed)
        for (k, i_val_norm) in enumerate(i_vals_norm)
            prob   = NonlinearProblem(nl_sys!, u_curr, (p_opt, get_u_input(i_val_norm)))
            sol_nl = solve(prob, NewtonRaphson(); maxiters = 1000)
            if SciMLBase.successful_retcode(sol_nl)
                u_curr   = sol_nl.u
                eq[:, k] = u_curr
            else
                eq[:, k] .= NaN
            end
        end
        return eq
    end

    eq_forward  = sweep(I_forward_norm,  u_at_I_min)
    eq_backward = sweep(I_backward_norm, u_at_I_max)

    return I_forward, I_backward, eq_forward, eq_backward, target_input
end

# ==============================================================================
#                          RENDERING FUNCTIONS (PAPER STYLE)
# ==============================================================================

const COLOR_REF = "#1f77b4" # Blue
const COLOR_DIS = "#d62728" # Red

function apply_paper_theme!(gp::GridPosition, ax::Axis, label::String)
    ax.topspinevisible = false
    ax.rightspinevisible = false
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.xtickalign = 1.0
    ax.ytickalign = 1.0

    ax.spinewidth = 2.5
    ax.xtickwidth = 2.5
    ax.ytickwidth = 2.5
    ax.xticksize = 10.0
    ax.yticksize = 10.0

    ax.xlabelsize = 22
    ax.ylabelsize = 22
    ax.xticklabelsize = 18
    ax.yticklabelsize = 18
    
    Label(gp, label, font = :bold, fontsize = 24,
          halign = :left, valign = :top, padding = (10, 0, 5, 0),
          tellwidth = false, tellheight = false)
end

function render_timeseries!(gp::GridPosition, t_kyr, X_real, sim_data, var_name, sim_ok, var_idx, label)
    ax = Axis(gp; xlabel = "Time (kyr)", ylabel = var_name)
    apply_paper_theme!(gp, ax, label)
    

    scatterlines!(ax, t_kyr, X_real; 
                  color = (COLOR_REF, 0.45),      
                  markercolor = COLOR_REF, 
                  linewidth = 3.0,                
                  markersize = 6.0)               
                  
    if sim_ok
        
        scatterlines!(ax, t_kyr, sim_data[var_idx, :]; 
                      color = (COLOR_DIS, 0.45),  
                      markercolor = COLOR_DIS,    
                      linewidth = 3.0,            
                      markersize = 6.0)           
    end
end

function render_derivatives!(gp::GridPosition, t_kyr, dX_true, dX_pred, var_name, label)
    ax = Axis(gp; xlabel = "Time (kyr)", ylabel = "d$(var_name)/dt")
    apply_paper_theme!(gp, ax, label)
    
    scatterlines!(ax, t_kyr, dX_true; 
                  color = (COLOR_REF, 0.45), 
                  markercolor = COLOR_REF, 
                  linewidth = 3.0, 
                  markersize = 6.0)
                  
    scatterlines!(ax, t_kyr, dX_pred; 
                  color = (COLOR_DIS, 0.45), 
                  markercolor = COLOR_DIS, 
                  linewidth = 3.0, 
                  markersize = 6.0)
end

function render_phase_portraits!(gp1::GridPosition, gp2::GridPosition, X_real, dX_true, sim_data, dX_pred_norm, deriv_scale, var_name, sim_ok, var_idx, label)
    ax3 = Axis(gp1; xlabel = var_name, ylabel = "d$(var_name)/dt")
    apply_paper_theme!(gp1, ax3, label * "₁")
    
    scatterlines!(ax3, X_real, dX_true; 
                  color = (COLOR_REF, 0.35), 
                  markercolor = COLOR_REF, 
                  linewidth = 2.5, 
                  markersize = 5.0)
    ax4 = Axis(gp2; xlabel = var_name, ylabel = "d$(var_name)/dt")
    apply_paper_theme!(gp2, ax4, label * "₂")
    
    if sim_ok
        dX_sim_real = vec(dX_pred_norm[var_idx, :]) .* deriv_scale
        
        scatterlines!(ax4, sim_data[var_idx, :], dX_sim_real; 
                      color = (COLOR_DIS, 0.35),      
                      markercolor = COLOR_DIS,        
                      linewidth = 2.5,               
                      markersize = 5.0)               
    end
end

function render_psd!(gp::GridPosition, t_kyr, X_real, sim_data, var_idx, sim_ok, label)
    ax = Axis(gp; xlabel = "Frequency (1/kyr)", ylabel = "Power Spectral Density")
    apply_paper_theme!(gp, ax, label)
    
    freqs, power = compute_psd(X_real, t_kyr)
    lines!(ax, freqs, power; color = COLOR_REF, linewidth = 3.0)

    if sim_ok
        freqs_sim, power_sim = compute_psd(sim_data[var_idx, :], t_kyr)
        lines!(ax, freqs_sim, power_sim; color = COLOR_DIS, linewidth = 3.0, linestyle = :dash)
    end
    
    milankovitch_periods = [100.0, 41.0, 23.0, 19.0]
    milankovitch_freqs   = 1.0 ./ milankovitch_periods
    
    ax_top = Axis(gp; xaxisposition = :top, yaxisposition = :right, xlabel = "Period Equivalent (kyr)")
    ax_top.topspinevisible = true
    ax_top.rightspinevisible = false
    ax_top.leftspinevisible = false
    ax_top.bottomspinevisible = false
    ax_top.xgridvisible = false
    ax_top.ygridvisible = false
    ax_top.xtickalign = 1.0
    ax_top.ytickalign = 1.0
    ax_top.spinewidth = 2.5
    ax_top.xtickwidth = 2.5
    ax_top.xticksize = 10.0
    ax_top.xlabelsize = 18
    ax_top.xticklabelsize = 16
    
    linkyaxes!(ax, ax_top)
    ax_top.xticks = (milankovitch_freqs, ["100k", "41k", "23k", "19k"])
    hideydecorations!(ax_top)
    
    vlines!(ax, milankovitch_freqs; color = (:gray, 0.25), linestyle = :dot, linewidth = 1.5)
    xlims!(ax, 0.0, 0.06) 
    xlims!(ax_top, 0.0, 0.06)
end

function render_hysteresis!(gp::GridPosition, I_forward, I_backward, V_forward_real, V_backward_real, target_input, var_name, label)
    ax = Axis(gp; xlabel = "Forcing ($target_input)", ylabel = var_name)
    apply_paper_theme!(gp, ax, label)
    
    lines!(ax, collect(I_forward),  V_forward_real;  color = COLOR_DIS, linewidth = 3.0)
    lines!(ax, collect(I_backward), V_backward_real; color = COLOR_DIS, linewidth = 3.0, linestyle = :dash)
    
    Legend(gp, [[LineElement(color = COLOR_DIS)], [LineElement(color = COLOR_DIS, linestyle = :dash)]],
           ["Forward sweep", "Backward sweep"]; 
           tellwidth = false, tellheight = false, font = :italic,
           halign = :right, valign = :bottom, margin = (15, 15, 15, 15), framevisible = false,
           labelsize = 18)
end

# ==============================================================================
#                      MAIN PLOT RESULTS FUNCTION
# ==============================================================================

function plot_results(
    exp_key::String, output_path::String, sol::DataDrivenSolution,
    sim_denorm::DenormalizedTrajectoryData, ddprob, normalized_data::NormalizedTrajectoryData,
    t_raw::Vector{<:AbstractFloat}, state_data::Dict{String, Vector{Float64}},
    input_data::Dict{String, Vector{Float64}}, state_vars::Vector{String},
    input_vars::Vector{String}, experiment::ExperimentConfig
)::Vector{String}

    cfg = experiment.plots
    plot_vars = isempty(cfg.variables) ? experiment.state_vars : cfg.variables

    discovered_sys    = get_basis(sol)
    p_opt             = get_parameter_values(discovered_sys)
    scaling           = normalized_data.scaling
    dX_predicted_norm = discovered_sys(normalized_data.X, p_opt, normalized_data.t, normalized_data.U)
    dX_true_norm      = ddprob.DX

    t_kyr  = t_raw ./ 1000
    sim_ok = size(sim_denorm.X, 2) == length(t_kyr)
    plot_files = String[]

    active_panels = Tuple{Symbol, Int}[]
    cfg.generate_timeseries      && push!(active_panels, (:timeseries, 1))
    cfg.generate_derivatives     && push!(active_panels, (:derivatives, 1))
    cfg.generate_phase_portraits && push!(active_panels, (:phase, 2)) 
    cfg.generate_psd             && push!(active_panels, (:psd, 1))
    
    run_hysteresis = cfg.generate_hysteresis && !isempty(input_vars)
    run_hysteresis               && push!(active_panels, (:hysteresis, 1))

    n_rows = length(active_panels)
    if n_rows == 0 return plot_files end

    if run_hysteresis
        I_forward, I_backward, eq_forward, eq_backward, target_input = 
            compute_hysteresis_sweeps(sol, normalized_data, input_vars, input_data, state_vars)
    end

    panel_letters = ["a", "b", "c", "d", "e"]

    for var_name in plot_vars
        var_idx = findfirst(isequal(var_name), state_vars)
        if isnothing(var_idx) continue end

        μ, r        = scaling.state[var_name]
        _, r_t      = scaling.time
        deriv_scale = r / r_t

        dX_true_real      = vec(dX_true_norm[var_idx, :])      .* deriv_scale
        dX_predicted_real = vec(dX_predicted_norm[var_idx, :]) .* deriv_scale
        X_real            = state_data[var_name]

        fig = Figure(size = (1200, 260 * n_rows + 60), backgroundcolor = :white)

        for (row_idx, (panel_type, _)) in enumerate(active_panels)
            letter = "($(panel_letters[row_idx]))"
            
            if panel_type == :timeseries
                render_timeseries!(fig[row_idx, 1:2], t_kyr, X_real, sim_denorm.X, var_name, sim_ok, var_idx, letter)
            elseif panel_type == :derivatives
                render_derivatives!(fig[row_idx, 1:2], t_kyr, dX_true_real, dX_predicted_real, var_name, letter)
            elseif panel_type == :phase
                render_phase_portraits!(fig[row_idx, 1], fig[row_idx, 2], X_real, dX_true_real, sim_denorm.X, dX_predicted_norm, deriv_scale, var_name, sim_ok, var_idx, letter)
            elseif panel_type == :psd
                render_psd!(fig[row_idx, 1:2], t_kyr, X_real, sim_denorm.X, var_idx, sim_ok, letter)
            elseif panel_type == :hysteresis
                V_forward_real  = eq_forward[var_idx, :]  .* r .+ μ
                V_backward_real = eq_backward[var_idx, :] .* r .+ μ
                render_hysteresis!(fig[row_idx, 1:2], I_forward, I_backward, V_forward_real, V_backward_real, target_input, var_name, letter)
            end
        end

        Legend(fig[n_rows + 1, 1:2],
                [[LineElement(color = COLOR_REF, linewidth = 2.5), MarkerElement(color = COLOR_REF, marker = :circle, markersize = 12)],
                 [LineElement(color = COLOR_DIS, linewidth = 2.5), MarkerElement(color = COLOR_DIS, marker = :circle, markersize = 12)]],
                ["Reference Data (PACCO)", "Identified Framework (SINDy)"];
                orientation = :horizontal, tellwidth = false, tellheight = true, 
                framevisible = false, labelsize = 18)

        plot_file = joinpath(output_path, "$(exp_key)_$(var_name).png")
        
        save(plot_file, fig; px_per_unit = 4) 
        push!(plot_files, plot_file)
    end

    return plot_files
end
# ==============================================================================
#                        ENSEMBLE STATISTICS PLOT
# ==============================================================================

function plot_ensemble_statistics(
    exp_key::String, output_path::String, inclusion_rates::Vector{Float64},
    n_ensemble::Int, n_success::Int, state_vars::Vector{String}
)::Vector{String}

    fig = Figure(size = (1200, 500), backgroundcolor = :white)

    ax1 = Axis(fig[1, 1]; xlabel = "Library term index", ylabel = "Inclusion rate")
    ax1.topspinevisible = false
    ax1.rightspinevisible = false
    ax1.xgridvisible = false
    ax1.ygridvisible = false
    Label(fig[1, 1], "(a)", font = :bold, fontsize = 16, halign = :left, valign = :top, padding = (10, 0, 5, 0), tellwidth=false, tellheight=false)

    colors = [r > 0.8 ? "#2ca02c" : r > 0.5 ? "#ff7f0e" : "#d62728" for r in inclusion_rates]
    barplot!(ax1, 1:length(inclusion_rates), inclusion_rates; color = colors, width = 0.7)
    hlines!(ax1, [0.8, 0.5]; color = (:gray, 0.4), linestyle = :dash, linewidth = 1)
    ylims!(ax1, 0, 1.05)

    ax2 = Axis(fig[1, 2]; ylabel = "Count")
    ax2.topspinevisible = false
    ax2.rightspinevisible = false
    ax2.xgridvisible = false
    ax2.ygridvisible = false
    Label(fig[1, 2], "(b)", font = :bold, fontsize = 16, halign = :left, valign = :top, padding = (10, 0, 5, 0), tellwidth=false, tellheight=false)

    success_rate = n_success / n_ensemble
    n_failed     = n_ensemble - n_success
    barplot!(ax2, [1, 2], [n_success, n_failed]; color = ["#2ca02c", "#d62728"], width = 0.5)
    ax2.xticks = ([1, 2], ["Successful", "Failed"])
    
    text!(ax2, 1.0, n_success / 2;
          text = "$(round(success_rate * 100, digits=1))%",
          align = (:center, :center), fontsize = 14, color = :white, font = :bold)
    ylims!(ax2, 0, n_ensemble * 1.1)

    plot_file = joinpath(output_path, "$(exp_key)_ensemble_statistics.png")
    
    save(plot_file, fig; px_per_unit = 3)
    
    return [plot_file]
end