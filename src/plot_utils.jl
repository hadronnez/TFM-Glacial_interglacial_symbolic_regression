using CairoMakie
using DataDrivenDiffEq
using NonlinearSolve
using Statistics

"""
    plot_results(exp_key, output_path, sol, sim_sol, ddprob, X, U,
                 t_norm_final, t_raw, state_data, state_vars, scaling, plot_vars) -> plot_files
 
Generate and save a four-panel diagnostic figure for a selection of state variables:
 
1. **Time series** — observed data vs. simulated ODE trajectory.
2. **Derivative fit** — true vs. predicted `dX/dt`.
3. **True Phase portrait** — `X` vs. true `dX/dt` with 50-kyr time annotations.
4. **Predicted Phase portrait** — `X` vs. predicted `dX/dt` with 50-kyr time annotations.
"""
function plot_results(
    exp_key, output_path, sol, sim_sol, ddprob,
    X, U, t_norm_final, t_raw,
    state_data::Dict{String, Vector{Float64}},
    state_vars::Vector{String},
    scaling,
    plot_vars::Vector{String} 
)
    println("\n📊 Plotting results for variables: ", join(plot_vars, ", "))
 
    discovered_sys    = get_basis(sol)
    p_opt             = get_parameter_values(discovered_sys)
 
    dX_predicted_norm = discovered_sys(X, p_opt, t_norm_final, U)
    dX_true_norm      = ddprob.DX
 
    # Time axis annotations every 50 kyr
    target_times = range(
        ceil(t_raw[1]    / 50000) * 50000,
        floor(t_raw[end] / 50000) * 50000,
        step = 50000
    )
    label_idx   = [argmin(abs.(t_raw .- tt)) for tt in target_times]
    time_labels = [string(round(Int, t_raw[i] / 1000), "k") for i in label_idx]
 
    plot_files = String[]
    
    for var_name in plot_vars
        var_idx = findfirst(isequal(var_name), state_vars)
        
        if var_idx === nothing
            println("⚠️ WARNING: Variable '$var_name' not found in state_vars. Skipping.")
            continue
        end

        # Denormalise for the current state variable
        μ, σ              = scaling.state[var_name]
        deriv_scale       = σ / scaling.t_scale
     
        dX_true_real      = vec(dX_true_norm[var_idx, :])      .* deriv_scale
        dX_predicted_real = vec(dX_predicted_norm[var_idx, :]) .* deriv_scale
        X_real            = state_data[var_name] .* σ .+ μ
     
        fig = Figure(size = (1400, 800))
     
        # --- Panel 1: Time series (Top Left) ---
        ax1 = Axis(fig[1, 1],
                   xlabel = "Time",
                   ylabel = var_name)
     
        scatterlines!(ax1, t_raw, X_real,
                      label = "Data", 
                      color = (:blue, 0.4), markercolor = :blue, 
                      linewidth = 1, markersize = 4)
     
        if sim_sol.retcode == ReturnCode.Success
            scatterlines!(ax1, t_raw, sim_sol[var_idx, :] .* σ .+ μ,
                          label = "Simulated ODE", 
                          color = (:red, 0.5), markercolor = :red, 
                          linewidth = 1, markersize = 4)
        else
            lines!(ax1, sim_sol.t .* scaling.t_scale, sim_sol[var_idx, :] .* σ .+ μ,
                   label = "Failed ODE (exploded)", color = :orange, linestyle = :dot)
            println("⚠️ WARNING: ODE exploded or stalled for $var_name — plotting partial trajectory.")
        end
        axislegend(ax1, position = :rb) # :rb = Right Bottom
     
        # --- Panel 2: Derivatives (Bottom Left) ---
        ax2 = Axis(fig[2, 1],
                   xlabel = "Time",
                   ylabel = "d$(var_name)/dt")
     
        scatterlines!(ax2, t_raw, dX_true_real,      
                      label = "True",      
                      color = (:blue, 0.4), markercolor = :blue, 
                      linewidth = 1, markersize = 4)
                      
        scatterlines!(ax2, t_raw, dX_predicted_real, 
                      label = "Predicted", 
                      color = (:red, 0.5), markercolor = :red, 
                      linewidth = 1, markersize = 4)
        axislegend(ax2, position = :rb)
     
        # --- Panel 3: TRUE Phase portrait (Top Right) ---
        ax3 = Axis(fig[1, 2],
                   xlabel = var_name,
                   ylabel = "d$(var_name)/dt (True)")
     
        scatterlines!(ax3, X_real, dX_true_real,
                      color = (:blue, 0.4), markercolor = :blue, 
                      linewidth = 1, markersize = 4)
                      
        scatter!(ax3, X_real[label_idx], dX_true_real[label_idx],
                 color = :darkblue, markersize = 8)
        text!(ax3, X_real[label_idx], dX_true_real[label_idx],
              text = time_labels, color = :darkblue, offset = (7, 7))
              
        # --- Panel 4: PREDICTED Phase portrait (Bottom Right) ---
        ax4 = Axis(fig[2, 2],
                   xlabel = var_name,
                   ylabel = "d$(var_name)/dt (Predicted)")
                   
        scatterlines!(ax4, X_real, dX_predicted_real,
                      color = (:red, 0.5), markercolor = :red, 
                      linewidth = 1, markersize = 4)
                      
        scatter!(ax4, X_real[label_idx], dX_predicted_real[label_idx],
                 color = :darkred, markersize = 8)
        text!(ax4, X_real[label_idx], dX_predicted_real[label_idx],
              text = time_labels, color = :darkred, offset = (7, 7))
     
        # Save dynamically named file per variable (sim_time removed)
        plot_file = joinpath(output_path, "$(exp_key)_$(var_name).png")
        save(plot_file, fig)
        display(fig)
        push!(plot_files, plot_file)
    end
    
    return plot_files
end


function plot_hysteresis(
    exp_key, output_path, sol, X, t_norm_final, 
    state_vars::Vector{String}, input_vars::Vector{String},
    state_data, input_data, scaling
)
    if isempty(input_vars)
        println("⚠️ No input variables available for hysteresis plot. Skipping.")
        return String[]
    end
    
    target_input = input_vars[1]
    U_real = input_data[target_input] 
    
    discovered_sys = get_basis(sol)
    p_opt          = get_parameter_values(discovered_sys)
    plot_files     = String[]
    
    I_min, I_max = minimum(U_real), maximum(U_real)
    I_pad = (I_max - I_min) * 0.05
    n_points = 100
    
    I_forward  = range(I_min - I_pad, I_max + I_pad, length=n_points)
    I_backward = range(I_max + I_pad, I_min - I_pad, length=n_points)
    
    function get_u_input(i_val)
        u_in = zeros(length(input_vars))
        for (j, iv) in enumerate(input_vars)
            u_in[j] = (iv == target_input) ? i_val : mean(input_data[iv])
        end
        return u_in
    end

    # The Nonlinear System Wrapper: We want this to equal 0
    function nl_sys!(du, u, p)
        p_sys, u_in = p
        du .= discovered_sys(u, p_sys, 0.0, u_in)
    end

    u_start = [mean(state_data[v]) for v in state_vars]
    
    # --- FORWARD SWEEP ---
    eq_forward = zeros(length(state_vars), n_points)
    u_curr = copy(u_start)
    
    for (k, i_val) in enumerate(I_forward)
        prob = NonlinearProblem(nl_sys!, u_curr, (p_opt, get_u_input(i_val)))
        # Use Newton-Raphson to find the exact root
        sol_nl = solve(prob, NewtonRaphson(), maxiters=1000)
        
        if SciMLBase.successful_retcode(sol_nl)
            u_curr = sol_nl.u # Update guess for next parameter step
            eq_forward[:, k] = u_curr
        else
            println("⚠️ No equilibrium root found at I = $(round(i_val, digits=2)).")
            eq_forward[:, k] .= NaN
            # Do not update u_curr, keep trying from the last known root
        end
    end
    
    # --- BACKWARD SWEEP ---
    eq_backward = zeros(length(state_vars), n_points)
    
    for (k, i_val) in enumerate(I_backward)
        prob = NonlinearProblem(nl_sys!, u_curr, (p_opt, get_u_input(i_val)))
        sol_nl = solve(prob, NewtonRaphson(), maxiters=1000)
        
        if SciMLBase.successful_retcode(sol_nl)
            u_curr = sol_nl.u
            eq_backward[:, k] = u_curr
        else
            eq_backward[:, k] .= NaN
        end
    end
    
    # --- Plotting ---
    for (i, var_name) in enumerate(state_vars)
        μ_v, σ_v = scaling.state[var_name]

        V_real_traj     = state_data[var_name] .* σ_v .+ μ_v
        V_forward_real  = eq_forward[i, :] .* σ_v .+ μ_v
        V_backward_real = eq_backward[i, :] .* σ_v .+ μ_v

        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1], 
                  xlabel = target_input, 
                  ylabel = var_name,
                  title = "Algebraic Equilibrium (Hysteresis Sweep)")
        
        lines!(ax, U_real, V_real_traj, 
               color = (:gray, 0.4), linewidth = 1, label = "Observed Trajectory")
               
        lines!(ax, I_forward, V_forward_real, 
               color = :red, linewidth = 2.5, label = "Forward Sweep")
               
        lines!(ax, I_backward, V_backward_real, 
               color = :blue, linewidth = 2.5, linestyle = :dash, label = "Backward Sweep")
        
        axislegend(ax, position = :rb)
        
        plot_file = joinpath(output_path, "$(exp_key)_hysteresis_algebraic_$(var_name)_vs_$(target_input).png")
        save(plot_file, fig)
        display(fig)
        push!(plot_files, plot_file)
    end
    
    return plot_files
end

function plot_quasiequilibrium(
    exp_key, output_path, sol, X, t_norm_final, 
    state_vars::Vector{String}, input_vars::Vector{String},
    state_data, input_data, scaling
)
    if isempty(input_vars)
        println("⚠️ No input variables available. Skipping.")
        return String[]
    end
    
    target_input = input_vars[1]
    
    # 1. Obtener parámetros para convertir TODO a físico desde el inicio
    μ_i, σ_i = scaling.input[target_input]
    
    # Convertimos los datos de entrada normalizados a físicos para definir el rango
    U_phys = input_data[target_input] .* σ_i .+ μ_i
    I_phys_min, I_phys_max = minimum(U_phys), maximum(U_phys)
    I_phys_pad = (I_phys_max - I_phys_min) * 0.05
    
    I_start_phys = I_phys_min - I_phys_pad
    I_end_phys   = I_phys_max + I_phys_pad

    discovered_sys = get_basis(sol)
    p_opt          = get_parameter_values(discovered_sys)
    plot_files     = String[]

    # 2. Configuración de la simulación temporal
    T_total = 2000.0 
    t_span  = (0.0, T_total)
    
    # Rampa de forzamiento en UNIDADES FÍSICAS (W/m²)
    function slow_forcing_phys(t)
        t_mid = T_total / 2
        if t <= t_mid
            return I_start_phys + (I_end_phys - I_start_phys) * (t / t_mid)
        else
            return I_end_phys - (I_end_phys - I_start_phys) * ((t - t_mid) / t_mid)
        end
    end

    # 3. Definición de la ODE en espacio físico
    function f_quasi!(du, u, p, t)
        I_val_phys = slow_forcing_phys(t)
        
        # IMPORTANTE: Si el modelo SINDy espera entradas normalizadas, 
        # aquí debemos normalizar I_val_phys ANTES de pasárselo al basis.
        I_val_norm = (I_val_phys - μ_i) / σ_i
        
        # Mantener otras entradas en su media normalizada
        u_in = [ (iv == target_input) ? I_val_norm : mean(input_data[iv]) for iv in input_vars ]
        
        du .= discovered_sys(u, p, t, u_in)
    end

    # Condición inicial (media de los estados normalizados)
    u0 = [mean(state_data[v]) for v in state_vars]
    
    println("⏳ Integrando cuasi-equilibrio con forzamiento físico [$I_start_phys, $I_end_phys]...")
    prob = ODEProblem(f_quasi!, u0, t_span, p_opt)
    sol_ode = solve(prob, Rosenbrock23(), reltol=1e-6, abstol=1e-6)

    # --- PLOTTING ---
    # El eje X ya es físico directamente desde la función de forzamiento
    I_vals_plot = [slow_forcing_phys(t) for t in sol_ode.t]
    mid_idx = findfirst(t -> t >= T_total/2, sol_ode.t)

    for (i, var_name) in enumerate(state_vars)
        μ_v, σ_v = scaling.state[var_name]
        
        # Denormalizamos el estado (Eje Y) para la visualización
        V_phys = sol_ode[i, :] .* σ_v .+ μ_v
        V_real_traj = state_data[var_name] .* σ_v .+ μ_v

        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1], 
                  xlabel = "$target_input (W/m²)", 
                  ylabel = var_name,
                  title = "Dynamic Quasi-Equilibrium (Physical Forcing)")

        # Datos observados
        lines!(ax, U_phys, V_real_traj, color = (:gray, 0.4), label = "Observed Data")
               
        # Ramas dinámica
        lines!(ax, I_vals_plot[1:mid_idx], V_phys[1:mid_idx], 
               color = :red, linewidth = 2, label = "Forward Sweep")
               
        lines!(ax, I_vals_plot[mid_idx:end], V_phys[mid_idx:end], 
               color = :blue, linewidth = 2, linestyle = :dash, label = "Backward Sweep")
        
        axislegend(ax, position = :rb)
        
        plot_file = joinpath(output_path, "$(exp_key)_quasiequilibrium_PHYS_$(var_name).png")
        save(plot_file, fig)
        display(fig)
        push!(plot_files, plot_file)
    end
    
    return plot_files
end