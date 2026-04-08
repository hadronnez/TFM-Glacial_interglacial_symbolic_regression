using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra
using NCDatasets
using CairoMakie
using Statistics
using StatsAPI
using DataInterpolations
using Dates 
using JSON  

function find_times(t0, tend, time_vector) 
     # findmin(vector)[2] extracts only the index
     t0_index = findmin(abs.(t0 .- time_vector))[2]
     tend_index = findmin(abs.(tend .- time_vector))[2]
     return t0_index, tend_index
end

function normalization(vec, p, scale = 1e5)
    if p == "z"
        m, s = mean(vec), std(vec)
        norm_vec = (vec .- m) ./ s
        # Returns the vector, the mean, and the std
        return norm_vec, m, s 
    elseif p == "max"
        norm_vec = vec ./ scale
        # Returns the vector, and 'nothing' for mean/std since they aren't used
        return norm_vec, nothing, nothing 
    end
end


ROOT = normpath(joinpath(@__DIR__, "..", ".."))
data_path = joinpath(ROOT, "analysis", "experiment-A", "input", "BASE.nc")
output_path = joinpath(ROOT, "analysis", "experiment-A", "output")
log_file = joinpath(output_path, "experiments_log.jsonl")
mkpath(output_path)

# ─── 1. Loading Data and defining DataDrivenProblem ──────────────────────────

exp_key = "EXP_" * Dates.format(now(), "yyyymmdd_HHMMSS")
println("==========================================")
println(" STARTING EXPERIMENT: $exp_key")
println("==========================================\n")

println("📂 Loading NetCDF data...")
ds = NCDataset(data_path)

variables = ["time", "I", "C", "Vol"]
tspan_idx = find_times(-800e3, -400e3, vec(ds["time"][:]))

t = vec(ds["time"][tspan_idx[1]:tspan_idx[2]]) 
I = vec(ds["I"][tspan_idx[1]:tspan_idx[2]])
C = vec(ds["C"][tspan_idx[1]:tspan_idx[2]])
V = vec(ds["Vol"][tspan_idx[1]:tspan_idx[2]])

close(ds)

t_scale = 1e5
V_norm, V_mean, V_std = normalization(V, "z")
I_norm, I_mean, I_std = normalization(I, "z")
C_norm, C_mean, C_std = normalization(C, "z")
t_norm, _, _          = normalization(t, "max", t_scale)

X = reshape(vcat(V_norm), 1, :) 
U = reshape(vcat(I_norm, C_norm), 2, :) 

println("  → Data matrix shape: $(size(X)) (vars × time)")
println("  → Time span (normalized): $(t_norm[1]) to $(t_norm[end]), N=$(length(t_norm))")

ddprob = ContinuousDataDrivenProblem(X, t_norm, U=U)

# ─── 2. Building Basis ──────────────────────────────────────────────────────

@parameters τ
@variables i(τ) v(τ) c(τ)
u_vars = [v]            
c_vars = [i, c]         
vars = [i, v, c]        
function_library=
    [
    polynomial_basis(vars, 3);
    ]

basis = Basis(function_library, u_vars, controls=c_vars, iv = τ) 

# ─── 3. Solver ──────────────────────────────────────────────────

println("\n🚀 Solving with SR3")

threshold = exp10.(range(-5, 0, length=100))
opt = SR3(threshold)
sol = solve(ddprob, basis, opt, options=DataDrivenCommonOptions(maxiters=10000))

# ─── 4. Analysis ────────────────────────────────────────────────────────────

println(get_basis(sol))
println("Parameters: $(get_parameter_values(sol.basis))\n")
println("Degrees of freedom: $(dof(sol))\n")
println("Residual sum of squares: $(rss(sol))\n")
println("r^2: $(r2(sol))\n")
println("log-likelihood: $(loglikelihood(sol))\n")
println("Summary stats: $(summarystats(sol))")

# ─── 5. Plotting ────────────────────────────────────────────────────────────

println("\n📊 Plotting results...")

discovered_sys = get_basis(sol)
p_opt = get_parameter_values(discovered_sys)

dX_predicted_norm = discovered_sys(X, p_opt, t_norm, U)
dX_true_norm = ddprob.DX 

I_interp = LinearInterpolation(I_norm, t_norm)
C_interp = LinearInterpolation(C_norm, t_norm)

function discovered_ode!(du, u, p, t_val)
    u_control = [I_interp(t_val), C_interp(t_val)]
    du .= discovered_sys(u, p, t_val, u_control)
end

u0_norm = [V_norm[1]]
tspan_norm = (t_norm[1], t_norm[end])
ode_prob = ODEProblem(discovered_ode!, u0_norm, tspan_norm, p_opt)

sim_sol = solve(ode_prob, Tsit5(), saveat=t_norm)

# ==========================================
#       DENORMALIZATION FOR PLOTTING
# ==========================================
sim_V_real = sim_sol[1, :] .* V_std .+ V_mean
deriv_scale = V_std / t_scale
dX_true_real = dX_true_norm .* deriv_scale
dX_predicted_real = dX_predicted_norm .* deriv_scale

# ==========================================
#       TEXT LABEL SETUP (Phase Space Only)
# ==========================================
# 1. Define the target times (every 50,000 years) within your time limits
target_times = range(ceil(t[1]/50000)*50000, floor(t[end]/50000)*50000, step=50000)

# 2. Find the index in your actual data that is closest to each target time
label_idx = [findmin(abs.(t .- tt))[2] for tt in target_times]

# 3. Create the text labels (e.g., "-800k")
time_labels = [string(round(Int, t[i] / 1000), "k") for i in label_idx]
# ==========================================

fig = Figure(size = (1300, 700))

# Top Left: Time vs V
ax1 = Axis(fig[1, 1], title = "State Variable: V(t) [$exp_key]", xlabel = "Time", ylabel = "V")
scatterlines!(ax1, t, V, label = "Real Data (V)", color = :blue, linewidth=0.75, linestyle=:dash, markersize=4)

if sim_sol.retcode == ReturnCode.Success || sim_sol.retcode == ReturnCode.Terminated
    scatterlines!(ax1, t, sim_V_real, label = "Simulated ODE", color = :red, linewidth=0.75, linestyle=:dash, markersize=4)
else
    println("⚠️ ODE simulation failed to complete.")
    t_failed_real = sim_sol.t .* t_scale
    scatterlines!(ax1, t_failed_real, sim_V_real, label = "Simulated ODE (Failed/Truncated)", color = :orange, linewidth=0.75, linestyle=:dot, markersize=4)
end
axislegend(ax1, position = :rt)

# Bottom Left: Time vs dV/dt
ax2 = Axis(fig[2, 1], title = "Derivatives: dV/dt", xlabel = "Time", ylabel = "dV/dt")
scatterlines!(ax2, t, dX_true_real[1, :], label = "Numerical Derivative (Data)", color = :blue, linewidth=0.75, linestyle=:dash, markersize=4)
scatterlines!(ax2, t, dX_predicted_real[1, :], label = "Predicted Derivative (Model)", color = :red, linewidth=0.75, linestyle=:dash, markersize=4)
axislegend(ax2, position = :rt)

# Right Side: Phase Portrait (V vs dV/dt)
ax3 = Axis(fig[1:2, 2], title = "Phase Portrait: V vs dV/dt", xlabel = "V", ylabel = "dV/dt")

# --- TRUE PHASE SPACE (BLUE) ---
scatterlines!(ax3, V, vec(dX_true_real[1, :]), label = "True Phase Space", color = :blue, linewidth=0.9, linestyle=:dash, markersize=4)
# Highlight the labeled points by plotting larger, darker markers directly over them
scatter!(ax3, V[label_idx], vec(dX_true_real[1, :])[label_idx], color = :darkblue, markersize = 10)
# Add labels offset upwards
text!(ax3, V[label_idx], vec(dX_true_real[1, :])[label_idx], text = time_labels, fontsize = 11, color = :darkblue, offset = (6, 6), font = :bold)

# --- PREDICTED PHASE SPACE (RED) ---
scatterlines!(ax3, V, vec(dX_predicted_real[1, :]), label = "Predicted Phase Space", color = :red, linewidth=0.9, linestyle=:dash, markersize=4)
# Highlight the labeled points by plotting larger, darker markers directly over them
scatter!(ax3, V[label_idx], vec(dX_predicted_real[1, :])[label_idx], color = :darkred, markersize = 10)
# Add labels offset downwards (to prevent overlapping with the blue text)
text!(ax3, V[label_idx], vec(dX_predicted_real[1, :])[label_idx], text = time_labels, fontsize = 11, color = :darkred, offset = (6, -14), font = :bold)

axislegend(ax3, position = :rt)

plot_file = joinpath(output_path, "$(exp_key).png")
save(plot_file, fig)
display(fig)

println("\nWould you like to log the metadata (and the results) of this experiment? (y/[n])")

save_metadata = lowercase(readline()) == "y"

if save_metadata
    metadata = Dict(
        "experiment_key" => exp_key,                   
        "data_path" => data_path,
        "tspan" => (t[1], t[end]),
        "variables" => variables,
        "basis_functions" => string(get_basis(sol)),
        "parameters" => get_parameter_values(get_basis(sol)),
        "dof" => dof(sol),
        "r2" => r2(sol),
        "solver" => string(get_algorithm(sol)),
    )
    
    open(log_file, "a") do io
        JSON.print(io, metadata)
        println(io)
    end
    println("✅ Metadata appended to registry: $log_file")
    println("✅ Plot saved to: $plot_file")

else
    rm(plot_file, force=true)
    println("Metadata and results are not saved.")
end