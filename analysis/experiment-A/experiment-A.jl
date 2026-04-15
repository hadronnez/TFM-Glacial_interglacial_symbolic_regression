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
using MLUtils

function find_times(t0, tend, time_vector) 
     # findmin(vector)[2] extracts only the index
     t0_index = findmin(abs.(t0 .- time_vector))[2]
     tend_index = findmin(abs.(tend .- time_vector))[2]
     return t0_index, tend_index
end

function normalization(vec, p; scale = nothing)
    if p == "z"
        m, s = mean(vec), std(vec)
        norm_vec = (vec .- m) ./ s
        return norm_vec, m, s 
    elseif p == "max"
        # Use the provided scale or find the maximum absolute value
        s = isnothing(scale) ? maximum(abs.(vec)) : scale
        norm_vec = vec ./ s
        # Return 0.0 for mean to keep the 3-value structure consistent
        return norm_vec, 0.0, s 
    else
        error("Normalization type 'p' must be 'z' or 'max'")
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

variables = ["time", "I", "C", "H"]
tspan_idx = find_times(-800e3, -0e3, vec(ds["time"][:]))

t = vec(ds["time"][tspan_idx[1]:tspan_idx[2]]) 
I = vec(ds["I"][tspan_idx[1]:tspan_idx[2]])
C = vec(ds["C"][tspan_idx[1]:tspan_idx[2]])
H = vec(ds["H"][tspan_idx[1]:tspan_idx[2]])

close(ds)

t_scale = 1e5
H_norm, H_mean, H_std = normalization(H, "max")
I_norm, I_mean, I_std = normalization(I, "max")
C_norm, C_mean, C_std = normalization(C, "max")
t_norm, _, _          = normalization(t, "max", t_scale)

X = reshape(vcat(H_norm), 1, :) 
U = reshape(vcat(I_norm, C_norm), 2, :) 

println("  → Data matrix shape: $(size(X)) (vars × time)")
println("  → Time span (normalized): $(t_norm[1]) to $(t_norm[end]), N=$(length(t_norm))")

ddprob = ContinuousDataDrivenProblem(X, t_norm, U=U)

# ─── 2. Building Basis ──────────────────────────────────────────────────────

@parameters τ
@variables i(τ) h(τ) c(τ)
u_vars = [h]            
c_vars = [i, c]         
vars = [i, h, c]        
function_library = [
    polynomial_basis(vars, 4);
]

basis = Basis(function_library, u_vars, controls=c_vars, iv = τ) 

# ─── 3. Solver ──────────────────────────────────────────────────

println("\n🚀 Solving with SR3")

sampler = DataProcessing(split = 0.8, shuffle = false, batchsize = 50, rng=MLUtils.Xoshiro(42))
λs = exp10.(-10:0.1:-1)
opt = ADMM(λs)
sol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(data_processing = sampler, digits = 4, maxiters = 15000))

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

u0_norm = [H_norm[1]]
tspan_norm = (t_norm[1], t_norm[end])
ode_prob = ODEProblem(discovered_ode!, u0_norm, tspan_norm, p_opt)

sim_sol = solve(ode_prob, Tsit5(), saveat=t_norm)

# ==========================================
#       DENORMALIZATION FOR PLOTTING
# ==========================================
sim_H_real = sim_sol[1, :] .* H_std .+ H_mean
deriv_scale = H_std / t_scale
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

# Top Left: Time vs H
ax1 = Axis(fig[1, 1], title = "State Variable: H(t) [$exp_key]", xlabel = "Time", ylabel = "H")
scatterlines!(ax1, t, H, label = "Real Data (H)", color = :blue, linewidth=0.75, linestyle=:dash, markersize=4)

if sim_sol.retcode == ReturnCode.Success || sim_sol.retcode == ReturnCode.Terminated
    scatterlines!(ax1, t, sim_H_real, label = "Simulated ODE", color = :red, linewidth=0.75, linestyle=:dash, markersize=4)
else
    println("⚠️ ODE simulation failed to complete.")
    t_failed_real = sim_sol.t .* t_scale
    scatterlines!(ax1, t_failed_real, sim_H_real, label = "Simulated ODE (Failed/Truncated)", color = :orange, linewidth=0.75, linestyle=:dot, markersize=4)
end
axislegend(ax1, position = :rt)

# Bottom Left: Time vs dH/dt
ax2 = Axis(fig[2, 1], title = "Derivatives: dH/dt", xlabel = "Time", ylabel = "dH/dt")
scatterlines!(ax2, t, dX_true_real[1, :], label = "Numerical Derivative (Data)", color = :blue, linewidth=0.75, linestyle=:dash, markersize=4)
scatterlines!(ax2, t, dX_predicted_real[1, :], label = "Predicted Derivative (Model)", color = :red, linewidth=0.75, linestyle=:dash, markersize=4)
axislegend(ax2, position = :rt)

# Right Side: Phase Portrait (H vs dH/dt)
ax3 = Axis(fig[1:2, 2], title = "Phase Portrait: H vs dH/dt", xlabel = "H", ylabel = "dH/dt")

# --- TRUE PHASE SPACE (BLUE) ---
scatterlines!(ax3, H, vec(dX_true_real[1, :]), label = "True Phase Space", color = :blue, linewidth=0.9, linestyle=:dash, markersize=4)
# Highlight the labeled points by plotting larger, darker markers directly over them
scatter!(ax3, H[label_idx], vec(dX_true_real[1, :])[label_idx], color = :darkblue, markersize = 10)
# Add labels offset upwards
text!(ax3, H[label_idx], vec(dX_true_real[1, :])[label_idx], text = time_labels, fontsize = 11, color = :darkblue, offset = (6, 6), font = :bold)

# --- PREDICTED PHASE SPACE (RED) ---
scatterlines!(ax3, H, vec(dX_predicted_real[1, :]), label = "Predicted Phase Space", color = :red, linewidth=0.9, linestyle=:dash, markersize=4)
# Highlight the labeled points by plotting larger, darker markers directly over them
scatter!(ax3, H[label_idx], vec(dX_predicted_real[1, :])[label_idx], color = :darkred, markersize = 10)
# Add labels offset downwards (to prevent overlapping with the blue text)
text!(ax3, H[label_idx], vec(dX_predicted_real[1, :])[label_idx], text = time_labels, fontsize = 11, color = :darkred, offset = (6, -14), font = :bold)

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