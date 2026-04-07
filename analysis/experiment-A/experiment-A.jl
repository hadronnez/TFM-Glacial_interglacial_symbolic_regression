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

function find_times(t0, tend, time_vector) 
     # findmin(vector)[2] extracts only the index
     t0_index = findmin(abs.(t0 .- time_vector))[2]
     tend_index = findmin(abs.(tend .- time_vector))[2]
     return t0_index, tend_index
end

ROOT = normpath(joinpath(@__DIR__, "..", ".."))
data_path = joinpath(ROOT, "analysis", "experiment-A", "input", "BASE.nc")
output_path = joinpath(ROOT, "analysis", "experiment-A", "output")
mkpath(output_path)

# ─── 1. Loading Data and defining DataDrivenProblem (only in the late Pleistocene)──────────────────────────

println("📂 Loading NetCDF data...")
ds = NCDataset(data_path)

variables = ["time", "I", "C", "Vol"]
tspan_idx = find_times(-2000e3, -1000e3, vec(ds["time"][:]))

t = vec(ds["time"][tspan_idx[1]:tspan_idx[2]]) 
I = vec(ds["I"][tspan_idx[1]:tspan_idx[2]])
C = vec(ds["C"][tspan_idx[1]:tspan_idx[2]])
V = vec(ds["Vol"][tspan_idx[1]:tspan_idx[2]])

close(ds)

# ==========================================
# 🟢 NORMALIZATION BLOCK 🟢
# ==========================================
# Scale time by 1e5 so it ranges roughly from -8.0 to 0.0
t_scale = 1e5
t_norm = t ./ t_scale

# Z-score standardization for state and control variables
V_mean, V_std = mean(V), std(V)
V_norm = (V .- V_mean) ./ V_std

I_mean, I_std = mean(I), std(I)
I_norm = (I .- I_mean) ./ I_std

C_mean, C_std = mean(C), std(C)
C_norm = (C .- C_mean) ./ C_std
# ==========================================

# Feed the NORMALIZED data into the problem matrices
X = reshape(vcat(V_norm), 1, :) 
U = reshape(vcat(I_norm, C_norm), 2, :) 

println("  → Data matrix shape: $(size(X)) (vars × time)")
println("  → Time span (normalized): $(t_norm[1]) to $(t_norm[end]), N=$(length(t_norm))")

# Note we use t_norm here!
ddprob = ContinuousDataDrivenProblem(X, t_norm, U=U)

# ─── 2. Building Basis ──────────────────────────────────────────────────────

@parameters τ
@variables i(τ) c(τ) v(τ)
u_vars = [v]            # State variables
c_vars = [i, c]         # Control variables
vars = [i, v, c]        # All variables for the basis (state + control)
basis = Basis(polynomial_basis(vars, 5), u_vars, controls=c_vars, iv = τ) 

# ─── 3. Solver ──────────────────────────────────────────────────

println("\n🚀 Solving with SR3")

# Using a more robust optimizer for sparse regression
opt = SR3(1)

# Solve with cross-validation if possible, or simple solve
sol = solve(ddprob, basis, opt)

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

# --- 1. Evaluate the predicted derivatives (in normalized space) ---
dX_predicted_norm = discovered_sys(X, p_opt, t_norm, U)
dX_true_norm = ddprob.DX 

# --- 2. Simulate the Discovered ODE (in normalized space) ---
# Interpolate using normalized variables
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
# 🔴 DENORMALIZATION FOR PLOTTING 🔴
# ==========================================
# 1. Denormalize the State Variable: V_real = (V_norm * V_std) + V_mean
sim_V_real = sim_sol[1, :] .* V_std .+ V_mean

# 2. Denormalize the Derivatives
# Chain rule: dV_norm/dt_norm = (t_scale / V_std) * dV/dt
# Therefore:  dV/dt = (V_std / t_scale) * dV_norm/dt_norm
deriv_scale = V_std / t_scale
dX_true_real = dX_true_norm .* deriv_scale
dX_predicted_real = dX_predicted_norm .* deriv_scale
# ==========================================

# --- 3. Create CairoMakie Figure ---
fig = Figure(size = (900, 700))

# Top subplot: Simulated State vs Real State
ax1 = Axis(fig[1, 1], title = "State Variable: V(t) (Real vs. Simulated)", xlabel = "Time", ylabel = "V")
lines!(ax1, t, V, label = "Real Data (V)", color = :blue, linewidth=2)

if sim_sol.retcode == ReturnCode.Success || sim_sol.retcode == ReturnCode.Terminated
    lines!(ax1, t, sim_V_real, label = "Simulated ODE", color = :red, linestyle = :dash, linewidth=2)
else
    println("⚠️ ODE simulation failed to complete (likely due to unstable model parameters blowing up to infinity).")
    # For a failed solve, we must recover the unscaled time steps that it managed to compute
    t_failed_real = sim_sol.t .* t_scale
    lines!(ax1, t_failed_real, sim_V_real, label = "Simulated ODE (Failed/Truncated)", color = :orange, linestyle = :dot, linewidth=2)
end
axislegend(ax1, position = :rt)

# Bottom subplot: Predicted Derivative vs True Derivative
ax2 = Axis(fig[2, 1], title = "Derivatives: dV/dt (Numerical vs. Model Prediction)", xlabel = "Time", ylabel = "dV/dt")
lines!(ax2, t, dX_true_real[1, :], label = "Numerical Derivative (Data)", color = :blue, linewidth=2)
lines!(ax2, t, dX_predicted_real[1, :], label = "Predicted Derivative (Model)", color = :red, linestyle = :dash, linewidth=2)
axislegend(ax2, position = :rt)

plot_file = joinpath(output_path, "solution_vs_real.png")
save(plot_file, fig)
display(fig)

println("✅ Plot saved to: $plot_file")