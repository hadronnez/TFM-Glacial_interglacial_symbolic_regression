using OrdinaryDiffEq
using Statistics

# ── Sistema de prueba: Duffing doble pozo ─────────────────────────────────
# dx/dt = y
# dy/dt = -δ*y + x - x³ + I
# Parámetros: δ = 0.3 (amortiguamiento)
# Histéresis conocida: bifurcaciones en I ≈ ±0.38

function duffing(du, u, p, t, u_in)
    δ    = 0.3
    x, y = u
    I    = u_in[1]
    du[1] = y
    du[2] = -δ*y + x - x^3 + I
end

# ── Versión standalone de compute_hysteresis_sweeps ───────────────────────
function compute_hysteresis_sweeps_test(; n_points = 50, t_end = 200.0)

    I_min, I_max = -0.6, 0.6

    I_forward  = range(I_min, I_max; length = n_points)
    I_backward = range(I_max, I_min; length = n_points)

    function find_eq(u0, i_val)
        prob = ODEProblem(
            (du, u, p, t) -> duffing(du, u, p, t, [i_val]),
            u0, (0.0, t_end), nothing
        )
        s = solve(prob, Tsit5(), save_everystep=false, saveat=[t_end],
                  abstol=1e-8, reltol=1e-8)
        return Array(s)[:, end]
    end

    H_fwd = zeros(n_points)
    H_bwd = zeros(n_points)

    u_curr = [-1.0, 0.0]   # arranca en el pozo negativo

    for (q, lq) in enumerate(I_forward)
        println("Forward $q / $n_points  I = $(round(lq, digits=3))")
        u_curr   = find_eq(u_curr, lq)
        H_fwd[q] = u_curr[1]   # guardamos x
    end

    for (q, lq) in enumerate(I_backward)
        println("Backward $q / $n_points  I = $(round(lq, digits=3))")
        u_curr   = find_eq(u_curr, lq)
        H_bwd[q] = u_curr[1]
    end

    return collect(I_forward), collect(I_backward), H_fwd, H_bwd
end

# ── Ejecutar y pintar ──────────────────────────────────────────────────────
I_fwd, I_bwd, H_fwd, H_bwd = compute_hysteresis_sweeps_test()

using CairoMakie
fig = Figure(size = (600, 400))
ax  = Axis(fig[1,1], xlabel = "Forcing (I)", ylabel = "x equilibrium")
lines!(ax, I_fwd, H_fwd, color = :blue,  label = "Forward")
lines!(ax, I_bwd, H_bwd, color = :red,   linestyle = :dash, label = "Backward")
axislegend(ax)
display(fig)