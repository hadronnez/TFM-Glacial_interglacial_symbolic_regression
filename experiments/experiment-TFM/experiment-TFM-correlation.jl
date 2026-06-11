
using NCDatasets
using CairoMakie
using Statistics
using DataStructures
using LinearAlgebra

# ============================================================
#  1. Cálculo de métricas de colinealidad
# ============================================================

function get_colineality_advanced(H, T, C, Hsed, B, I, a, z)

    # ── Construir términos en escala física (sin normalizar previamente)
    terms = OrderedDict(
        "T"        => T,
        "I"        => I,
        "C"        => C,
        "H"        => H,
        "B"        => B,
        "a"        => a,
        "aI"       => a .* I,
        "lnC"      => log.(C .+ 3),
        "z"        => z,
        "H4z3"     => H.^4 .* z.^3,
        "H3z2"     => H.^3 .* z.^2,
        "H5z3"     => H.^5 .* z.^3,
        "H2z2"     => H.^2 .* z.^2,
        "HsedH4z3" => Hsed .* H.^4 .* z.^3
    )

    names    = collect(keys(terms))
    Θ_raw    = hcat([terms[k] for k in names]...)

    # ── Normalizar columnas de Θ por z-score (media=0, std=1)
    #    Se normaliza DESPUÉS de construir los términos, no antes
    Θ = similar(Θ_raw, Float64)
    for j in 1:size(Θ_raw, 2)
        col      = Float64.(Θ_raw[:, j])
        Θ[:, j]  = (col .- mean(col)) ./ std(col)
    end

    # ── Pearson
    pearson  = cor(Θ)

    # ── Número de condición
    κ        = cond(Θ)

    # ── Valores singulares (SVD)
    _, σ, V  = svd(Θ)

    # ── VIF: para cada columna j, regresión OLS contra el resto
    n_cols   = size(Θ, 2)
    vif      = zeros(n_cols)
    for j in 1:n_cols
        y        = Θ[:, j]
        X        = Θ[:, setdiff(1:n_cols, j)]
        β        = X \ y
        ŷ        = X * β
        ss_res   = sum((y .- ŷ).^2)
        ss_tot   = sum((y .- mean(y)).^2)
        r²       = 1.0 - ss_res / ss_tot
        vif[j]   = 1.0 / max(1.0 - r², 1e-10)
    end

    println("─────────────────────────────────────")
    println("Número de condición κ = $(round(κ, digits=1))")
    κ < 100                  && println("  ✓ Bien condicionada")
    100 <= κ < 1000          && println("  ⚠ Colinealidad moderada")
    κ >= 1000                && println("  ✗ Colinealidad severa — revisar librería")
    println("─────────────────────────────────────")
    println("VIF por término:")
    for (i, n) in enumerate(names)
        flag = vif[i] > 10 ? " ✗" : vif[i] > 5 ? " ⚠" : " ✓"
        println("  $(rpad(n, 12)) VIF = $(round(vif[i], digits=1))$flag")
    end
    println("─────────────────────────────────────")

    return (pearson=pearson, vif=vif, σ=σ, V=V, κ=κ, names=names, Θ=Θ)
end

# ============================================================
#  2. Plot Pearson
# ============================================================

function plot_colineality_matrix(pearson, names, output_path)
    n        = length(names)
    fig      = Figure(size=(900, 800), fontsize=14)
    plot_mat = fill(NaN, n, n)
    for i in 1:n, j in 1:n
        j > i && (plot_mat[i, j] = pearson[i, j])
    end
    ax = Axis(fig[1, 1],
              xticks=(1:n, names),
              yticks=(1:n, names),
              xticklabelrotation=π/4,
              aspect=AxisAspect(1))
    hm = heatmap!(ax, plot_mat, colormap=:coolwarm, colorrange=(-1, 1))
    for i in 1:n, j in 1:n
        if j > i
            val = pearson[i, j]
            text!(ax, i, j, text=string(round(val, digits=2)),
                  color=abs(val) > 0.6 ? :white : :black,
                  align=(:center, :center), fontsize=10)
        end
    end
    Colorbar(fig[1, 2], hm, label="Correlación de Pearson (r)", width=20)
    save(joinpath(output_path, "colineality_pearson.png"), fig)
    println("Pearson guardado.")
end

# ============================================================
#  3. Plot VIF
# ============================================================

function plot_vif(vif, names, output_path)
    n   = length(names)
    fig = Figure(size=(800, 500), fontsize=13)
    ax  = Axis(fig[1, 1],
               xticks=(1:n, names),
               xticklabelrotation=π/4,
               ylabel="VIF",
               title="Variance Inflation Factor por término de librería")

    colors = [v > 10 ? :tomato : v > 5 ? :orange : :steelblue for v in vif]
    barplot!(ax, 1:n, vif, color=colors)

    hlines!(ax, [5.0],  color=:orange, linestyle=:dash, label="VIF = 5 (moderado)")
    hlines!(ax, [10.0], color=:tomato,  linestyle=:dash, label="VIF = 10 (severo)")
    axislegend(ax, position=:rt)

    save(joinpath(output_path, "colineality_vif.png"), fig)
    println("VIF guardado.")
end

# ============================================================
#  4. Plot valores singulares (SVD)
# ============================================================

function plot_singular_values(σ, κ, output_path)
    fig = Figure(size=(700, 400), fontsize=13)
    ax  = Axis(fig[1, 1],
               xlabel="Índice",
               ylabel="Valor singular σ",
               title="SVD de la librería Θ  —  κ = $(round(κ, digits=1))",
               yscale=log10)

    scatterlines!(ax, 1:length(σ), σ, color=:steelblue, markersize=8)

    save(joinpath(output_path, "colineality_svd.png"), fig)
    println("SVD guardado.")
end

# ============================================================
#  Ejecución
# ============================================================

ROOT        = normpath(joinpath(@__DIR__, "..", ".."))
data_path   = joinpath(ROOT, "pacco", "output", "BASE", "BASE.nc")
output_path = joinpath(ROOT, "experiments", "experiment-TFM", "output")

ds = NCDataset(data_path)

# Variables en escala física — sin normalizar antes de construir Θ
H    = ds["H"][:]
T    = ds["T"][:]
C    = ds["C"][:]
Hsed = ds["Hsed"][:]
B    = ds["B"][:]
I    = ds["I"][:]
a    = ds["a"][:]
z    = B .+ Hsed .+ H

result = get_colineality_advanced(H, T, C, Hsed, B, I, a, z)

plot_colineality_matrix(result.pearson, result.names, output_path)
plot_vif(result.vif,                   result.names, output_path)
plot_singular_values(result.σ, result.κ,             output_path)