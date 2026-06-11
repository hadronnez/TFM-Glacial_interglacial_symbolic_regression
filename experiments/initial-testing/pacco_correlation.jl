using NCDatasets
using CairoMakie
using Statistics

ROOT = normpath(joinpath(@__DIR__, "..", "..")) 
data_path = joinpath(ROOT, "pacco", "output", "BASE", "BASE.nc")
output_path = joinpath(ROOT, "experiments", "initial-testing")

ds = NCDataset(data_path)

H = ds["H"][:]
T = ds["T"][:]
C = ds["C"][:]
Hsed = ds["Hsed"][:]
B = ds["B"][:]
I = ds["I"][:]
a = ds["a"][:]

struct colineality_test_variables 
    H::Vector{Float64}
    T::Vector{Float64}
    C::Vector{Float64}
    Hsed::Vector{Float64}
    B::Vector{Float64}
    I::Vector{Float64}
    a::Vector{Float64}
end

function colineality_matrix(vars::colineality_test_variables)
    data_matrix = hcat(vars.H, vars.Hsed, vars.T, vars.C, vars.B, vars.a, vars.I)
    return cor(data_matrix)
end

function plot_colineality_matrix(colineality_mat::Matrix{Float64}, variable_names::Vector{String}, output_path::String)
    fig = Figure(size = (850, 750), fontsize = 18)
    n = length(variable_names)
    
    plot_mat = copy(colineality_mat)
    for i in 1:n, j in 1:n
        if j <= i  
            plot_mat[i, j] = NaN  
        end
    end

    ax = Axis(fig[1, 1], 
              xticks = (1:n, variable_names), 
              yticks = (1:n, variable_names),
              title = "",
              aspect = AxisAspect(1))
    
    current_colormap = :coolwarm 
    hm = heatmap!(ax, plot_mat, colormap = current_colormap, colorrange = (-1, 1))
    
    for i in 1:n, j in 1:n
        if j > i
            val = round(colineality_mat[i, j], digits = 2)
            
            text_color = abs(val) > 0.5 ? :white : :black
            
            text!(ax, i, j, 
                  text = string(val), 
                  color = text_color, 
                  align = (:center, :center), 
                  fontsize = 14)
        end
    end
    
    Colorbar(fig[1, 2], hm, label = "Pearson Correlation Coefficient (r)", width = 20)
    
    target_file = joinpath(output_path, "colineality_matrix_upper_left.png")
    save(target_file, fig)
    println("¡Éxito! Gráfico guardado en: $target_file")
end

variables_ordenadas = ["I", "a", "B", "C", "T","Hsed", "H"]
matrix = colineality_matrix(colineality_test_variables(H, T, C, Hsed, B, I, a))
plot_colineality_matrix(matrix, variables_ordenadas, output_path)