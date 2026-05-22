using Pkg

env_path = joinpath(@__DIR__, "sindy")
Pkg.activate(env_path)
Pkg.instantiate()

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using Revise
using MyProject

println("Setup complete.")
