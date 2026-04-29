
using Pkg

env_path = joinpath(@__DIR__, "sindy")
Pkg.activate(env_path)
Pkg.instantiate()

# Link your local module folder (src)
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using Revise
using MyProject

cd("analysis//")
println("Setup complete.")