
using NCDatasets, CairoMakie

ROOT = normpath(joinpath(@__DIR__, "..", "..")) 
data_path = joinpath(ROOT, "pacco", "output", "BASE", "BASE.nc")
output_path = joinpath(ROOT, "analysis", "pacco_validation", "output")

ds = NCDataset(data_path)

ds
