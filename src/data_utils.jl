using NCDatasets

"""
    find_times(t0, tend, time_vector)

Finds indices closest to t0 and tend using a non-allocating search.
"""
function find_times(t0::T, tend::T, time_vector::AbstractVector{T}) where T <: Real
     t0_index = argmin(abs(t - t0) for t in time_vector)
     tend_index = argmin(abs(t - tend) for t in time_vector)
     return t0_index, tend_index
end



"""
    normalization(vec, p)
    normalization(vec, p, scale)
"""
function normalization(vec::AbstractVector{T}, p::Union{String, Char}) where T <: Number
    method = string(p)
    if method == "z"
        m, s = mean(vec), std(vec)
        return (vec .- m) ./ s, m, s
    elseif method == "max" || method == "m"
        s = maximum(abs, vec)
        return vec ./ s, zero(eltype(vec ./ s)), s
    else
        error("Normalization type 'p' must be 'z', 'm', or 'max'")
    end
end

function normalization(vec::AbstractVector{T}, p::Union{String, Char}, scale::Number) where T <: Number
    method = string(p)
    if method == "max" || method == "m"
        return vec ./ scale, zero(eltype(vec ./ scale)), scale
    else
        error("Manual scale only supported for 'max' or 'm'")
    end
end




"""
    load_raw_data(data_path, t_start, t_end, vars) -> (t_raw, data)
 
Load time series variables from a NetCDF file over a specified time window.
 
# Arguments
- `data_path`: Path to the `.nc` file.
- `t_start`, `t_end`: Time window bounds (same units as the NetCDF time axis).
- `vars`: Variable names to load (e.g. `["H", "I", "C"]`).
 
# Returns
- `t_raw`: Raw time vector for the selected window.
- `data`: Dict mapping each variable name to its time series.
"""
function load_raw_data(
    data_path::String,
    t_start::Real,
    t_end::Real,
    vars::Vector{String}
)
    println("📂 Loading NetCDF data...")
    ds = NCDataset(data_path)
 
    tspan_idx = MyProject.find_times(t_start, t_end, vec(ds["time"][:]))
    t_raw     = vec(ds["time"][tspan_idx[1]:tspan_idx[2]])
    data      = Dict{String, Vector{Float64}}()
 
    for v in vars
        data[v] = vec(ds[v][tspan_idx[1]:tspan_idx[2]])
    end
 
    close(ds)
    return t_raw, data
end
