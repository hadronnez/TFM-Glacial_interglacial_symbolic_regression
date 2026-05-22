
"""
    minmax_normalization(vec::AbstractVector{T}) -> (normalized_vec, min_val, range_val)

Applies a local MinMax scaling transformation to a 1D vector, mapping values strictly to [0.0, 1.0].
Returns the normalized vector along with the minimum value and range required for inversion.
"""
function minmax_normalization(vec::AbstractVector{T}) where T <: Number
    x_min, x_max = minimum(vec), maximum(vec)
    x_range = x_max - x_min
    
    if x_min == x_max
        return zeros(Float64, length(vec)), x_min, one(Float64)
    end
    
    return (vec .- x_min) ./ x_range, x_min, x_range
end