
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


"""
    iqr_normalization(vec) -> (normalized_vec, center, scale)

Normalizes using median centering and IQR scaling.
More robust than MinMax for variables with outliers or very small ranges.
Maps typical values to roughly [-1, 1] rather than [0, 1].
Returns the normalized vector plus center and scale for inversion.
"""
function iqr_normalization(vec::AbstractVector{T}) where T <: Number
    center = median(vec)
    q75, q25 = quantile(vec, 0.75), quantile(vec, 0.25)
    scale = q75 - q25

    if scale < eps(Float64)
        scale = std(vec)
    end

    if scale < eps(Float64)
        return zeros(Float64, length(vec)), center, one(Float64)
    end

    return (vec .- center) ./ scale, center, scale
end
