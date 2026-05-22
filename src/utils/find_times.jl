
"""
    find_times(t0, tend, time_vector)

Finds indices closest to t0 and tend using a non-allocating search.
"""
function find_times(t0::T, tend::T, time_vector::AbstractVector{T}) where T <: Real
     t0_index = argmin(abs(t - t0) for t in time_vector)
     tend_index = argmin(abs(t - tend) for t in time_vector)
     return t0_index, tend_index
end

