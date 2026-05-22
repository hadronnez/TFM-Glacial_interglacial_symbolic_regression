using DataDrivenDiffEq
using Dates


# ==========================================
#             ABSTRACT TYPES
# ==========================================

abstract type AbstractSindyMethod end
abstract type DataFlow end
abstract type SINDyResult end


# ==========================================
#              SINDY METHODS
# ==========================================

struct StandardSindy <: AbstractSindyMethod end

struct EnsembleSindy <: AbstractSindyMethod
    n_ensemble::Int
    subsample_fraction::Float64
    inclusion_threshold::Float64
    
    function EnsembleSindy(;
        n_ensemble::Int = 15,
        subsample_fraction::Float64 = 0.85,
        inclusion_threshold::Float64 = 0.70
    )
        return new(n_ensemble, subsample_fraction, inclusion_threshold)
    end
end

# ==========================================
#             EXPERIMENT CONFIG
# ==========================================


struct SindyConfig{M<:AbstractSindyMethod, F}
    threshold::Vector{Float64}
    optimizer_type::Any          
    threshold_function::Any      
    split::Float64
    shuffle::Bool
    maxiters::Int
    digits::Int
    method::M
    basis_library::F

    function SindyConfig(
        threshold::Vector{Float64},
        optimizer_type,
        threshold_function,
        split::Float64,
        shuffle::Bool,
        maxiters::Int,
        digits::Int,
        method::M,
        basis_library::F
    ) where {M<:AbstractSindyMethod, F}
        return new{M, F}(threshold, optimizer_type, threshold_function,
                         split, shuffle, maxiters, digits, method, basis_library)
    end
end

function SindyConfig(;
    threshold::Union{Float64, AbstractVector{Float64}}  = exp10.(-2.0:0.1:1.0),
    optimizer_type     = SR3,
    threshold_function = HardThreshold,
    split              = 0.8,
    shuffle            = false,
    maxiters           = 30000,
    digits             = 4,
    method             = StandardSindy(),
    basis_library      = nothing
)
    threshold_vec = isa(threshold, AbstractVector) ? Vector{Float64}(threshold) : [Float64(threshold)]
    
    return SindyConfig(threshold_vec, optimizer_type, threshold_function,
                       Float64(split), shuffle, Int(maxiters), Int(digits),
                       method, basis_library)
end


struct ODESolverConfig
    algorithm::Any
    reltol::Float64
    abstol::Float64
    maxiters::Int
    saveat::Union{Vector, Nothing}
    variable_bounds::Dict{String, Tuple{Float64, Float64}}

    function ODESolverConfig(;
        algorithm       = Rosenbrock23(),
        reltol          = 1e-5,
        abstol          = 1e-5,
        maxiters        = 10_000,
        saveat          = nothing,
        variable_bounds = Dict{String, Tuple{Float64, Float64}}()
    )
        return new(algorithm, Float64(reltol), Float64(abstol),
                   Int(maxiters), saveat, variable_bounds)
    end
end


struct PlotConfig
    generate_timeseries::Bool
    generate_derivatives::Bool
    generate_phase_portraits::Bool
    generate_psd::Bool
    generate_hysteresis::Bool
    generate_quasiequilibrium::Bool
    variables::Vector{String}
    output_format::String

    function PlotConfig(;
        variables                 = String[],
        generate_timeseries       = true,
        generate_derivatives      = true,
        generate_phase_portraits  = true,
        generate_psd              = true,
        generate_hysteresis       = false,
        generate_quasiequilibrium = false,
        output_format             = "png"
    )
        return new(
            generate_timeseries,
            generate_derivatives,
            generate_phase_portraits,
            generate_psd,
            generate_hysteresis,
            generate_quasiequilibrium,
            variables,
            output_format
        )
    end
end


struct ExperimentConfig
    state_vars::Vector{String}
    input_vars::Vector{String}
    t_start::Float64
    t_end::Float64
    key::String
    data_path::String
    output_path::String
    log_file::String
    exp_dir::String
    sindy::SindyConfig
    ode::ODESolverConfig
    plots::PlotConfig

    function ExperimentConfig(;
        state_vars,
        input_vars,
        t_start::Float64,
        t_end::Float64,
        experiment_folder::String,
        sindy::SindyConfig,
        ode::ODESolverConfig,
        plots::PlotConfig     = PlotConfig(),
        data_filename::String = "BASE.nc",
        root_dir::String      = normpath(joinpath(@__DIR__, "..", ".."))
    )
        key         = "EXP_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        exp_path    = joinpath(root_dir, "experiments", experiment_folder)
        data_path   = joinpath(exp_path, "input", data_filename)
        output_path = joinpath(exp_path, "output")
        exp_dir     = joinpath(output_path, key)
        log_file    = joinpath(output_path, "experiments_log.jsonl")

        mkpath(exp_dir)

        return new(Vector{String}(state_vars), Vector{String}(input_vars),
                   t_start, t_end, key, data_path,
                   output_path, log_file, exp_dir, sindy, ode, plots)
    end
end


# ==========================================
#               DATA MANAGEMENT
# ==========================================

const ScalingParams = NamedTuple{
    (:state, :input, :time),
    Tuple{
        Dict{String, Tuple{Float64, Float64}},
        Dict{String, Tuple{Float64, Float64}},
        Tuple{Float64, Float64}
    }
}

struct TrajectoryData{T<:AbstractFloat} <: DataFlow
    X::Matrix{T}
    U::Matrix{T}
    t::Vector{T}
end

struct NormalizedTrajectoryData{T<:AbstractFloat} <: DataFlow
    X::Matrix{T}
    U::Matrix{T}
    t::Vector{T}
    scaling::ScalingParams
end

struct DenormalizedTrajectoryData{T<:AbstractFloat} <: DataFlow
    X::Matrix{T}
    U::Matrix{T}
    t::Vector{T}
    state_vars::Vector{String}
    input_vars::Vector{String}
end


# ==========================================
#              SINDy OUTPUTS
# ==========================================

struct StandardSINDyResult <: SINDyResult
    basis::Basis
    sol::DataDrivenSolution
    ddprob::Any
    parameters::Vector{Float64}
end

struct EnsembleSINDyResult <: SINDyResult
    base_result::StandardSINDyResult
    Ξ_ensemble::Vector{Float64}
    inclusion_rates::Vector{Float64}
    n_ensemble::Int
    n_success::Int
    subsample_fraction::Float64
end
