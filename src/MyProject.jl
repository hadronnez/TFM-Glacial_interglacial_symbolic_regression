module MyProject

using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using DataInterpolations
using DiffEqCallbacks
using NCDatasets
using CairoMakie
using NonlinearSolve
using SciMLBase
using Statistics
using StatsBase
using FFTW
using JSON
using Dates
using MLUtils

include("utils/find_times.jl")
include("utils/minmax_normalization.jl")
include("abstractions/types.jl")
include("data/loading.jl")
include("data/preprocess.jl")
include("data/logging.jl")
include("sindy/utils.jl")
include("sindy/solvers.jl")
include("ode_solvers/ode_integrate.jl")
include("analysis/plotting.jl")
include("abstractions/dispatcher.jl")

export AbstractSindyMethod, DataFlow, SINDyResult
export StandardSindy, EnsembleSindy
export SindyConfig, ODESolverConfig, PlotConfig, ExperimentConfig
export ScalingParams, TrajectoryData, NormalizedTrajectoryData, DenormalizedTrajectoryData
export StandardSINDyResult, EnsembleSINDyResult
export find_times, minmax_normalization
export load_experiment_data, preprocess_data, save_experiment_log
export build_sindy_basis, solve_standard_sindy, solve_ensemble_sindy
export simulate
export plot_results, plot_hysteresis, plot_ensemble_statistics
export run_sindy, plot_from_experiment, run_experiment

end
