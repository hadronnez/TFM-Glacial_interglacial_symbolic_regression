using DataDrivenDiffEq
using DataDrivenSparse
using OrdinaryDiffEq


#########################################################
#  EXPERIMENTO A3: REFACTORIZACIÓN PARA ENSEMBLE SINDY  #
#########################################################


config_sindy = SindyConfig(
    threshold          = exp10.(1.15:0.01:1.25),  
    optimizer_type     = ADMM,
    threshold_function = SoftThreshold,
    maxiters           = 50000,
    digits             = 8,
    method             = StandardSindy(),
    basis_library = (h, c, b, t, hsed, i, a) -> [
        (h+hsed+b); 
        1;
        h; b; c; t; i; hsed; a;
        h^5*(h+hsed+b)^3;
        h^3*(h+hsed+b)^2;
        a*i;
    ]
)


config_ode = ODESolverConfig(
    algorithm       = Rosenbrock23(),
    reltol          = 1e-6,
    abstol          = 1e-6,
    maxiters        = 50000,
    variable_bounds = Dict(
        "H"    => (0.0,    2000.0),
        "Hsed" => (5.0,    5.15),
        "C"    => (200.0,  300.0),
        "T"    => (255.0,  280.0),
        "B"    => (-200.0, 470.0),
        "a"    => (0.0,    1.0)
    )
)

config_plots = PlotConfig(
    variables                = ["H", "C", "B", "T", "Hsed", "a"],
    generate_timeseries      = true,
    generate_derivatives     = true,
    generate_phase_portraits = false,
    generate_psd             = false,
    generate_hysteresis      = false
)

experiment = ExperimentConfig(
    state_vars        = ["H", "C", "B", "T", "Hsed", "a"],
    input_vars        = ["I"],
    t_start           = -800e3,
    t_end             = -0e3,
    experiment_folder = "experiment-A3",
    data_filename     = "BASE.nc",
    sindy             = config_sindy,
    ode               = config_ode,
    plots             = config_plots
)

run_experiment(experiment)