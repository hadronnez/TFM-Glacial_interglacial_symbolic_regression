using DataDrivenDiffEq
using DataDrivenSparse
using OrdinaryDiffEq

#########################################################
#  EXPERIMENTO A3: REFACTORIZACIÓN PARA ENSEMBLE SINDY  #
#########################################################

config_sindy = SindyConfig(
    threshold          = exp10.(1:0.01:1.2),
    optimizer_type     = SR3,
    threshold_function = HardThreshold,
    maxiters           = 50000,
    digits             = 4,
    method             = StandardSindy(),
    basis_library = (h, i, c, b, t, hsed) -> [
        h; hsed; c; t; i; b;
        h^2; h^3; h^4; h^5;
        t^2; t^3; t^4; t^5
        h*t; h^2*t; h^3*t; h^4*t; h^5*t;
        h*hsed; h^2*hsed; h^3*hsed;
    ]

    # EnsembleSindy(
    #     n_ensemble = 5,
    #     subsample_fraction = 0.80,
    #     inclusion_threshold = 0.80
    # ),
)

config_ode = ODESolverConfig(
    algorithm       = Rosenbrock23(),
    reltol          = 1e-4,
    abstol          = 1e-4,
    maxiters        = 50000,
    variable_bounds = Dict(
        "H"    => (0.0,    2100.0),
        "Hsed" => (5.0,    5.5),
        "C"    => (200.0,  320.0),
        "T"    => (255.0,  280.0),
        "B"    => (-200.0, 470.0)
    )
)

config_plots = PlotConfig(
    variables                = ["H"],
    generate_timeseries      = false,
    generate_derivatives     = false,
    generate_phase_portraits = false,
    generate_psd             = true,
    generate_hysteresis      = false
)

experiment = ExperimentConfig(
    state_vars        = ["H", "C", "B", "T", "Hsed"],
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