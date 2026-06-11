using DataDrivenDiffEq
using DataDrivenSparse
using OrdinaryDiffEq


#########################################################
#                    EXPERIMENTOS TFM                   #
#########################################################


config_sindy = SindyConfig(
    threshold          = exp10.(1.50:0.1:2),  
    optimizer_type     = ADMM,
    threshold_function = SoftThreshold,
    maxiters           = 50000,
    digits             = 8,
    method             = StandardSindy(),
    basis_library = (h, c, b, t, i) -> [
        1;
        h; b; t; i; c;
        (h+b); log(c+3);
        h^5*(h+b)^3;
        h^3*(h+b)^2;
        h^2*(h+b)^2;
        h^4*(h+b)^3;
    ]
)


config_ode = ODESolverConfig(
    algorithm       = Tsit5(),
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
    variables                = ["H", "C", "B", "T"],
    generate_timeseries      = true,
    generate_derivatives     = true,
    generate_phase_portraits = false,
    generate_psd             = false,
    generate_hysteresis      = true
)

experiment = ExperimentConfig(
    state_vars        = ["H", "C", "B", "T"],
    input_vars        = ["I"],
    t_start           = -800e3,
    t_end             = -0e3,
    experiment_folder = "experiment-TFM",
    data_filename     = "BASE.nc",
    sindy             = config_sindy,
    ode               = config_ode,
    plots             = config_plots
)

run_experiment(experiment)