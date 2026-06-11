using DataDrivenDiffEq
using DataDrivenSparse
using OrdinaryDiffEq


#########################################################
#                    EXPERIMENTOS TFM                   #
#########################################################


config_sindy = SindyConfig(
    threshold          = exp10.(2.10),  
    optimizer_type     = STLSQ,
    threshold_function = HardThreshold,
    maxiters           = 100000,
    digits             = 8,
    method             = StandardSindy(),
    basis_library = (h, c, b, t, hsed, a, i) -> [
        1;
        h; b; c; t; i; a;
        a*i; (h+b+hsed); log(c+3);
        h^5*(h+b+hsed)^3;
        h^3*(h+b+hsed)^2;
        h^2*(h+b+hsed)^2;
        h^4*(h+b+hsed)^3;
        hsed*h^4*(h+b+hsed)^3;
    ]
)


config_ode = ODESolverConfig(
    algorithm       = Tsit5(),
    reltol          = 1e-4,
    abstol          = 1e-4,
    maxiters        = 100000,
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
    generate_hysteresis      = true
)

experiment = ExperimentConfig(
    state_vars        = ["H", "C", "B", "T", "Hsed", "a"],
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