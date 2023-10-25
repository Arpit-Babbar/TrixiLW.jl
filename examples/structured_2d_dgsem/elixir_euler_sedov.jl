using TrixiLW
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  E  = 1.0
  p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
  p0_outer = 1.0e-5 # = true Sedov setup

  # Calculate primitive variables
  rho = 1.0
  v1  = 0.0
  v2  = 0.0
  p   = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_sedov_blast_wave

# Get the DG approximation space
surface_flux = flux_lax_friedrichs
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)

volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(indicator_sc;
                                                      volume_flux_fv=surface_flux,
                                                      reconstruction = TrixiLW.FirstOrderReconstruction()
                                                      # reconstruction = TrixiLW.MUSCLHancockReconstruction()
                                                     )

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral
               )

# Get the curved quad mesh from a mapping function
# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta)
  y = eta + 0.125 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta))

  x = xi + 0.125 * (cos(0.5 * pi * xi) * cos(2 * pi * y))

  return SVector(x, y)
end

cells_per_dimension = (20, 20)

mesh = StructuredMesh(cells_per_dimension, mapping, periodicity=true)

# create the semidiscretization
cfl_number = 0.5
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
 equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)

lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=10)

save_solution = SaveSolutionCallback(interval=300,
                                     save_initial_solution=true,
                                     save_final_solution=true)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                     variables=(Trixi.density, Trixi.pressure))

callbacks = ( analysis_callback, alive_callback,   save_solution)
time_int_tol = 1e-6
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-2;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                      time_step_computation = TrixiLW.Adaptive(),
                      # time_step_computation = TrixiLW.CFLBased(cfl_number),
                      limiters = (;stage_limiter!)
                      );

summary_callback() # print the timer summary
