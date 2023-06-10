using Trixi
using TrixiLW

###############################################################################
# semidiscretization of the linear advection-diffusion equation

diffusivity() = 5e-02
advection_velocity = (1.0, 0.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs,
               volume_integral = TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-1.0, -0.5) # minimum coordinates (min(x), min(y))
coordinates_max = ( 0.0,  0.5) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                periodicity=false,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# Example setup taken from
# - Truman Ellis, Jesse Chan, and Leszek Demkowicz (2016).
#   Robust DPG methods for transient convection-diffusion.
#   In: Building bridges: connections and challenges in modern approaches
#   to numerical partial differential equations.
#   [DOI](https://doi.org/10.1007/978-3-319-41640-3_6).
function initial_condition_eriksson_johnson(x, t, equations)
  l = 4
  epsilon = diffusivity() # TODO: this requires epsilon < .6 due to sqrt
  lambda_1 = (-1 + sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
  lambda_2 = (-1 - sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
  r1 = (1 + sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
  s1 = (1 - sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
  u = exp(-l * t) * (exp(lambda_1 * x[1]) - exp(lambda_2 * x[1])) +
      cos(pi * x[2]) * (exp(s1 * x[1]) - exp(r1 * x[1])) / (exp(-s1) - exp(-r1))
  return SVector{1}(u)
end

initial_condition = initial_condition_eriksson_johnson

boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                         y_neg = BoundaryConditionDirichlet(initial_condition),
                         y_pos = BoundaryConditionDirichlet(initial_condition),
                         x_pos = boundary_condition_do_nothing)

boundary_conditions_parabolic = BoundaryConditionDirichlet(initial_condition)

cfl_number = 0.98

# A semidiscretization collects data structures and functions for the spatial discretization
semi = TrixiLW.SemidiscretizationHyperbolicParabolic(mesh,
                                                  TrixiLW.get_time_discretization(solver),
                                                  (equations, equations_parabolic),
                                                  initial_condition, solver;
                                                  boundary_conditions=(boundary_conditions,
                                                                        boundary_conditions_parabolic),
                                                  initial_caches = ((;cfl_number, dt = zeros(1)),(;cfl_number)))


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.0)
lw_update = TrixiLW.semidiscretize(semi, TrixiLW.get_time_discretization(solver), tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=analysis_interval)


# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = (
  analysis_callback,
  alive_callback,
);

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1.0e-14
tolerances = (;abstol = time_int_tol, reltol = time_int_tol)
dt_initial = 2.5e-01
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                      time_step_computation = TrixiLW.CFLBased(cfl_number),
                      # time_step_computation = TrixiLW.Adaptive(),
                      );
