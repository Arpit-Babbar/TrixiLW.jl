# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh
using TrixiLW
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs,
               volume_integral = TrixiLW.VolumeIntegralFR(TrixiLW.MDRK()))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (1, 1)

# Create P4estMesh with 8 x 8 trees and 16 x 16 elements (because level = 1)
mesh = P4estMesh(trees_per_dimension, polydeg=3,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=6)

# A semidiscretization collects data structures and functions for the spatial discretization
cfl_number = 0.4
semi = SemidiscretizationHyperbolic(mesh,
                                    get_time_discretization(solver),
                                    equations,
                                    initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
# ode = semidiscretize(semi, (0.0, 1.0));
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)


callbacks = (; analysis_callback, save_solution,
               alive_callback,
            )


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-3;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );

# Print the timer summary
summary_callback()
