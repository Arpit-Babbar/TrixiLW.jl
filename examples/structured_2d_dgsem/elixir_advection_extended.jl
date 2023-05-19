using Trixi
using TrixiLW

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

# you can either use a single function to impose the BCs weakly in all
# 1*ndims == 2 directions or you can pass a tuple containing BCs for
# each direction
# Note: "boundary_condition_periodic" indicates that it is a periodic boundary and can be omitted on
#       fully periodic domains. Here, however, it is included to allow easy override during testing
boundary_conditions = boundary_condition_periodic

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.MDRK()))

# The initial condition is 2-periodic
coordinates_min = (-1.5, 1.3) # minimum coordinates (min(x), min(y))
coordinates_max = ( 0.5, 5.3) # maximum coordinates (max(x), max(y))

cells_per_dimension = (38, 74)

# Create curved mesh with 19 x 37 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
cfl_number = 0.2
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
 equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 0.1)

lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# VisualizationCallback
visualization_callback = VisualizationCallback(interval=100,
   solution_variables=cons2prim)

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy, energy_total))

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=10)

# The SaveRestartCallback allows to save a file from which a Trixi simulation can be restarted
save_restart = SaveRestartCallback(interval=100, save_final_restart=true)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = (; analysis_callback, alive_callback, save_restart,
               save_solution,
               # visualization_callback
               );

###############################################################################
# run the simulation

time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-3;

sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );
