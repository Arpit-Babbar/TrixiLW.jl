using TrixiLW
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

polydeg = 3
# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=polydeg, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = TrixiLW.SemidiscretizationHyperbolic(mesh,
get_time_discretization(solver), equations,
 initial_condition_convergence_test, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
# ode = semidiscretize(semi, (0.0, 1.0));
tspan = (0.0, 1.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=1000)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=1000,
                                     solution_variables=cons2prim)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = (;
             analysis_callback, save_solution,
            );


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks

time_int_tol = 1e-5
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-3;
# 0.9 works for 2-staged
cfl_number = TrixiLW.trixi2lw(0.71, solver)
sol, summary = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                      time_step_computation = TrixiLW.Adaptive(),
                     #  time_step_computation = TrixiLW.CFLBased(cfl_number),
                      stages = TrixiLW.TwoStaged()
                      );

# Print the timer summary
summary()
