# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using TrixiLW
using TrixiLW: AMRCallbackLW
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

polydeg = 4

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=polydeg, surface_flux=flux_lax_friedrichs,
   volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (1, 1)

# Create P4estMesh with 8 x 8 trees and 16 x 16 elements (because level = 1)
mesh = P4estMesh(trees_per_dimension, polydeg=polydeg,
   coordinates_min=coordinates_min, coordinates_max=coordinates_max,
   initial_refinement_level=5)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh,
   get_time_discretization(solver),
   equations,
   initial_condition_convergence_test,
   solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 5.0)
# ode = semidiscretize(semi, (0.0, 1.0));
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
alive_callback = AliveCallback(analysis_interval=1)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
   solution_variables=cons2prim)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
   base_level=4,
   med_level=5, med_threshold=0.1,
   max_level=6, max_threshold=0.6)

amr_callback = AMRCallback(semi, amr_controller,
   interval=5,
   adapt_initial_condition=false,
   adapt_initial_condition_only_refine=true)

amr_callback_lw = AMRCallbackLW(amr_callback, get_time_discretization(solver))

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
summary_callback = SummaryCallback()
callbacks = (analysis_callback, save_solution, alive_callback,
             r_callback_lw, summary_callback
            )

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1e-6
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-3;
cfl_number = 0.2
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
   time_step_computation=TrixiLW.Adaptive()
   #  time_step_computation = TrixiLW.CFLBased(cfl_number)
);

# Print the timer summary
summary_callback()
