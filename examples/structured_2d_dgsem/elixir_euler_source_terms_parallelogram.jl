using TrixiLW
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

# Define faces for a parallelogram that looks like this
#
#             (0,1) __________ (2, 1)
#                ⟋         ⟋
#             ⟋         ⟋
#          ⟋         ⟋
# (-2,-1) ‾‾‾‾‾‾‾‾‾‾ (0,-1)

mapping(xi, eta) = SVector(xi + eta, eta)

cells_per_dimension = (20, 20)

mesh = StructuredMesh(cells_per_dimension, mapping)

cfl_number = 0.2
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, time_discretization(solver),
 equations, initial_condition, solver, source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
# ode = semidiscretize(semi, tspan);
lw_update = TrixiLW.semidiscretize(semi, time_discretization(solver), tspan);

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = (; analysis_callback, alive_callback, save_restart,
               save_solution);

###############################################################################
# run the simulation

time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-3;
sol, summary_callback = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );
summary_callback() # print the timer summary
