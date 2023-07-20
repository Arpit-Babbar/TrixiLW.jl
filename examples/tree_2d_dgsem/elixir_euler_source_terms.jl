using Trixi
using TrixiLW

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=10_000)


cfl_number = 0.2
semi = TrixiLW.SemidiscretizationHyperbolic(mesh,
                                         get_time_discretization(solver),
                                         equations, initial_condition,
                                         solver,
                                         source_terms = source_terms_convergence_test)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)

lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)
callbacks = (;
               analysis_callback, alive_callback,
               save_solution
            )
###############################################################################
# run the simulation

time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol)
dt_initial = 1e-3
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );

summary_callback() # print the timer summary
