using TrixiLW
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

# you can either use a single function to impose the BCs weakly in all
# 2*ndims == 4 directions or you can pass a tuple containing BCs for each direction
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (x_neg=boundary_condition,
                       x_pos=boundary_condition,
                       y_neg=boundary_condition,
                       y_pos=boundary_condition)

solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)
mesh = StructuredMesh((40, 40), coordinates_min, coordinates_max, periodicity=false)

cfl_number = 0.5
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
 equations, initial_condition, solver, source_terms = source_terms_convergence_test,
 boundary_conditions = boundary_conditions, initial_cache = (;cfl_number, dt = zeros(1)))


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)

lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

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
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );
summary_callback() # print the timer summary
