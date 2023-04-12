using TrixiLW
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_gauss

# you can either use a single function to impose the BCs weakly in all
# 2*ndims == 4 directions or you can pass a tuple containing BCs for each direction
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-5.0, -5.0)
coordinates_max = ( 5.0,  5.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=30_000,
                periodicity=false)

cfl_number = 0.2
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, solver.volume_integral.time_discretization,
 equations, initial_condition, solver,
 boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
# ode = semidiscretize(semi, tspan)
lw_update = TrixiLW.semidiscretize(semi, solver.volume_integral.time_discretization, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = (;analysis_callback, alive_callback,
              save_solution)
###############################################################################
# run the simulation

time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-1;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
