using Downloads: download
using Trixi
using TrixiLW

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg=6, surface_flux=flux_lax_friedrichs,
               volume_integral = TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)

default_mesh_file = joinpath(@__DIR__, "mesh_periodic_square_with_twist.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/12ce661d7c354c3d94c74b964b0f1c96/raw/8275b9a60c6e7ebbdea5fc4b4f091c47af3d5273/mesh_periodic_square_with_twist.mesh",
                                       default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file, periodicity=true)

###############################################################################
# create the semi discretization object
cfl_number = 0.5
semi = SemidiscretizationHyperbolic(mesh,
                                    time_discretization(solver),
                                    equations,
                                    initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
# ode = semidiscretize(semi, (0.0, 1.0));
lw_update = TrixiLW.semidiscretize(semi, time_discretization(solver), tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 100
# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

callbacks = (; analysis_callback, save_solution, alive_callback)

###############################################################################
# run the simulation


time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-3;
run(`clear`)
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );

summary_callback() # print the timer summary
