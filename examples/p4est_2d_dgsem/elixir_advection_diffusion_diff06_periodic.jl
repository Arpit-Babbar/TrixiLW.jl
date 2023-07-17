using Trixi, TrixiLW
using Plots

###############################################################################
# semidiscretization of the linear advection-diffusion equation

diffusivity() = 1e-6
advection_velocity = (1.5, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t,
   equation::LinearScalarAdvectionEquation2D)
   # Store translated coordinate for easy use of exact solution
   x_trans = x - equation.advection_velocity * t

   nu = diffusivity()
   c = 1.0
   A = 0.5
   L = 2
   f = 1/L
   omega = 2 * pi * f
   scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
   return SVector(scalar)
end

initial_condition = initial_condition_diffusive_convergence_test

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=1, surface_flux=flux_lax_friedrichs,
   volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
   polydeg=1, initial_refinement_level=4,
   coordinates_min=coordinates_min, coordinates_max=coordinates_max,
   periodicity=true)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = TrixiLW.SemidiscretizationHyperbolicParabolic(mesh,
   get_time_discretization(solver),
   (equations, equations_parabolic),
   initial_condition, solver,
   initial_caches=((; dt=zeros(1)), (;)))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=analysis_interval)

visualization_callback = VisualizationCallback(interval=100,
   save_initial_solution=true,
   save_final_solution=true)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = (
   summary_callback,
   analysis_callback,
   alive_callback,
   visualization_callback
);

###############################################################################
# run the simulation

cfl_number = 100.0
time_int_tol = 1e-8
tolerances = (; abstol=time_int_tol, reltol=time_int_tol)
dt_initial = 1.0
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
   # time_step_computation = TrixiLW.Adaptive()
   time_step_computation=TrixiLW.CFLBased(cfl_number)
);
summary_callback()
