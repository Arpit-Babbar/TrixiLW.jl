using Trixi, TrixiLW
using Plots

###############################################################################
# semidiscretization of the linear advection-diffusion equation

diffusivity() = 5.0e-2
advection_velocity = (1.0, 0.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# Example setup taken from
# - Truman Ellis, Jesse Chan, and Leszek Demkowicz (2016).
#   Robust DPG methods for transient convection-diffusion.
#   In: Building bridges: connections and challenges in modern approaches
#   to numerical partial differential equations.
#   [DOI](https://doi.org/10.1007/978-3-319-41640-3_6).
function initial_condition_eriksson_johnson(x, t, equations)
  l = 4
  epsilon = diffusivity() # TODO: this requires epsilon < .6 due to sqrt
  lambda_1 = (-1 + sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
  lambda_2 = (-1 - sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
  r1 = (1 + sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
  s1 = (1 - sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
  u = exp(-l * t) * (exp(lambda_1 * x[1]) - exp(lambda_2 * x[1])) +
      cos(pi * x[2]) * (exp(s1 * x[1]) - exp(r1 * x[1])) / (exp(-s1) - exp(-r1))
  return SVector{1}(u)
end

initial_condition = initial_condition_eriksson_johnson

boundary_conditions = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition),
                           :y_neg => BoundaryConditionDirichlet(initial_condition),
                           :y_pos => BoundaryConditionDirichlet(initial_condition),
                           :x_pos => boundary_condition_do_nothing)

boundary_conditions_parabolic = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition),
                                     :x_pos => BoundaryConditionDirichlet(initial_condition),
                                     :y_neg => BoundaryConditionDirichlet(initial_condition),
                                     :y_pos => BoundaryConditionDirichlet(initial_condition))

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=1, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-1.0, -0.5)
coordinates_max = ( 0.0,  0.5)

# This maps the domain [-1, 1]^2 to [-1, 0] x [-0.5, 0.5] while also
# introducing a curved warping to interior nodes.
function mapping(xi, eta)
    x = xi  + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return SVector(0.5 * (1 + x) - 1, 0.5 * y)
end

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=1, initial_refinement_level=4,
                 mapping=mapping, periodicity=(false, false))

# A semidiscretization collects data structures and functions for the spatial discretization
semi = TrixiLW.SemidiscretizationHyperbolicParabolic(mesh,
   get_time_discretization(solver),
   (equations, equations_parabolic),
   initial_condition, solver,
   initial_caches=((; dt=zeros(1)), (;)),
   boundary_conditions = (boundary_conditions, boundary_conditions_parabolic))


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

visualization_callback = VisualizationCallback(interval=300,
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
cfl_number = 10000
time_int_tol = 1e-8
tolerances = (; abstol=time_int_tol, reltol=time_int_tol)
dt_initial = 1.0
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
   # time_step_computation = TrixiLW.Adaptive()
   time_step_computation=TrixiLW.CFLBased(cfl_number)
);
summary_callback()
