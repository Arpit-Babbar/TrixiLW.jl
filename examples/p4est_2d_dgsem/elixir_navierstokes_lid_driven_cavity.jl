using Trixi, TrixiLW
using Plots

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu() = 0.001

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = TrixiLW.CompressibleNavierStokesDiffusion2D(equations, mu=mu(), Prandtl=prandtl_number(),
  gradient_variables=TrixiLW.GradientVariablesConservative())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=1, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh
trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=1, initial_refinement_level=1,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=(false, false))

function initial_condition_cavity(x, t, equations::CompressibleEulerEquations2D)
  Ma = 0.1
  rho = 1.0
  u, v = 0.0, 0.0
  p = 1.0 / (Ma^2 * equations.gamma)
  return prim2cons(SVector(rho, u, v, p), equations)
end
initial_condition = initial_condition_cavity

# BC types
velocity_bc_lid = NoSlip((x, t, equations) -> SVector(1.0, 0.0))
velocity_bc_cavity = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_lid = BoundaryConditionNavierStokesWall(velocity_bc_lid, heat_bc)
boundary_condition_cavity = BoundaryConditionNavierStokesWall(velocity_bc_cavity, heat_bc)

# define periodic boundary conditions everywhere
boundary_conditions = (;
                       x_neg = TrixiLW.boundary_condition_slip_wall_vertical,
                       x_pos = TrixiLW.boundary_condition_slip_wall_vertical,
                       y_neg = TrixiLW.boundary_condition_slip_wall_horizontal,
                       y_pos = TrixiLW.boundary_condition_slip_wall_horizontal
                      )

boundary_conditions = Dict(
                       :x_neg => TrixiLW.boundary_condition_slip_wall_vertical,
                       :x_pos => TrixiLW.boundary_condition_slip_wall_vertical,
                       :y_neg => TrixiLW.boundary_condition_slip_wall_horizontal,
                       :y_pos => TrixiLW.boundary_condition_slip_wall_horizontal
                      )

boundary_conditions_parabolic = Dict( :x_neg => boundary_condition_cavity,
                                      :y_neg => boundary_condition_cavity,
                                      :y_pos => boundary_condition_lid,
                                      :x_pos => boundary_condition_cavity)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = TrixiLW.SemidiscretizationHyperbolicParabolic(mesh,
                                                  get_time_discretization(solver),
                                             (equations, equations_parabolic), initial_condition, solver;
                                             boundary_conditions=(boundary_conditions, boundary_conditions_parabolic),
                                             initial_caches = ((;dt = ones(1)), (;)))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 25.0)
lw_update = TrixiLW.semidiscretize(semi,
                                   get_time_discretization(solver),
                                   tspan);

summary_callback = SummaryCallback()
save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

alive_callback = AliveCallback(alive_interval=100)
analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

visualization_callback = VisualizationCallback(interval=10000,
   save_initial_solution=true,
   save_final_solution=true)

callbacks = (
  save_solution,
  analysis_callback,
  alive_callback,
  visualization_callback
);

###############################################################################
# run the simulation

time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol)
dt_initial = 2.5e-01
cfl_number = 10
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                         time_step_computation = TrixiLW.CFLBased(cfl_number),
                         # time_step_computation = TrixiLW.Adaptive(),
                        );
