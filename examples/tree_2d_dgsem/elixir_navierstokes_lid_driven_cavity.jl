using OrdinaryDiffEq
using Trixi
using TrixiLW

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu() = 0.001

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(),
                                                          Prandtl=prandtl_number(),
                                                          gradient_variables=TrixiLW.GradientVariablesConservative())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                periodicity=false,
                n_cells_max=30_000) # set maximum capacity of tree data structure


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
boundary_conditions = boundary_condition_slip_wall

boundary_conditions = (; x_neg = TrixiLW.boundary_condition_slip_wall_vertical,
                       x_pos = TrixiLW.boundary_condition_slip_wall_vertical,
                       y_neg = TrixiLW.boundary_condition_slip_wall_horizontal,
                       y_pos = TrixiLW.boundary_condition_slip_wall_horizontal
                      )

boundary_conditions_parabolic = (; x_neg = boundary_condition_cavity,
                                   y_neg = boundary_condition_cavity,
                                   y_pos = boundary_condition_lid,
                                   x_pos = boundary_condition_cavity)

# A semidiscretization collects data structures and functions for the spatial discretization
cfl_number = 0.98

semi = TrixiLW.SemidiscretizationHyperbolicParabolic(mesh,
                                                  time_discretization(solver),
                                             (equations, equations_parabolic), initial_condition, solver;
                                             boundary_conditions=(boundary_conditions, boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 25.0)
lw_update = TrixiLW.semidiscretize(semi,
                                   time_discretization(solver),
                                   tspan);

summary_callback = SummaryCallback()

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

alive_callback = AliveCallback(alive_interval=100)
analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

callbacks = (
  save_solution,
  analysis_callback,
  alive_callback,
);

###############################################################################
# run the simulation

time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol)
dt_initial = 2.5e-01
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                      time_step_computation = TrixiLW.CFLBased(cfl_number),
                      # time_step_computation = TrixiLW.Adaptive(),
                      );
