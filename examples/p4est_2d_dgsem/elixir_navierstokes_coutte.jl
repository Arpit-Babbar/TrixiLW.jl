using Trixi, TrixiLW
using Plots

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

H() = 2.0 # Height of domain
T0() = 0.8
T1() = 0.85
Ma() = 0.1
R() = 1.0 # Weird but that was setup in the paper
Re() = 100.0 # Chosen corresponding to the top plate
ps() = 0.85
a1(equations) = sqrt(equations.gamma * R() * T1())
rho1(equations) = equations.gamma * ps() / (a1(equations)^2)
U1(equations) = Ma() * a1(equations)
# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu(equations) = rho1(equations) * U1(equations) * H() / Re()

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = TrixiLW.CompressibleNavierStokesDiffusion2D(equations, mu=mu(equations),
    Prandtl=prandtl_number(),
    gradient_variables=TrixiLW.GradientVariablesConservative())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
    volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (2 * H(), H()) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh
trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
    polydeg=1, initial_refinement_level=3,
    coordinates_min=coordinates_min, coordinates_max=coordinates_max,
    periodicity=(true, false))

function initial_condition_coutte(x, t, equations::CompressibleEulerEquations2D)
    y = x[2]
    gamma = equations.gamma
    T0_ = T0()
    T1_ = T1()
    p = ps()
    Pr = prandtl_number()
    R_ = R()
    U1_ = U1(equations)
    kappa = (equations.gamma * equations.inv_gamma_minus_one * mu(equations) * R_) / Pr
    cp = Pr * kappa / mu(equations)
    T = T0_ + (T1_ - T0_) * (y / H())
    T += Pr * U1_^2 / (2.0 * cp) * (y * (H() - y) / H()^2)
    a = sqrt(equations.gamma * R_ * T)
    rho = gamma * p / a^2
    u = v = 0.0
    return prim2cons(SVector(rho, u, v, p), equations)
end
initial_condition = initial_condition_coutte

# BC types
velocity_top = NoSlip((x, t, equations) -> SVector(U1(equations), 0.0))
heat_top = Isothermal((x, t, equations) -> T1())
velocity_bottom = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bottom = Isothermal((x, t, equations) -> T0())
boundary_condition_ns_top = BoundaryConditionNavierStokesWall(velocity_top, heat_top)
boundary_condition_ns_bottom = BoundaryConditionNavierStokesWall(velocity_bottom, heat_bottom)

boundary_conditions = Dict(:y_neg => TrixiLW.boundary_condition_slip_wall_horizontal,
    :y_pos => TrixiLW.boundary_condition_slip_wall_horizontal)

boundary_conditions_parabolic = Dict(:y_neg => boundary_condition_ns_bottom,
    :y_pos => boundary_condition_ns_top)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = TrixiLW.SemidiscretizationHyperbolicParabolic(mesh,
    get_time_discretization(solver),
    (equations, equations_parabolic), initial_condition, solver;
    boundary_conditions=(boundary_conditions, boundary_conditions_parabolic),
    initial_caches=((; dt=ones(1)), (;)))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 100.0)
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

visualization_callback = VisualizationCallback(interval=1000,
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
tolerances = (; abstol=time_int_tol, reltol=time_int_tol)
dt_initial = 2.5e-01
cfl_number = 10
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
    # time_step_computation=TrixiLW.CFLBased(cfl_number),
    time_step_computation = TrixiLW.Adaptive(),
);

# 6e-05
