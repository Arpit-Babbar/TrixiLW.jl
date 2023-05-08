using Downloads: download
using TrixiLW
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)

Compressible Euler setup for a Mach 3 wind tunnel flow with a forward facing step.
Results in strong shock interactions as well as Kelvin-Helmholtz instabilities at later times.
See Section IV b on the paper below for details.

- Paul Woodward and Phillip Colella (1984)
  The Numerical Simulation of Two-Dimensional Fluid Flows with Strong Shocks.
  [DOI: 10.1016/0021-9991(84)90142-6](https://doi.org/10.1016/0021-9991(84)90142-6)
"""
@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
   # set the freestream flow parameters
   rho_freestream = 1.4
   v1 = 3.0
   v2 = 0.0
   p_freestream = 1.0

   prim = SVector(rho_freestream, v1, v2, p_freestream)
   return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach3_flow

@inline function boundary_condition_inflow_forward_step(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization)
   u_outer = initial_condition_mach3_flow(x, t, equations)
   flux = Trixi.flux(u_outer, normal_direction, equations)

   return flux
end

boundary_conditions = Dict(:Bottom => TrixiLW.slip_wall_approximate,
   :Step_Front => TrixiLW.slip_wall_approximate,
   :Step_Top => TrixiLW.slip_wall_approximate,
   :Top => TrixiLW.slip_wall_approximate,
   :Right => TrixiLW.boundary_condition_outflow,
   :Left => boundary_condition_inflow_forward_step)

surface_flux = flux_lax_friedrichs

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
   alpha_max=1.0,
   alpha_min=0.001,
   alpha_smooth=true,
   variable=density_pressure)
volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(
   shock_indicator;
   volume_flux_fv=surface_flux,
   # reconstruction=TrixiLW.FirstOrderReconstruction()
   # reconstruction=TrixiLW.MUSCLReconstruction()
   reconstruction=TrixiLW.MUSCLHancockReconstruction()
)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
   volume_integral=volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "abaqus_forward_step.inp")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/b346ee6aa5446687f128eab8b37d52a7/raw/cd1e1d43bebd8d2631a07caec45585ec8456ca4c/abaqus_forward_step.inp",
   default_mesh_file)
mesh_file = default_mesh_file

mesh = P4estMesh{2}(mesh_file)

semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
  equations, initial_condition, solver, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
   extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=2000,
   save_initial_solution=true,
   save_final_solution=true,
   solution_variables=cons2prim)

callbacks = (; analysis_callback, alive_callback, save_solution)

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
   variables=(Trixi.density, pressure))

###############################################################################
# run the simulation
time_int_tol = 1e-7
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-6;
cfl_number = 0.1
sol, summary_callback = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
  time_step_computation = TrixiLW.Adaptive(),
#   time_step_computation=TrixiLW.CFLBased(cfl_number),
  limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary
