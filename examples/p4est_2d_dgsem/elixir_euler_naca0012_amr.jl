using Downloads: download
using TrixiLW
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
   # set the freestream flow parameters
   gasGam = 1.4
   mach_inf = 0.85
   aoa = 10.0
   aoa *= pi / 180.0
   rho_inf = 1.0
   U_inf = 1.0
   pre_inf = rho_inf * U_inf^2 / (gasGam * mach_inf^2)

   v1 = U_inf * cos(aoa)
   v2 = U_inf * sin(aoa)

   prim = SVector(rho_inf, v1, v2, pre_inf)
   return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach3_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(u_inner, U_inner, f_inner,
   outer_cache, normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization)
   u_boundary = initial_condition_mach3_flow(x, t, equations)
   flux = Trixi.flux(u_boundary, normal_direction, equations)

   return flux
end

boundary_conditions = Dict(:Left => boundary_condition_supersonic_inflow,
   :Right => TrixiLW.boundary_condition_outflow,
   :Top => TrixiLW.boundary_condition_outflow,
   :Bottom => TrixiLW.boundary_condition_outflow,
   :AirfoilBottom => TrixiLW.slip_wall_approximate,
   :AirfoilTop => TrixiLW.slip_wall_approximate)

surface_flux = flux_lax_friedrichs

polydeg = 7
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
   alpha_max=1.0,
   alpha_min=0.001,
   alpha_smooth=true,
   variable=density_pressure)

volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(
   shock_indicator;
   volume_flux_fv=surface_flux,
   reconstruction=TrixiLW.FirstOrderReconstruction()
   #   reconstruction = TrixiLW.MUSCLHancockReconstruction()
   # reconstruction=TrixiLW.MUSCLReconstruction()
)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
   volume_integral=volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "NACA0012.inp")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/Arpit-Babbar/8e44898b95ea7edad054044aa63671e6/raw/ea3d59e435132f7fcca74d42aa1a77303452d0e6/NACA0012.inp",
   default_mesh_file)
mesh_file = default_mesh_file

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=0)

semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
   equations, initial_condition, solver, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers

tspan = (0.0, 1.0)
# ode = semidiscretize(semi, tspan)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
   save_initial_solution=true,
   save_final_solution=true,
   solution_variables=cons2prim)

amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level=3, med_threshold=0.05,
                                      max_level=5, max_threshold=0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = (;
   # analysis_callback,
   alive_callback,
   # save_solution
   amr_callback
   )

# positivity limiter necessary for this example with strong shocks. Very sensitive
# to the order of the limiter variables, pressure must come first.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-7, 1.0e-6),
   variables=(Trixi.pressure, Trixi.density))

###############################################################################
# run the simulation
time_int_tol = 1e-5
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-4;
cfl_number = 0.14
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
    time_step_computation=TrixiLW.Adaptive(),
   # time_step_computation=TrixiLW.CFLBased(cfl_number),
   limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary
