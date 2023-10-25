using TrixiLW
using Trixi

###############################################################################

equations = CompressibleEulerEquations2D(5.0/3.0)

@inline function initial_condition_astro_jet(x, t, equations::CompressibleEulerEquations2D)
  rho  = 0.5
  v1 = 0.0
  v2 = 0.0
  p = 0.4127
  prim = SVector(rho, v1, v2, p)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_astro_jet

@inline function boundary_condition_astro_jet(x, t, equations::CompressibleEulerEquations2D)
  if t > 0 && x[2] >= -0.05 && x[2] <= 0.05 && x[1] ≈ 0.0
    rho  = 5.0
    v1 = 800
    v2 = 0.0
    p  = 0.4127
  else
    rho  = 0.5
    v1 = 0.0
    v2 = 0.0
    p  = 0.4127
  end

  prim = SVector(rho, v1, v2, p)
  return prim2cons(prim, equations)
end

@inline function boundary_condition_supersonic_inflow(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)

   u_boundary = boundary_condition_astro_jet(x, t, equations)
   flux = Trixi.flux(u_boundary, normal_direction, equations)

   return flux
end


@inline function boundary_condition_outflow(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)
   # flux = Trixi.flux(u_inner, normal_direction, equations)

   # return flux
   return f_inner
end

boundary_conditions = Dict( :y_neg => boundary_condition_outflow,
                            :y_pos  => boundary_condition_outflow,
                            :x_pos  => boundary_condition_outflow,
                            :x_neg  => boundary_condition_supersonic_inflow   )
surface_flux = flux_lax_friedrichs

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=1.0,
                                            alpha_min=0.0001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(
   shock_indicator;
   volume_flux_fv=surface_flux,
   reconstruction=TrixiLW.FirstOrderReconstruction()
   # reconstruction=TrixiLW.MUSCLReconstruction()
   # reconstruction=TrixiLW.MUSCLHancockReconstruction()
)
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)


coordinates_min = (0.0, -0.5) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0,  0.5) # maximum coordinates (max(x), max(y))

mesh_file = joinpath(@__DIR__, "M2000.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/Arpit-Babbar/88216acbd3d6a257b12f3d03e03c3584/raw/db2b7a23767276a5afc4ca145ecf25aae6ea8e19/M2000.inp", mesh_file)
#                              mesh_file)isfile(mesh_file) || download("https://gist.githubusercontent.com/Arpit-Babbar/627f19ef40127b84624429b2f0b9e7f0/raw/c0f4adc777bd5eda7c38002f57f09de9277659f3/M2000_bigger.inp",
#                              mesh_file)
# mesh = P4estMesh{2}(mesh_file, initial_refinement_level = 0)

trees_per_dimension = (2, 2)

# Create P4estMesh with 8 x 8 trees and 16 x 16 elements (because level = 1)
mesh = P4estMesh(trees_per_dimension, polydeg=4,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=6, periodicity = (false, false))

# A semidiscretization collects data structures and functions for the spatial discretization
cfl_number = 0.1
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
  equations, initial_condition, solver, boundary_conditions=boundary_conditions,
  initial_cache=(; cfl_number, dt=zeros(1)))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.001
tspan = (0.0, 0.001)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=1.0,
                                          alpha_min=0.0001,
                                          alpha_smooth=false,
                                          variable=Trixi.density)

# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level=0,
#                                       med_level=3, med_threshold=0.05,
#                                       max_level=5, max_threshold=0.1)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=2,
                                      med_level=0, med_threshold=0.0003,
                                      max_level=8, max_threshold=0.003)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)
callbacks = (;analysis_callback, alive_callback,
              save_solution, amr_callback
              )

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                     variables=(Trixi.density, pressure))

###############################################################################
# run the simulation
time_int_tol = 1e-6 # Works a lot better than 1e-5!
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-7;
sol = TrixiLW.solve_lwfr(
  lw_update, callbacks, dt_initial, tolerances,
  time_step_computation = TrixiLW.Adaptive(),
  # time_step_computation=TrixiLW.CFLBased(cfl_number),
  limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary
