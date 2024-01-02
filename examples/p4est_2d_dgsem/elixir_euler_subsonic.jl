using Trixi, TrixiLW
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach01_flow(x, t, equations::CompressibleEulerEquations2D)
   # set the freestream flow parameters
   rho_freestream = 1.4
   v1 = 0.1 # This is Mach number
   v2 = 0.0
   p_freestream = 1.0

   prim = SVector(rho_freestream, v1, v2, p_freestream)
   return prim2cons(prim, equations)
 end

initial_condition = initial_condition_mach01_flow

solver = DGSEM(polydeg=4,
               surface_flux = flux_lax_friedrichs,
               surface_flux = TrixiLW.flux_hll,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

refinement_level = 0
cells_per_dimension = (2^refinement_level * 16, 2^refinement_level * 16)

function mapping2ellipse(ξ_, η_)
   ξ, η = 0.5*(ξ_+1), 0.5*(η_+1.0)
   R = 50.0 # Bigger circle
   r = 0.5  # Smaller circle
   amp = (R-r)*ξ + r
   x = amp * cos((2.0*pi) * η)
   y = amp * sin((2.0*pi) * η)
   return (x,y)
end

mesh = P4estMesh(cells_per_dimension, mapping = mapping2ellipse, polydeg = 4,
                  periodicity = (false, true))

cfl_number = 0.15
@inline function boundary_condition_subsonic_constant(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1.0)

   u_boundary = initial_condition_mach01_flow(x, t, equations)

   return Trixi.flux_hll(u_inner, u_boundary, normal_direction, equations)
end

boundary_conditions = Dict(
  :x_neg => TrixiLW.slip_wall_approximate,
  :x_pos => boundary_condition_subsonic_constant,
  )
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
 equations, initial_condition, solver, boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 200.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=100)

visualization_callback = VisualizationCallback(interval=100,
   save_initial_solution=true,
   save_final_solution=true)

save_restart = SaveRestartCallback(interval=10000,
                                   save_final_restart=true)
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

cylinder_center = [0.0, 0.0]
cylinder_radius = 0.5
amr_indicator = RadialIndicator(cylinder_center, 4.0*cylinder_radius)

@inline function mach_number(u, equations)
  rho, v1, v2, p = cons2prim(u, equations)
  return sqrt(v1^2+v2^2)/sqrt(equations.gamma * p / rho)
end
# amr_indicator = IndicatorLöhner(semi, variable=mach_number)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level = 2, med_threshold=0.0001, # It is a zero-one indicator
                                      max_level = 2, max_threshold=0.001)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=1000000,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = ( analysis_callback, alive_callback,
               # save_restart,
               save_solution,
               # visualization_callback,
               amr_callback
            );

###############################################################################
# run the simulation

time_int_tol = 1e-14
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-3;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                      time_step_computation = TrixiLW.Adaptive()
                     #  time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );
summary_callback() # print the timer summary
