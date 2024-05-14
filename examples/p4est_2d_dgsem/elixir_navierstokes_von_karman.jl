using Downloads: download
using TrixiLW
using TrixiLW: AnalysisSurfaceIntegral, AnalysisSurfaceIntegralViscous, slip_wall_approximate_trixi
using Trixi
using LinearAlgebra
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

Re() = 100.0
prandtl_number() = 0.72

rho_inf() = 1.0
pre_inf() = 160.714285714286
T_inf() = pre_inf()
linf() = 0.1
H() = 0.41
um() = 1.5 # Maximum velocity
u(y) = 4.0 * um() * y/H() * ( 1.0- y/H() )
mach_inf() = 0.1
mu() = 1e-3
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(),
                                                          Prandtl=prandtl_number(),
                                                          gradient_variables=TrixiLW.GradientVariablesConservative())
@inline function initial_condition_mach01_flow(x, t, equations)
   rho = rho_inf()
   pre = pre_inf()
   U_inf = u(x[2])

   v1 = U_inf
   v2 = 0.0

   prim = SVector(rho, v1, v2, pre)
   return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach01_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_subsonic(U_inner, f_inner, u_inner,
   outer_cache, normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)
   u_boundary = initial_condition_mach01_flow(x, t, equations)
   flux = Trixi.flux_hll(u_inner, u_boundary, normal_direction, equations)
   return flux
end


# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)
   # flux = Trixi.flux(u_inner, normal_direction, equations)

   # return flux
   return f_inner
end

@inline function slip_wall_approximate(U_inner, f_inner, u_inner,
   normal_direction::AbstractVector,
   x, t, dt,
   surface_flux_function,
   equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)

   u_outer = Trixi.get_reflection(u_inner, normal_direction, equations)

   flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

   return flux
end

@inline function slip_wall_approximate(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector,
   x, t, dt,
   surface_flux_function,
   equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)

   norm_ = norm(normal_direction)
   # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
   normal = normal_direction / norm_

   # rotate the internal solution state
   u_local = Trixi.rotate_to_x(u_inner, normal, equations)

   # compute the primitive variables
   rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

   # Get the solution of the pressure Riemann problem
   # See Section 6.3.3 of
   # Eleuterio F. Toro (2009)
   # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
   # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
   if v_normal <= 0.0
      sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
      a_ = 1.0 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed
      if a_ >= 0.0
         p_star = p_local * (a_)^(2.0 * equations.gamma * equations.inv_gamma_minus_one)
      else
         u_outer = TrixiLW.get_reflection(u_inner, normal_direction, equations)

         flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

         return flux
      end
   else # v_normal > 0.0
      # @show v_normal, p_local
      A = 2.0 / ((equations.gamma + 1) * rho_local)
      B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
      p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
   end

   # For the slip wall we directly set the flux as the normal velocity is zero
   return SVector(zero(eltype(u_inner)),
      p_star * normal[1],
      p_star * normal[2],
      zero(eltype(u_inner))) * norm_
end


boundary_conditions = Dict(
   :Left => boundary_condition_subsonic, # TODO - Put HLL here?
   :Right => boundary_condition_subsonic, # TODO - Put HLL here?
   :Top => TrixiLW.boundary_condition_slip_wall_horizontal,
   :Bottom => TrixiLW.boundary_condition_slip_wall_horizontal,
   :circle => slip_wall_approximate_trixi
   )
velocity_wall_bc = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
velocity_exit_bc = TrixiLW.OutflowBC((x, t, equations) -> nothing)
heat_bc = Isothermal((x, t, equations) -> T_inf())
boundary_condition_ns_wall = BoundaryConditionNavierStokesWall(velocity_wall_bc, heat_bc) # THIS IS NOT A WALL EITHER!! This is inflow. It prolly worked in NACA because nothing was happening in the boundary the
boundary_condition_ns_exit = BoundaryConditionNavierStokesWall(velocity_exit_bc, heat_bc) # TODO - This is not a wall!!

boundary_condition_ns_inflow = BoundaryConditionsNavierStokesInflow(initial_condition_mach01_flow)

boundary_conditions_parabolic = Dict(
   :Left => boundary_condition_ns_inflow,
   :Right => boundary_condition_ns_exit,
   :Top => boundary_condition_ns_wall,
   :Bottom => boundary_condition_ns_wall,
   :circle => boundary_condition_ns_wall
   )

surface_flux = flux_lax_friedrichs

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=polydeg, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = joinpath(@__DIR__, "cylinder_von_karman.inp")

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=0)

cfl_number = 0.001
semi = TrixiLW.SemidiscretizationHyperbolicParabolic(mesh,
                                                  get_time_discretization(solver),
                                             (equations, equations_parabolic), initial_condition, solver;
                                             boundary_conditions=(boundary_conditions, boundary_conditions_parabolic),
                                             initial_caches = ((;dt = ones(1)), (;)))

###############################################################################
# ODE solvers

tspan = (0.0, 10.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000

aoa = 0.0*pi/180.0

# TODO - Fix this. IT MUST be passed as a proper function for AMR. It requires
# boundary_conditions to be a part of the cache object though
indices = cache -> semi.boundary_conditions.boundary_indices[1]
drag_force = AnalysisSurfaceIntegral(indices, TrixiLW.DragForcePressure(
   aoa, rho_inf(), u(centre()), linf()))
lift_force = AnalysisSurfaceIntegral(indices, TrixiLW.LiftForcePressure(
   aoa, rho_inf(), u(centre()), linf()))
drag_force_viscous = AnalysisSurfaceIntegralViscous(indices, TrixiLW.DragForceViscous(
   aoa, rho_inf(), u(centre()), linf()))
lift_force_viscous = AnalysisSurfaceIntegralViscous(indices, TrixiLW.LiftForceViscous(
   aoa, rho_inf(), u(centre()), linf()))

analysis_callback = AnalysisCallback(semi, interval=5000,
                                     analysis_errors = Symbol[],
                                     output_directory = "analysis_results", save_analysis = true,
                                     analysis_integrals = (
                                       drag_force, lift_force, drag_force_viscous, lift_force_viscous,
                                       TrixiLW.SaveSurfacePrimitives(indices)))

alive_callback = AliveCallback(analysis_interval=1000)

save_solution = SaveSolutionCallback(interval=5000,
   save_initial_solution=true,
   save_final_solution=true,
   solution_variables=cons2prim)

visualization_callback = VisualizationCallback(interval=200,
   save_initial_solution=true,
   save_final_solution=true)

callbacks = (
   analysis_callback,
   alive_callback,
   # visualization_callback,
   save_solution
   )

# positivity limiter necessary for this example with strong shocks. Very sensitive
# to the order of the limiter variables, pressure must come first.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-7, 1.0e-6),
   variables=(Trixi.pressure, Trixi.density))

###############################################################################
# run the simulation
time_int_tol = 1e-10
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-6;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
   time_step_computation=TrixiLW.Adaptive(),
   # time_step_computation=TrixiLW.CFLBased(cfl_number),
   # limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary
