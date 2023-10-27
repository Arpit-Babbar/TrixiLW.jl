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
   dg, time_discretization, scaling_factor = 1)
   u_boundary = initial_condition_mach3_flow(x, t, equations)
   flux = Trixi.flux(u_boundary, normal_direction, equations)

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

@inline function slip_wall_approximate(u_inner, U_inner, f_inner,
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

boundary_conditions = Dict(:Left => boundary_condition_supersonic_inflow,
   :Right => boundary_condition_outflow,
   :Top => boundary_condition_outflow,
   :Bottom => boundary_condition_outflow,
   :AirfoilBottom => slip_wall_approximate,
   :AirfoilTop => slip_wall_approximate)

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
default_mesh_file = joinpath(@__DIR__, "NACA0012_N6.inp")
# default_mesh_file = joinpath(@__DIR__, "NACA0012_refined.inp")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/Arpit-Babbar/8e44898b95ea7edad054044aa63671e6/raw/ea3d59e435132f7fcca74d42aa1a77303452d0e6/NACA0012.inp",
   default_mesh_file)
mesh_file = default_mesh_file

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=0)

cfl_number = 0.1
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
   equations, initial_condition, solver, boundary_conditions=boundary_conditions,
   initial_cache=(; cfl_number, dt=zeros(1)))

###############################################################################
# ODE solvers

tspan = (0.0, 10.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors = (:residual,),
                                     output_directory = "analysis_results", save_analysis = true,
                                     analysis_integrals = (TrixiLW.RhoRes(),TrixiLW.CFLComputation(),
                                     TrixiLW.CFLComputationMax()))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
   save_initial_solution=true,
   save_final_solution=true,
   solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.15)
summary_callback = SummaryCallback()

callbacks = (
   analysis_callback,
   alive_callback,
   summary_callback
   # save_solution
   )

# positivity limiter necessary for this example with strong shocks. Very sensitive
# to the order of the limiter variables, pressure must come first.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-7, 1.0e-6),
   variables=(Trixi.pressure, Trixi.density))

###############################################################################
# run the simulation
time_int_tol = 1e-6
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-4;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
   #  time_step_computation=TrixiLW.Adaptive(),
   time_step_computation=TrixiLW.CFLBased(cfl_number),
   limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary


# plot_mesh(semi, mesh, solver, xlimits = (-.5,1.25), ylimits = (-0.25, 0.25))
