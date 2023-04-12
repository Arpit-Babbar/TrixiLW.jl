# Channel flow around a cylinder at Mach 3
#
# Boundary conditions are supersonic Mach 3 inflow at the left portion of the domain
# and supersonic outflow at the right portion of the domain. The top and bottom of the
# channel as well as the cylinder are treated as Euler slip wall boundaries.
# This flow results in strong shock refletions / interactions as well as Kelvin-Helmholtz
# instabilities at later times as two Mach stems form above and below the cylinder.
#
# For complete details on the problem setup see Section 5.7 of the paper:
# - Jean-Luc Guermond, Murtazo Nazarov, Bojan Popov, and Ignacio Tomas (2018)
#   Second-Order Invariant Domain Preserving Approximation of the Euler Equations using Convex Limiting.
#   [DOI: 10.1137/17M1149961](https://doi.org/10.1137/17M1149961)
#
# Keywords: supersonic flow, shock capturing, AMR, unstructured curved mesh, positivity preservation, compressible Euler, 2D

using Downloads: download
using Trixi
using TrixiLW
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

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

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(U_inner, f_inner, u_inner,
  outer_cache,
  normal_direction::AbstractVector, x, t, dt,
  surface_flux_function, equations::CompressibleEulerEquations2D,
  dg, time_discretization)

  u_boundary = initial_condition_mach3_flow(x, t, equations)
  flux = Trixi.flux(u_boundary, normal_direction, equations)

  return flux
end

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition)

# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(U_inner, f_inner, u_inner,
  outer_cache,
  normal_direction::AbstractVector, x, t, dt,
  surface_flux_function, equations::CompressibleEulerEquations2D,
  dg, time_discretization)
  # flux = Trixi.flux(u_inner, normal_direction, equations)

  # return flux
  return f_inner
end

@inline function slip_wall_approximate(U_inner, f_inner, u_inner,
  outer_cache,
  normal_direction::AbstractVector,
  x, t, dt,
  surface_flux_function,
  equations::CompressibleEulerEquations2D,
  dg, time_discretization)

  # u_outer = Trixi.get_reflection(u_inner, normal_direction, equations)

  # flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

  # return flux


  # TRIXI WAY!
  # return boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function, equations)
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

boundary_conditions = Dict(:Bottom => slip_wall_approximate,
  :Circle => slip_wall_approximate,
  :Top => slip_wall_approximate,
  :Right => boundary_condition_outflow,
  :Left => boundary_condition_supersonic_inflow)

surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
  alpha_max=1.0,
  alpha_min=0.001,
  alpha_smooth=true,
  variable=density_pressure)
volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(
  shock_indicator;
  volume_flux_fv=surface_flux,
  reconstruction = TrixiLW.FirstOrderReconstruction(),
  # reconstruction=TrixiLW.MUSCLReconstruction()
  # reconstruction=TrixiLW.MUSCLHancockReconstruction()
)

# volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
#                                                  volume_flux_dg=volume_flux,
#                                                  volume_flux_fv=surface_flux)
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
  volume_integral=volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "abaqus_cylinder_in_channel.inp")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/a08f78f6b185b63c3baeff911a63f628/raw/addac716ea0541f588b9d2bd3f92f643eb27b88f/abaqus_cylinder_in_channel.inp",
  default_mesh_file)
mesh_file = default_mesh_file

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=0)

cfl_number = 0.15
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, time_discretization(solver),
  equations, initial_condition, solver, boundary_conditions=boundary_conditions,
  initial_cache=(; cfl_number, dt=zeros(1)))

###############################################################################
# ODE solvers

tspan = (0.0, 2.0)
# ode = semidiscretize(semi, tspan)
lw_update = TrixiLW.semidiscretize(semi, time_discretization(solver), tspan);

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=10)

save_solution = SaveSolutionCallback(interval=100,
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

callbacks = (; summary_callback, analysis_callback, alive_callback, save_solution,
              #  amr_callback
            )

# positivity limiter necessary for this example with strong shocks. Very sensitive
# to the order of the limiter variables, pressure must come first.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-7, 1.0e-6),
  variables=(Trixi.pressure, Trixi.density))

###############################################################################
# run the simulation
time_int_tol = 1e-6
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-6;
sol, summary_callback = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
  time_step_computation = TrixiLW.Adaptive(),
  # time_step_computation=TrixiLW.CFLBased(cfl_number),
  limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary
