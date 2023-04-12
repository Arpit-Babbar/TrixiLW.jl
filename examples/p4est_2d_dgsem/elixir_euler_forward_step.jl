using Downloads: download
using TrixiLW
using Trixi

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

@inline function boundary_condition_inflow_lo(U_inner, f_inner, u_inner,
  outer_cache,
  normal_direction::AbstractVector, x, t, dt,
  surface_flux_function, equations::CompressibleEulerEquations2D,
  dg, time_discretization)
  u_outer = initial_condition_mach3_flow(x, t, equations)
  flux = Trixi.flux(u_outer, normal_direction, equations)

  return flux
end

# Outflow boundary condition.
# FIXME: For now only works for supersonic outflow where all values come from the internal state.
# The bones are here for the subsonic outflow as well. One simply needs to pass the reference pressure
# to set it from the outside, the rest comes from the internal solution.
# Once fixed this function should probably move to `compressible_euler_2d.jl`
# See the reference below for a discussion on inflow/outflow boundary conditions.
#
# - Jan-Reneé Carlson (2011)
#   Inflow/Outflow Boundary Conditions with Application to FUN3D.
#   [NASA TM 20110022658](https://ntrs.nasa.gov/citations/20110022658)
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function, equations::CompressibleEulerEquations2D)
  # # This would be for the general case where we need to check the magnitude of the local Mach number
  # norm_ = norm(normal_direction)
  # # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  # normal = normal_direction / norm_

  # # Rotate the internal solution state
  # u_local = Trixi.rotate_to_x(u_inner, normal, equations)

  # # Compute the primitive variables
  # rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

  # # Compute local Mach number
  # a_local = sqrt( equations.gamma * p_local / rho_local )
  # Mach_local = abs( v_normal / a_local )
  # if Mach_local <= 1.0
  #   p_local = # Set to the external reference pressure value (somehow? maybe stored in `equations`)
  # end

  # # Create the `u_surface` solution state where the local pressure is possibly set from an external value
  # prim = SVector(rho_local, v_normal, v_tangent, p_local)
  # u_boundary = prim2cons(prim, equations)
  # u_surface = Trixi.rotate_from_x(u_boundary, normal, equations)

  # Compute the flux using the appropriate mixture of internal / external solution states
  # flux = Trixi.flux(u_surface, normal_direction, equations)

  # NOTE: Only for the supersonic outflow is this strategy valid
  # Calculate the boundary flux entirely from the internal solution state
  flux = Trixi.flux(u_inner, normal_direction, equations)

  return flux
end

@inline function boundary_condition_slip_wall_horizontal(u_inner, U_inner, f_inner,
                                              normal_direction::AbstractVector,
                                              x, t, dt,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquations2D,
                                              dg, time_discretization::TrixiLW.LW)


  # f_outer = SVector(-f_inner[1], -f_inner[2], f_inner[3], -f_inner[4])
  # U_outer = SVector(U_inner[1], U_inner[2], -U_inner[3], U_inner[4])
  # u_outer = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4])
  # norm_ = Trixi.norm(normal_direction)
  # # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  # normal = normal_direction / norm_

  # flux = surface_flux_function(f_inner, f_outer, u_inner, u_outer, U_inner, U_outer,
  #                               normal_direction, equations)

  # return flux

  u_outer = Trixi.rotate_to_x_wall(u_inner, normal_direction, equations)

  flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)
end

@inline function boundary_condition_slip_wall_vertical(u_inner, U_inner, f_inner,
                                              normal_direction::AbstractVector,
                                              x, t, dt,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquations2D,
                                              dg, time_discretization::TrixiLW.LW)

  # f_outer = SVector(-f_inner[1], f_inner[2], -f_inner[3], -f_inner[4])
  # U_outer = SVector(U_inner[1], -U_inner[2], U_inner[3], U_inner[4])
  # u_outer = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])
  # norm_ = Trixi.norm(normal_direction)
  # # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  # normal = normal_direction / norm_
  # flux = surface_flux_function(f_inner, f_outer, u_inner, u_outer, U_inner, U_outer,
  #                              normal_direction, equations)

  u_outer = Trixi.rotate_to_x_wall(u_inner, normal_direction, equations)

  flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

  # return flux
end

@inline function boundary_condition_outflow(u_inner, U_inner, f_inner,
                                            normal_direction::AbstractVector, x, t, dt,
                                            surface_flux_function, equations::CompressibleEulerEquations2D,
                                            dg, timer_discretization)
  # NOTE: Only for the supersonic outflow is this strategy valid
  # Calculate the boundary flux entirely from the internal solution state
  # return flux(u_inner, normal_direction, equations)
  return f_inner
  #  Trixi.flux(u_inner, normal_direction, equations)
end



boundary_conditions = Dict( :Bottom      => boundary_condition_slip_wall_horizontal, # boundary_condition_slip_wall,
                            :Step_Front  => boundary_condition_slip_wall_vertical, # boundary_condition_slip_wall,
                            :Step_Top    => boundary_condition_slip_wall_horizontal, # boundary_condition_slip_wall,
                            :Top         => boundary_condition_slip_wall_horizontal, # boundary_condition_slip_wall,
                            :Right       => boundary_condition_outflow,
                            :Left        => boundary_condition_inflow_lo   )

volume_flux = flux_ranocha
surface_flux = flux_lax_friedrichs

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=1.0,
                                            alpha_min=0.001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
# volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
#                                                  volume_flux_dg=volume_flux,
#                                                  volume_flux_fv=surface_flux)
volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(
  shock_indicator;
  volume_flux_fv=surface_flux,
   reconstruction = TrixiLW.FirstOrderReconstruction(),
  # reconstruction = Trixi.MUSCLHancockReconstruction(),
  )
  volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(
    shock_indicator;
    volume_flux_fv=surface_flux,
    reconstruction=TrixiLW.FirstOrderReconstruction()
    # reconstruction=TrixiLW.MUSCLReconstruction()
    # reconstruction=TrixiLW.MUSCLHancockReconstruction()
 )

 # volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
 #                                                  volume_flux_dg=volume_flux,
 #                                                  volume_flux_fv=surface_flux)
 solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
    volume_integral=volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "abaqus_forward_step.inp")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/b346ee6aa5446687f128eab8b37d52a7/raw/cd1e1d43bebd8d2631a07caec45585ec8456ca4c/abaqus_forward_step.inp",
                                      default_mesh_file)
mesh_file = default_mesh_file

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=2)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval=2000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)


callbacks = (;analysis_callback, alive_callback,
                        save_solution
                        )

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
   variables=(Trixi.density, pressure))

###############################################################################
# run the simulation
time_int_tol = 1e-7
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-6;
sol, summary_callback = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
#   time_step_computation = TrixiLW.Adaptive(),
  time_step_computation=TrixiLW.CFLBased(cfl_number),
  limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary