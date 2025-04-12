using Downloads: download
using TrixiLW
using TrixiLW.Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)

Compressible Euler setup for a double Mach reflection problem.
Involves strong shock interactions as well as steady / unsteady flow structures.
Also exercises special boundary conditions along the bottom of the domain that is a mixture of
Dirichlet and slip wall.
See Section IV c on the paper below for details.

- Paul Woodward and Phillip Colella (1984)
  The Numerical Simulation of Two-Dimensional Fluid Flows with Strong Shocks.
  [DOI: 10.1016/0021-9991(84)90142-6](https://doi.org/10.1016/0021-9991(84)90142-6)
"""

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

@inline function initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)

  if x[1] < 1 / 6 + (x[2] + 20 * t) / sqrt(3)
    phi = pi / 6
    sin_phi, cos_phi = sincos(phi)

    rho =  8
    v1  =  8.25 * cos_phi
    v2  = -8.25 * sin_phi
    p   =  116.5
  else
    rho = 1.4
    v1  = 0
    v2  = 0
    p   = 1
  end

  prim = SVector(rho, v1, v2, p)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_double_mach_reflection


@inline function boundary_condition_inflow(U_inner, F_inner, u_inner,
  outer_cache,
  orientation_or_normal, direction, x, t, dt,
  surface_flux_function, equations::CompressibleEulerEquations2D,
  dg, time_discretization, scaling_factor = 1)

  u_boundary = initial_condition_double_mach_reflection(x, t, equations)
  flux = Trixi.flux(u_boundary, orientation_or_normal, equations)

  return flux
end

# Supersonic outflow boundary condition. Solution is taken entirely from the internal state.
# See `examples/p4est_2d_dgsem/elixir_euler_forward_step_amr.jl` for complete documentation.
@inline function boundary_condition_outflow(U_inner, F_inner, u_inner,
  outer_cache,
  orientation_or_normal, direction, x, t, dt,
  surface_flux_function, equations::CompressibleEulerEquations2D,
  dg, time_discretization, scaling_factor = 1)
  # NOTE: Only for the supersonic outflow is this strategy valid
  # Calculate the boundary flux entirely from the internal solution state
  return flux(u_inner, orientation_or_normal, equations)
end

# Special mixed boundary condition type for the :Bottom of the domain.
# It is Dirichlet when x < 1/6 and a slip wall when x >= 1/6
@inline function boundary_condition_mixed_dirichlet_wall(U_inner, F_inner, u_inner,
   outer_cache,
   orientation_or_normal, direction, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)
  if x[1] < 1 / 6
    # From the BoundaryConditionDirichlet
    # get the external value of the solution
    u_outer = initial_condition_double_mach_reflection(x, t, equations)
    # Calculate boundary flux
    # Calculate boundary flux
    if iseven(direction) # U_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
    else # u_boundary is "left" of boundary, U_inner is "right" of boundary
      flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
    end
  else # x[1] >= 1 / 6
    # Use the free slip wall BC otherwise
    # flux = TrixiLW.boundary_condition_slip_wall_horizontal(U_inner, F_inner, u_inner, outer_cache,
    # orientation_or_normal, direction,
    # x, t, dt,
    # surface_flux_function,
    # equations,
    # dg, time_discretization, scaling_factor)
    flux = boundary_condition_slip_wall(u_inner, orientation_or_normal, direction, x, t, surface_flux_function, equations)
  end

  return flux
end

boundary_conditions = (; y_neg = boundary_condition_mixed_dirichlet_wall,
                         y_pos = boundary_condition_inflow,
                        #  y_pos = BoundaryConditionDirichlet(initial_condition_double_mach_reflection),
                         x_pos = boundary_condition_outflow,
                         x_neg = boundary_condition_inflow
                          # x_neg = BoundaryConditionDirichlet(initial_condition_double_mach_reflection),
                         )

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
  volume_integralFR = TrixiLW.VolumeIntegralFR(TrixiLW.LW()),
  volume_flux_fv=surface_flux,
  reconstruction = TrixiLW.FirstOrderReconstruction(),
  # reconstruction=TrixiLW.MUSCLReconstruction()
  # reconstruction=TrixiLW.MUSCLHancockReconstruction()
)

# volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW())

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
  volume_integral=volume_integral)


# Get the unstructured quad mesh from a file (downloads the file if not available locally)
coordinates_min = (0.0, 0.0)
coordinates_max = (4.0, 4.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000,
                periodicity = (false, false)) # set maximum capacity of tree data structure

cfl_number = 0.8
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
  equations, initial_condition, solver, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)

lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# visualization_callback = VisualizationCallback(interval=100,
#    save_initial_solution=true,
#    save_final_solution=true,
#    solution_variables=cons2prim)

callbacks = (analysis_callback, alive_callback, save_solution,
              # visualization_callback
              )

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                     variables=(Trixi.density, pressure))

###############################################################################
# run the simulation
time_int_tol = 1e-2

# For MDRK, initial_refinement_level = 2, time_int_tol = 1e-2 work.
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-6;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
  # time_step_computation = TrixiLW.Adaptive(),
  time_step_computation=TrixiLW.CFLBased(cfl_number),
  limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary
