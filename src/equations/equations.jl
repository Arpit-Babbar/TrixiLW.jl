import Trixi: BoundaryConditionDirichlet, CompressibleEulerEquations2D
import Trixi
using SimpleUnPack
using MuladdMacro
@muladd begin
#!= no indent

# Dirichlet-type boundary condition for use with TreeMesh or StructuredMesh
@inline function (boundary_condition::BoundaryConditionDirichlet)(U_inner, F_inner, u_inner,
   outer_cache, orientation_or_normal, direction, x, t, dt, surface_flux_function, equations, dg,
   time_discretization::AbstractLWTimeDiscretization, scaling_factor = 1)
   @unpack nodes, weights = dg.basis
   U_outer, F_outer = outer_cache.F_U_cache[Threads.threadid()]
   fill!(U_outer, zero(eltype(U_outer)))
   fill!(F_outer, zero(eltype(F_outer)))
   dt_scaled = scaling_factor * dt
   for i in eachnode(dg) # Loop over intermediary time levels
      ts = t + 0.5 * dt_scaled * (nodes[i] + 1.0)
      # get the external value of the solution
      u_boundary = boundary_condition.boundary_value_function(x, ts, equations)
      f_boundary = Trixi.flux(u_boundary, orientation_or_normal, equations)
      for n in eachvariable(equations)
         U_outer[n] += 0.5 * scaling_factor * u_boundary[n] * weights[i]
         F_outer[n] += 0.5 * scaling_factor * f_boundary[n] * weights[i]
      end
   end

   u_outer = boundary_condition.boundary_value_function(x, t, equations)

   # Calculate boundary flux
   if iseven(direction) # U_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(F_inner, F_outer, U_inner, U_outer, u_inner, u_outer,
         orientation_or_normal, equations)
   else # u_boundary is "left" of boundary, U_inner is "right" of boundary
      flux = surface_flux_function(F_outer, F_inner, U_outer, U_inner, u_outer, u_inner,
         orientation_or_normal, equations)
   end

   return flux
end

@inline function time_averaged_bc_dirichlet(U_inner, F_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector,
   x, t, dt,
   surface_flux_function, equations, dg, boundary_value_function,
   time_discretization::AbstractLWTimeDiscretization,
   scaling_factor = 1)
   @unpack nodes, weights = dg.basis
   U_outer, F_outer = outer_cache.F_U_cache[Threads.threadid()]
   fill!(U_outer, zero(eltype(U_outer)))
   fill!(F_outer, zero(eltype(F_outer)))
   dt_scaled = scaling_factor * dt
   for i in eachnode(dg) # Loop over intermediary time levels
      ts = t + 0.5 * dt_scaled * (nodes[i] + 1.0)
      # get the external value of the solution
      u_boundary = boundary_value_function(x, ts, equations)
      f_boundary = Trixi.flux(u_boundary, normal_direction, equations)
      U_outer .+= u_boundary * scaling_factor * weights[i] / 2.0
      F_outer .+= f_boundary * scaling_factor * weights[i] / 2.0
   end

   # Calculate boundary flux
   flux = surface_flux_function(F_inner, F_outer, u_inner, U_outer, U_inner, U_outer,
      normal_direction, equations)

   return flux
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(U_inner, F_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector,
   x, t, dt,
   surface_flux_function, equations, dg,
   time_discretization::AbstractLWTimeDiscretization,
   scaling_factor = 1)
   @unpack nodes, weights = dg.basis
   U_outer, F_outer = outer_cache.F_U_cache[Threads.threadid()]
   fill!(U_outer, zero(eltype(U_outer)))
   fill!(F_outer, zero(eltype(F_outer)))
   dt_scaled = scaling_factor * dt
   for i in eachnode(dg) # Loop over intermediary time levels
      ts = t + 0.5 * dt_scaled * (nodes[i] + 1.0)
      # get the external value of the solution
      u_boundary = boundary_condition.boundary_value_function(x, ts, equations)
      f_boundary = Trixi.flux(u_boundary, normal_direction, equations)
      U_outer .+= u_boundary * scaling_factor * weights[i] / 2.0
      F_outer .+= f_boundary * scaling_factor * weights[i] / 2.0
   end

   # Calculate boundary flux
   flux = surface_flux_function(F_inner, F_outer, u_inner, U_outer, U_inner, U_outer,
      normal_direction, equations)

   return flux
end

@inline function boundary_condition_slip_wall_vertical(U_inner, F_inner, u_inner, outer_cache,
   orientation_or_normal, direction,
   x, t, dt,
   surface_flux_function,
   equations::CompressibleEulerEquations2D,
   dg, time_discretization::AbstractLWTimeDiscretization, scaling_factor = 1)

   F_outer = SVector(-F_inner[1], F_inner[2], -F_inner[3], -F_inner[4])
   U_outer = SVector(U_inner[1], -U_inner[2], U_inner[3], U_inner[4])
   u_outer = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])

   # Calculate boundary flux
   if iseven(direction) # U_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(F_inner, F_outer, U_inner, U_outer, u_inner, u_outer,
         orientation_or_normal, equations)
   else # u_boundary is "left" of boundary, U_inner is "right" of boundary
      flux = surface_flux_function(F_outer, F_inner, U_outer, U_inner, u_outer, u_inner,
         orientation_or_normal, equations)
   end

   return flux
end

@inline function boundary_condition_slip_wall_horizontal(U_inner, F_inner, u_inner, outer_cache,
   orientation_or_normal, direction,
   x, t, dt,
   surface_flux_function,
   equations::CompressibleEulerEquations2D,
   dg, time_discretization::AbstractLWTimeDiscretization, scaling_factor = 1)

   F_outer = SVector(-F_inner[1], -F_inner[2], F_inner[3], -F_inner[4])
   U_outer = SVector(U_inner[1], U_inner[2], -U_inner[3], U_inner[4])
   u_outer = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4])

   # Calculate boundary flux
   if iseven(direction) # U_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(F_inner, F_outer, U_inner, U_outer, u_inner, u_outer,
         orientation_or_normal, equations)
   else # u_boundary is "left" of boundary, U_inner is "right" of boundary
      flux = surface_flux_function(F_outer, F_inner, U_outer, U_inner, u_outer, u_inner,
         orientation_or_normal, equations)
   end

   return flux
end

@inline function boundary_condition_slip_wall_vertical(U_inner, F_inner, u_inner, outer_cache,
   normal_direction,
   x, t, dt,
   surface_flux_function,
   equations::CompressibleEulerEquations2D,
   dg, time_discretization::AbstractLWTimeDiscretization, scaling_factor = 1)

   F_outer = SVector(-F_inner[1], F_inner[2], -F_inner[3], -F_inner[4])
   U_outer = SVector(U_inner[1], -U_inner[2], U_inner[3], U_inner[4])
   u_outer = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])

   flux = surface_flux_function(F_inner, F_outer, u_inner, u_outer, U_inner, U_outer,
   normal_direction, equations)

   return flux
end

@inline function boundary_condition_slip_wall_horizontal(U_inner, F_inner, u_inner, outer_cache,
   normal_direction,
   x, t, dt,
   surface_flux_function,
   equations::CompressibleEulerEquations2D,
   dg, time_discretization::AbstractLWTimeDiscretization, scaling_factor = 1)

   F_outer = SVector(-F_inner[1], -F_inner[2], F_inner[3], -F_inner[4])
   U_outer = SVector(U_inner[1], U_inner[2], -U_inner[3], U_inner[4])
   u_outer = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4])

   flux = surface_flux_function(F_inner, F_outer, u_inner, u_outer, U_inner, U_outer,
      normal_direction, equations)
   return flux
end

is_admissible(u, equations::AbstractEquations) = @assert false "Please define specialized method to avoid bugs!"

@inline function get_reflection(u, normal_direction, equations::CompressibleEulerEquations2D)
   rho, v1, v2, p = cons2prim(u, equations)
   norm_sqr = Trixi.norm(normal_direction)^2
   v_dot_n = (v1*normal_direction[1] + v2*normal_direction[2])/norm_sqr
   v1 = v1 - 2.0*v_dot_n*normal_direction[1]
   v2 = v2 - 2.0*v_dot_n*normal_direction[2]
   return prim2cons((rho, v1, v2, p), equations)
end

@inline function slip_wall_approximate(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector,
   x, t, dt,
   surface_flux_function,
   equations::CompressibleEulerEquations2D,
   dg, time_discretization, scaling_factor = 1)

   u_outer = TrixiLW.get_reflection(u_inner, normal_direction, equations)

   flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

   return flux
end

@inline function slip_wall_approximate_trixi(U_inner, f_inner, u_inner,
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

# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(U_inner, f_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization , scaling_factor = 1)

   flux = Trixi.flux(u_inner, normal_direction, equations)

   return flux
end

function limit_variable_slope(eq, variable, slope, u_star_ll, u_star_rr, ue, xl, xr)
   # The MUSCL-Hancock scheme is guaranteed to be admissibility preserving if
   # slope is chosen so that
   # u_star_ll = ue + 2.0*slope*xl, u_star_rr = ue+2.0*slope*xr are admissible
   # ue is already admissible and we know we can find sequences of thetas
   # to make theta*u_star_ll+(1-theta)*ue is admissible.
   # This is equivalent to replacing u_star_ll by
   # u_star_ll = ue + 2.0*theta*s*xl.
   # Thus, we simply have to update the slope by multiplying by theta.

   # By Jensen's inequality, we can find theta's directly for the primitives

   var_star_ll, var_star_rr = variable(u_star_ll, eq), variable(u_star_rr, eq)
   var_low = variable(ue, eq)
   eps = 1e-10
   threshold = 0.1*var_low
   if var_star_ll < eps || var_star_rr < eps
      ratio_ll = abs(threshold - var_low) / (abs(var_star_ll - var_low) + 1e-13)
      ratio_rr = abs(threshold - var_low) / (abs(var_star_rr - var_low) + 1e-13)
      theta = min(min(ratio_ll, ratio_rr), 1.0)
      slope *= theta
      u_star_ll = ue + (2.0*theta)*(xl*slope)
      u_star_rr = ue + (2.0*theta)*(xr*slope)
   end
   return slope, u_star_ll, u_star_rr
end

struct SlipWallHO{Semi}
   semi::Semi
end



@inline function pre_process_boundary_condition_ho(boundary_condition,
   i_index, j_index, boundary_element,
   inner_node, inner_direction,
   boundary_index, boundary_node,
   direction,
   t, dt, equations, dg, time_discretization, scaling_factor = 1)
   @unpack volume_integral = dg
   @unpack nodes, weights = dg.basis
   dt_scaled = scaling_factor * dt

   @unpack semi = boundary_condition
   @unpack mesh, solver, cache, source_terms = semi
   @unpack element_cache, boundary_cache = cache
   @unpack outer_cache = boundary_cache
   _, F_outer = outer_cache.F_U_cache[Threads.threadid()]
   fill!(F_outer, zero(eltype(F_outer)))

   @unpack contravariant_vectors = cache.elements

   # Outward-pointing normal direction (not normalized)
   normal_direction = get_normal_direction(direction, contravariant_vectors,
      i_index, j_index, boundary_element)
   time_int_tol = NaN
   tolerances = (; abstol=time_int_tol, reltol=time_int_tol); # Dummy, doesn't matter
   @unpack u_inner_reflected, dummy_du = cache.lw_res_cache.boundary_arrays[Threads.threadid()]
   dummy_element_number = Threads.threadid() # Wow, what a hack!
   # TODO - Move u_inner_big to LWBoundariesContainer?
   u_inner_big = @view cache.element_cache.u[:,:,:,boundary_element]
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u_inner_big, equations, dg, i, j, 1)
      u_node_reflected = get_reflection(u_node, normal_direction, equations)
      set_node_vars!(u_inner_reflected, u_node_reflected, equations, dg, i, j)
   end
   correct_temporal_error = cache.temporal_errors[dummy_element_number] # This is the correct value but will be overwritten
   lw_volume_kernel_4!(dummy_du, u_inner_reflected, t, dt, tolerances, dummy_element_number, mesh,
                       have_nonconservative_terms(equations), source_terms, equations, dg, cache, 0.0)
   cache.temporal_errors[dummy_element_number] = correct_temporal_error # Fix overwritten value

   inv_jacobian = cache.elements.inverse_jacobian[i_index, j_index, boundary_element]
   fn_inner = get_node_vars(boundary_cache.fn_low, equations, dg, boundary_node, boundary_index)
   c_ll = inner_direction * dt * inv_jacobian / weights[inner_node]
   outer_cache.c_ll[Threads.threadid()] .= c_ll
   set_node_vars!(outer_cache.fn_inner[Threads.threadid()], fn_inner, equations, dg, 1)
   outer_cache.alpha[Threads.threadid()] .= get_alpha(volume_integral, boundary_element)

   F = get_flux_vars(element_cache.F, equations, dg, i_index, j_index, dummy_element_number)
   F_outer .= normal_product(F, equations, normal_direction)

   return nothing
end

@inline function pre_process_boundary_condition(boundary_condition::SlipWallHO,
   U_inner, f_inner, u_inner, i_index, j_index, boundary_element, outer_cache, normal_direction,
   x, t, dt, surface_flux, equations, dg, time_discretization, scaling_factor = 1)

   return pre_process_boundary_condition_ho(boundary_condition,
   U_inner, f_inner, u_inner, i_index, j_index, boundary_element, outer_cache, normal_direction,
   x, t, dt, surface_flux, equations, dg, time_discretization, scaling_factor)

   return nothing
end

@inline function (slip_wall_ho::SlipWallHO)(U_inner, F_inner, u_inner, outer_cache,
   normal_direction::AbstractVector, x, t, dt,
   surface_flux_function, equations::CompressibleEulerEquations2D,
   dg, time_discretization , scaling_factor = 1)
   _, F_outer = outer_cache.F_U_cache[Threads.threadid()]

   c_ll = - outer_cache.c_ll[Threads.threadid()][1]
   fn_inner = get_node_vars(outer_cache.fn_inner[Threads.threadid()], equations, dg, 1)
   alpha = outer_cache.alpha[Threads.threadid()][1]


   # return @inline slip_wall_approximate(U_inner, F_inner, u_inner,
   # outer_cache,
   # normal_direction,
   # x, t, dt,
   # surface_flux_function,
   # equations,
   # dg, time_discretization, scaling_factor)
   # TODO - Add flux correction here. Only Fn has to be corrected. You just need the
   # Jl here. Get that into the outer_cache somehow.
   u_outer = get_reflection(u_inner, normal_direction, equations)
   U_outer = get_reflection(U_inner, normal_direction, equations)
   Fn = surface_flux_function(F_inner, F_outer, u_inner, u_outer, U_inner, U_outer,
                              normal_direction, equations)
   fn = surface_flux_function(u_inner, u_outer, normal_direction, equations)
   Fn = (1.0 - alpha) * Fn + alpha * fn
   test_update = u_inner - c_ll * (Fn - fn_inner)
   low_update = u_inner - c_ll * (fn - fn_inner)
   if is_admissible(test_update, equations) == false
      Fn = zhang_shu_flux_fix(equations, u_inner, low_update, Fn, fn_inner, fn, c_ll)
   end
   return Fn

   # Don't know how to map the element time averaged flux to the correct face. Damn!
   # They are available in calc_boundary_flux! function though. They are the i_index, j_index over there.
end

end # muladd
