import Trixi: BoundaryConditionDirichlet, CompressibleEulerEquations2D
import Trixi
using SimpleUnPack

# Dirichlet-type boundary condition for use with TreeMesh or StructuredMesh
@inline function (boundary_condition::BoundaryConditionDirichlet)(U_inner, F_inner, u_inner,
   outer_cache, orientation_or_normal, direction, x, t, dt, surface_flux_function, equations, dg,
   time_discretization::LW)
   @unpack nodes, weights = dg.basis
   U_outer, F_outer = outer_cache[Threads.threadid()]
   fill!(U_outer, zero(eltype(U_outer)))
   fill!(F_outer, zero(eltype(F_outer)))
   for i in eachnode(dg) # Loop over intermediary time levels
      ts = t + 0.5 * dt * (nodes[i] + 1.0)
      # get the external value of the solution
      u_boundary = boundary_condition.boundary_value_function(x, ts, equations)
      f_boundary = Trixi.flux(u_boundary, orientation_or_normal, equations)
      for n in eachvariable(equations)
         U_outer[n] += 0.5 * u_boundary[n] * weights[i]
         F_outer[n] += 0.5 * f_boundary[n] * weights[i]
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

@inline function (boundary_condition::BoundaryConditionDirichlet)(U_inner, F_inner, u_inner,
   outer_cache,
   normal_direction::AbstractVector,
   x, t, dt,
   surface_flux_function, equations, dg,
   time_discretization::LW)
   @unpack nodes, weights = dg.basis
   U_outer, F_outer = outer_cache[Threads.threadid()]
   for i in eachnode(dg) # Loop over intermediary time levels
      ts = t + 0.5 * dt * (nodes[i] + 1.0)
      # get the external value of the solution
      u_boundary = boundary_condition.boundary_value_function(x, ts, equations)
      f_boundary = Trixi.flux(u_boundary, normal_direction, equations)
      U_outer .+= u_boundary * weights[i] / 2.0
      F_outer .+= f_boundary * weights[i] / 2.0
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
   dg, time_discretization::LW)

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
   dg, time_discretization::LW)

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

is_admissible(u, equations::AbstractEquations) = @assert false "Please define specialized method to avoid bugs!"

function get_reflection(u, normal_direction, equations::CompressibleEulerEquations2D)
   rho, v1, v2, p = cons2prim(u, equations)
   norm_sqr = Trixi.norm(normal_direction)^2
   v_dot_n = (v1*normal_direction[1] + v2*normal_direction[2])/norm_sqr
   v1 = v1 - 2.0*v_dot_n*normal_direction[1]
   v2 = v2 - 2.0*v_dot_n*normal_direction[2]
   return prim2cons((rho, v1, v2, p), equations)
end
