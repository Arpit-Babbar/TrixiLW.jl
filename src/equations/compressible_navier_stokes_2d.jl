using Trixi: NoSlip, Adiabatic, Isothermal, Gradient, Divergence,
             TreeMesh, polydeg, eachelement, eachnode, max_abs_speeds,
             prim2cons

import Trixi: CompressibleNavierStokesDiffusion2D, cons2prim,
              convert_derivative_to_primitive,
              BoundaryConditionNavierStokesWall,
              pressure, cons2cons,
              convert_transformed_to_primitive,
              max_dt

using StaticArrays

struct GradientVariablesConservative end

gradient_variable_transformation(::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative}) = cons2cons

pressure(u, equations::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative}
) = pressure(u, equations.equations_hyperbolic)

@inline cons2cons(u, equations::CompressibleNavierStokesDiffusion2D) = u

@inline function convert_transformed_to_primitive(u_transformed, equations::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative})
   return cons2prim(u_transformed, equations)
end

# the first argument is always the "transformed" variables.
@inline function convert_derivative_to_primitive(w, gradient_cons_vars,
   equations::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative})

   # TODO: parabolic. This is inefficient to pass in transformed variables but then transform them back.
   # We can fix this if we directly compute v1, v2, T from the entropy variables
   # w = (ρ, ρ*v1, ρ*v2, ρ*T/(gamma-1) + )
   u = cons2prim(w, equations) # calls a "modified" entropy2cons defined for CompressibleNavierStokesDiffusion2D
   ρ, v1, v2, T = u

   W = gradient_cons_vars #= W = (ρ_x,
                                   ρ_x*v1 + ρ*v1_x,
                                   ρ_x*v2 + ρ*v2_x,
                                   (ρ_x*T + ρ*T_x) / (gamma-1) + 0.5*ρ_x*(v1^2 + v2^2)
                                                   + ρ*(v1*v1_x + v2*v2_x) ) =#
   ρ_x = W[1]
   v1_x = (W[2] - ρ_x * v1) / ρ
   v2_x = (W[3] - ρ_x * v2) / ρ
   p_x = (W[4] - 0.5 * ρ_x * (v1^2 + v2^2) - ρ * (v1 * v1_x + v2 * v2_x)) * (equations.gamma - 1.0)
   T_x = (p_x - ρ_x * T) / ρ

   return SVector(ρ_x, v1_x, v2_x, T_x)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,<:Adiabatic})(
   flux_inner, u_inner, normal::AbstractVector, x, t, operator_type::Divergence,
   equations::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative},
   ::AbstractLWTimeDiscretization)
   # rho, v1, v2, _ = u_inner
   normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x, t, equations)
   v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
   _, tau_1n, tau_2n, _ = flux_inner # extract fluxes for 2nd and 3rd equations
   normal_energy_flux = v1 * tau_1n + v2 * tau_2n + normal_heat_flux
   return SVector(flux_inner[1], flux_inner[2], flux_inner[3], normal_energy_flux)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,<:Isothermal})(flux_inner, u_inner, normal::AbstractVector,
   x, t, operator_type::Gradient,
   equations::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative})
   v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
   T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t, equations)
   return SVector(u_inner[1], v1, v2, T)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,<:Isothermal})(
   flux_inner, u_inner, normal::AbstractVector,
   x, t, operator_type::Divergence,
   equations::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative})
   return flux_inner
end

# specialized BC impositions for GradientVariablesConservative.

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,<:Adiabatic})(flux_inner, u_inner, normal::AbstractVector,
   x, t, operator_type::Gradient,
   equations::CompressibleNavierStokesDiffusion2D{GradientVariablesConservative})
   v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
   rho = u_inner[1]
   p = pressure(u_inner, equations)

   # TODO - Is u_outer the correct name?

   u_outer = prim2cons((rho, v1, v2, p), equations)
   return u_outer
end


function max_dt(u, t, mesh::TreeMesh{2}, equations_parabolic::CompressibleNavierStokesDiffusion2D,
   dg, cache)
   N = polydeg(dg)
   max_diffusion = nextfloat(zero(t))
   max_lam_a = max_lam_v = nextfloat(zero(t))
   dim = 2

   equations = equations_parabolic.equations_hyperbolic
   # compute max_lam_v

   for element in eachelement(dg, cache)
      max_λ1 = max_λ2 = zero(max_lam_a)
      for j in eachnode(dg), i in eachnode(dg)
         u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
         λ1, λ2 = max_abs_speeds(u_node, equations)
         max_λ1 = max(max_λ1, λ1)
         max_λ2 = max(max_λ2, λ2)
      end
      inv_jacobian = cache.elements.inverse_jacobian[element]
      max_lam_a = max(max_lam_a, inv_jacobian * (max_λ1 + max_λ2))
   end

   mu = equations_parabolic.mu
   Pr = equations_parabolic.Pr
   kappa = equations_parabolic.kappa

   for element in eachelement(dg, cache)
      inv_jacobian = cache.elements.inverse_jacobian[element]
      for j in eachnode(dg), i in eachnode(dg)
         u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
         rho = u_node[1]
         lam_v = inv_jacobian^2 * max(4.0 * mu / (3.0 * rho), mu * kappa / (rho * Pr))
         max_lam_v = max(max_lam_v, lam_v)
      end
   end

   max_lam_v *= 2.0 * (2.0 * N + 1.0)

   dt = 1.0 / ((max_lam_a + max_lam_v) * (dim * (2.0 * N + 1.0)))

   return dt


   # CGSEM style
   # mu = equations_parabolic.mu # dynamic viscosity
   # dim = 2
   # N = polydeg(dg)
   # for element in eachelement(dg, cache)
   #   for j in eachnode(dg), i in eachnode(dg)
   #     rho = u[1, i, j, element]
   #     nu = mu / rho # kinematic viscosity
   #     inv_jacobian = cache.elements.inverse_jacobian[element]
   #     max_diffusion = max(max_diffusion, inv_jacobian^2 * nu)
   #   end
   # end

   # return 1/(N^4 * max_diffusion)
end