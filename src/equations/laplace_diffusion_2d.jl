using Trixi: LaplaceDiffusion2D, TreeMesh, Divergence
import Trixi: max_dt, BoundaryConditionDirichlet

function max_dt(u, t, mesh::Union{TreeMesh{2}, P4estMesh{2}}, equations_parabolic::LaplaceDiffusion2D, dg, cache)
   max_diffusion = nextfloat(zero(t))
   N = polydeg(dg)
   mu = equations_parabolic.diffusivity

   # ADER style

   N = polydeg(dg)
   max_diffusion = nextfloat(zero(t))
   max_lam_a = max_lam_v = nextfloat(zero(t))
   dim = 2

   equations = equations_parabolic.equations_hyperbolic

   max_λ1, max_λ2 = max_abs_speeds(equations)
   for element in eachelement(dg, cache)
      inv_jacobian = cache.elements.inverse_jacobian[element]
      max_lam_a = max(max_lam_a, inv_jacobian * (max_λ1 + max_λ2))
   end

   mu = equations_parabolic.diffusivity

   for element in eachelement(dg, cache)
      inv_jacobian = cache.elements.inverse_jacobian[element]
      lam_v = inv_jacobian^2 * mu
      max_lam_v = max(max_lam_v, lam_v)
   end

   max_lam_v *= 2.0 * (2.0 * N + 1.0)

   dt = 1.0 / ((max_lam_a + max_lam_v) * (dim * (2.0 * N + 1.0)))

   return dt
   # CGSEM style
   # for element in eachelement(dg, cache)
   #   inv_jacobian = cache.elements.inverse_jacobian[element]
   #   max_diffusion = max(max_diffusion, inv_jacobian^2 * mu)
   # end
   # return 1/(N^4 * max_diffusion)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner, normal::AbstractVector,
   x, t, operator_type::Divergence,
   equations_parabolic::LaplaceDiffusion2D, time_discretization::AbstractLWTimeDiscretization,
   scaling_factor = 1)
   return flux_inner
end
