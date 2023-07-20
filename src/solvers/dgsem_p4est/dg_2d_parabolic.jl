import Trixi: create_cache
using Trixi: dot

function contravariant_fluxes(u, grad_u, i, j, element, contravariant_vectors,
   equations::AbstractEquations{2}, equations_parabolic::AbstractEquationsParabolic{2})
   Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
   Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
   (fa, ga), (fv, gv) = fluxes(u, grad_u, equations, equations_parabolic)

   # Contravariant fluxes
   cv_fa = Ja11 * fa + Ja12 * ga
   cv_ga = Ja21 * fa + Ja22 * ga
   cv_fv = Ja11 * fv + Ja12 * gv
   cv_gv = Ja21 * fv + Ja22 * gv

   cv_f = cv_fa - cv_fv
   cv_g = cv_ga - cv_gv

   # TODO - Remove other cv terms?
   return fa, ga, fv, gv, cv_f, cv_g
end

function contravariant_fluxes(u, grad_u, Ja,
   equations::AbstractEquations{2}, equations_parabolic::AbstractEquationsParabolic{2})
   (Ja11, Ja12), (Ja21, Ja22) = Ja
   (fa, ga), (fv, gv) = fluxes(u, grad_u, equations, equations_parabolic)

   # Contravariant fluxes
   cv_fa = Ja11 * fa + Ja12 * ga
   cv_ga = Ja21 * fa + Ja22 * ga
   cv_fv = Ja11 * fv + Ja12 * gv
   cv_gv = Ja21 * fv + Ja22 * gv

   cv_f = cv_fa - cv_fv
   cv_g = cv_ga - cv_gv

   # TODO - Remove other cv terms?
   return fa, ga, fv, gv, cv_f, cv_g
end