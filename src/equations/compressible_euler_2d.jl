@inline function is_admissible(u, equations::CompressibleEulerEquations2D)
   prim = cons2prim(u, equations)
   return prim[1] > 0.0 && prim[4] > 0.0
end

function limit_slope(eq::CompressibleEulerEquations2D, slope, ufl, u_star_ll,
   ufr, u_star_rr, ue, xl, xr)
   slope, u_star_ll, u_star_rr = limit_variable_slope(
      eq, Trixi.density, slope, u_star_ll, u_star_rr, ue, xl, xr)

   slope, u_star_ll, u_star_rr = limit_variable_slope(
      eq, Trixi.pressure, slope, u_star_ll, u_star_rr, ue, xl, xr)

   ufl = ue + slope * xl
   ufr = ue + slope * xr

   return ufl, ufr, slope
end