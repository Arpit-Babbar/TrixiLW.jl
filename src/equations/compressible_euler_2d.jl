import Trixi: FluxHLL

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


@inline function (numflux::FluxHLL)(f_ll, f_rr, U_ll, U_rr, u_ll, u_rr,
                                    orientation_or_normal_direction, equations)
    λ_min, λ_max = numflux.min_max_speed(u_ll, u_rr, orientation_or_normal_direction,
                                         equations)

    if λ_min >= 0 && λ_max >= 0
        return f_ll
    elseif λ_max <= 0 && λ_min <= 0
        return f_rr
    else
        inv_λ_max_minus_λ_min = inv(λ_max - λ_min)
        factor_ll = λ_max * inv_λ_max_minus_λ_min
        factor_rr = λ_min * inv_λ_max_minus_λ_min
        factor_diss = λ_min * λ_max * inv_λ_max_minus_λ_min
        return factor_ll * f_ll - factor_rr * f_rr + factor_diss * (U_rr - U_ll)
    end
end
