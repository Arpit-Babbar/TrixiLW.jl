import Trixi: FluxHLL, FluxRotated
using Trixi: rotate_to_x, rotate_from_x, flux_hllc

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

# Rotated surface flux computation (2D version)
@inline function (flux_rotated::FluxRotated)(f_ll, f_rr, U_ll, U_rr, u_ll, u_rr,
   normal_direction::AbstractVector,
   equations::AbstractEquations{2})

   @unpack numerical_flux = flux_rotated

   norm_ = norm(normal_direction)
   # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
   normal_vector = normal_direction / norm_

   u_ll_rotated = rotate_to_x(u_ll, normal_vector, equations)
   u_rr_rotated = rotate_to_x(u_rr, normal_vector, equations)

   f = numerical_flux(u_ll_rotated, u_rr_rotated, 1, equations)

   return rotate_from_x(f, normal_vector, equations) * norm_
end

const flux_hllc_rotated = FluxRotated(flux_hllc)
