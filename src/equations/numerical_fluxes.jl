using Trixi: AbstractEquations

import Trixi: flux_central, FluxPlusDissipation, DissipationLocalLaxFriedrichs

# TODO - This is type piracy?

@inline function flux_central(f_ll, f_rr, u_ll, u_rr,
  orientation_or_normal_direction,
  eq::AbstractEquations)
  # Average regular fluxes
  return 0.5 * (f_ll + f_rr)
end

@inline function (numflux::FluxPlusDissipation)(f_ll, f_rr, u_ll, u_rr, U_ll, U_rr,
  orientation_or_normal_direction,
  eq)
  @unpack numerical_flux, dissipation = numflux

  return (numerical_flux(f_ll, f_rr, u_ll, u_rr, orientation_or_normal_direction, eq)
          +
          dissipation(u_ll, u_rr, U_ll, U_rr, orientation_or_normal_direction, eq))
end

@inline function (numflux::FluxPlusDissipation)(f_ll, f_rr, u_ll, u_rr,
  orientation_or_normal_direction,
  eq)
  @unpack numerical_flux, dissipation = numflux

  return (numerical_flux(f_ll, f_rr, u_ll, u_rr, orientation_or_normal_direction, eq)
          +
          dissipation(u_ll, u_rr, u_ll, u_rr, orientation_or_normal_direction, eq))
end

@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
  U_ll, U_rr, orientation_or_normal_direction, eq)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, eq)
  return -0.5 * λ * (U_rr - U_ll)
end
