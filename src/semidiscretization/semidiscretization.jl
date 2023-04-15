using Trixi: SemidiscretizationHyperbolic, SemidiscretizationHyperbolicParabolic
using Trixi: compute_coefficients

function semidiscretize(semi::Union{SemidiscretizationHyperbolic, SemidiscretizationHyperbolicParabolic},
   time_discretization::AbstractLWTimeDiscretization,
   tspan)

   # Create copies of u_ode here!!
   u0_ode  = compute_coefficients(first(tspan), semi)
   du_ode  = similar(u0_ode)

   soln_arrays = (; u0_ode, du_ode)

   return LWUpdate(rhs!, soln_arrays, tspan, semi)
 end