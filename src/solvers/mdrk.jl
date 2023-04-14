function perform_step!(integrator, limiters, callbacks, lw_update,
                       time_step_computation::CFLBased, ::TwoStaged)
   semi = integrator.p
   @unpack mesh, cache = semi
   dt = compute_dt(semi, mesh, time_step_computation, integrator)
   dt = set_dt!(integrator, dt)
   @unpack u, uprev, epsilon = integrator
   @unpack rhs!, soln_arrays = lw_update
   @unpack du_ode, u0_ode = soln_arrays         # Vectors form for compability with callbacks
   @.. uprev = u

   us_copy   = deepcopy(uprev)
   us_copy_2 = deepcopy(uprev)
   @unpack _us = cache.element_cache
   # Compute du = u^{n+1/2}-0.5*dt*u^n
   time_int_tol = 1e-8
   dummy_tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
   rhs!(du_ode, u, semi, integrator.t, dummy_tolerances, calc_volume_integral_mdrk1!)
   update_soln!(integrator, _us, uprev, du_ode) # us = uprev + dt * du
   us_copy_2 .= uprev + 2*integrator.dt * du_ode
   @unpack _u_low = cache.element_cache
   apply_limiters!(limiters, integrator)

   us_copy .= _us

   rhs!(du_ode, u, semi, integrator.t, dummy_tolerances,
   calc_volume_integral_mdrk2!
        )
   update_soln!(integrator, u, uprev, du_ode) # u += dt * du

#    @show maximum(abs.(u-us_copy))
#    @show maximum(abs.(u-_u_low))
#    @show maximum(abs.(u-us_copy_2))

   set_t_and_iter!(integrator, dt)

   return nothing
end
