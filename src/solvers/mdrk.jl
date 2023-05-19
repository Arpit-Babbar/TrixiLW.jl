function perform_step!(integrator, limiters, callbacks, lw_update,
                       time_step_computation::CFLBased, ::MDRK)
   semi = integrator.p
   @unpack mesh, cache = semi
   dummy_tolerances = integrator.opts.tolerances
   dt = compute_dt(semi, mesh, time_step_computation, integrator)
   dt = set_dt!(integrator, dt)
   @unpack u, uprev, epsilon = integrator
   @unpack rhs!, soln_arrays = lw_update
   @unpack du_ode, u0_ode = soln_arrays         # Vectors form for compability with callbacks
   @.. uprev = u

   # First stage
   @unpack _us = cache.element_cache.mdrk_cache
   # Compute du = u^{n+1/2}-0.5*dt*u^n
   rhs_mdrk1!(du_ode, u, semi, integrator.t, dummy_tolerances)
   update_soln!(integrator, _us, uprev, du_ode) # us = uprev + dt * du
   @unpack _u_low = cache.element_cache.mdrk_cache
   apply_limiters!(limiters, integrator)

   # Second stage
   # TODO - Just pass u* here!
   rhs_mdrk2!(du_ode, u, semi, integrator.t, dummy_tolerances)
   update_soln!(integrator, u, uprev, du_ode) # u += dt * du

   set_t_and_iter!(integrator, dt)

   return nothing
end
