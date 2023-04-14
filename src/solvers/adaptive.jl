using Trixi: get_node_vars

@inline function test_rhs!(du_ode, u_ode, semi, t, integrator, tolerances,
                           volume_integral = calc_volume_integral!)
   # TODO - Does this try catch block cause a performance issue?
   try
      rhs!(du_ode, u_ode, semi, t, tolerances, volume_integral) # Compute du = u^{n+1}-dt*u^n
   catch e
      if isa(e, DomainError) || isa(e, TaskFailedException) # Second exception is for multithreading. TODO - Get a more specific second exception

         println("Adjusting time step to maintain admissibility")
         domain_valid = false
         return domain_valid
      else
         println("Some non-domain error occured at t=$t, iter = $(integrator.iter), rethrowing...")
         rethrow(e)
      end
   end

   domain_valid = true

   return domain_valid
end

function test_updated_solution(u_ode, semi)
   @unpack mesh, solver, cache, equations = semi

   for element in eachelement(solver, cache)
      for j in eachnode(solver), i in eachnode(solver)
         u  = wrap_array(u_ode, mesh, equations, solver, cache)
         u_node = Trixi.get_node_vars(u, equations, solver, i, j, element)
         if is_admissible(u_node, equations) == false
            domain_valid = false
            return domain_valid
         end
      end
   end

   domain_valid = true

   return domain_valid
end

@inline function load_temporal_errors!(epsilon, temporal_errors, ndofs, redo)
   total_temporal_error = sqrt(sum(temporal_errors) / ndofs)

   # epsilon[i] = 1/total_temporal_error[i]
   # epsilon[1] contains the prediction of the upcoming time
   # while the epsilon[0], epsilon[-1] contain previous info
   if !redo # If it is a redo, we have already put correct previous epsilons
      epsilon[-1], epsilon[0] = epsilon[0], epsilon[1]
   end

   tol = 1e-12
   epsilon[1] = 1.0 / (total_temporal_error + tol)

   return epsilon
end

@inline function compute_and_load_temporal_errors!(u_ode, u_low_ode, semi, epsilon, ndofs, tolerances, redo)
   @unpack mesh, equations, solver, cache = semi
   dg = solver

   u     = wrap_array(u_ode    , mesh, equations, solver, cache)
   u_low = wrap_array(u_low_ode, mesh, equations, solver, cache)

   @unpack abstol, reltol = tolerances
   temporal_error = 0.0
   for element in eachelement(dg, cache)
      for j in eachnode(dg), i in eachnode(dg)
         u_node = get_node_vars(u, equations, dg, i, j, element)
         u_low_node = get_node_vars(u_low, equations, dg, i, j, element)
         for v in eachvariable(equations)
            local_error = u_node[v] - u_low_node[v]
            normalizing_factor = abstol + reltol * max(abs(u_node[v]), abs(u_low_node[v]))
            temporal_error += (local_error / normalizing_factor)^2
         end
      end
   end

   total_temporal_error = sqrt(temporal_error / ndofs)


   # epsilon[i] = 1/total_temporal_error[i]
   # epsilon[1] contains the prediction of the upcoming time
   # while the epsilon[0], epsilon[-1] contain previous info
   if !redo # If it is a redo, we have already put correct previous epsilons
      epsilon[-1], epsilon[0] = epsilon[0], epsilon[1]
   end

   tol = 1e-12
   epsilon[1] = 1.0 / (total_temporal_error + tol)

   return epsilon
end

function dt_factor(epsilon, k, controller)
   beta1, beta2, beta3 = controller
   factor_ = epsilon[1]^(beta1 / k) * epsilon[0]^(beta2 / k) * epsilon[-1]^(beta3 / k)
   factor = 1.0 + atan(factor_ - 1) # limiting function
   return factor
end

function perform_step!(integrator, limiters, callbacks, lw_update,
   time_step_computation::Adaptive, stages::SingleStaged, redo=false)
   semi = integrator.p
   @unpack tolerances, controller = integrator.opts
   @unpack u, uprev, epsilon = integrator
   @unpack rhs!, soln_arrays = lw_update
   @unpack du_ode, u0_ode = soln_arrays         # Vectors form for compability with callbacks
   @.. uprev = u

   domain_valid = true # Checks for domain errors
   error_valid = true # Checks if factor is large / small
   @unpack t, dt = integrator
   dt_next = dt

   @unpack temporal_errors = semi.cache

   # Compute du_ode and check for domain_error
   domain_valid = test_rhs!(du_ode, u, semi, t, integrator, tolerances)

   # Update the solution with obtained RHS
   update_soln!(integrator, u, uprev, du_ode) # u = uprev + dt*du_ode

   # Apply positivity limiter
   apply_limiters!(limiters, integrator)

   # TODO - This min is prolly not needed, we can just doc
   # domain_valid = test_updated_solution(u, semi)
   domain_valid = min(domain_valid, test_updated_solution(u, semi))

   # put appropriate temporal errors in epsilon
   epsilon = load_temporal_errors!(epsilon, temporal_errors,
      Trixi.ndofs(semi), redo)
   # Use epsilon to compute dt_factor
   factor = dt_factor(epsilon, Trixi.nnodes(semi.solver), controller)

   # Reject solution if factor is small
   if factor <= 0.81
      error_valid = false
   end

   # Reject solution if obtained u^{n+1} is inadmissible
   # if !(is_admissible(u, semi.equations))
   #    domain_valid = false
   # end

   if !(domain_valid && error_valid)
      dt = min(factor, 0.95) * dt
      println("Redoing time step to decrease $(integrator.dt) to $dt")
      @show domain_valid, error_valid
      set_dt!(integrator, dt)
      redo = true
      @.. u = uprev
      # Go back to beginning of function
      perform_step!(integrator, limiters, callbacks, lw_update, time_step_computation,
         stages, redo)
      return nothing
   end

   dt_next = min(factor, 1.5) * dt

   set_t_and_iter!(integrator, dt)

   # increase next dt if needed, also sets it to Tf-dt if t + dt > Tf
   set_dt!(integrator, dt_next)

   return nothing
end


function perform_step!(integrator, limiters, callbacks, lw_update,
                       time_step_computation::Adaptive, stages::TwoStaged, redo=false)
   semi = integrator.p
   @unpack mesh, cache = semi
   @unpack tolerances, controller = integrator.opts
   @unpack u, uprev, epsilon = integrator
   @unpack rhs!, soln_arrays = lw_update
   @unpack du_ode, u0_ode = soln_arrays         # Vectors form for compability with callbacks
   @unpack _us = cache.element_cache
   @.. uprev = u

   domain_valid = true # Checks for domain errors
   error_valid = true # Checks if factor is large / small
   @unpack t, dt = integrator
   dt_next = dt

   # First stage

   # Compute du_ode and check for domain_error
   domain_valid = min(domain_valid,
                      test_rhs!(du_ode, u, semi, t, integrator, tolerances,
                                calc_volume_integral_mdrk1!)
                     )

   # Update to us
   update_soln!(integrator, _us, uprev, du_ode) # us = uprev + dt * du

   # Positivity limiter
   apply_limiters!(limiters, integrator, _us)

   # TODO - This min is prolly not needed, we can just doc
   # domain_valid = test_updated_solution(u, semi)
   domain_valid = min(domain_valid, test_updated_solution(_us, semi))

   # Second stage

   # Compute du_ode and check for domain_error
   domain_valid = min(domain_valid,
                      test_rhs!(du_ode, u, semi, t, integrator, tolerances,
                                calc_volume_integral_mdrk2!)
                     )

   # Update the solution with obtained RHS
   update_soln!(integrator, u, uprev, du_ode) # u += dt * du

   # Apply positivity limiter
   apply_limiters!(limiters, integrator)

   # TODO - This min is prolly not needed, we can just doc
   # domain_valid = test_updated_solution(u, semi)
   domain_valid = min(domain_valid, test_updated_solution(u, semi))

   # put appropriate temporal errors in epsilon
   @unpack _u_low = cache.element_cache
   epsilon = compute_and_load_temporal_errors!(u, _u_low, semi, epsilon, Trixi.ndofs(semi), tolerances, redo)

   # Use epsilon to compute dt_factor
   factor = dt_factor(epsilon, Trixi.nnodes(semi.solver)-1, controller)

   # Reject solution if factor is small
   if factor <= 0.81
      error_valid = false
   end

   if !(domain_valid && error_valid)
      dt = min(factor, 0.95) * dt
      println("Redoing time step to decrease $(integrator.dt) to $dt")
      @show domain_valid, error_valid
      set_dt!(integrator, dt)
      redo = true
      @.. u = uprev
      # Go back to beginning of function
      perform_step!(integrator, limiters, callbacks, lw_update, time_step_computation,
         stages, redo)
      return nothing
   end

   dt_next = min(factor, 1.5) * dt

   set_t_and_iter!(integrator, dt)

   # increase next dt if needed, also sets it to Tf-dt if t + dt > Tf
   set_dt!(integrator, dt_next)

   return nothing
end