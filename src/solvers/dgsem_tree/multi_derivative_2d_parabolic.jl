function rhs_mdrk1!(du_ode, u_ode,
   semi::SemidiscretizationHyperbolicParabolic,
   t, tolerances = (;abstol = 0.0, reltol = 0.0))
  @unpack mesh, equations, equations_parabolic, initial_condition, boundary_conditions,
  solver_parabolic, cache, cache_parabolic, boundary_conditions_parabolic, source_terms, solver = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "rhs!" rhs_mdrk1!(du, u, t, mesh, equations, equations_parabolic,
    initial_condition, boundary_conditions,
    boundary_conditions_parabolic, source_terms, solver,
    solver_parabolic,
    get_time_discretization(solver), cache,
    cache_parabolic,
    tolerances)
  runtime = time_ns() - time_start
  # This is a struct in Trixi with two elements for counting
  # times of parabolic and hyperbolic part separately.
  # For Lax-Wendroff there needs to be only one because
  # we handle parabolic and hyperbolic part together.
  put!(semi.performance_counter.counters[1], runtime)
  put!(semi.performance_counter.counters[2], runtime)

  return nothing
end

function rhs_mdrk2!(du_ode, u_ode,
   semi::SemidiscretizationHyperbolicParabolic,
   t, tolerances = (;abstol = 0.0, reltol = 0.0))
  @unpack mesh, equations, equations_parabolic, initial_condition, boundary_conditions,
  solver_parabolic, cache, cache_parabolic, boundary_conditions_parabolic, source_terms, solver = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "rhs!" rhs_mdrk2!(du, u, t, mesh, equations, equations_parabolic,
    initial_condition, boundary_conditions,
    boundary_conditions_parabolic, source_terms, solver,
    solver_parabolic,
    get_time_discretization(solver), cache,
    cache_parabolic,
    tolerances)
  runtime = time_ns() - time_start
  # This is a struct in Trixi with two elements for counting
  # times of parabolic and hyperbolic part separately.
  # For Lax-Wendroff there needs to be only one because
  # we handle parabolic and hyperbolic part together.
  put!(semi.performance_counter.counters[1], runtime)
  put!(semi.performance_counter.counters[2], runtime)

  return nothing
end

function rhs_mdrk1!(du, u,
   t, mesh::Union{TreeMesh{2},P4estMesh{2}}, equations, equations_parabolic,
   initial_condition, boundary_conditions, boundary_conditions_parabolic,
   source_terms, dg::DG, parabolic_scheme,
   time_discretization::MDRK, cache, cache_parabolic,
   tolerances::NamedTuple)
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)
    @unpack u_transformed, gradients, flux_viscous = cache_parabolic

    dt = cache.dt[1]

    # Convert conservative variables to a form more suitable for viscous flux calculations
    @trixi_timeit timer() "transform variables" transform_variables!(
       u_transformed, u, mesh, equations_parabolic, dg, parabolic_scheme, cache, cache_parabolic)

    # Compute the gradients of the transformed variables
    @trixi_timeit timer() "calculate gradient" calc_gradient!(
       gradients, u_transformed, t, mesh, equations_parabolic, boundary_conditions_parabolic, dg,
       cache, cache_parabolic)

    # Compute and store the viscous fluxes computed with S variable
    @trixi_timeit timer() "calculate viscous fluxes" calc_viscous_fluxes!(
       flux_viscous, gradients, u_transformed, mesh, equations_parabolic, dg, cache, cache_parabolic)

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" calc_volume_integral_mdrk1!(
       du, flux_viscous, gradients, u_transformed, u, t, dt, tolerances, mesh,
       have_nonconservative_terms(equations), source_terms, equations, equations_parabolic,
       dg.volume_integral, time_discretization, dg, cache, cache_parabolic)

    # # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
       cache, u, mesh, equations, dg.surface_integral, dg)

    # Prolong F, U to interfaces
    @trixi_timeit timer() "prolong2interfaces" prolong2interfaces_lw_parabolic!(
       cache, cache_parabolic, u, mesh, equations, dg.surface_integral, dg)

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" calc_interface_flux_hyperbolic_parabolic!(
       cache.elements.surface_flux_values,
       mesh,
       # have_nonconservative_terms(equations),
       equations, equations_parabolic,
       dg.surface_integral, dg, cache, cache_parabolic)

    # Prolong u to boundaries
    @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
       cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
       cache, t, dt, boundary_conditions, mesh, equations, dg.surface_integral, time_discretization, dg)

    # Prolong viscous flux to boundaries
    @trixi_timeit timer() "prolong2boundaries" prolong2boundaries_visc_lw!(
       cache_parabolic, flux_viscous, mesh, equations_parabolic, dg.surface_integral, dg, cache)

    # Calculate viscous surface fluxes on boundaries
    @trixi_timeit timer() "boundary flux" calc_boundary_flux_divergence_lw!(
       cache_parabolic, cache, t, boundary_conditions_parabolic, mesh, equations_parabolic,
       dg.surface_integral, dg)

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" calc_surface_integral!(
       du, u, mesh, equations, dg.surface_integral, dg, cache)

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(
       du, mesh, equations, dg, cache)

   return nothing
end

function rhs_mdrk2!(du, u,
   t, mesh::Union{TreeMesh{2},P4estMesh{2}}, equations, equations_parabolic,
   initial_condition, boundary_conditions, boundary_conditions_parabolic,
   source_terms, dg::DG, parabolic_scheme,
   time_discretization::MDRK, cache, cache_parabolic,
   tolerances::NamedTuple)

   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)
   @unpack u_transformed, gradients, flux_viscous = cache_parabolic

   dt = cache.dt[1]

   @unpack us = cache.element_cache.mdrk_cache

   # Convert conservative variables to a form more suitable for viscous flux calculations
   @trixi_timeit timer() "transform variables" transform_variables!(
      u_transformed, us, mesh, equations_parabolic, dg, parabolic_scheme, cache, cache_parabolic)

   # Compute the gradients of the transformed variables
   @trixi_timeit timer() "calculate gradient" calc_gradient!(
      gradients, u_transformed, t, mesh, equations_parabolic, boundary_conditions_parabolic, dg,
      cache, cache_parabolic)

   # Compute and store the viscous fluxes computed with S variable
   @trixi_timeit timer() "calculate viscous fluxes" calc_viscous_fluxes!(
      flux_viscous, gradients, u_transformed, mesh, equations_parabolic, dg, cache, cache_parabolic)

   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

   # Calculate volume integral
   @trixi_timeit timer() "volume integral" calc_volume_integral_mdrk2!(
      du, flux_viscous, gradients, u_transformed, us, t, dt, tolerances, mesh,
      have_nonconservative_terms(equations), source_terms, equations, equations_parabolic,
      dg.volume_integral, time_discretization, dg, cache, cache_parabolic)

   # # Prolong solution to interfaces
   @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
      cache, u, mesh, equations, dg.surface_integral, dg)

   # Prolong F, U to interfaces
   @trixi_timeit timer() "prolong2interfaces" prolong2interfaces_lw_parabolic!(
      cache, cache_parabolic, u, mesh, equations, dg.surface_integral, dg)

   # Calculate interface fluxes
   @trixi_timeit timer() "interface flux" calc_interface_flux_hyperbolic_parabolic!(
      cache.elements.surface_flux_values,
      mesh,
      # have_nonconservative_terms(equations),
      equations, equations_parabolic,
      dg.surface_integral, dg, cache, cache_parabolic)

   # Prolong u to boundaries
   @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
      cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Calculate boundary fluxes
   @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
      cache, t, dt, boundary_conditions, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Prolong viscous flux to boundaries
   @trixi_timeit timer() "prolong2boundaries" prolong2boundaries_visc_lw!(
      cache_parabolic, flux_viscous, mesh, equations_parabolic, dg.surface_integral, dg, cache)

   # Calculate viscous surface fluxes on boundaries
   @trixi_timeit timer() "boundary flux" calc_boundary_flux_divergence_lw!(
      cache_parabolic, cache, t, boundary_conditions_parabolic, mesh, equations_parabolic,
      dg.surface_integral, dg)

   # Calculate surface integrals
   @trixi_timeit timer() "surface integral" calc_surface_integral!(
      du, u, mesh, equations, dg.surface_integral, dg, cache)

   # Apply Jacobian from mapping to reference element
   @trixi_timeit timer() "Jacobian" apply_jacobian!(
      du, mesh, equations, dg, cache)


   return nothing
end

@inline function mdrk_kernel_1!(
   du, flux_viscous, gradients, u_transformed,
   u, t, dt, tolerances,
   mesh::TreeMesh{2},
   nonconservative_terms::False, source_terms, equations,
   equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::MDRK,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack mdrk_cache = element_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   id = Threads.threadid()

   refresh!(arr) = fill!(arr, zero(eltype(u)))

   Fa, Ga, Fa2, Ga2, S2, ut, U, U2, S = cell_arrays[id]

   utx, uty, Fv, Gv, Fv2, Gv2 = cache_parabolic.lw_res_cache.cell_arrays[id]
   @unpack u_low = element_cache.mdrk_cache
   refresh!.((ut, utx, uty))
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      flux_adv_1, flux_adv_2 = fluxes(u_node, equations)
      flux_visc_1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, element)
      flux_visc_2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, element)
      flux1 = flux_adv_1 - flux_visc_1
      flux2 = flux_adv_2 - flux_visc_2

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1,
            equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
      end

      set_node_vars!(Fa, 0.5*flux_adv_1, equations, dg, i, j)
      set_node_vars!(Ga, 0.5*flux_adv_2, equations, dg, i, j)
      set_node_vars!(Fv, 0.5*flux_visc_1, equations, dg, i, j)
      set_node_vars!(Gv, 0.5*flux_visc_2, equations, dg, i, j)

      set_node_vars!(Fa2, flux_adv_1, equations, dg, i, j)
      set_node_vars!(Ga2, flux_adv_2, equations, dg, i, j)
      set_node_vars!(Fv2, flux_visc_1, equations, dg, i, j)
      set_node_vars!(Gv2, flux_visc_2, equations, dg, i, j)

      set_node_vars!(U, 0.5*u_node, equations, dg, i, j)
      set_node_vars!(U2, u_node, equations, dg, i, j)
      set_node_vars!(u_low, u_node, equations, dg, i, j, element)
   end

   # Scale ut
   for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
         ut[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to ut and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      s_node = calc_source(u_node, x, t, source_terms, equations, dg, cache)
      set_node_vars!(S, 0.5*s_node, equations, dg, i, j)
      set_node_vars!(S2, s_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇u_t
   for j in eachnode(dg), i in eachnode(dg)
      ut_node = get_node_vars(ut, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(utx, derivative_matrix[ii, i], ut_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(uty, derivative_matrix[jj, j], ut_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇u_t
   for j in eachnode(dg), i in eachnode(dg)
      # inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utx[v, i, j] *= inv_jacobian
         uty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      u_node  = get_node_vars(u, equations, dg, i, j, element)
      ut_node  = get_node_vars(ut,  equations, dg, i, j)
      utx_node = get_node_vars(utx, equations, dg, i, j)
      uty_node = get_node_vars(uty, equations, dg, i, j)

      ux_node = get_node_vars(gradients_x, equations, dg, i, j, element)
      uy_node = get_node_vars(gradients_y, equations, dg, i, j, element)
      up  = u_node +  ut_node
      um  = u_node -  ut_node
      upp = u_node +2*ut_node
      umm = u_node -2*ut_node

      upx  = ux_node +  utx_node
      umx  = ux_node -  utx_node
      upy  = uy_node +  uty_node
      umy  = uy_node -  uty_node

      uppx = ux_node +2*utx_node
      ummx = ux_node -2*utx_node
      uppy = uy_node +2*uty_node
      ummy = uy_node -2*uty_node

      (fma, gma), (fmv, gmv) = fluxes(um, (umx, umy), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up, (upx, upy), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm, (ummx, ummy), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp, (uppx, uppy), equations, equations_parabolic)

      fa, ga = 2*get_node_vars(Fa, equations, dg, i, j), 2*get_node_vars(Ga, equations, dg, i, j)
      fv, gv = 2*get_node_vars(Fv, equations, dg, i, j), 2*get_node_vars(Gv, equations, dg, i, j)
      f = fa-fv
      g = ga-gv
      s = get_node_vars(S2,  equations, dg, i, j)

      fta = 1.0 / 12.0 * (-fppa + 8.0 * fpa - 8.0 * fma + fmma)
      multiply_add_to_node_vars!(Fa, 0.125, fta, equations, dg, i, j)
      multiply_add_to_node_vars!(Fa2, 1.0/6.0, fta, equations, dg, i, j)
      gta = 1.0 / 12.0 * (-gppa + 8.0 * gpa - 8.0 * gma + gmma)
      multiply_add_to_node_vars!(Ga, 0.125, gta, equations, dg, i, j)
      multiply_add_to_node_vars!(Ga2, 1.0/6.0, gta, equations, dg, i, j)

      ftv = 1.0 / 12.0 * (-fppv + 8.0 * fpv - 8.0 * fmv + fmmv)
      multiply_add_to_node_vars!(Fv, 0.125, ftv, equations, dg, i, j)
      multiply_add_to_node_vars!(Fv2, 1.0/6.0, ftv, equations, dg, i, j)
      gtv = 1.0 / 12.0 * (-gppv + 8.0 * gpv - 8.0 * gmv + gmmv)
      multiply_add_to_node_vars!(Gv, 0.125, gtv, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv2, 1.0/6.0, gtv, equations, dg, i, j)

      multiply_add_to_node_vars!(U,  0.125, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(U2, 1.0/6.0, ut_node, equations, dg, i, j)

      ft = fta - ftv
      gt = gta - gtv

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N34(u_node, up, upp, um, umm, x, t, dt,
                             source_terms, equations, dg, cache)

      F_node_low = f + 0.5 * ft
      G_node_low = g + 0.5 * gt
      S_node_low = s + 0.5 * st

      Fa_node = get_node_vars(Fa, equations, dg, i, j)
      Fv_node = get_node_vars(Fv, equations, dg, i, j)

      F = Fa_node - Fv_node

      Ga_node = get_node_vars(Ga, equations, dg, i, j)
      Gv_node = get_node_vars(Gv, equations, dg, i, j)

      G = Ga_node - Gv_node

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F,
                                    equations, dg, ii, j, element)
         multiply_add_to_node_vars!(u_low, -dt * inv_jacobian * derivative_matrix[ii, i],
                                    F_node_low, equations, dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G,
                                    equations, dg, i, jj, element)
         multiply_add_to_node_vars!(u_low, -dt * inv_jacobian * derivative_matrix[jj, j],
                                    G_node_low, equations, dg, i, jj, element)
      end

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model

      # TODO - Move them out of dof loop and then use @turbo
      U_node = get_node_vars(U, equations, dg, i, j)
      U2_node = get_node_vars(U2, equations, dg, i, j)

      set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)
      set_node_vars!(cache.element_cache.F, Fa_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Fv_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache.element_cache.F, Ga_node, equations, dg, 2, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Gv_node, equations, dg, 2, i, j, element)

      Fa_node2 = get_node_vars(Fa2, equations, dg, i, j)
      Fv_node2 = get_node_vars(Fv2, equations, dg, i, j)
      Ga_node2 = get_node_vars(Ga2, equations, dg, i, j)
      Gv_node2 = get_node_vars(Gv2, equations, dg, i, j)

      set_node_vars!(mdrk_cache.F2, Fa_node2, equations, dg, 1, i, j, element)
      set_node_vars!(cache_parabolic.mdrk_cache.Fv2, Fv_node2, equations, dg, 1, i, j, element)
      set_node_vars!(mdrk_cache.F2, Ga_node2, equations, dg, 2, i, j, element)
      set_node_vars!(cache_parabolic.mdrk_cache.Fv2, Gv_node2, equations, dg, 2, i, j, element)

      set_node_vars!(mdrk_cache.U2, U2_node, equations, dg, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations,
                                       dg, i, j, element)
      multiply_add_to_node_vars!(u_low, dt , S_node_low, equations,
                                       dg, i, j, element)

   end

   return nothing
end

function calc_volume_integral_mdrk1!(
   du, flux_viscous, gradients, u_transformed, u, t, dt, tolerances::NamedTuple,
   mesh::TreeMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::MDRK,
   dg::DGSEM, cache, cache_parabolic)

   degree = polydeg(dg)
   @threaded for element in eachelement(dg, cache)
      mdrk_kernel_1!(du, flux_viscous, gradients, u_transformed, u, t, dt,
         tolerances, mesh,
         have_nonconservative_terms, source_terms,
         equations, equations_parabolic,
         volume_integral, time_discretization,
         dg, cache, cache_parabolic, element)
   end

end

function calc_volume_integral_mdrk2!(
   du, flux_viscous, gradients, u_transformed, u, t, dt, tolerances::NamedTuple,
   mesh::TreeMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::MDRK,
   dg::DGSEM, cache, cache_parabolic)

   degree = polydeg(dg)
   @threaded for element in eachelement(dg, cache)
      mdrk_kernel_2!(du, flux_viscous, gradients, u_transformed, u, t, dt,
         tolerances, mesh,
         have_nonconservative_terms, source_terms,
         equations, equations_parabolic,
         volume_integral, time_discretization,
         dg, cache, cache_parabolic, element)
   end

end

@inline function mdrk_kernel_2!(
   du, flux_viscous, gradients, u_transformed,
   us, t, dt, tolerances,
   mesh::TreeMesh{2},
   nonconservative_terms::False, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::MDRK,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack mdrk_cache = element_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   @unpack U2, F2, S2 = mdrk_cache
   @unpack Fv2 = cache_parabolic.mdrk_cache

   id = Threads.threadid()

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   Fa, Ga, ust, U, S = cell_arrays[id]

   ustx, usty, Fv, Gv = cache_parabolic.lw_res_cache.cell_arrays[id]
   # Load U2, F2 in local arrays
   for j in eachnode(dg), i in eachnode(dg), n in eachvariable(equations)
      Fa[n,i,j] = F2[n,1,i,j,element]
      Ga[n,i,j] = F2[n,2,i,j,element]
      Fv[n,i,j] = Fv2[n,1,i,j,element]
      Gv[n,i,j] = Fv2[n,2,i,j,element]
      U[n,i,j]  = U2[n,i,j,element]
   end

   refresh!.((ust, ustx, usty))
   for j in eachnode(dg), i in eachnode(dg)
      us_node = get_node_vars(us, equations, dg, i, j, element)

      flux_adv_1, flux_adv_2 = fluxes(us_node, equations)
      flux_visc_1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, element)
      flux_visc_2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, element)
      flux1 = flux_adv_1 - flux_visc_1
      flux2 = flux_adv_2 - flux_visc_2

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ust, -dt * derivative_matrix[ii, i], flux1,
            equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ust, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
      end
   end



   # Scale ust
   for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
         ust[v, i, j] *= inv_jacobian
      end
   end

   # Load S2 into S
   for j in eachnode(dg), i in eachnode(dg)
      s2_node = get_node_vars(S2, equations, dg, i, j, element)
      set_node_vars!(S, s2_node, equations, dg, i, j)
   end

   # Add source term contribution to ut and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      us_node = get_node_vars(us, equations, dg, i, j, element)
      s_node = calc_source(us_node, x, t, source_terms, equations, dg, cache)
      set_node_vars!(S, s_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ust, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇u_t
   for j in eachnode(dg), i in eachnode(dg)
      ust_node = get_node_vars(ust, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ustx, derivative_matrix[ii, i], ust_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(usty, derivative_matrix[jj, j], ust_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇us_t
   for j in eachnode(dg), i in eachnode(dg)
      # inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         ustx[v, i, j] *= inv_jacobian
         usty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(us, equations, dg, i, j, element)
      ut_node = get_node_vars(ust, equations, dg, i, j)
      ux_node = get_node_vars(gradients_x, equations, dg, i, j, element)
      uy_node = get_node_vars(gradients_y, equations, dg, i, j, element)
      utx_node = get_node_vars(ustx, equations, dg, i, j)
      uty_node = get_node_vars(usty, equations, dg, i, j)

      up  = u_node +  ut_node
      um  = u_node -  ut_node
      upp = u_node +2*ut_node
      umm = u_node -2*ut_node

      upx  = ux_node +  utx_node
      umx  = ux_node -  utx_node
      upy  = uy_node +  uty_node
      umy  = uy_node -  uty_node

      uppx = ux_node +2*utx_node
      ummx = ux_node -2*utx_node
      uppy = uy_node +2*uty_node
      ummy = uy_node -2*uty_node

      (fma, gma), (fmv, gmv) = fluxes(um, (umx, umy), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up, (upx, upy), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm, (ummx, ummy), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp, (uppx, uppy), equations, equations_parabolic)

      fta = 1.0 / 12.0 * (-fppa + 8.0 * fpa - 8.0 * fma + fmma)
      multiply_add_to_node_vars!(Fa, 1.0/3.0, fta, equations, dg, i, j)
      gta = 1.0 / 12.0 * (-gppa + 8.0 * gpa - 8.0 * gma + gmma)
      multiply_add_to_node_vars!(Ga, 1.0/3.0, gta, equations, dg, i, j)

      ftv = 1.0 / 12.0 * (-fppv + 8.0 * fpv - 8.0 * fmv + fmmv)
      multiply_add_to_node_vars!(Fv, 1.0/3.0, ftv, equations, dg, i, j)
      gtv = 1.0 / 12.0 * (-gppv + 8.0 * gpv - 8.0 * gmv + gmmv)
      multiply_add_to_node_vars!(Gv, 1.0/3.0, gtv, equations, dg, i, j)

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N34(u_node, up, upp, um, umm, x, t, dt,
                             source_terms, equations, dg, cache)

      multiply_add_to_node_vars!(U, 1.0/3.0, ut_node, equations, dg, i, j)

      multiply_add_to_node_vars!(S, 1.0/3.0, st, equations, dg, i, j) # Source term

      Fa_node = get_node_vars(Fa, equations, dg, i, j)
      Fv_node = get_node_vars(Fv, equations, dg, i, j)

      F = Fa_node - Fv_node

      Ga_node = get_node_vars(Ga, equations, dg, i, j)
      Gv_node = get_node_vars(Gv, equations, dg, i, j)

      G = Ga_node - Gv_node

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F, equations, dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G, equations, dg, i, jj, element)
      end

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)

      set_node_vars!(element_cache.U, U_node,  equations, dg,    i, j, element)
      set_node_vars!(element_cache.F, Fa_node, equations, dg, 1, i, j, element)
      set_node_vars!(element_cache.F, Ga_node, equations, dg, 2, i, j, element)

      set_node_vars!(cache_parabolic.Fv, Fv_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Gv_node, equations, dg, 2, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations,
                                       dg, i, j, element)
   end

   return nothing
end