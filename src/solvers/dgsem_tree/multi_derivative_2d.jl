function rhs_mdrk1!(du, u,
   t, mesh::Union{TreeMesh{2},P4estMesh{2}}, equations,
   initial_condition, boundary_conditions, source_terms, dg::DG,
   time_discretization::AbstractLWTimeDiscretization, cache, tolerances::NamedTuple)
   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

   dt = cache.dt[1]

   # Update dt in cache and the callback will just take it from there

   # Calculate volume integral
   @trixi_timeit timer() "volume integral" calc_volume_integral_mdrk1!(
      du,
      u,
      t, dt, tolerances, mesh,
      have_nonconservative_terms(equations), source_terms, equations,
      dg.volume_integral, time_discretization, dg, cache)
   # Prolong solution to interfaces
   @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
      cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Calculate interface fluxes
   @trixi_timeit timer() "interface flux" calc_interface_flux!(
      cache.elements.surface_flux_values, mesh,
      have_nonconservative_terms(equations), equations,
      dg.surface_integral, dt, time_discretization, dg, cache)

   # Prolong solution to boundaries
   @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
      cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Calculate boundary fluxes
   @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
      cache, t, dt, boundary_conditions, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Prolong solution to mortars
   @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
      cache, u, mesh, equations, dg.mortar, dg.surface_integral, time_discretization, dg)
   # @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
   #       cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

   # Calculate mortar fluxes
   @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
      cache.elements.surface_flux_values, mesh,
      have_nonconservative_terms(equations), equations,
      dg.mortar, dg.surface_integral, dt, time_discretization, dg, cache)

   # Calculate surface integrals
   @trixi_timeit timer() "surface integral" calc_surface_integral!(
      du, u, mesh, equations, dg.surface_integral, dg, cache)

   # Apply Jacobian from mapping to reference element
   @trixi_timeit timer() "Jacobian" apply_jacobian!(
      du, mesh, equations, dg, cache)

   return nothing
end

function rhs_mdrk2!(du, u,
   t, mesh::Union{TreeMesh{2},P4estMesh{2}}, equations,
   initial_condition, boundary_conditions, source_terms, dg::DG,
   time_discretization::AbstractLWTimeDiscretization, cache, tolerances::NamedTuple)
   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

   dt = cache.dt[1]

   # Update dt in cache and the callback will just take it from there

   @unpack us = cache.element_cache.mdrk_cache

   # Calculate volume integral
   alpha = @trixi_timeit timer() "volume integral" calc_volume_integral_mdrk2!(
      du, us, t, dt, tolerances, mesh,
      have_nonconservative_terms(equations), source_terms, equations,
      dg.volume_integral, time_discretization, dg, cache)

   # Prolong solution to interfaces
   @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
      cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Calculate interface fluxes
   @trixi_timeit timer() "interface flux" calc_interface_flux!(
      cache.elements.surface_flux_values, mesh,
      have_nonconservative_terms(equations), equations,
      dg.surface_integral, dt, time_discretization, dg, cache)

   # Prolong solution to boundaries
   @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
      cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Calculate boundary fluxes
   @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
      cache, t, dt, boundary_conditions, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Prolong solution to mortars
   @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
      cache, u, mesh, equations, dg.mortar, dg.surface_integral, time_discretization, dg)
   # @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
   #       cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

   # Calculate mortar fluxes
   @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
      cache.elements.surface_flux_values, mesh,
      have_nonconservative_terms(equations), equations,
      dg.mortar, dg.surface_integral, dt, time_discretization, dg, cache)

   # Calculate surface integrals
   @trixi_timeit timer() "surface integral" calc_surface_integral!(
      du, u, mesh, equations, dg.surface_integral, dg, cache)

   # Apply Jacobian from mapping to reference element
   @trixi_timeit timer() "Jacobian" apply_jacobian!(
      du, mesh, equations, dg, cache)

   return nothing
end

function calc_volume_integral_mdrk1!(du, u,
   t, dt, tolerances::NamedTuple,
   mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms, source_terms, equations,
   volume_integral::VolumeIntegralFR,
   time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache)
   @threaded for element in eachelement(dg, cache)
      mdrk_kernel_1!(du, u,
         t, dt, tolerances, element, mesh,
         nonconservative_terms, source_terms, equations,
         dg, cache)
   end
   return nothing
end

function calc_volume_integral_mdrk2!(du, u,
   t, dt, tolerances::NamedTuple,
   mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms, source_terms, equations,
   volume_integral::VolumeIntegralFR,
   time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache)
   @threaded for element in eachelement(dg, cache)
      mdrk_kernel_2!(du, u,
         t, dt, tolerances, element, mesh,
         nonconservative_terms, source_terms, equations,
         dg, cache)
   end
   return nothing
end

function calc_volume_integral_mdrk1!(du, u, t, dt, tolerances,
   mesh::Union{
      TreeMesh{2},
      StructuredMesh{2},
      UnstructuredMesh2D,
      P4estMesh{2}
   },
   nonconservative_terms, source_terms, equations,
   volume_integral::VolumeIntegralFRShockCapturing,
   ::AbstractLWTimeDiscretization,
   dg::DGSEM, cache)

   @unpack element_ids_dg, element_ids_dgfv = cache
   @unpack volume_flux_fv, indicator = volume_integral

   # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
   alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg, cache)

   # Determine element ids for DG-only and blended DG-FV volume integral
   pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

   # Loop over pure DG elements
   @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
      element = element_ids_dg[idx_element]
      alpha_element = alpha[element]

      # Calculate DG volume integral contribution
      mdrk_kernel_1!(du, u, t, dt, tolerances, element, mesh,
         nonconservative_terms, source_terms, equations,
         dg, cache, 1 - alpha_element)

      # Calculate fn_low because it is needed for admissibility preservation proof
      calc_fn_low_kernel!(du, u,
         mesh,
         nonconservative_terms, equations,
         volume_flux_fv, dg, cache, element, alpha_element)
   end

   # Loop over blended DG-FV elements
   @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
      element = element_ids_dgfv[idx_element]
      alpha_element = alpha[element]

      # Calculate DG volume integral contribution
      mdrk_kernel_1!(du, u, t, dt, tolerances, element, mesh,
         nonconservative_terms, source_terms, equations,
         dg, cache, 1 - alpha_element)

      fv_kernel!(du, u, 0.5*dt, volume_integral.reconstruction, mesh,
         nonconservative_terms, equations, volume_flux_fv,
         dg, cache, element, alpha_element)
   end

   return alpha
end

function calc_volume_integral_mdrk2!(du, u, t, dt, tolerances,
   mesh::Union{
      TreeMesh{2},
      StructuredMesh{2},
      UnstructuredMesh2D,
      P4estMesh{2}
   },
   nonconservative_terms, source_terms, equations,
   volume_integral::VolumeIntegralFRShockCapturing,
   ::AbstractLWTimeDiscretization,
   dg::DGSEM, cache)

   @unpack element_ids_dg, element_ids_dgfv = cache
   @unpack volume_flux_fv, indicator = volume_integral

   # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
   alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg, cache)

   # Determine element ids for DG-only and blended DG-FV volume integral
   pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

   # Loop over pure DG elements
   @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
      element = element_ids_dg[idx_element]
      alpha_element = alpha[element]

      # Calculate DG volume integral contribution
      mdrk_kernel_2!(du, u, t, dt, tolerances, element, mesh,
         nonconservative_terms, source_terms, equations,
         dg, cache, 1 - alpha_element)

      # Calculate fn_low because it is needed for admissibility preservation proof
      calc_fn_low_kernel!(du, u,
         mesh,
         nonconservative_terms, equations,
         volume_flux_fv, dg, cache, element, alpha_element)
   end

   # Loop over blended DG-FV elements
   @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
      element = element_ids_dgfv[idx_element]
      alpha_element = alpha[element]

      # Calculate DG volume integral contribution
      mdrk_kernel_2!(du, u, t, dt, tolerances, element, mesh,
         nonconservative_terms, source_terms, equations,
         dg, cache, 1 - alpha_element)

      fv_kernel!(du, u, dt, volume_integral.reconstruction, mesh,
         nonconservative_terms, equations, volume_flux_fv,
         dg, cache, element, alpha_element)
   end

   return alpha
end

@inline function mdrk_kernel_1!(du, u,
   t, dt, tolerances,
   element, mesh::TreeMesh{2},
   nonconservative_terms::False, source_terms, equations,
   dg::DGSEM, cache, alpha=true)
   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack mdrk_cache = element_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   id = Threads.threadid()

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   # TODO - Are the local F2, G2, U2 needed? This is a performance question
   # Benchmark and find out
   F, F2, G, G2, ut, U, U2, S, S2 = cell_arrays[id]
   @unpack u_low = mdrk_cache
   refresh!.((ut,))
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      flux1, flux2 = fluxes(u_node, equations)

      set_node_vars!(F,  flux1, equations, dg, i, j)
      set_node_vars!(F2, flux1, equations, dg, i, j)
      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1,
            equations, dg, ii, j)
      end

      set_node_vars!(G,  flux2, equations, dg, i, j)
      set_node_vars!(G2, flux2, equations, dg, i, j)
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
      end

      set_node_vars!(U, u_node, equations, dg, i, j)
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
      set_node_vars!(S, s_node, equations, dg, i, j)
      set_node_vars!(S2, s_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      u_node  = get_node_vars(u, equations, dg, i, j, element)
      ut_node = get_node_vars(ut, equations, dg, i, j)

      um  = u_node - ut_node
      up  = u_node + ut_node
      umm = u_node - 2*ut_node
      upp = u_node + 2*ut_node

      fm, gm = fluxes(um, equations)
      fp, gp = fluxes(up, equations)
      fmm, gmm = fluxes(umm, equations)
      fpp, gpp = fluxes(upp, equations)

      ft = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
      gt = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N34(u_node, up, upp, um, umm, x, t, dt,
                             source_terms, equations, dg, cache)

      f_node = get_node_vars(F, equations, dg, i, j)
      g_node = get_node_vars(G, equations, dg, i, j)
      s_node = get_node_vars(S, equations, dg, i, j)
      F_node_low = f_node + 0.5 * ft
      G_node_low = g_node + 0.5 * gt
      S_node_low = s_node + 0.5 * st

      multiply_add_to_node_vars!(F, 0.25, ft, equations, dg, i, j)
      multiply_add_to_node_vars!(G, 0.25, gt, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 0.25, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(S, 0.25, st, equations, dg, i, j) # Source term

      multiply_add_to_node_vars!(F2, 1.0/6.0, ft, equations, dg, i, j)
      multiply_add_to_node_vars!(G2, 1.0/6.0, gt, equations, dg, i, j)
      multiply_add_to_node_vars!(U2, 1.0/6.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(S2, 1.0/6.0, st, equations, dg, i, j) # Source term

      F_node = get_node_vars(F, equations, dg, i, j)
      G_node = get_node_vars(G, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], F_node,
                                    equations, dg, ii, j, element)
         multiply_add_to_node_vars!(u_low, -dt * inv_jacobian * derivative_matrix[ii, i],
                                    F_node_low, equations, dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], G_node,
                                    equations, dg, i, jj, element)
         multiply_add_to_node_vars!(u_low, -dt * inv_jacobian * derivative_matrix[jj, j],
                                    G_node_low, equations, dg, i, jj, element)
      end

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)
      U2_node = get_node_vars(U2, equations, dg, i, j)

      F2_node = get_node_vars(F2, equations, dg, i, j)
      G2_node = get_node_vars(G2, equations, dg, i, j)

      set_node_vars!(element_cache.U, U_node, equations, dg,    i, j, element)
      set_node_vars!(element_cache.F, F_node, equations, dg, 1, i, j, element)
      set_node_vars!(element_cache.F, G_node, equations, dg, 2, i, j, element)

      set_node_vars!(mdrk_cache.U2, U2_node, equations, dg,    i, j, element)
      set_node_vars!(mdrk_cache.F2, F2_node, equations, dg, 1, i, j, element)
      set_node_vars!(mdrk_cache.F2, G2_node, equations, dg, 2, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      S2_node = get_node_vars(S2, equations, dg, i, j)
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations,
                                       dg, i, j, element)
      multiply_add_to_node_vars!(u_low, dt , S_node_low, equations,
                                       dg, i, j, element)

      set_node_vars!(mdrk_cache.S2, S2_node, equations, dg, i, j, element)

   end

   return nothing
end

@inline function mdrk_kernel_2!(du, us,
   t, dt, tolerances,
   element, mesh::TreeMesh{2},
   nonconservative_terms::False, source_terms, equations,
   dg::DGSEM, cache, alpha=true)

   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack mdrk_cache = element_cache

   @unpack U2, F2, S2 = mdrk_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   id = Threads.threadid()

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   F, G, ut, ust, U, S = cell_arrays[id]

   # Load U2, F2 in local arrays
   for j in eachnode(dg), i in eachnode(dg), n in eachvariable(equations)
      F[n,i,j] = F2[n,1,i,j,element]
      G[n,i,j] = F2[n,2,i,j,element]
      U[n,i,j] = U2[n,i,j,element]
   end

   refresh!.((ut, ust))
   for j in eachnode(dg), i in eachnode(dg)
      us_node = get_node_vars(us, equations, dg, i, j, element)

      flux1, flux2 = fluxes(us_node, equations)

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
      # Add source term contribution to ust
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      us_node = get_node_vars(us, equations, dg, i, j, element)
      s_node = calc_source(us_node, x, t, source_terms, equations, dg, cache)
      set_node_vars!(S, s_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ust, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      us_node = get_node_vars(us, equations, dg, i, j, element)
      ust_node = get_node_vars(ust, equations, dg, i, j)

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)

      um  = us_node -   ust_node
      fm, gm  = fluxes(um, equations)
      up  = us_node +   ust_node
      fp, gp  = fluxes(up, equations)
      umm = us_node - 2*ust_node
      fmm, gmm  = fluxes(umm, equations)
      upp = us_node + 2*ust_node
      fpp, gpp  = fluxes(upp, equations)
      st = calc_source_t_N34(us_node, up, upp, um, umm, x, t, dt,
                             source_terms, equations, dg, cache)

      ft = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
      gt = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)

      multiply_add_to_node_vars!(F, 1.0/3.0, ft, equations, dg, i, j)

      multiply_add_to_node_vars!(G, 1.0/3.0, gt, equations, dg, i, j)

      multiply_add_to_node_vars!(U, 1.0/3.0, ust_node, equations, dg, i, j)

      multiply_add_to_node_vars!(S, 1.0/3.0, st, equations, dg, i, j) # Source term

      F_node = get_node_vars(F, equations, dg, i, j)
      G_node = get_node_vars(G, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], F_node, equations, dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], G_node, equations, dg, i, jj, element)
      end

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)

      set_node_vars!(element_cache.U, U_node, equations, dg,    i, j, element)
      set_node_vars!(element_cache.F, F_node, equations, dg, 1, i, j, element)
      set_node_vars!(element_cache.F, G_node, equations, dg, 2, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations,
                                       dg, i, j, element)
   end

   return nothing
end
