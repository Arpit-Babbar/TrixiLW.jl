function rhs_mdrk1!(du, u, t,
   mesh::StructuredMesh{2}, equations,
   initial_condition, boundary_conditions, source_terms,
   dg::DG, time_discretization::AbstractLWTimeDiscretization, cache, tolerances::NamedTuple)

   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

   dt = cache.dt[1]

   # Update dt in cache and the callback will just take it from there

   # Calculate volume integral
   alpha = @trixi_timeit timer() "volume integral" calc_volume_integral_mdrk1!(
      du, u, t, dt, tolerances, mesh,
      have_nonconservative_terms(equations), source_terms, equations,
      dg.volume_integral, time_discretization, dg, cache)

   # Calculate interface fluxes
   @trixi_timeit timer() "interface flux" calc_interface_flux!(
      cache, u, dt, mesh,
      have_nonconservative_terms(equations), equations,
      dg.surface_integral, time_discretization, alpha, dg)

   # Calculate boundary fluxes
   @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
      cache, u, t, dt, boundary_conditions, mesh, equations, dg.surface_integral,
      time_discretization, dg)

   # Calculate surface integrals
   @trixi_timeit timer() "surface integral" calc_surface_integral!(
      du, u, mesh, equations, dg.surface_integral, dg, cache)

   # Apply Jacobian from mapping to reference element
   @trixi_timeit timer() "Jacobian" apply_jacobian!(
      du, mesh, equations, dg, cache)

   return nothing
end

function rhs_mdrk2!(du, u,
   t, mesh::StructuredMesh{2}, equations,
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

   # Calculate interface fluxes
   @trixi_timeit timer() "interface flux" calc_interface_flux!(
      cache, u, dt, mesh, # TODO - us or u or both?
      have_nonconservative_terms(equations), equations,
      dg.surface_integral, time_discretization, alpha, dg)

   # Calculate boundary fluxes
   @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
      cache, u, t, dt, boundary_conditions, mesh, equations, dg.surface_integral,
      time_discretization, dg)

   # Calculate surface integrals
   @trixi_timeit timer() "surface integral" calc_surface_integral!(
      du, u, mesh, equations, dg.surface_integral, dg, cache)

   # Apply Jacobian from mapping to reference element
   @trixi_timeit timer() "Jacobian" apply_jacobian!(
      du, mesh, equations, dg, cache)

   return nothing
end

@inline function mdrk_kernel_1!(du, u, t, dt, tolerances, element,
   mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
   nonconservative_terms::False, source_terms, equations, dg::DGSEM, cache, alpha=true)
   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack contravariant_vectors, node_coordinates, inverse_jacobian = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack mdrk_cache = element_cache

   id = Threads.threadid()

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   # TODO - Are the local F2, G2, U2 needed? This is a performance question
   # Benchmark and find out
   Ftilde, Gtilde, ut, U, U2, S, S2 = cell_arrays[id]
   @unpack u_low = mdrk_cache
   refresh!.((ut,))
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      flux1, flux2, cv_flux1, cv_flux2 = contravariant_flux(u_node, Ja, equations)

      set_node_vars!(element_cache.F, 0.5*flux1, equations, dg, 1, i, j, element)
      set_node_vars!(element_cache.F, 0.5*flux2, equations, dg, 2, i, j, element)
      set_node_vars!(mdrk_cache.F2, flux1, equations, dg, 1, i, j, element)
      set_node_vars!(mdrk_cache.F2, flux2, equations, dg, 2, i, j, element)

      set_node_vars!(Ftilde, 0.5*cv_flux1, equations, dg, i, j)
      set_node_vars!(Gtilde, 0.5*cv_flux2, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1,
            equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2,
            equations, dg, i, jj)
      end

      set_node_vars!(U, 0.5*u_node, equations, dg, i, j)
      set_node_vars!(U2, u_node, equations, dg, i, j)
      set_node_vars!(u_low, u_node, equations, dg, i, j, element)
   end

   # Scale ut
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
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

   for j in eachnode(dg), i in eachnode(dg)
      u_node  = get_node_vars(u,  equations, dg, i, j, element)
      ut_node = get_node_vars(ut, equations, dg, i, j)

      um  = u_node - ut_node
      up  = u_node + ut_node
      umm = u_node - 2*ut_node
      upp = u_node + 2*ut_node

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fm, gm, cv_fm, cv_gm = contravariant_flux(um, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp, Ja, equations)

      ft = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
      gt = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)
      ftilde_t = 1.0 / 12.0 * (-cv_fpp + 8.0 * cv_fp - 8.0 * cv_fm + cv_fmm)
      gtilde_t = 1.0 / 12.0 * (-cv_gpp + 8.0 * cv_gp - 8.0 * cv_gm + cv_gmm)

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N34(u_node, up, upp, um, umm, x, t, dt,
                             source_terms, equations, dg, cache)

      ftilde_node = 2.0*get_node_vars(Ftilde, equations, dg, i, j)
      gtilde_node = 2.0*get_node_vars(Gtilde, equations, dg, i, j)
      s_node = 2.0*get_node_vars(S, equations, dg, i, j)
      Ftilde_node_low = ftilde_node + 0.5 * ftilde_t
      Gtilde_node_low = gtilde_node + 0.5 * gtilde_t
      S_node_low = s_node + 0.5 * st

      multiply_add_to_node_vars!(element_cache.F, 0.125, ft, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 0.125, gt, equations, dg, 2, i, j, element)
      multiply_add_to_node_vars!(Ftilde, 0.125, ftilde_t, equations, dg, i, j)
      multiply_add_to_node_vars!(Gtilde, 0.125, gtilde_t, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 0.125, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(S, 0.125, st, equations, dg, i, j) # Source term

      multiply_add_to_node_vars!(mdrk_cache.F2, 1.0/6.0, ft, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(mdrk_cache.F2, 1.0/6.0, gt, equations, dg, 2, i, j, element)
      multiply_add_to_node_vars!(U2, 1.0/6.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(S2, 1.0/6.0, st, equations, dg, i, j) # Source term

      F_node = get_node_vars(Ftilde, equations, dg, i, j)
      G_node = get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], F_node,
                                    equations, dg, ii, j, element)
         inv_jacobian = inverse_jacobian[ii,j,element]
         multiply_add_to_node_vars!(u_low, -dt * inv_jacobian * derivative_matrix[ii, i],
                                    Ftilde_node_low, equations, dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], G_node,
                                    equations, dg, i, jj, element)
         inv_jacobian = inverse_jacobian[i,jj,element]
         multiply_add_to_node_vars!(u_low, -dt * inv_jacobian * derivative_matrix[jj, j],
                                    Gtilde_node_low, equations, dg, i, jj, element)
      end

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node  = get_node_vars(U,  equations, dg, i, j)
      U2_node = get_node_vars(U2, equations, dg, i, j)

      set_node_vars!(element_cache.U, U_node,  equations, dg, i, j, element)
      set_node_vars!(mdrk_cache.U2,   U2_node, equations, dg, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      S2_node = get_node_vars(S2, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i,j,element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations,
                                 dg, i, j, element)
      multiply_add_to_node_vars!(u_low, dt, S_node_low, equations, dg, i, j, element)

      set_node_vars!(mdrk_cache.S2, S2_node, equations, dg, i, j, element)
   end

   return nothing
end

@inline function mdrk_kernel_2!(du, us, t, dt, tolerances, element,
   mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
   nonconservative_terms::False, source_terms, equations, dg::DGSEM, cache, alpha=true)

   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack mdrk_cache = element_cache

   @unpack U2, F2, S2 = mdrk_cache

   id = Threads.threadid()

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   Ftilde, Gtilde, F, G, ut, ust, U, S = cell_arrays[id]

   # Load U2, F2 in local arrays
   for j in eachnode(dg), i in eachnode(dg)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      (Ja11, Ja12), (Ja21, Ja22) = Ja
      flux1 = get_node_vars(F2, equations, dg, 1, i, j, element)
      flux2 = get_node_vars(F2, equations, dg, 2, i, j, element)
      for n in eachvariable(equations)
         F[n, i, j] = flux1[n]
         G[n, i, j] = flux2[n]
         contravariant_flux1 = Ja11 * flux1[n] + Ja12 * flux2[n]
         contravariant_flux2 = Ja21 * flux1[n] + Ja22 * flux2[n]
         Ftilde[n, i, j] = contravariant_flux1
         Gtilde[n, i, j] = contravariant_flux2
         U[n,i,j] = U2[n,i,j,element]
      end
   end

   refresh!.((ut, ust))
   for j in eachnode(dg), i in eachnode(dg)
      us_node = get_node_vars(us, equations, dg, i, j, element)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      _, _, cv_flux1, cv_flux2 = contravariant_flux(us_node, Ja, equations)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ust, -dt * derivative_matrix[ii, i], cv_flux1,
            equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ust, -dt * derivative_matrix[jj, j], cv_flux2,
            equations, dg, i, jj)
      end
   end

   # Scale ust
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
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
      us_node = get_node_vars(us,   equations, dg, i, j, element)
      ust_node = get_node_vars(ust, equations, dg, i, j)

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      um  = us_node -   ust_node
      fm, gm, cv_fm, cv_gm = contravariant_flux(um, Ja, equations)
      up  = us_node +   ust_node
      fp, gp, cv_fp, cv_gp = contravariant_flux(up, Ja, equations)
      umm = us_node - 2*ust_node
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm, Ja, equations)
      upp = us_node + 2*ust_node
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp, Ja, equations)
      st = calc_source_t_N34(us_node, up, upp, um, umm, x, t, dt,
                             source_terms, equations, dg, cache)

      ft = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
      gt = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)
      ftilde_t = 1.0 / 12.0 * (-cv_fpp + 8.0 * cv_fp - 8.0 * cv_fm + cv_fmm)
      gtilde_t = 1.0 / 12.0 * (-cv_gpp + 8.0 * cv_gp - 8.0 * cv_gm + cv_gmm)

      multiply_add_to_node_vars!(F, 1.0/3.0, ft, equations, dg, i, j)
      multiply_add_to_node_vars!(G, 1.0/3.0, gt, equations, dg, i, j)

      multiply_add_to_node_vars!(Ftilde, 1.0/3.0, ftilde_t, equations, dg, i, j)
      multiply_add_to_node_vars!(Gtilde, 1.0/3.0, gtilde_t, equations, dg, i, j)

      multiply_add_to_node_vars!(U, 1.0/3.0, ust_node, equations, dg, i, j)
      multiply_add_to_node_vars!(S, 1.0/3.0, st, equations, dg, i, j) # Source term

      Ftilde_node = get_node_vars(Ftilde, equations, dg, i, j)
      Gtilde_node = get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], Ftilde_node, equations, dg,
            ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], Gtilde_node, equations, dg,
            i, jj, element)
      end

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)
      F_node = get_node_vars(F, equations, dg, i, j)
      G_node = get_node_vars(G, equations, dg, i, j)

      set_node_vars!(element_cache.U, U_node, equations, dg,    i, j, element)
      set_node_vars!(element_cache.F, F_node, equations, dg, 1, i, j, element)
      set_node_vars!(element_cache.F, G_node, equations, dg, 2, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
   end

   return nothing
end
