import Trixi: create_cache
using Trixi: dot

function contravariant_fluxes(u, grad_u, i, j, element, contravariant_vectors,
   equations::AbstractEquations{2}, equations_parabolic::AbstractEquationsParabolic{2})
   Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
   Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
   (fa, ga), (fv, gv) = fluxes(u, grad_u, equations, equations_parabolic)

   # Contravariant fluxes
   cv_fa = Ja11 * fa + Ja12 * ga
   cv_ga = Ja21 * fa + Ja22 * ga
   cv_fv = Ja11 * fv + Ja12 * gv
   cv_gv = Ja21 * fv + Ja22 * gv

   cv_f = cv_fa - cv_fv
   cv_g = cv_ga - cv_gv

   return fa, ga, fv, gv, cv_f, cv_g
end

function contravariant_fluxes(u, grad_u, Ja,
   equations::AbstractEquations{2}, equations_parabolic::AbstractEquationsParabolic{2})
   (Ja11, Ja12), (Ja21, Ja22) = Ja
   (fa, ga), (fv, gv) = fluxes(u, grad_u, equations, equations_parabolic)

   # Contravariant fluxes
   cv_fa = Ja11 * fa + Ja12 * ga
   cv_ga = Ja21 * fa + Ja22 * ga
   cv_fv = Ja11 * fv + Ja12 * gv
   cv_gv = Ja21 * fv + Ja22 * gv

   cv_f = cv_fa - cv_fv
   cv_g = cv_ga - cv_gv

   return fa, ga, fv, gv, cv_f, cv_g
end

function lw_volume_kernel_1!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::P4estMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates, contravariant_vectors = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack inverse_jacobian = cache.elements

   id = Threads.threadid()

   F, G, ut, U, up, um, S = cell_arrays[id]

   utx, uty, upx, upy, umx, umy,
   u_np1, u_np1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   refresh!.((ut, utx, uty))

   # Calculate volume terms in one element
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      flux_adv_1, flux_adv_2, cv_flux_adv_1, cv_flux_adv_2 = contravariant_flux(u_node, Ja, equations)

      flux_visc_1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, element)
      flux_visc_2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, element)
      (Ja11, Ja12), (Ja21, Ja22) = Ja
      cv_flux_visc_1 = Ja11 * flux_visc_1 + Ja12 * flux_visc_2
      cv_flux_visc_2 = Ja21 * flux_visc_1 + Ja22 * flux_visc_2

      set_node_vars!(element_cache.F, flux_adv_1, equations, dg, 1, i, j, element) # This is Fa
      set_node_vars!(cache_parabolic.Fv, flux_visc_1, equations, dg, 1, i, j, element)
      set_node_vars!(element_cache.F, flux_adv_2, equations, dg, 2, i, j, element) # This is Ga
      set_node_vars!(cache_parabolic.Fv, flux_visc_2, equations, dg, 2, i, j, element)

      cv_flux1 = cv_flux_adv_1 - cv_flux_visc_1
      cv_flux2 = cv_flux_adv_2 - cv_flux_visc_2

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end

      set_node_vars!(F, cv_flux1, equations, dg, i, j)
      set_node_vars!(G, cv_flux2, equations, dg, i, j)

      set_node_vars!(u_np1, u_node, equations, dg, i, j)
      set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

      set_node_vars!(um, u_node, equations, dg, i, j)
      set_node_vars!(up, u_node, equations, dg, i, j)

      ux_node = get_node_vars(gradients_x, equations, dg, i, j, element)
      set_node_vars!(upx, ux_node, equations, dg, i, j)
      set_node_vars!(umx, ux_node, equations, dg, i, j)

      uy_node = get_node_vars(gradients_y, equations, dg, i, j, element)
      set_node_vars!(upy, uy_node, equations, dg, i, j)
      set_node_vars!(umy, uy_node, equations, dg, i, j)

      set_node_vars!(U, u_node, equations, dg, i, j)
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
      set_node_vars!(S, s_node, equations, dg, i, j)
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
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utx[v, i, j] *= inv_jacobian
         uty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = get_node_vars(ut, equations, dg, i, j)
      utx_node = get_node_vars(utx, equations, dg, i, j)
      uty_node = get_node_vars(uty, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upx, 1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, -1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 1.0, uty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, -1.0, uty_node, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      fta = 0.5 * (fpa - fma)
      gta = 0.5 * (gpa - gma)
      multiply_add_to_node_vars!(element_cache.F, 0.5, fta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 0.5, gta, equations, dg, 2, i, j, element)

      ftv = 0.5 * (fpv - fmv)
      gtv = 0.5 * (gpv - gmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, ftv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, gtv, equations, dg, 2, i, j, element)

      ft = 0.5 * (cv_fp - cv_fm)
      gt = 0.5 * (cv_gp - cv_gm)
      multiply_add_to_node_vars!(F, 0.5, ft, equations, dg, i, j)
      multiply_add_to_node_vars!(G, 0.5, gt, equations, dg, i, j)

      F_node = get_node_vars(F, equations, dg, i, j)
      G_node = get_node_vars(G, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F_node, equations,
            dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G_node, equations,
            dg, i, jj, element)
      end

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N12(up_node, um_node, x, t, dt, source_terms, equations,
         dg, cache)
      multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)

      # Ub = UT * V
      # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
      set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
   end

   return nothing
end

function lw_volume_kernel_2!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::P4estMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates, contravariant_vectors = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack inverse_jacobian = cache.elements

   id = Threads.threadid()

   cv_F, cv_G, fa, ga, cv_f, cv_g, ut, utt, U, up, um, S = cell_arrays[id]

   fv, gv, utx, uty, uttx, utty, upx, upy, umx, umy,
   u_np1, u_np1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   refresh!.((ut, utx, uty, utt, uttx, utty))

   # Calculate volume terms in one element
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      flux_adv_1, flux_adv_2, cv_flux_adv_1, cv_flux_adv_2 = contravariant_flux(u_node, Ja, equations)

      flux_visc_1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, element)
      flux_visc_2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, element)
      (Ja11, Ja12), (Ja21, Ja22) = Ja
      cv_flux_visc_1 = Ja11 * flux_visc_1 + Ja12 * flux_visc_2
      cv_flux_visc_2 = Ja21 * flux_visc_1 + Ja22 * flux_visc_2

      set_node_vars!(element_cache.F, flux_adv_1, equations, dg, 1, i, j, element) # Fa
      set_node_vars!(fa, flux_adv_1, equations, dg, i, j)

      set_node_vars!(cache_parabolic.Fv, flux_visc_1, equations, dg, 1, i, j, element) # Fv
      set_node_vars!(fv, flux_visc_1, equations, dg, i, j)

      set_node_vars!(element_cache.F, flux_adv_2, equations, dg, 2, i, j, element) # Ga
      set_node_vars!(ga, flux_adv_2, equations, dg, i, j)

      set_node_vars!(cache_parabolic.Fv, flux_visc_2, equations, dg, 2, i, j, element) # Gv
      set_node_vars!(gv, flux_visc_2, equations, dg, i, j)

      cv_flux1 = cv_flux_adv_1 - cv_flux_visc_1
      cv_flux2 = cv_flux_adv_2 - cv_flux_visc_2

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end

      set_node_vars!(cv_f, cv_flux1, equations, dg, i, j)
      set_node_vars!(cv_g, cv_flux2, equations, dg, i, j)

      set_node_vars!(cv_F, cv_flux1, equations, dg, i, j)
      set_node_vars!(cv_G, cv_flux2, equations, dg, i, j)

      set_node_vars!(u_np1, u_node, equations, dg, i, j)
      set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

      set_node_vars!(um, u_node, equations, dg, i, j)
      set_node_vars!(up, u_node, equations, dg, i, j)

      ux_node = get_node_vars(gradients_x, equations, dg, i, j, element)
      set_node_vars!(upx, ux_node, equations, dg, i, j)
      set_node_vars!(umx, ux_node, equations, dg, i, j)

      uy_node = get_node_vars(gradients_y, equations, dg, i, j, element)
      set_node_vars!(upy, uy_node, equations, dg, i, j)
      set_node_vars!(umy, uy_node, equations, dg, i, j)

      set_node_vars!(U, u_node, equations, dg, i, j)
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
      set_node_vars!(S, s_node, equations, dg, i, j)
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
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utx[v, i, j] *= inv_jacobian
         uty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = get_node_vars(ut, equations, dg, i, j)
      utx_node = get_node_vars(utx, equations, dg, i, j)
      uty_node = get_node_vars(uty, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upx, 1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, -1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 1.0, uty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, -1.0, uty_node, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      fta = 0.5 * (fpa - fma)
      gta = 0.5 * (gpa - gma)
      multiply_add_to_node_vars!(element_cache.F, 0.5, fta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 0.5, gta, equations, dg, 2, i, j, element)

      ftv = 0.5 * (fpv - fmv)
      gtv = 0.5 * (gpv - gmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, ftv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, gtv, equations, dg, 2, i, j, element)

      cv_ft = 0.5 * (cv_fp - cv_fm)
      cv_gt = 0.5 * (cv_gp - cv_gm)
      multiply_add_to_node_vars!(cv_F, 0.5, cv_ft, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 0.5, cv_gt, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], cv_ft, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], cv_gt, equations,
            dg, i, jj)
      end
   end

   # Scale utt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to utt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N12(up_node, um_node, x, t, dt, source_terms,
         equations, dg, cache)
      multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)
      multiply_add_to_node_vars!(utt, dt, st, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇utt
   for j in eachnode(dg), i in eachnode(dg)
      utt_node = get_node_vars(utt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(uttx, derivative_matrix[ii, i], utt_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utty, derivative_matrix[jj, j], utt_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇u_tt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         uttx[v, i, j] *= inv_jacobian
         utty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      utt_node = get_node_vars(utt, equations, dg, i, j)

      multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)

      uttx_node = get_node_vars(uttx, equations, dg, i, j)
      multiply_add_to_node_vars!(upx, 0.5, uttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, 0.5, uttx_node, equations, dg, i, j)

      utty_node = get_node_vars(utty, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 0.5, utty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, 0.5, utty_node, equations, dg, i, j)

      fa_node = get_node_vars(fa, equations, dg, i, j)
      ga_node = get_node_vars(ga, equations, dg, i, j)
      fv_node = get_node_vars(fv, equations, dg, i, j)
      gv_node = get_node_vars(gv, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      ftta, gtta = fpa - 2.0 * fa_node + fma, gpa - 2.0 * ga_node + gma
      fttv, gttv = fpv - 2.0 * fv_node + fmv, gpv - 2.0 * gv_node + gmv
      multiply_add_to_node_vars!(element_cache.F, 1.0/6.0, ftta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 1.0/6.0, gtta, equations, dg, 2, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0/6.0, fttv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0/6.0, gttv, equations, dg, 2, i, j, element)

      cv_f_node = get_node_vars(cv_f, equations, dg, i, j)
      cv_g_node = get_node_vars(cv_g, equations, dg, i, j)
      cv_ftt, cv_gtt = cv_fp - 2.0 * cv_f_node + cv_fm, cv_gp - 2.0 * cv_g_node + cv_gm
      multiply_add_to_node_vars!(cv_F, 1.0/6.0, cv_ftt, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 1.0/6.0, cv_gtt, equations, dg, i, j)

      F_node = get_node_vars(cv_F, equations, dg, i, j)
      G_node = get_node_vars(cv_G, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F_node, equations,
            dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G_node, equations,
            dg, i, jj, element)
      end

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      stt = calc_source_tt_N23(u_node, up_node, um_node, x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, equations, dg, i, j)


      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)

      # Ub = UT * V
      # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
      set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
   end
   return nothing
end

function lw_volume_kernel_3!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::P4estMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates, contravariant_vectors = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack inverse_jacobian = cache.elements

   id = Threads.threadid()

   cv_F, cv_G, fa, ga, cv_f, cv_g, ut, utt, uttt, U, up, um, upp, umm, S = cell_arrays[id]

   fv, gv, utx, uty, uttx, utty, utttx, uttty, upx, upy, umx, umy, uppx, uppy, ummx, ummy,
   u_np1, u_np1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   refresh!.((ut, utx, uty, utt, uttx, utty, uttt, utttx, uttty))

   # Calculate volume terms in one element
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      flux_adv_1, flux_adv_2, cv_flux_adv_1, cv_flux_adv_2 = contravariant_flux(u_node, Ja, equations)

      flux_visc_1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, element)
      flux_visc_2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, element)
      (Ja11, Ja12), (Ja21, Ja22) = Ja
      cv_flux_visc_1 = Ja11 * flux_visc_1 + Ja12 * flux_visc_2
      cv_flux_visc_2 = Ja21 * flux_visc_1 + Ja22 * flux_visc_2

      set_node_vars!(element_cache.F, flux_adv_1, equations, dg, 1, i, j, element) # Fa
      set_node_vars!(fa, flux_adv_1, equations, dg, i, j)

      set_node_vars!(cache_parabolic.Fv, flux_visc_1, equations, dg, 1, i, j, element) # Fv
      set_node_vars!(fv, flux_visc_1, equations, dg, i, j)

      set_node_vars!(element_cache.F, flux_adv_2, equations, dg, 2, i, j, element) # Ga
      set_node_vars!(ga, flux_adv_2, equations, dg, i, j)

      set_node_vars!(cache_parabolic.Fv, flux_visc_2, equations, dg, 2, i, j, element) # Gv
      set_node_vars!(gv, flux_visc_2, equations, dg, i, j)

      cv_flux1 = cv_flux_adv_1 - cv_flux_visc_1
      cv_flux2 = cv_flux_adv_2 - cv_flux_visc_2

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end

      set_node_vars!(cv_f, cv_flux1, equations, dg, i, j)
      set_node_vars!(cv_g, cv_flux2, equations, dg, i, j)

      set_node_vars!(cv_F, cv_flux1, equations, dg, i, j)
      set_node_vars!(cv_G, cv_flux2, equations, dg, i, j)

      set_node_vars!(u_np1, u_node, equations, dg, i, j)
      set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

      set_node_vars!(um, u_node, equations, dg, i, j)
      set_node_vars!(up, u_node, equations, dg, i, j)
      set_node_vars!(umm, u_node, equations, dg, i, j)
      set_node_vars!(upp, u_node, equations, dg, i, j)

      ux_node = get_node_vars(gradients_x, equations, dg, i, j, element)
      set_node_vars!(upx, ux_node, equations, dg, i, j)
      set_node_vars!(umx, ux_node, equations, dg, i, j)
      set_node_vars!(uppx, ux_node, equations, dg, i, j)
      set_node_vars!(ummx, ux_node, equations, dg, i, j)

      uy_node = get_node_vars(gradients_y, equations, dg, i, j, element)
      set_node_vars!(upy, uy_node, equations, dg, i, j)
      set_node_vars!(umy, uy_node, equations, dg, i, j)
      set_node_vars!(uppy, uy_node, equations, dg, i, j)
      set_node_vars!(ummy, uy_node, equations, dg, i, j)

      set_node_vars!(U, u_node, equations, dg, i, j)
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
      set_node_vars!(S, s_node, equations, dg, i, j)
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
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utx[v, i, j] *= inv_jacobian
         uty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = get_node_vars(ut, equations, dg, i, j)
      utx_node = get_node_vars(utx, equations, dg, i, j)
      uty_node = get_node_vars(uty, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umm, -2.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upp, 2.0, ut_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upx, 1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, -1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 1.0, uty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, -1.0, uty_node, equations, dg, i, j)

      multiply_add_to_node_vars!(ummx, -2.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppx, 2.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummy, -2.0, uty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppy, 2.0, uty_node, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umm_node = get_node_vars(umm, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)

      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)

      ummx_node = get_node_vars(ummx, equations, dg, i, j)
      uppx_node = get_node_vars(uppx, equations, dg, i, j)
      ummy_node = get_node_vars(ummy, equations, dg, i, j)
      uppy_node = get_node_vars(uppy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      fmma, gmma, fmmv, gmmv, cv_fmm, cv_gmm = contravariant_fluxes(
         umm_node, (ummx_node, ummy_node), Ja, equations, equations_parabolic)
      fppa, gppa, fppv, gppv, cv_fpp, cv_gpp = contravariant_fluxes(
         upp_node, (uppx_node, uppy_node), Ja, equations, equations_parabolic)

      fta = 1.0 / 12.0 * (-fppa + 8.0 * fpa - 8.0 * fma + fmma)
      gta = 1.0 / 12.0 * (-gppa + 8.0 * gpa - 8.0 * gma + gmma)
      multiply_add_to_node_vars!(element_cache.F, 0.5, fta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 0.5, gta, equations, dg, 2, i, j, element)

      ftv = 1.0 / 12.0 * (-fppv + 8.0 * fpv - 8.0 * fmv + fmmv)
      gtv = 1.0 / 12.0 * (-gppv + 8.0 * gpv - 8.0 * gmv + gmmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, ftv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, gtv, equations, dg, 2, i, j, element)

      cv_ft = 1.0 / 12.0 * (-cv_fpp + 8.0 * cv_fp - 8.0 * cv_fm + cv_fmm)
      cv_gt = 1.0 / 12.0 * (-cv_gpp + 8.0 * cv_gp - 8.0 * cv_gm + cv_gmm)

      multiply_add_to_node_vars!(cv_F, 0.5, cv_ft, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 0.5, cv_gt, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], cv_ft, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], cv_gt, equations,
            dg, i, jj)
      end
   end

   # Scale utt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to utt and some to S
   # Add source term contribution to utt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N34(u_node, up_node, upp_node, um_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(utt, dt, st, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇utt
   for j in eachnode(dg), i in eachnode(dg)
      utt_node = get_node_vars(utt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(uttx, derivative_matrix[ii, i], utt_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utty, derivative_matrix[jj, j], utt_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇u_tt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         uttx[v, i, j] *= inv_jacobian
         utty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      utt_node = get_node_vars(utt, equations, dg, i, j)
      uttx_node = get_node_vars(uttx, equations, dg, i, j)
      utty_node = get_node_vars(utty, equations, dg, i, j)

      multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upp, 2.0, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umm, 2.0, utt_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upx, 0.5, uttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, 0.5, uttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upy, 0.5, utty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, 0.5, utty_node, equations, dg, i, j)

      multiply_add_to_node_vars!(uppx, 2.0, uttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummx, 2.0, uttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(uppy, 2.0, utty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummy, 2.0, utty_node, equations, dg, i, j)

      fa_node = get_node_vars(fa, equations, dg, i, j)
      ga_node = get_node_vars(ga, equations, dg, i, j)
      fv_node = get_node_vars(fv, equations, dg, i, j)
      gv_node = get_node_vars(gv, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      ftta, gtta = fpa - 2.0 * fa_node + fma, gpa - 2.0 * ga_node + gma
      fttv, gttv = fpv - 2.0 * fv_node + fmv, gpv - 2.0 * gv_node + gmv
      multiply_add_to_node_vars!(element_cache.F, 1.0/6.0, ftta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 1.0/6.0, gtta, equations, dg, 2, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0/6.0, fttv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0/6.0, gttv, equations, dg, 2, i, j, element)

      cv_f_node = get_node_vars(cv_f, equations, dg, i, j)
      cv_g_node = get_node_vars(cv_g, equations, dg, i, j)
      cv_ftt, cv_gtt = cv_fp - 2.0 * cv_f_node + cv_fm, cv_gp - 2.0 * cv_g_node + cv_gm
      multiply_add_to_node_vars!(cv_F, 1.0/6.0, cv_ftt, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 1.0/6.0, cv_gtt, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], cv_ftt, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], cv_gtt, equations,
            dg, i, jj)
      end
   end

   # Apply Jacobian to uttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i,j,element]
      for v in eachvariable(equations)
         uttt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to uttt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = get_node_vars(u, equations, dg, i, j, element)
      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stt = calc_source_tt_N23(u_node, up_node, um_node, x, t, dt, source_terms,
         equations, dg, cache)
      multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, equations, dg, i, j)
      multiply_add_to_node_vars!(uttt, dt, stt, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇uttt
   for j in eachnode(dg), i in eachnode(dg)
      uttt_node = get_node_vars(uttt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(utttx, derivative_matrix[ii, i], uttt_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(uttty, derivative_matrix[jj, j], uttt_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇uttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utttx[v, i, j] *= inv_jacobian
         uttty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      uttt_node = get_node_vars(uttt, equations, dg, i, j)
      utttx_node = get_node_vars(utttx, equations, dg, i, j)
      uttty_node = get_node_vars(uttty, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, equations, dg, i, j)

      multiply_add_to_node_vars!(umx, -1.0 / 6.0, utttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upx, 1.0 / 6.0, utttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummx, -4.0 / 3.0, utttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppx, 4.0 / 3.0, utttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(umy, -1.0 / 6.0, uttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 1.0 / 6.0, uttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummy, -4.0 / 3.0, uttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppy, 4.0 / 3.0, uttty_node, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umm_node = get_node_vars(umm, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)

      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      ummx_node = get_node_vars(ummx, equations, dg, i, j)
      uppx_node = get_node_vars(uppx, equations, dg, i, j)

      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)
      ummy_node = get_node_vars(ummy, equations, dg, i, j)
      uppy_node = get_node_vars(uppy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      fmma, gmma, fmmv, gmmv, cv_fmm, cv_gmm = contravariant_fluxes(
         umm_node, (ummx_node, ummy_node), Ja, equations, equations_parabolic)
      fppa, gppa, fppv, gppv, cv_fpp, cv_gpp = contravariant_fluxes(
         upp_node, (uppx_node, uppy_node), Ja, equations, equations_parabolic)

      fttta = 0.5 * (fppa - 2.0 * fpa + 2.0 * fma - fmma)
      multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, fttta, equations, dg, 1, i, j, element)
      gttta = 0.5 * (gppa - 2.0 * gpa + 2.0 * gma - gmma)
      multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, gttta, equations, dg, 2, i, j, element)

      ftttv = 0.5 * (fppv - 2.0 * fpv + 2.0 * fmv - fmmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0 / 24.0, ftttv, equations, dg, 1, i, j, element)
      gtttv = 0.5 * (gppv - 2.0 * gpv + 2.0 * gmv - gmmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0 / 24.0, gtttv, equations, dg, 2, i, j, element)

      cv_fttt = 0.5 * (cv_fpp - 2.0 * cv_fp + 2.0 * cv_fm - cv_fmm)
      cv_gttt = 0.5 * (cv_gpp - 2.0 * cv_gp + 2.0 * cv_gm - cv_gmm)

      multiply_add_to_node_vars!(cv_F, 1.0 / 24.0, cv_fttt, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 1.0 / 24.0, cv_gttt, equations, dg, i, j)

      F_node = get_node_vars(cv_F, equations, dg, i, j)
      G_node = get_node_vars(cv_G, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F_node, equations,
            dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G_node, equations,
            dg, i, jj, element)
      end

      u_node = get_node_vars(u, equations, dg, i, j, element)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      sttt = calc_source_ttt_N34(u_node, up_node, um_node, upp_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, equations, dg, i, j)


      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)

      # Ub = UT * V
      # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
      set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
   end
   return nothing
end

function lw_volume_kernel_4!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::P4estMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates, contravariant_vectors = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   @unpack inverse_jacobian = cache.elements

   id = Threads.threadid()

   cv_F, cv_G, fa, ga, cv_f, cv_g, ut, utt, uttt, utttt, U, up, um, upp, umm, S = cell_arrays[id]

   fv, gv, utx, uty, uttx, utty, utttx, uttty, uttttx, utttty, upx, upy, umx, umy, uppx, uppy,
   ummx, ummy, u_np1, u_np1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   refresh!.((ut, utx, uty, utt, uttx, utty, uttt, utttx, uttty, utttt, uttttx, utttty))

   # Calculate volume terms in one element
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      flux_adv_1, flux_adv_2, cv_flux_adv_1, cv_flux_adv_2 = contravariant_flux(u_node, Ja, equations)

      flux_visc_1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, element)
      flux_visc_2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, element)
      (Ja11, Ja12), (Ja21, Ja22) = Ja
      cv_flux_visc_1 = Ja11 * flux_visc_1 + Ja12 * flux_visc_2
      cv_flux_visc_2 = Ja21 * flux_visc_1 + Ja22 * flux_visc_2

      set_node_vars!(element_cache.F, flux_adv_1, equations, dg, 1, i, j, element) # This is Fa
      set_node_vars!(fa, flux_adv_1, equations, dg, i, j)

      set_node_vars!(cache_parabolic.Fv, flux_visc_1, equations, dg, 1, i, j, element)
      set_node_vars!(fv, flux_visc_1, equations, dg, i, j)

      set_node_vars!(element_cache.F, flux_adv_2, equations, dg, 2, i, j, element) # This is Ga
      set_node_vars!(ga, flux_adv_2, equations, dg, i, j)

      set_node_vars!(cache_parabolic.Fv, flux_visc_2, equations, dg, 2, i, j, element)
      set_node_vars!(gv, flux_visc_2, equations, dg, i, j)

      cv_flux1 = cv_flux_adv_1 - cv_flux_visc_1
      cv_flux2 = cv_flux_adv_2 - cv_flux_visc_2

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end

      set_node_vars!(cv_f, cv_flux1, equations, dg, i, j)
      set_node_vars!(cv_g, cv_flux2, equations, dg, i, j)

      set_node_vars!(cv_F, cv_flux1, equations, dg, i, j)
      set_node_vars!(cv_G, cv_flux2, equations, dg, i, j)

      set_node_vars!(u_np1, u_node, equations, dg, i, j)
      set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

      set_node_vars!(um, u_node, equations, dg, i, j)
      set_node_vars!(up, u_node, equations, dg, i, j)
      set_node_vars!(umm, u_node, equations, dg, i, j)
      set_node_vars!(upp, u_node, equations, dg, i, j)

      ux_node = get_node_vars(gradients_x, equations, dg, i, j, element)
      set_node_vars!(upx, ux_node, equations, dg, i, j)
      set_node_vars!(umx, ux_node, equations, dg, i, j)
      set_node_vars!(uppx, ux_node, equations, dg, i, j)
      set_node_vars!(ummx, ux_node, equations, dg, i, j)

      uy_node = get_node_vars(gradients_y, equations, dg, i, j, element)
      set_node_vars!(upy, uy_node, equations, dg, i, j)
      set_node_vars!(umy, uy_node, equations, dg, i, j)
      set_node_vars!(uppy, uy_node, equations, dg, i, j)
      set_node_vars!(ummy, uy_node, equations, dg, i, j)

      set_node_vars!(U, u_node, equations, dg, i, j)
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
      set_node_vars!(S, s_node, equations, dg, i, j)
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
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utx[v, i, j] *= inv_jacobian
         uty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = get_node_vars(ut, equations, dg, i, j)
      utx_node = get_node_vars(utx, equations, dg, i, j)
      uty_node = get_node_vars(uty, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umm, -2.0, ut_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upp, 2.0, ut_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upx, 1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, -1.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 1.0, uty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, -1.0, uty_node, equations, dg, i, j)

      multiply_add_to_node_vars!(ummx, -2.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppx, 2.0, utx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummy, -2.0, uty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppy, 2.0, uty_node, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umm_node = get_node_vars(umm, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)

      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)

      ummx_node = get_node_vars(ummx, equations, dg, i, j)
      uppx_node = get_node_vars(uppx, equations, dg, i, j)
      ummy_node = get_node_vars(ummy, equations, dg, i, j)
      uppy_node = get_node_vars(uppy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      fmma, gmma, fmmv, gmmv, cv_fmm, cv_gmm = contravariant_fluxes(
         umm_node, (ummx_node, ummy_node), Ja, equations, equations_parabolic)
      fppa, gppa, fppv, gppv, cv_fpp, cv_gpp = contravariant_fluxes(
         upp_node, (uppx_node, uppy_node), Ja, equations, equations_parabolic)

      fta = 1.0 / 12.0 * (-fppa + 8.0 * fpa - 8.0 * fma + fmma)
      gta = 1.0 / 12.0 * (-gppa + 8.0 * gpa - 8.0 * gma + gmma)
      multiply_add_to_node_vars!(element_cache.F, 0.5, fta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 0.5, gta, equations, dg, 2, i, j, element)

      ftv = 1.0 / 12.0 * (-fppv + 8.0 * fpv - 8.0 * fmv + fmmv)
      gtv = 1.0 / 12.0 * (-gppv + 8.0 * gpv - 8.0 * gmv + gmmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, ftv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 0.5, gtv, equations, dg, 2, i, j, element)

      cv_ft = 1.0 / 12.0 * (-cv_fpp + 8.0 * cv_fp - 8.0 * cv_fm + cv_fmm)
      cv_gt = 1.0 / 12.0 * (-cv_gpp + 8.0 * cv_gp - 8.0 * cv_gm + cv_gmm)

      multiply_add_to_node_vars!(cv_F, 0.5, cv_ft, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 0.5, cv_gt, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], cv_ft, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], cv_gt, equations,
            dg, i, jj)
      end
   end

   # Scale utt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to utt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N34(u_node, up_node, upp_node, um_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(utt, dt, st, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇utt
   for j in eachnode(dg), i in eachnode(dg)
      utt_node = get_node_vars(utt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(uttx, derivative_matrix[ii, i], utt_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utty, derivative_matrix[jj, j], utt_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇u_tt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         uttx[v, i, j] *= inv_jacobian
         utty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      utt_node = get_node_vars(utt, equations, dg, i, j)
      uttx_node = get_node_vars(uttx, equations, dg, i, j)
      utty_node = get_node_vars(utty, equations, dg, i, j)

      multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upp, 2.0, utt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umm, 2.0, utt_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upx, 0.5, uttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, 0.5, uttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upy, 0.5, utty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, 0.5, utty_node, equations, dg, i, j)

      multiply_add_to_node_vars!(uppx, 2.0, uttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummx, 2.0, uttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(uppy, 2.0, utty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummy, 2.0, utty_node, equations, dg, i, j)

      fa_node = get_node_vars(fa, equations, dg, i, j)
      ga_node = get_node_vars(ga, equations, dg, i, j)
      fv_node = get_node_vars(fv, equations, dg, i, j)
      gv_node = get_node_vars(gv, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umm_node = get_node_vars(umm, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)

      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      ummx_node = get_node_vars(ummx, equations, dg, i, j)
      uppx_node = get_node_vars(uppx, equations, dg, i, j)

      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)
      ummy_node = get_node_vars(ummy, equations, dg, i, j)
      uppy_node = get_node_vars(uppy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)
      fmma, gmma, fmmv, gmmv, cv_fmm, cv_gmm = contravariant_fluxes(
         umm_node, (ummx_node, ummy_node), Ja, equations, equations_parabolic)
      fppa, gppa, fppv, gppv, cv_fpp, cv_gpp = contravariant_fluxes(
         upp_node, (uppx_node, uppy_node), Ja, equations, equations_parabolic)

      ftta = (1.0 / 12.0) * (-fppa + 16.0 * fpa - 30.0 * fa_node + 16.0 * fma - fmma)
      gtta = (1.0 / 12.0) * (-gppa + 16.0 * gpa - 30.0 * ga_node + 16.0 * gma - gmma)
      fttv = (1.0 / 12.0) * (-fppv + 16.0 * fpv - 30.0 * fv_node + 16.0 * fmv - fmmv)
      gttv = (1.0 / 12.0) * (-gppv + 16.0 * gpv - 30.0 * gv_node + 16.0 * gmv - gmmv)
      multiply_add_to_node_vars!(element_cache.F, 1.0/6.0, ftta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 1.0/6.0, gtta, equations, dg, 2, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0/6.0, fttv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0/6.0, gttv, equations, dg, 2, i, j, element)

      cv_f_node = get_node_vars(cv_f, equations, dg, i, j)
      cv_g_node = get_node_vars(cv_g, equations, dg, i, j)
      cv_ftt = (1.0 / 12.0) * (-cv_fpp + 16.0 * cv_fp - 30.0 * cv_f_node + 16.0 * cv_fm - cv_fmm)
      cv_gtt = (1.0 / 12.0) * (-cv_gpp + 16.0 * cv_gp - 30.0 * cv_g_node + 16.0 * cv_gm - cv_gmm)
      multiply_add_to_node_vars!(cv_F, 1.0/6.0, cv_ftt, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 1.0/6.0, cv_gtt, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], cv_ftt, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], cv_gtt, equations,
            dg, i, jj)
      end
   end

   # Apply Jacobian to uttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i,j,element]
      for v in eachvariable(equations)
         uttt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to uttt and some to S
   # Add source term contribution to uttt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stt = calc_source_tt_N4(u_node, up_node, upp_node, um_node, umm_node, x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(uttt, dt, stt, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇uttt
   for j in eachnode(dg), i in eachnode(dg)
      uttt_node = get_node_vars(uttt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(utttx, derivative_matrix[ii, i], uttt_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(uttty, derivative_matrix[jj, j], uttt_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇uttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utttx[v, i, j] *= inv_jacobian
         uttty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      uttt_node = get_node_vars(uttt, equations, dg, i, j)
      utttx_node = get_node_vars(utttx, equations, dg, i, j)
      uttty_node = get_node_vars(uttty, equations, dg, i, j)
      multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, equations, dg, i, j)

      multiply_add_to_node_vars!(umx, -1.0 / 6.0, utttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upx, 1.0 / 6.0, utttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummx, -4.0 / 3.0, utttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppx, 4.0 / 3.0, utttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(umy, -1.0 / 6.0, uttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 1.0 / 6.0, uttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummy, -4.0 / 3.0, uttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppy, 4.0 / 3.0, uttty_node, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umm_node = get_node_vars(umm, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)

      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      ummx_node = get_node_vars(ummx, equations, dg, i, j)
      uppx_node = get_node_vars(uppx, equations, dg, i, j)

      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)
      ummy_node = get_node_vars(ummy, equations, dg, i, j)
      uppy_node = get_node_vars(uppy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      fmma, gmma, fmmv, gmmv, cv_fmm, cv_gmm = contravariant_fluxes(
         umm_node, (ummx_node, ummy_node), Ja, equations, equations_parabolic)
      fppa, gppa, fppv, gppv, cv_fpp, cv_gpp = contravariant_fluxes(
         upp_node, (uppx_node, uppy_node), Ja, equations, equations_parabolic)

      fttta = 0.5 * (fppa - 2.0 * fpa + 2.0 * fma - fmma)
      multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, fttta, equations, dg, 1, i, j, element)
      gttta = 0.5 * (gppa - 2.0 * gpa + 2.0 * gma - gmma)
      multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, gttta, equations, dg, 2, i, j, element)

      ftttv = 0.5 * (fppv - 2.0 * fpv + 2.0 * fmv - fmmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0 / 24.0, ftttv, equations, dg, 1, i, j, element)
      gtttv = 0.5 * (gppv - 2.0 * gpv + 2.0 * gmv - gmmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0 / 24.0, gtttv, equations, dg, 2, i, j, element)

      cv_fttt = 0.5 * (cv_fpp - 2.0 * cv_fp + 2.0 * cv_fm - cv_fmm)
      cv_gttt = 0.5 * (cv_gpp - 2.0 * cv_gp + 2.0 * cv_gm - cv_gmm)

      multiply_add_to_node_vars!(cv_F, 1.0 / 24.0, cv_fttt, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 1.0 / 24.0, cv_gttt, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[ii, i], cv_fttt, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[jj, j], cv_gttt, equations,
            dg, i, jj)
      end
   end

   # Apply jacobian on utttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i,j,element]
      for v in eachvariable(equations)
         utttt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to utttt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = get_node_vars(u, equations, dg, i, j, element)
      um_node = get_node_vars(um, equations, dg, i, j)
      umm_node = get_node_vars(umm, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      sttt = calc_source_ttt_N34(u_node, up_node, upp_node, um_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, equations, dg, i, j)
      multiply_add_to_node_vars!(utttt, dt, sttt, equations, dg, i, j) # has no jacobian factor
   end

   # Compute ∇utttt
   for j in eachnode(dg), i in eachnode(dg)
      utttt_node = get_node_vars(utttt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(uttttx, derivative_matrix[ii, i], utttt_node, equations, dg,
            ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utttty, derivative_matrix[jj, j], utttt_node, equations, dg,
            i, jj)
      end
   end

   # Scale ∇utttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         uttttx[v, i, j] *= inv_jacobian
         utttty[v, i, j] *= inv_jacobian
      end
   end

   for j in eachnode(dg), i in eachnode(dg)
      utttt_node = get_node_vars(utttt, equations, dg, i, j)
      uttttx_node = get_node_vars(uttttx, equations, dg, i, j)
      utttty_node = get_node_vars(utttty, equations, dg, i, j)

      multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(um, 1.0 / 24.0, utttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(up, 1.0 / 24.0, utttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umm, 2.0 / 3.0, utttt_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upp, 2.0 / 3.0, utttt_node, equations, dg, i, j)

      multiply_add_to_node_vars!(umx, 1.0 / 24.0, uttttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upx, 1.0 / 24.0, uttttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummx, 2.0 / 3.0, uttttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppx, 2.0 / 3.0, uttttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(umy, 1.0 / 24.0, utttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(upy, 1.0 / 24.0, utttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(ummy, 2.0 / 3.0, utttty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(uppy, 2.0 / 3.0, utttty_node, equations, dg, i, j)

      fa_node = get_node_vars(fa, equations, dg, i, j)
      fv_node = get_node_vars(fv, equations, dg, i, j)

      ga_node = get_node_vars(ga, equations, dg, i, j)
      gv_node = get_node_vars(gv, equations, dg, i, j)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)
      umm_node = get_node_vars(umm, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)

      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      ummx_node = get_node_vars(ummx, equations, dg, i, j)
      uppx_node = get_node_vars(uppx, equations, dg, i, j)

      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)
      ummy_node = get_node_vars(ummy, equations, dg, i, j)
      uppy_node = get_node_vars(uppy, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fma, gma, fmv, gmv, cv_fm, cv_gm = contravariant_fluxes(
         um_node, (umx_node, umy_node), Ja, equations, equations_parabolic)
      fpa, gpa, fpv, gpv, cv_fp, cv_gp = contravariant_fluxes(
         up_node, (upx_node, upy_node), Ja, equations, equations_parabolic)

      fmma, gmma, fmmv, gmmv, cv_fmm, cv_gmm = contravariant_fluxes(
         umm_node, (ummx_node, ummy_node), Ja, equations, equations_parabolic)
      fppa, gppa, fppv, gppv, cv_fpp, cv_gpp = contravariant_fluxes(
         upp_node, (uppx_node, uppy_node), Ja, equations, equations_parabolic)

      ftttta = 0.5 * (fppa - 4.0 * fpa + 6.0 * fa_node - 4.0 * fma + fmma)
      gtttta = 0.5 * (gppa - 4.0 * gpa + 6.0 * ga_node - 4.0 * gma + gmma)
      multiply_add_to_node_vars!(element_cache.F, 1.0 / 120.0, ftttta, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 1.0 / 120.0, gtttta, equations, dg, 2, i, j, element)

      fttttv = 0.5 * (fppv - 4.0 * fpv + 6.0 * fv_node - 4.0 * fmv + fmmv)
      gttttv = 0.5 * (gppv - 4.0 * gpv + 6.0 * gv_node - 4.0 * gmv + gmmv)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0 / 120.0, fttttv, equations, dg, 1, i, j, element)
      multiply_add_to_node_vars!(cache_parabolic.Fv, 1.0 / 120.0, gttttv, equations, dg, 2, i, j, element)

      cv_f_node = get_node_vars(cv_f, equations, dg, i, j)
      cv_g_node = get_node_vars(cv_g, equations, dg, i, j)
      cv_ftttt = 0.5 * (cv_fpp - 4.0 * cv_fp + 6.0 * cv_f_node - 4.0 * cv_fm + cv_fmm)
      cv_gtttt = 0.5 * (cv_gpp - 4.0 * cv_gp + 6.0 * cv_g_node - 4.0 * cv_gm + cv_gmm)

      multiply_add_to_node_vars!(cv_F, 1.0 / 120.0, cv_ftttt, equations, dg, i, j)
      multiply_add_to_node_vars!(cv_G, 1.0 / 120.0, cv_gtttt, equations, dg, i, j)

      F_node = get_node_vars(cv_F, equations, dg, i, j)
      G_node = get_node_vars(cv_G, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F_node, equations, dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G_node, equations, dg, i, jj, element)
      end

      u_node = get_node_vars(u, equations, dg, i, j, element)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stttt = calc_source_tttt_N4(u_node, up_node, um_node, upp_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      multiply_add_to_node_vars!(S, 1.0 / 120.0, stttt, equations, dg, i, j)

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = get_node_vars(U, equations, dg, i, j)

      set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
         i, j, element)
   end
   return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces_lw_parabolic!(cache, cache_parabolic, u,
   mesh::P4estMesh{2}, equations,
   surface_integral, dg::DG)

   (; interfaces, element_cache, interface_cache) = cache
   (; contravariant_vectors) = cache_parabolic.elements
   @unpack U, F = cache.element_cache
   @unpack Fv = cache_parabolic
   index_range = eachnode(dg)

   @threaded for interface in eachinterface(dg, cache)
      # Copy solution data from the primary element using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.
      # Note that in the current implementation, the interface will be
      # "aligned at the primary element", i.e., the index of the primary side
      # will always run forwards.
      primary_element = interfaces.neighbor_ids[1, interface]
      primary_indices = interfaces.node_indices[1, interface]
      primary_direction = indices2direction(primary_indices)

      i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
         index_range)
      j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
         index_range)

      i_primary = i_primary_start
      j_primary = j_primary_start
      for i in eachnode(dg)
         # this is the outward normal direction on the primary element
         normal_direction = get_normal_direction(primary_direction,
            contravariant_vectors,
            i_primary, j_primary, primary_element)

         f = get_flux_vars(F, equations, dg, i_primary, j_primary, primary_element)
         fv = get_flux_vars(Fv, equations, dg, i_primary, j_primary, primary_element)
         fn_adv = normal_product(f, equations, normal_direction)
         fn_visc = normal_product(fv, equations, normal_direction)
         for v in eachvariable(equations)
            interface_cache.u[1, v, i, interface] = u[v, i_primary, j_primary, primary_element]
            interface_cache.U[1, v, i, interface] = U[v, i_primary, j_primary, primary_element]

            interface_cache.f[1, v, i, interface] = fn_adv[v]
            cache_parabolic.Fb[1, v, i, interface] = fn_visc[v]
         end
         i_primary += i_primary_step
         j_primary += j_primary_step
      end

      # Copy solution data from the secondary element using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.
      secondary_element = interfaces.neighbor_ids[2, interface]
      secondary_indices = interfaces.node_indices[2, interface]
      secondary_direction = indices2direction(secondary_indices)

      i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
         index_range)
      j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
         index_range)

      i_secondary = i_secondary_start
      j_secondary = j_secondary_start

      for i in eachnode(dg)
         # This is the outward normal direction on the secondary element.
         # Here, we assume that normal_direction on the secondary element is
         # the negative of normal_direction on the primary element.
         normal_direction = get_normal_direction(secondary_direction,
            contravariant_vectors,
            i_secondary, j_secondary,
            secondary_element)
         f = get_flux_vars(F, equations, dg, i_secondary, j_secondary, secondary_element)
         fv = get_flux_vars(Fv, equations, dg, i_secondary, j_secondary, secondary_element)
         fn_adv = normal_product(f, equations, normal_direction)
         fn_visc = normal_product(fv, equations, normal_direction)
         for v in eachvariable(equations)
            interface_cache.u[2, v, i, interface] = u[v, i_secondary, j_secondary, secondary_element]
            interface_cache.U[2, v, i, interface] = U[v, i_secondary, j_secondary, secondary_element]

            interface_cache.f[2, v, i, interface]  = -fn_adv[v]
            cache_parabolic.Fb[2, v, i, interface] = -fn_visc[v]
         end

         i_secondary += i_secondary_step
         j_secondary += j_secondary_step
      end
   end

   return nothing
end

function calc_interface_flux_hyperbolic_parabolic!(surface_flux_values, mesh::P4estMesh{2},
   # nonconservative_terms::Val{false},
   equations,
   equations_parabolic,
   surface_integral, dg::DG, cache, cache_parabolic)
   (; interface_cache ) = cache
   (; neighbor_ids, node_indices) = cache_parabolic.interfaces
   (; contravariant_vectors) = cache_parabolic.elements
   index_range = eachnode(dg)
   index_end = last(index_range)
   @unpack u = cache.interfaces
   @unpack surface_flux = surface_integral

   @threaded for interface in eachinterface(dg, cache_parabolic)
      # Get element and side index information on the primary element
      primary_element = neighbor_ids[1, interface]
      primary_indices = node_indices[1, interface]
      primary_direction_index = indices2direction(primary_indices)

      # Create the local i,j indexing on the primary element used to pull normal direction information
      i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
         index_range)
      j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
         index_range)

      i_primary = i_primary_start
      j_primary = j_primary_start

      # Get element and side index information on the secondary element
      secondary_element = neighbor_ids[2, interface]
      secondary_indices = node_indices[2, interface]
      secondary_direction_index = indices2direction(secondary_indices)

      # Initiate the secondary index to be used in the surface for loop.
      # This index on the primary side will always run forward but
      # the secondary index might need to run backwards for flipped sides.
      if :i_backward in secondary_indices
         node_secondary = index_end
         node_secondary_step = -1
      else
         node_secondary = 1
         node_secondary_step = 1
      end

      for node in eachnode(dg)
         U_ll, U_rr = get_surface_node_vars(interface_cache.U, equations_parabolic, dg, node,
                                            interface)
         u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg, node, interface)
         f_ll, f_rr = get_surface_node_vars(interface_cache.f, equations_parabolic, dg, node,
                                            interface)
         normal_direction = get_normal_direction(primary_direction_index,
            contravariant_vectors,
            i_primary, j_primary, primary_element)
         F_adv = surface_flux(f_ll, f_rr, u_ll, u_rr, U_ll, U_rr, normal_direction,
            equations)
         viscous_flux_normal_ll, viscous_flux_normal_rr = get_surface_node_vars(cache_parabolic.Fb,
            equations_parabolic,
            dg, node,
            interface)

         F_visc = 0.5 * (viscous_flux_normal_ll + viscous_flux_normal_rr)

         Fn = F_adv - F_visc

         # TODO - compute fn here and blend with Fn

         for v in eachvariable(equations_parabolic)
            surface_flux_values[v, node, primary_direction_index, primary_element] = Fn[v]
            surface_flux_values[v, node_secondary, secondary_direction_index, secondary_element] = -Fn[v]
         end

         # Increment primary element indices to pull the normal direction
         i_primary += i_primary_step
         j_primary += j_primary_step
         # Increment the surface node index along the secondary element
         node_secondary += node_secondary_step
      end
   end

   return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
function prolong2boundaries_visc_lw!(cache_parabolic, flux_viscous,
   mesh::P4estMesh{2},
   equations_parabolic::AbstractEquationsParabolic,
   surface_integral, dg::DG, cache)
   (; boundaries, Fv) = cache_parabolic
   (; contravariant_vectors) = cache_parabolic.elements
   index_range = eachnode(dg)
   flux_viscous_x, flux_viscous_y = flux_viscous

   @threaded for boundary in eachboundary(dg, cache_parabolic)
      # Copy solution data from the element using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.
      element = boundaries.neighbor_ids[boundary]
      node_indices = boundaries.node_indices[boundary]
      direction = indices2direction(node_indices)

      i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
      j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

      i_node = i_node_start
      j_node = j_node_start
      for i in eachnode(dg)
         # this is the outward normal direction on the primary element
         normal_direction = get_normal_direction(direction, contravariant_vectors,
            i_node, j_node, element)

         for v in eachvariable(equations_parabolic)
            flux_viscous = SVector(Fv[v, 1, i_node, j_node, element],
                                   Fv[v, 2, i_node, j_node, element])

            boundaries.u[v, i, boundary] = dot(flux_viscous, normal_direction)
         end

         i_node += i_node_step
         j_node += j_node_step
      end
   end

   return nothing
end

function calc_boundary_flux_divergence_lw!(
   cache_parabolic, cache_hyperbolic, t, boundary_conditions_parabolic::BoundaryConditionPeriodic,
   mesh::P4estMesh{2},
   equations_parabolic::AbstractEquationsParabolic, surface_integral, dg::DG, scaling_factor=1)
   @assert isempty(eachboundary(dg, cache_hyperbolic))
end

function calc_boundary_flux_divergence_lw!(cache, cache_hyperbolic, t,
   boundary_conditions,
   mesh::P4estMesh{2}, equations::AbstractEquationsParabolic,
   surface_integral, dg::DG, scaling_factor=1)

   (; boundary_condition_types, boundary_indices) = boundary_conditions

   calc_boundary_flux_by_type_lw!(cache, cache_hyperbolic, t, boundary_condition_types, boundary_indices,
      Divergence(), mesh, equations, surface_integral, dg,
      scaling_factor)
   return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type_lw!(cache, cache_hyperbolic, t, BCs::NTuple{N,Any},
   BC_indices::NTuple{N,Vector{Int}},
   operator_type,
   mesh::P4estMesh,
   equations, surface_integral, dg::DG,
   scaling_factor) where {N}
   # Extract the boundary condition type and index vector
   boundary_condition = first(BCs)
   boundary_condition_indices = first(BC_indices)
   # Extract the remaining types and indices to be processed later
   remaining_boundary_conditions = Base.tail(BCs)
   remaining_boundary_condition_indices = Base.tail(BC_indices)

   # process the first boundary condition type
   calc_boundary_flux_lw!(cache, cache_hyperbolic, t, boundary_condition, boundary_condition_indices,
      operator_type, mesh, equations, surface_integral, dg, scaling_factor)

   # recursively call this method with the unprocessed boundary types
   calc_boundary_flux_by_type_lw!(cache, cache_hyperbolic, t, remaining_boundary_conditions,
      remaining_boundary_condition_indices,
      operator_type,
      mesh, equations, surface_integral, dg, scaling_factor)

   return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type_lw!(cache, cache_hyperbolic, t, BCs::Tuple{}, BC_indices::Tuple{},
   operator_type, mesh::P4estMesh, equations,
   surface_integral, dg::DG, scaling_factor)
   nothing
end

function calc_boundary_flux_lw!(cache, cache_hyperbolic, t,
   boundary_condition_parabolic, # works with Dict types
   boundary_condition_indices,
   operator_type, mesh::P4estMesh{2},
   equations_parabolic::AbstractEquationsParabolic,
   surface_integral, dg::DG, scaling_factor)
   (; boundaries) = cache
   (; node_coordinates) = cache.elements
   (; surface_flux_values) = cache_hyperbolic.elements
   (; contravariant_vectors) = cache.elements
   index_range = eachnode(dg)

   @threaded for local_index in eachindex(boundary_condition_indices)
      # Use the local index to get the global boundary index from the pre-sorted list
      boundary_index = boundary_condition_indices[local_index]

      # Get information on the adjacent element, compute the surface fluxes,
      # and store them
      element = boundaries.neighbor_ids[boundary_index]
      node_indices = boundaries.node_indices[boundary_index]
      direction_index = indices2direction(node_indices)

      i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
      j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

      i_node = i_node_start
      j_node = j_node_start
      for node_index in eachnode(dg)
         # Extract solution data from boundary container
         u_inner = get_node_vars(boundaries.u, equations_parabolic, dg, node_index,
            boundary_index)

         # Outward-pointing normal direction (not normalized)
         normal_direction = get_normal_direction(direction_index, contravariant_vectors,
            i_node, j_node, element)

         # TODO: (Trixi.jl comment) revisit if we want more general boundary treatments.
         # This assumes the gradient numerical flux at the boundary is the gradient variable,
         # which is consistent with BR1, LDG.
         flux_inner = u_inner

         # Coordinates at boundary node
         x = get_node_coords(node_coordinates, equations_parabolic, dg, i_node, j_node,
            element)

         flux_ = boundary_condition_parabolic(flux_inner, u_inner, normal_direction,
            x, t, operator_type, equations_parabolic, get_time_discretization(dg), scaling_factor)

         # Copy flux to element storage in the correct orientation
         for v in eachvariable(equations_parabolic)
            surface_flux_values[v, node_index, direction_index, element] -= flux_[v]
         end

         i_node += i_node_step
         j_node += j_node_step
      end
   end
end
