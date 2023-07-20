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

   # TODO - Remove other cv terms?
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

   # TODO - Remove other cv terms?
   return fa, ga, fv, gv, cv_f, cv_g
end

function weak_form_kernel_1!(
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

            # TODO - Try with get_flux_vars instead?
            # flux_advectv = SVector(F[v, 1, i_primary, j_primary, primary_element],
            #                        F[v, 2, i_primary, j_primary, primary_element])
            # flux_viscous = SVector(Fv[v, 1, i_primary, j_primary, primary_element],
            #                        Fv[v, 2, i_primary, j_primary, primary_element])

            # fn_adv = dot(flux_advectv, normal_direction)
            # fn_visc = dot(flux_viscous, normal_direction)

            # interface_cache.f[1, v, i, interface] = dot(flux_advectv, normal_direction)
            # cache_parabolic.Fb[1, v, i, interface] = dot(flux_viscous, normal_direction)

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

            # flux_advectv = SVector(F[v, 1, i_secondary, j_secondary, secondary_element],
            #                        F[v, 2, i_secondary, j_secondary, secondary_element])
            # flux_viscous = SVector(Fv[v, 1, i_secondary, j_secondary, secondary_element],
            #                        Fv[v, 2, i_secondary, j_secondary, secondary_element])

            # store the normal flux with respect to the primary normal direction
            # interface_cache.f[2, v, i, interface] = -dot(flux_advectv, normal_direction)
            # cache_parabolic.Fb[2, v, i, interface] = -dot(flux_viscous, normal_direction)

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

         # TODO: revisit if we want more general boundary treatments.
         # This assumes the gradient numerical flux at the boundary is the gradient variable,
         # which is consistent with BR1, LDG.
         flux_inner = u_inner

         # Coordinates at boundary node
         x = get_node_coords(node_coordinates, equations_parabolic, dg, i_node, j_node,
            element)

         flux_ = boundary_condition_parabolic(flux_inner, u_inner, normal_direction,
            x, t, operator_type, equations_parabolic, get_time_discretization(dg), scaling_factor)

         # Copy flux to element storage in the correct orientation
         # print("Before update ")
         # @show surface_flux_values[1, node_index, direction_index, element]
         for v in eachvariable(equations_parabolic)
            surface_flux_values[v, node_index, direction_index, element] -= flux_[v]
            # @show surface_flux_values[v, node_index, direction_index, element]
            # surface_flux_values[v, node_index, direction_index, element] = 1000.0
         end
         # print("After update ")
         # @show surface_flux_values[1, node_index, direction_index, element]

         i_node += i_node_step
         j_node += j_node_step
      end
   end
end
