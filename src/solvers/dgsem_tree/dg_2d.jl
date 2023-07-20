import Trixi: create_cache, calc_volume_integral!, ninterfaces, nboundaries,
   prolong2interfaces!, calc_interface_flux!, prolong2boundaries!,
   calc_boundary_flux!, prolong2mortars!, calc_mortar_flux!,
   element_solutions_to_mortars!, calc_fstar!

# TODO - Reorder these
using Trixi: TreeMesh, P4estMesh, BoundaryConditionPeriodic,
   pure_and_blended_element_ids!, have_nonconservative_terms,
   calc_surface_integral!, apply_jacobian!, reset_du!,
   max_dt, calcflux_fv!, index_to_start_step_2d,
   AbstractMesh, StructuredMesh, UnstructuredMesh2D,
   LobattoLegendreMortarL2,
   AbstractContainer,
   DG, DGSEM, ndims, polydeg, nnodes, eachelement, nelements, nmortars,
   eachmortar, eachnode, eachboundary, multiply_add_to_node_vars!,
   multiply_dimensionwise!, mortar_fluxes_to_elements!, False,
   @threaded, get_surface_node_vars
using MuladdMacro
using LoopVectorization: @turbo

@muladd begin

   # By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
   # Since these FMAs can increase the performance of many numerical algorithms,
   # we need to opt-in explicitly.
   # See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

   function rhs!(du, u, t, mesh::Union{TreeMesh{2},P4estMesh{2}}, equations,
      initial_condition, boundary_conditions, source_terms, dg::DG,
      time_discretization::AbstractLWTimeDiscretization, cache, tolerances::NamedTuple)

      # Reset du
      @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

      dt = cache.dt[1]

      # Update dt in cache and the callback will just take it from there

      # Calculate volume integral
      @trixi_timeit timer() "volume integral" calc_volume_integral!(
         du, u, t, dt, tolerances, mesh,
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
         cache, t, dt, boundary_conditions, mesh, equations, dg.surface_integral,
         time_discretization, dg)

      # Prolong solution to mortars
      @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
         cache, u, mesh, equations, dg.mortar, dg.surface_integral, time_discretization, dg)
      # @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
      #       cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

      # Calculate mortar fluxes
      @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
         cache.elements.surface_flux_values, mesh,
         have_nonconservative_terms(equations), equations,
         dg.mortar, dg.surface_integral, time_discretization, dg, cache)

      # Calculate surface integrals
      @trixi_timeit timer() "surface integral" calc_surface_integral!(
         du, u, mesh, equations, dg.surface_integral, dg, cache)

      # Apply Jacobian from mapping to reference element
      @trixi_timeit timer() "Jacobian" apply_jacobian!(
         du, mesh, equations, dg, cache)

      return nothing
   end

   # If there is no source_term, Trixi gets a nothing object in its place.
   # With that information, we can use multiple dispatch to create a source term function which
   # are zero functions when there is no source terms (i.e., it is a Nothing object)

   function calc_source(u, x, t, source_terms,
      eq::AbstractEquations{2}, dg, cache)
      return source_terms(u, x, t, eq)
   end

   function calc_source(u, x, t, source_terms::Nothing,
      eq::AbstractEquations{2}, dg, cache)
      return zero(u)
   end

   function calc_source_t_N12(up, um, x, t, dt, source_terms::Nothing,
      eq::AbstractEquations{2}, dg, cache)
      return zero(up)
   end

   function calc_source_t_N12(up, um, x, t, dt, source_terms,
      eq::AbstractEquations{2}, dg, cache)
      s(u_, Δt) = source_terms(u_, x, t + Δt, eq)
      s_t = 0.5 * (s(up, dt) - s(um, -dt))
      return s_t
   end

   function calc_source_t_N34(u, up, upp, um, umm, x, t, dt, source_terms::Nothing,
      eq::AbstractEquations{2}, dg, cache)
      return zero(u)
   end

   function calc_source_t_N34(u, up, upp, um, umm, x, t, dt, source_terms,
      eq::AbstractEquations{2}, dg, cache)
      s(u_, Δt) = source_terms(u_, x, t + Δt, eq)
      s_t = (1.0 / 12.0) * (-s(upp, 2.0 * dt) + 8.0 * s(up, dt)
                            -
                            8.0 * s(um, -dt) + s(umm, -2.0 * dt))
      return s_t
   end

   function calc_source_tt_N23(u, up, um, x, t, dt, source_terms::Nothing,
      eq::AbstractEquations{2}, dg, cache)
      return zero(u)
   end

   function calc_source_tt_N23(u, up, um, x, t, dt, source_terms,
      eq::AbstractEquations{2}, dg, cache)
      s(u_, Δt) = source_terms(u_, x, t + Δt, eq)
      s_tt = s(up, dt) - 2.0 * s(u, 0.0) + s(um, -dt)
      return s_tt
   end

   function calc_source_tt_N4(u, up, upp, um, umm, x, t, dt, source_terms::Nothing,
      eq::AbstractEquations{2}, dg, cache)
      return zero(u)
   end

   function calc_source_tt_N4(u, up, upp, um, umm, x, t, dt, source_terms,
      eq::AbstractEquations{2}, dg, cache)
      s(u_, Δt) = source_terms(u_, x, t + Δt, eq)
      s_tt = (1.0 / 12.0) * (-s(upp, 2.0 * dt) + 16.0 * s(up, dt)
                             -
                             30.0 * s(u, 0.0)
                             +
                             16.0 * s(um, -dt) - s(umm, -2.0 * dt))
      return s_tt
   end

   function calc_source_ttt_N34(u, up, upp, um, umm, x, t, dt, source_terms::Nothing,
      eq::AbstractEquations{2}, dg, cache)
      return zero(u)
   end

   function calc_source_ttt_N34(u, up, upp, um, umm, x, t, dt, source_terms,
      eq::AbstractEquations{2}, dg, cache)
      s(u_, Δt) = source_terms(u_, x, t + Δt, eq)
      s_ttt = 0.5 * (s(upp, 2.0 * dt) - 2.0 * s(up, dt)
                     +
                     2.0 * s(um, -dt) - s(umm, -2.0 * dt))
      return s_ttt
   end

   function calc_source_tttt_N4(u, up, upp, um, umm, x, t, dt, source_terms::Nothing,
      eq::AbstractEquations{2}, dg, cache)
      return zero(u)
   end

   function calc_source_tttt_N4(u, up, upp, um, umm, x, t, dt, source_terms,
      eq::AbstractEquations{2}, dg, cache)
      s(u_, Δt) = source_terms(u_, x, t + Δt, eq)
      s_tttt = s(upp, 2.0 * dt) - 4.0 * s(up, dt) + 6.0 * s(u, 0.0) - 4.0 * s(um, -dt) + s(umm, -2.0 * dt)
      return s_tttt
   end

   function calc_volume_integral!(du, u,
      t, dt, tolerances::NamedTuple,
      mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
      nonconservative_terms, source_terms, equations,
      volume_integral::VolumeIntegralFR,
      time_discretization::AbstractLWTimeDiscretization,
      dg::DGSEM, cache)

      degree = polydeg(dg)
      if degree == 1
         @threaded for element in eachelement(dg, cache)
            weak_form_kernel_1!(du, u,
               t, dt, tolerances, element, mesh,
               nonconservative_terms, source_terms, equations,
               dg, cache)
         end
      elseif degree == 2
         @threaded for element in eachelement(dg, cache)
            weak_form_kernel_2!(du, u,
               t, dt, tolerances, element, mesh,
               nonconservative_terms, source_terms, equations,
               dg, cache)
         end
      elseif degree == 3
         @threaded for element in eachelement(dg, cache)
            weak_form_kernel_3!(du, u,
               t, dt, tolerances, element, mesh,
               nonconservative_terms, source_terms, equations,
               dg, cache)
         end
      else
         @threaded for element in eachelement(dg, cache)
            weak_form_kernel_4!(du, u,
               t, dt, tolerances, element, mesh,
               nonconservative_terms, source_terms, equations,
               dg, cache)
         end
      end
      return nothing
   end

   function load_fn_low!(fn_low, mesh::Union{TreeMesh{2},StructuredMesh{2}}, dg,
      nonconservative_terms::False,
      fstar1_L, fstar2_L, ftar1_R, fstar2_R, element)
      # Without non-conservative terms fstar1_L, fstar1_R etc. are the same thing
      # The faces are numbered so that 1, nnodes(dg)+1 and the supercell faces and are
      # hence zero for now
      # TODO (URGENT) - Should this be nnodes(dg)-1??

      @views for k in eachnode(dg)
         fn_low[:, k, 1, element] .= fstar1_L[:, 2, k]          # left
         fn_low[:, k, 2, element] .= fstar1_L[:, nnodes(dg), k] # right
         fn_low[:, k, 3, element] .= fstar2_L[:, k, 2]          # bottom
         fn_low[:, k, 4, element] .= fstar2_L[:, k, nnodes(dg)] # top
      end
   end

   function load_fn_low!(fn_low, mesh::Union{UnstructuredMesh2D,P4estMesh{2}}, dg,
      nonconservative_terms::False,
      fstar1_L, fstar2_L, ftar1_R, fstar2_R, element)
      # Without non-conservative terms fstar1_L, fstar1_R etc. are the same thing
      # The faces are numbered so that 1, nnodes(dg)+1 and the supercell faces and are
      # hence zero for now
      @views for j in eachnode(dg)
         fn_low[:, j, 1, element] .= fstar2_L[:, j, 2]          # bottom
         fn_low[:, j, 2, element] .= fstar1_L[:, nnodes(dg), j] # right
         fn_low[:, j, 3, element] .= fstar2_L[:, j, nnodes(dg)] # top
         fn_low[:, j, 4, element] .= fstar1_L[:, 2, j]          # left
      end
   end


   @inline function fv_kernel!(du, u, dt, ::FirstOrderReconstruction,
      mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
      nonconservative_terms, equations,
      volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
      @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache
      @unpack fn_low = cache.element_cache
      @unpack inverse_weights = dg.basis

      # Calculate FV two-point fluxes
      fstar1_L = fstar1_L_threaded[Threads.threadid()]
      fstar2_L = fstar2_L_threaded[Threads.threadid()]
      fstar1_R = fstar1_R_threaded[Threads.threadid()]
      fstar2_R = fstar2_R_threaded[Threads.threadid()]
      calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
         nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

      for j in eachnode(dg), i in eachnode(dg)
         for v in eachvariable(equations)
            du[v, i, j, element] += (alpha *
                                     (inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                                      inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])))
         end
      end

      #= Since numbering is different for structured and unstructured meshes, it is
         done differently for both =#

      load_fn_low!(fn_low, mesh, dg, nonconservative_terms,
         fstar1_L, fstar2_L, fstar1_R, fstar2_R, element)

      return nothing
   end

   @inline function fv_kernel!(du, u, dt, ::MUSCLHancockReconstruction,
      mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
      nonconservative_terms, equations,
      volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
      @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded,
      # unph_threaded, uf_threaded
      mh_face_arrays,
      uext_threaded = cache
      @unpack x_subfaces, y_subfaces = cache
      @unpack fn_low = cache.element_cache
      @unpack node_coordinates = cache.elements
      @unpack inverse_weights = dg.basis

      # Calculate FV two-point fluxes
      fstar1_L = fstar1_L_threaded[Threads.threadid()]
      fstar2_L = fstar2_L_threaded[Threads.threadid()]
      fstar1_R = fstar1_R_threaded[Threads.threadid()]
      fstar2_R = fstar2_R_threaded[Threads.threadid()]
      unph, uf = mh_face_arrays[Threads.threadid()]
      uext = uext_threaded[Threads.threadid()]
      calcflux_mh!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, unph, uf, uext, alpha, u,
         dt, mesh, nonconservative_terms, equations, volume_flux_fv, dg,
         element, cache)

      # Calculate FV volume integral contribution
      for j in eachnode(dg), i in eachnode(dg)
         u_node = get_node_vars(u, equations, dg, i, j, element)
         for v in eachvariable(equations)
            du[v, i, j, element] += (alpha *
                                     (inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                                      inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])))
         end
      end

      #= Since numbering is different for structured and unstructured meshes, it is
         done differently for both =#
      load_fn_low!(fn_low, mesh, dg, nonconservative_terms,
         fstar1_L, fstar2_L, fstar1_R, fstar2_R, element)

      return nothing
   end

   @noinline function fv_kernel!(du, u, dt, ::MUSCLReconstruction,
      mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
      nonconservative_terms, equations,
      volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
      @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded,
      # unph_threaded, uf_threaded
      mh_face_arrays,
      uext_threaded = cache
      @unpack x_subfaces, y_subfaces = cache
      @unpack fn_low = cache.element_cache
      @unpack node_coordinates = cache.elements
      @unpack inverse_weights = dg.basis

      # Calculate FV two-point fluxes
      fstar1_L = fstar1_L_threaded[Threads.threadid()]
      fstar2_L = fstar2_L_threaded[Threads.threadid()]
      fstar1_R = fstar1_R_threaded[Threads.threadid()]
      fstar2_R = fstar2_R_threaded[Threads.threadid()]
      _, uf = mh_face_arrays[Threads.threadid()]
      uext = uext_threaded[Threads.threadid()]
      calcflux_muscl!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, uf, uext, alpha, u,
         dt, mesh, nonconservative_terms, equations, volume_flux_fv, dg,
         element, cache)

      # Calculate FV volume integral contribution
      for j in eachnode(dg), i in eachnode(dg)
         for v in eachvariable(equations)
            du[v, i, j, element] += (alpha *
                                     (inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                                      inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])))
         end
      end

      #= Since numbering is different for structured and unstructured meshes, it is
         done differently for both =#
      load_fn_low!(fn_low, mesh, dg, nonconservative_terms,
         fstar1_L, fstar2_L, fstar1_R, fstar2_R, element)

      return nothing
   end


   fluxes(u, equations::AbstractEquations{2}) = (Trixi.flux(u, 1, equations), Trixi.flux(u, 2, equations))

   @inline function weak_form_kernel_1!(du, u,
      t, dt,
      element, mesh::TreeMesh{2},
      nonconservative_terms::False, source_terms, equations,
      dg::DGSEM, cache, alpha=true)
      # true * [some floating point value] == [exactly the same floating point value]
      # This can (hopefully) be optimized away due to constant propagation.
      @unpack derivative_dhat, derivative_matrix = dg.basis
      @unpack node_coordinates = cache.elements

      @unpack lw_res_cache = cache
      @unpack cell_arrays, eval_data = lw_res_cache

      inv_jacobian = cache.elements.inverse_jacobian[element]

      id = Threads.threadid()

      F, G, ut, U, up, um, ft, gt, S = cell_arrays[id]

      refresh!(arr) = fill!(arr, zero(eltype(u)))

      refresh!.((ut, ft, gt))

      # Calculate volume terms in one element
      for j in eachnode(dg), i in eachnode(dg)
         u_node = get_node_vars(u, equations, dg, i, j, element)

         flux1, flux2 = fluxes(u_node, equations)
         for ii in eachnode(dg)
            # ut              += -lam * D * f for each variable
            # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
            multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
         end

         set_node_vars!(F, flux1, equations, dg, i, j)
         set_node_vars!(G, flux2, equations, dg, i, j)
         set_node_vars!(um, u_node, equations, dg, i, j)
         set_node_vars!(up, u_node, equations, dg, i, j)
         set_node_vars!(U, u_node, equations, dg, i, j)
      end

      # Scale ut
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i, j, element]
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

      for j in eachnode(dg), i in eachnode(dg)
         ut_node = get_node_vars(ut, equations, dg, i, j)
         multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         fm, gm = fluxes(um_node, 1, equations), fluxes(um_node, 2, equations)
         fp, gp = fluxes(up_node, 1, equations), fluxes(up_node, 2, equations)

         multiply_add_to_node_vars!(ft, 0.5, fp, equations, dg, i, j)
         multiply_add_to_node_vars!(ft, -0.5, fm, equations, dg, i, j)
         multiply_add_to_node_vars!(gt, 0.5, gp, equations, dg, i, j)
         multiply_add_to_node_vars!(gt, -0.5, gm, equations, dg, i, j)

         ft_node = get_node_vars(ft, equations, dg, i, j)
         gt_node = get_node_vars(gt, equations, dg, i, j)

         multiply_add_to_node_vars!(F, 0.5, ft_node, equations, dg, i, j)
         multiply_add_to_node_vars!(G, 0.5, gt_node, equations, dg, i, j)
         F_node = get_node_vars(F, equations, dg, i, j)
         G_node = get_node_vars(G, equations, dg, i, j)
         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], F_node, equations,
               dg, ii, j, element)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], G_node, equations,
               dg, i, jj, element)
         end

         set_node_vars!(cache.F, F_node, equations, dg, 1, i, j, element)
         set_node_vars!(cache.F, G_node, equations, dg, 2, i, j, element)

         x = get_node_coords(node_coordinates, equations, dg, i, j, element)
         st = calc_source_t_N12(up_node, um_node, x, t, dt, source_terms, equations,
            dg, cache)
         multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)

         # TODO - update to v1.8 and call with @inline
         # Give u1_ or U depending on dissipation model
         U_node = get_node_vars(U, equations, dg, i, j)

         # Ub = UT * V
         # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
         set_node_vars!(cache.U, U_node, equations, dg, i, j, element)

         S_node = get_node_vars(S, equations, dg, i, j)
         # inv_jacobian = inverse_jacobian[i, j, element]
         multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
      end

      return nothing
   end

   @inline function weak_form_kernel_2!(du, u,
      t, dt,
      element, mesh::TreeMesh{2},
      nonconservative_terms::False, source_terms, equations,
      dg::DGSEM, cache, alpha=true)
      # true * [some floating point value] == [exactly the same floating point value]
      # This can (hopefully) be optimized away due to constant propagation.
      @unpack derivative_dhat, derivative_matrix = dg.basis
      @unpack node_coordinates = cache.elements

      @unpack lw_res_cache = cache
      @unpack cell_arrays, eval_data = lw_res_cache

      inv_jacobian = cache.elements.inverse_jacobian[element]

      id = Threads.threadid()

      refresh!(arr) = fill!(arr, zero(eltype(u)))

      f, g, ft, gt, F, G, ut, utt, U, up, um, S = cell_arrays[id]
      refresh!.((ut, utt, ft, gt))
      for j in eachnode(dg), i in eachnode(dg)
         u_node = get_node_vars(u, equations, dg, i, j, element)

         flux1, flux2 = fluxes(u_node, equations)

         set_node_vars!(f, flux1, equations, dg, i, j)
         set_node_vars!(F, flux1, equations, dg, i, j)
         set_node_vars!(g, flux2, equations, dg, i, j)
         set_node_vars!(G, flux2, equations, dg, i, j)

         for ii in eachnode(dg)
            # ut              += -lam * D * f for each variable
            # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
            multiply_add_to_node_vars!(ut, derivative_matrix[ii, i], flux1, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(ut, derivative_matrix[jj, j], flux2, equations, dg, i, jj)
         end

         set_node_vars!(um, u_node, equations, dg, i, j)
         set_node_vars!(up, u_node, equations, dg, i, j)
         set_node_vars!(U, u_node, equations, dg, i, j)
      end

      # Scale ut
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
         for v in eachvariable(equations)
            ut[v, i, j] *= -dt * inv_jacobian
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

      for j in eachnode(dg), i in eachnode(dg)
         ut_node = get_node_vars(ut, equations, dg, i, j)
         multiply_add_to_node_vars!(U,
            0.5, ut_node,
            equations, dg, i, j)
         multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)

         multiply_add_to_node_vars!(ft, 0.5, fp, equations, dg, i, j)
         multiply_add_to_node_vars!(ft, -0.5, fm, equations, dg, i, j)
         multiply_add_to_node_vars!(gt, 0.5, gp, equations, dg, i, j)
         multiply_add_to_node_vars!(gt, -0.5, gm, equations, dg, i, j)

         ft_node = get_node_vars(ft, equations, dg, i, j)
         gt_node = get_node_vars(gt, equations, dg, i, j)

         multiply_add_to_node_vars!(F, 0.5, ft_node, equations, dg, i, j)
         multiply_add_to_node_vars!(G, 0.5, gt_node, equations, dg, i, j)

         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
            multiply_add_to_node_vars!(utt, derivative_matrix[ii, i], ft_node, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(utt, derivative_matrix[jj, j], gt_node, equations, dg, i, jj)
         end
      end

      # Apply Jacobian to utt
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
         for v in eachvariable(equations)
            utt[v, i, j] *= -dt * inv_jacobian
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

      for j in eachnode(dg), i in eachnode(dg)
         utt_node = get_node_vars(utt, equations, dg, i, j)
         multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)

         f_node = get_node_vars(f, equations, dg, i, j)
         g_node = get_node_vars(g, equations, dg, i, j)
         u_node = get_node_vars(u, equations, dg, i, j, element)
         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         ftt, gtt = fp - 2.0 * f_node + fm, gp - 2.0 * g_node + gm

         multiply_add_to_node_vars!(F, 1.0 / 6.0, ftt, equations, dg, i, j)
         multiply_add_to_node_vars!(G, 1.0 / 6.0, gtt, equations, dg, i, j)

         F_node = get_node_vars(F, equations, dg, i, j)
         G_node = get_node_vars(G, equations, dg, i, j)
         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], F_node, equations,
               dg, ii, j, element)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], G_node, equations,
               dg, i, jj, element)
         end

         x = get_node_coords(node_coordinates, equations, dg, i, j, element)
         stt = calc_source_tt_N23(u_node, up_node, um_node, x, t, dt, source_terms,
            equations, dg, cache)
         multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, equations, dg, i, j)

         # TODO - update to v1.8 and call with @inline
         # Give u1_ or U depending on dissipation model
         U_node = get_node_vars(U, equations, dg, i, j)

         # Ub = UT * V
         # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
         set_node_vars!(cache.U, U_node, equations, dg, i, j, element)
         set_node_vars!(cache.F, F_node, equations, dg, 1, i, j, element)
         set_node_vars!(cache.F, G_node, equations, dg, 2, i, j, element)

         S_node = get_node_vars(S, equations, dg, i, j)
         # inv_jacobian = inverse_jacobian[i, j, element]
         multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
      end

      return nothing
   end

   @inline function weak_form_kernel_3!(du, u,
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

      inv_jacobian = cache.elements.inverse_jacobian[element]

      id = Threads.threadid()

      refresh!(arr) = fill!(arr, zero(eltype(u)))

      f, g, ftilde, gtilde, F, G, ut, utt, uttt, U,
      up, um, upp, umm, S = cell_arrays[id]
      refresh!.((ut, utt, uttt))
      for j in eachnode(dg), i in eachnode(dg)
         u_node = get_node_vars(u, equations, dg, i, j, element)

         flux1, flux2 = fluxes(u_node, equations)

         set_node_vars!(f, flux1, equations, dg, i, j)
         set_node_vars!(g, flux2, equations, dg, i, j)

         set_node_vars!(F, flux1, equations, dg, i, j)
         for ii in eachnode(dg)
            # ut              += -lam * D * f for each variable
            # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
            multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1,
               equations, dg, ii, j)
         end

         set_node_vars!(G, flux2, equations, dg, i, j)
         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
         end

         set_node_vars!(um, u_node, equations, dg, i, j)
         set_node_vars!(up, u_node, equations, dg, i, j)
         set_node_vars!(umm, u_node, equations, dg, i, j)
         set_node_vars!(upp, u_node, equations, dg, i, j)
         set_node_vars!(U, u_node, equations, dg, i, j)
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
         multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
      end

      for j in eachnode(dg), i in eachnode(dg)
         ut_node = get_node_vars(ut, equations, dg, i, j)
         multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(umm, -2.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(upp, 2.0, ut_node, equations, dg, i, j)

         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)
         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         fmm, gmm = fluxes(umm_node, equations)
         fpp, gpp = fluxes(upp_node, equations)

         ft = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
         gt = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)

         multiply_add_to_node_vars!(F, 0.5, ft, equations, dg, i, j)
         multiply_add_to_node_vars!(G, 0.5, gt, equations, dg, i, j)
         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
            multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], ft, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], gt, equations, dg, i, jj)
         end
      end

      # Apply Jacobian to utt
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
         for v in eachvariable(equations)
            utt[v, i, j] *= inv_jacobian
         end
      end

      # Add source term contribution to utt and some to S
      for j in eachnode(dg), i in eachnode(dg)
         # Add source term contribution to ut
         u_node = get_node_vars(u, equations, dg, i, j, element)
         um_node = get_node_vars(um, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)
         x = get_node_coords(node_coordinates, equations, dg, i, j, element)
         st = calc_source_t_N34(u_node, up_node, upp_node, um_node, umm_node,
            x, t, dt, source_terms,
            equations, dg, cache)
         multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)
         multiply_add_to_node_vars!(utt, dt, st, equations, dg, i, j) # has no jacobian factor
      end

      for j in eachnode(dg), i in eachnode(dg)
         utt_node = get_node_vars(utt, equations, dg, i, j)
         multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(upp, 2.0, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(umm, 2.0, utt_node, equations, dg, i, j)

         f_node = get_node_vars(f, equations, dg, i, j)
         g_node = get_node_vars(g, equations, dg, i, j)
         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         ftt, gtt = fp - 2.0 * f_node + fm, gp - 2.0 * g_node + gm

         multiply_add_to_node_vars!(F, 1.0 / 6.0, ftt, equations, dg, i, j)
         multiply_add_to_node_vars!(G, 1.0 / 6.0, gtt, equations, dg, i, j)

         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
            multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], ftt, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], gtt, equations, dg, i, jj)
         end
      end

      # Apply Jacobian to uttt
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
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

      for j in eachnode(dg), i in eachnode(dg)
         uttt_node = get_node_vars(uttt, equations, dg, i, j)
         multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, equations, dg, i, j)

         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)

         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         fmm, gmm = fluxes(umm_node, equations)
         fpp, gpp = fluxes(upp_node, equations)
         fttt = 0.5 * (fpp - 2.0 * fp + 2.0 * fm - fmm)
         multiply_add_to_node_vars!(F, 1.0 / 24.0, fttt, equations, dg, i, j)
         gttt = 0.5 * (gpp - 2.0 * gp + 2.0 * gm - gmm)
         multiply_add_to_node_vars!(G, 1.0 / 24.0, gttt, equations, dg, i, j)

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
         set_node_vars!(element_cache.F, F_node, equations, dg, 1, i, j, element)
         set_node_vars!(element_cache.F, G_node, equations, dg, 2, i, j, element)

         S_node = get_node_vars(S, equations, dg, i, j)
         # inv_jacobian = inverse_jacobian[i, j, element]
         multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
            i, j, element)
      end

      return nothing
   end

   @inline function weak_form_kernel_4!(du, u,
      t, dt, tolerances,
      element, mesh::TreeMesh{2},
      nonconservative_terms::False,
      source_terms, equations,
      dg::DGSEM, cache, alpha=true)
      # true * [some floating point value] == [exactly the same floating point value]
      # This can (hopefully) be optimized away due to constant propagation.
      @unpack derivative_dhat, derivative_matrix = dg.basis
      @unpack node_coordinates = cache.elements

      @unpack lw_res_cache, element_cache = cache
      @unpack cell_arrays = lw_res_cache

      inv_jacobian = cache.elements.inverse_jacobian[element]

      id = Threads.threadid()

      refresh!(arr) = fill!(arr, zero(eltype(u)))

      f, g, F_cell, G_cell, ut, utt, uttt, utttt, U_cell, up, um, upp, umm, S,
      u_np1, u_np1_low = cell_arrays[id]
      refresh!.((ut, utt, uttt, utttt))
      for j in eachnode(dg), i in eachnode(dg)
         u_node = get_node_vars(u, equations, dg, i, j, element)

         flux1, flux2 = fluxes(u_node, equations)

         set_node_vars!(f, flux1, equations, dg, i, j)
         set_node_vars!(g, flux2, equations, dg, i, j)

         set_node_vars!(F_cell, flux1, equations, dg, i, j)
         for ii in eachnode(dg)
            # ut              += -lam * D * f for each variable
            # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
            multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1, equations, dg, ii, j)
         end

         set_node_vars!(G_cell, flux2, equations, dg, i, j)
         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
         end

         set_node_vars!(u_np1, u_node, equations, dg, i, j)
         set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

         set_node_vars!(um, u_node, equations, dg, i, j)
         set_node_vars!(up, u_node, equations, dg, i, j)
         set_node_vars!(umm, u_node, equations, dg, i, j)
         set_node_vars!(upp, u_node, equations, dg, i, j)
         set_node_vars!(U_cell, u_node, equations, dg, i, j)
      end
      # Scale ut
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
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

      for j in eachnode(dg), i in eachnode(dg)
         ut_node = get_node_vars(ut, equations, dg, i, j)
         multiply_add_to_node_vars!(U_cell, 0.5, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(umm, -2.0, ut_node, equations, dg, i, j)
         multiply_add_to_node_vars!(upp, 2.0, ut_node, equations, dg, i, j)

         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)
         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         fmm, gmm = fluxes(umm_node, equations)
         fpp, gpp = fluxes(upp_node, equations)

         f_t = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
         g_t = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)

         multiply_add_to_node_vars!(F_cell, 0.5, f_t, equations, dg, i, j)
         multiply_add_to_node_vars!(G_cell, 0.5, g_t, equations, dg, i, j)
         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
            multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], f_t, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], g_t, equations, dg, i, jj)
         end
      end

      # Apply Jacobian to utt
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
         for v in eachvariable(equations)
            utt[v, i, j] *= inv_jacobian
         end
      end

      # Add source term contribution to utt and some to S
      for j in eachnode(dg), i in eachnode(dg)
         # Add source term contribution to ut
         u_node = get_node_vars(u, equations, dg, i, j, element)
         um_node = get_node_vars(um, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)
         x = get_node_coords(node_coordinates, equations, dg, i, j, element)
         st = calc_source_t_N34(u_node, up_node, upp_node, um_node, umm_node,
            x, t, dt, source_terms,
            equations, dg, cache)
         multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)
         multiply_add_to_node_vars!(utt, dt, st, equations, dg, i, j) # has no jacobian factor
      end

      for j in eachnode(dg), i in eachnode(dg)
         utt_node = get_node_vars(utt, equations, dg, i, j)
         multiply_add_to_node_vars!(U_cell, 1.0 / 6.0, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(upp, 2.0, utt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(umm, 2.0, utt_node, equations, dg, i, j)

         f_node, g_node = get_node_vars(f, equations, dg, i, j), get_node_vars(g, equations, dg, i, j)
         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)
         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         fmm, gmm = fluxes(umm_node, equations)
         fpp, gpp = fluxes(upp_node, equations)
         f_tt = (1.0 / 12.0) * (-fpp + 16.0 * fp - 30.0 * f_node + 16.0 * fm - fmm)
         g_tt = (1.0 / 12.0) * (-gpp + 16.0 * gp - 30.0 * g_node + 16.0 * gm - gmm)

         multiply_add_to_node_vars!(F_cell, 1.0 / 6.0, f_tt, equations, dg, i, j)
         multiply_add_to_node_vars!(G_cell, 1.0 / 6.0, g_tt, equations, dg, i, j)

         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
            multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], f_tt, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], g_tt, equations, dg, i, jj)
         end
      end

      # Apply Jacobian to uttt
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
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

      for j in eachnode(dg), i in eachnode(dg)
         uttt_node = get_node_vars(uttt, equations, dg, i, j)
         multiply_add_to_node_vars!(U_cell, 1.0 / 24.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, equations, dg, i, j)

         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)

         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         fmm, gmm = fluxes(umm_node, equations)
         fpp, gpp = fluxes(upp_node, equations)
         fttt = 0.5 * (fpp - 2.0 * fp + 2.0 * fm - fmm)
         multiply_add_to_node_vars!(F_cell, 1.0 / 24.0, fttt, equations, dg, i, j)
         gttt = 0.5 * (gpp - 2.0 * gp + 2.0 * gm - gmm)
         multiply_add_to_node_vars!(G_cell, 1.0 / 24.0, gttt, equations, dg, i, j)

         for ii in eachnode(dg)
            # ut              += -lam * D * ft for each variable
            # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
            multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[ii, i], fttt, equations, dg, ii, j)
         end
         for jj in eachnode(dg)
            # C += -lam*gt*Dm' for each variable
            # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[jj, j], gttt, equations, dg, i, jj)
         end
      end

      # Apply jacobian on utttt
      for j in eachnode(dg), i in eachnode(dg)
         # inv_jacobian = inverse_jacobian[i,j,element]
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

      for j in eachnode(dg), i in eachnode(dg)
         utttt_node = get_node_vars(utttt, equations, dg, i, j)
         multiply_add_to_node_vars!(U_cell, 1.0 / 120.0, utttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(um, 1.0 / 24.0, utttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(up, 1.0 / 24.0, utttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(umm, 2.0 / 3.0, utttt_node, equations, dg, i, j)
         multiply_add_to_node_vars!(upp, 2.0 / 3.0, utttt_node, equations, dg, i, j)

         f_node = get_node_vars(f, equations, dg, i, j)
         g_node = get_node_vars(g, equations, dg, i, j)
         um_node = get_node_vars(um, equations, dg, i, j)
         up_node = get_node_vars(up, equations, dg, i, j)
         umm_node = get_node_vars(umm, equations, dg, i, j)
         upp_node = get_node_vars(upp, equations, dg, i, j)
         fm, gm = fluxes(um_node, equations)
         fp, gp = fluxes(up_node, equations)
         fmm, gmm = fluxes(umm_node, equations)
         fpp, gpp = fluxes(upp_node, equations)

         # Updating u_np1_low here
         F_ = get_node_vars(F_cell, equations, dg, i, j)
         G_ = get_node_vars(G_cell, equations, dg, i, j)

         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
            multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[ii, i],
               F_, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[jj, j],
               G_, equations, dg, i, jj)
         end

         # UPDATING u_np1_low ENDS!!!

         ftttt = 0.5 * (fpp - 4.0 * fp + 6.0 * f_node - 4.0 * fm + fmm)
         gtttt = 0.5 * (gpp - 4.0 * gp + 6.0 * g_node - 4.0 * gm + gmm)
         multiply_add_to_node_vars!(F_cell, 1.0 / 120.0, ftttt, equations, dg, i, j)
         multiply_add_to_node_vars!(G_cell, 1.0 / 120.0, gtttt, equations, dg, i, j)

         F_node = get_node_vars(F_cell, equations, dg, i, j)
         G_node = get_node_vars(G_cell, equations, dg, i, j)

         for ii in eachnode(dg)
            # res              += -lam * D * F for each variable
            # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], F_node, equations, dg, ii, j, element)

            multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[ii, i],
               F_node, equations, dg, ii, j)
         end

         for jj in eachnode(dg)
            # C += -lam*g*Dm' for each variable
            # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], G_node, equations, dg, i, jj, element)

            multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[jj, j],
               G_node, equations, dg, i, jj)
         end

         u_node = get_node_vars(u, equations, dg, i, j, element)
         x = get_node_coords(node_coordinates, equations, dg, i, j, element)
         stttt = calc_source_tttt_N4(u_node, up_node, um_node, upp_node, umm_node,
            x, t, dt, source_terms,
            equations, dg, cache)
         multiply_add_to_node_vars!(S, 1.0 / 120.0, stttt, equations, dg, i, j)

         # TODO - update to v1.8 and call with @inline
         # Give u1_ or U depending on dissipation model
         U_node = get_node_vars(U_cell, equations, dg, i, j)

         # Ub = UT * V
         # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
         set_node_vars!(element_cache.F, F_node, equations, dg, 1, i, j, element)
         set_node_vars!(element_cache.F, G_node, equations, dg, 2, i, j, element)
         set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

         S_node = get_node_vars(S, equations, dg, i, j)
         # inv_jacobian = inverse_jacobian[i, j, element]
         multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
            i, j, element)
      end

      @unpack temporal_errors = cache
      @unpack abstol, reltol = tolerances
      temporal_errors[element] = zero(dt)
      for j in eachnode(dg), i in eachnode(dg)
         u_np1_node = get_node_vars(u_np1, equations, dg, i, j)
         u_np1_low_node = get_node_vars(u_np1_low, equations, dg, i, j)
         # u_node = get_node_vars(u, equations, dg, i, j, element)
         for v in eachvariable(equations)
            temporal_errors[element] += (
               (u_np1_node[v] - u_np1_low_node[v])
               /
               (abstol + reltol * max(abs(u_np1_node[v]), abs(u_np1_low_node[v])))
            )^2
         end
      end
      return nothing
   end

   function prolong2interfaces!(cache, u,
      mesh::TreeMesh{2}, equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)

      @unpack interfaces, interface_cache, element_cache = cache
      @unpack orientations = interfaces

      @threaded for interface in eachinterface(dg, cache)
         left_element  = interfaces.neighbor_ids[1, interface]
         right_element = interfaces.neighbor_ids[2, interface]

         if orientations[interface] == 1
            # interface in x-direction
            for j in eachnode(dg), v in eachvariable(equations)
               # Solution
               interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j, left_element]
               interfaces.u[2, v, j, interface] = u[v, 1, j, right_element]

               # Time averaged solution
               interface_cache.U[1, v, j, interface] = element_cache.U[v, nnodes(dg), j, left_element]
               interface_cache.U[2, v, j, interface] = element_cache.U[v, 1, j, right_element]

               # Fluxes
               # TODO - RENAME THIS TO F ALREADY!!
               interface_cache.f[1, v, j, interface] = element_cache.F[v, 1, nnodes(dg), j, left_element]
               interface_cache.f[2, v, j, interface] = element_cache.F[v, 1, 1, j, right_element]
            end
            for v in eachvariable(equations), j in eachnode(dg)
               interface_cache.fn_low[1, v, j, interface] = element_cache.fn_low[
                  v, j, 2, left_element]
               interface_cache.fn_low[2, v, j, interface] = element_cache.fn_low[
                  v, j, 1, right_element]
            end
         else # if orientations[interface] == 2
            # interface in y-direction
            for i in eachnode(dg), v in eachvariable(equations)
               # Solution
               interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), left_element]
               interfaces.u[2, v, i, interface] = u[v, i, 1, right_element]

               # Time averaged solution
               interface_cache.U[1, v, i, interface] = element_cache.U[v, i, nnodes(dg), left_element]
               interface_cache.U[2, v, i, interface] = element_cache.U[v, i, 1, right_element]

               # Fluxes
               interface_cache.f[1, v, i, interface] = element_cache.F[v, 2, i, nnodes(dg), left_element]
               interface_cache.f[2, v, i, interface] = element_cache.F[v, 2, i, 1, right_element]
            end
            for v in eachvariable(equations), i in eachnode(dg)
               interface_cache.fn_low[1, v, i, interface] = element_cache.fn_low[
                  v, i, 4, left_element]
               interface_cache.fn_low[2, v, i, interface] = element_cache.fn_low[
                  v, i, 3, right_element]
            end
         end
      end

      return nothing
   end

   function prolong2boundaries!(cache, u,
      mesh::TreeMesh{2}, equations, surface_integral,
      time_discretization::AbstractLWTimeDiscretization, dg::DG)
      @unpack boundaries, boundary_cache = cache
      @unpack orientations, neighbor_sides = boundaries
      @unpack U, F = cache.element_cache

      @threaded for boundary in eachboundary(dg, cache)
         element = boundaries.neighbor_ids[boundary]

         if orientations[boundary] == 1
            # boundary in x-direction
            if neighbor_sides[boundary] == 1
               # element in -x direction of boundary
               for l in eachnode(dg), v in eachvariable(equations)
                  boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
                  boundary_cache.U[1, v, l, boundary] = U[v, nnodes(dg), l, element]
                  boundary_cache.f[1, v, l, boundary] = F[v, 1, nnodes(dg), l, element]
               end
            else # Element in +x direction of boundary
               for l in eachnode(dg), v in eachvariable(equations)
                  boundaries.u[2, v, l, boundary] = u[v, 1, l, element]
                  boundary_cache.U[2, v, l, boundary] = U[v, 1, l, element]
                  boundary_cache.f[2, v, l, boundary] = F[v, 1, 1, l, element]
               end
            end
         else # if orientations[boundary] == 2
            # boundary in y-direction
            if neighbor_sides[boundary] == 1
               # element in -y direction of boundary
               for l in eachnode(dg), v in eachvariable(equations)
                  boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
                  # TODO - boundaries_lw or lw_boundary_cache? Choose one!
                  boundary_cache.U[1, v, l, boundary] = U[v, l, nnodes(dg), element]
                  boundary_cache.f[1, v, l, boundary] = F[v, 2, l, nnodes(dg), element]
               end
            else
               # element in +y direction of boundary
               for l in eachnode(dg), v in eachvariable(equations)
                  boundaries.u[2, v, l, boundary] = u[v, l, 1, element]
                  boundary_cache.U[2, v, l, boundary] = U[v, l, 1, element]
                  boundary_cache.f[2, v, l, boundary] = F[v, 2, l, 1, element]
               end
            end
         end
      end

      return nothing
   end

   # TODO: Taal dimension agnostic
   function calc_boundary_flux!(cache, t, dt, boundary_condition::BoundaryConditionPeriodic,
      mesh::TreeMesh{2}, equations, surface_integral, time_discretization::AbstractLWTimeDiscretization,
      dg::DG, scaling_factor = 1)
      @assert isempty(eachboundary(dg, cache))
   end

   function calc_boundary_flux!(cache, t, dt, boundary_conditions::NamedTuple,
      ::TreeMesh{2}, equations, surface_integral,
      time_discretization::AbstractLWTimeDiscretization, dg::DG,
      scaling_factor = 1)
      @unpack surface_flux_values = cache.elements
      @unpack n_boundaries_per_direction = cache.boundaries

      # Calculate indices
      lasts = accumulate(+, n_boundaries_per_direction)
      firsts = lasts - n_boundaries_per_direction .+ 1

      # Calc boundary fluxes in each direction
      calc_adv_boundary_flux_by_direction!(surface_flux_values, t, dt, boundary_conditions[1],
         equations, surface_integral, time_discretization, dg, cache,
         1, firsts[1], lasts[1], scaling_factor)
      calc_adv_boundary_flux_by_direction!(surface_flux_values, t, dt, boundary_conditions[2],
         equations, surface_integral, time_discretization, dg, cache,
         2, firsts[2], lasts[2], scaling_factor)
      calc_adv_boundary_flux_by_direction!(surface_flux_values, t, dt, boundary_conditions[3],
         equations, surface_integral, time_discretization, dg, cache,
         3, firsts[3], lasts[3], scaling_factor)
      calc_adv_boundary_flux_by_direction!(surface_flux_values, t, dt, boundary_conditions[4],
         equations, surface_integral, time_discretization, dg, cache,
         4, firsts[4], lasts[4], scaling_factor)
   end

   function calc_adv_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,4}, t, dt,
      boundary_condition, equations,
      surface_integral, ::AbstractLWTimeDiscretization, dg::DG, cache,
      direction, first_boundary, last_boundary, scaling_factor)

      @unpack surface_flux = surface_integral
      @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries
      @unpack U, f, outer_cache = cache.boundary_cache

      @threaded for boundary in first_boundary:last_boundary
         # Get neighboring element
         neighbor = neighbor_ids[boundary]

         for i in eachnode(dg)
            # Get boundary flux
            u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, boundary)
            U_ll, U_rr = get_surface_node_vars(U, equations, dg, i, boundary)
            F_ll, F_rr = get_surface_node_vars(f, equations, dg, i, boundary)
            if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
               u_inner = u_ll
               U_inner = U_ll
               F_inner = F_ll
            else # Element is on the right, boundary on the left
               u_inner = u_rr
               U_inner = U_rr
               F_inner = F_rr
            end
            x = get_node_coords(node_coordinates, equations, dg, i, boundary)
            flux = boundary_condition(U_inner, F_inner, u_inner, outer_cache, orientations[boundary],
               direction, x, t, dt, surface_flux, equations, dg,
               get_time_discretization(dg), scaling_factor)

            # flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
            #                           equations)

            # Copy flux to left and right element storage
            for v in eachvariable(equations)
               surface_flux_values[v, i, direction, neighbor] = flux[v]
            end
         end
      end

      return nothing
   end

   function prolong2mortars!(cache, u,
      mesh::TreeMesh{2}, equations,
      mortar_l2::LobattoLegendreMortarL2, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DGSEM)

      @unpack lw_mortars = cache
      @unpack U, F = cache.element_cache

      @unpack U_upper, U_lower, F_upper, F_lower, fn_low_upper, fn_low_lower = lw_mortars

      @threaded for mortar in eachmortar(dg, cache)
         large_element = cache.mortars.neighbor_ids[3, mortar]
         upper_element = cache.mortars.neighbor_ids[2, mortar]
         lower_element = cache.mortars.neighbor_ids[1, mortar]

         # Copy solution small to small
         if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if cache.mortars.orientations[mortar] == 1
               # L2 mortars in x-direction
               for l in eachnode(dg)
                  for v in eachvariable(equations)
                     cache.mortars.u_upper[2, v, l, mortar] = u[v, 1, l, upper_element]
                     cache.mortars.u_lower[2, v, l, mortar] = u[v, 1, l, lower_element]

                     lw_mortars.U_upper[2, v, l, mortar] = U[v, 1, l, upper_element]
                     lw_mortars.U_lower[2, v, l, mortar] = U[v, 1, l, lower_element]

                     lw_mortars.F_upper[2, v, l, mortar] = F[v, 1, 1, l, upper_element]
                     lw_mortars.F_lower[2, v, l, mortar] = F[v, 1, 1, l, lower_element]
                  end
               end
            else
               # L2 mortars in y-direction
               for l in eachnode(dg)
                  for v in eachvariable(equations)
                     cache.mortars.u_upper[2, v, l, mortar] = u[v, l, 1, upper_element]
                     cache.mortars.u_lower[2, v, l, mortar] = u[v, l, 1, lower_element]

                     lw_mortars.U_upper[2, v, l, mortar] = U[v, l, 1, upper_element]
                     lw_mortars.U_lower[2, v, l, mortar] = U[v, l, 1, lower_element]

                     lw_mortars.F_upper[2, v, l, mortar] = F[v, 2, l, 1, upper_element]
                     lw_mortars.F_lower[2, v, l, mortar] = F[v, 2, l, 1, lower_element]
                  end
               end
            end
         else # large_sides[mortar] == 2 -> small elements on left side
            if cache.mortars.orientations[mortar] == 1
               # L2 mortars in x-direction
               for l in eachnode(dg)
                  for v in eachvariable(equations)
                     cache.mortars.u_upper[1, v, l, mortar] = u[v, nnodes(dg), l, upper_element]
                     cache.mortars.u_lower[1, v, l, mortar] = u[v, nnodes(dg), l, lower_element]

                     lw_mortars.U_upper[1, v, l, mortar] = U[v, nnodes(dg), l, upper_element]
                     lw_mortars.U_lower[1, v, l, mortar] = U[v, nnodes(dg), l, lower_element]

                     lw_mortars.F_upper[1, v, l, mortar] = F[v, 1, nnodes(dg), l, upper_element]
                     lw_mortars.F_lower[1, v, l, mortar] = F[v, 1, nnodes(dg), l, lower_element]
                  end
               end
            else
               # L2 mortars in y-direction
               for l in eachnode(dg)
                  for v in eachvariable(equations)
                     cache.mortars.u_upper[1, v, l, mortar] = u[v, l, nnodes(dg), upper_element]
                     cache.mortars.u_lower[1, v, l, mortar] = u[v, l, nnodes(dg), lower_element]

                     lw_mortars.U_upper[1, v, l, mortar] = U[v, l, nnodes(dg), upper_element]
                     lw_mortars.U_lower[1, v, l, mortar] = U[v, l, nnodes(dg), lower_element]

                     lw_mortars.F_upper[1, v, l, mortar] = F[v, 2, l, nnodes(dg), upper_element]
                     lw_mortars.F_lower[1, v, l, mortar] = F[v, 2, l, nnodes(dg), lower_element]
                  end
               end
            end
         end

         # Interpolate large element face data to small interface locations
         if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
            leftright = 1
            if cache.mortars.orientations[mortar] == 1
               # L2 mortars in x-direction
               u_large = view(u, :, nnodes(dg), :, large_element)
               U_large = view(U, :, nnodes(dg), :, large_element)
               F_large = view(F, :, 1, nnodes(dg), :, large_element)
               element_solutions_to_mortars!(cache.mortars, cache.lw_mortars, mortar_l2,
                  leftright, mortar, u_large, U_large, F_large,
                  time_discretization)
            else
               # L2 mortars in y-direction
               u_large = view(u, :, :, nnodes(dg), large_element)
               U_large = view(U, :, :, nnodes(dg), large_element)
               F_large = view(F, :, 2, :, nnodes(dg), large_element)
               element_solutions_to_mortars!(cache.mortars, cache.lw_mortars, mortar_l2,
                  leftright, mortar, u_large, U_large, F_large,
                  time_discretization)
            end
         else # large_sides[mortar] == 2 -> large element on right side
            leftright = 2
            if cache.mortars.orientations[mortar] == 1
               # L2 mortars in x-direction
               u_large = view(u, :, 1, :, large_element)
               U_large = view(U, :, 1, :, large_element)
               F_large = view(F, :, 1, 1, :, large_element)
               element_solutions_to_mortars!(cache.mortars, cache.lw_mortars, mortar_l2,
                  leftright, mortar, u_large, U_large, F_large,
                  time_discretization)
            else
               # L2 mortars in y-direction
               u_large = view(u, :, :, 1, large_element)
               U_large = view(U, :, :, 1, large_element)
               F_large = view(F, :, 2, :, 1, large_element)
               element_solutions_to_mortars!(cache.mortars, cache.lw_mortars, mortar_l2,
                  leftright, mortar, u_large, U_large, F_large,
                  time_discretization)
            end
         end
      end

      return nothing
   end

   @inline function element_solutions_to_mortars!(mortars, lw_mortars, mortar_l2::LobattoLegendreMortarL2, leftright, mortar,
      u_large::AbstractArray{<:Any,2}, U_large::AbstractArray{<:Any,2}, F_large::AbstractArray{<:Any,2},
      time_discretization::AbstractLWTimeDiscretization)
      multiply_dimensionwise!(view(mortars.u_upper, leftright, :, :, mortar),
         mortar_l2.forward_upper, u_large)
      multiply_dimensionwise!(view(mortars.u_lower, leftright, :, :, mortar),
         mortar_l2.forward_lower, u_large)

      multiply_dimensionwise!(view(lw_mortars.U_upper, leftright, :, :, mortar),
         mortar_l2.forward_upper, U_large)
      multiply_dimensionwise!(view(lw_mortars.U_lower, leftright, :, :, mortar),
         mortar_l2.forward_lower, U_large)

      multiply_dimensionwise!(view(lw_mortars.F_upper, leftright, :, :, mortar),
         mortar_l2.forward_upper, F_large)
      multiply_dimensionwise!(view(lw_mortars.F_lower, leftright, :, :, mortar),
         mortar_l2.forward_lower, F_large)
      return nothing
   end


   function calc_mortar_flux!(surface_flux_values,
      mesh::TreeMesh{2},
      nonconservative_terms::False, equations,
      mortar_l2::LobattoLegendreMortarL2,
      surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache)
      @unpack surface_flux = surface_integral
      @unpack u_lower, u_upper, orientations = cache.mortars
      @unpack U_lower, U_upper, F_lower, F_upper = cache.lw_mortars
      @unpack fstar_upper_threaded, fstar_lower_threaded = cache

      @threaded for mortar in eachmortar(dg, cache)
         # Choose thread-specific pre-allocated container
         fstar_upper = fstar_upper_threaded[Threads.threadid()]
         fstar_lower = fstar_lower_threaded[Threads.threadid()]

         # Calculate fluxes
         orientation = orientations[mortar]
         calc_fstar!(fstar_upper, equations, surface_flux, time_discretization,
            dg, u_upper, F_upper, U_upper, mortar, orientation)
         calc_fstar!(fstar_lower, equations, surface_flux, time_discretization,
            dg, u_lower, F_lower, U_lower, mortar, orientation)

         mortar_fluxes_to_elements!(surface_flux_values,
            mesh, equations, mortar_l2, dg, cache,
            mortar, fstar_upper, fstar_lower)
      end

      return nothing
   end

   @inline function calc_fstar!(destination::AbstractArray{<:Any,2}, equations,
      surface_flux, time_discretization::AbstractLWTimeDiscretization, dg::DGSEM,
      u_interfaces, F_interfaces, U_interfaces, interface, orientation)

      for i in eachnode(dg)
         # Call pointwise two-point numerical flux function
         u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, interface)
         F_ll, F_rr = get_surface_node_vars(F_interfaces, equations, dg, i, interface)
         U_ll, U_rr = get_surface_node_vars(U_interfaces, equations, dg, i, interface)
         flux = surface_flux(F_ll, F_rr, U_ll, U_rr, u_ll, u_rr, orientation, equations)
         # flux = surface_flux(u_ll, u_rr, orientation, equations)

         # Copy flux to left and right element storage
         set_node_vars!(destination, flux, equations, dg, i)
      end

      return nothing
   end

   function calc_volume_integral!(du, u, t, dt, tolerances,
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
      degree = polydeg(dg)

      # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
      alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg, cache)

      # Determine element ids for DG-only and blended DG-FV volume integral
      pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

      if degree == 1
         # Loop over pure DG elements
         @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
            element = element_ids_dg[idx_element]
            alpha_element = alpha[element]

            # Calculate DG volume integral contribution
            weak_form_kernel_1!(du, u, t, dt, element, mesh,
               nonconservative_terms, equations,
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
            weak_form_kernel_1!(du, u, t, dt, element, mesh,
               nonconservative_terms, equations,
               dg, cache, 1 - alpha_element)


            fv_kernel!(du, u, dt, volume_integral.reconstruction, mesh,
               nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, alpha_element)
         end
      elseif degree == 2
         # Loop over pure DG elements
         @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
            element = element_ids_dg[idx_element]
            alpha_element = alpha[element]

            # Calculate DG volume integral contribution
            weak_form_kernel_2!(du, u, t, dt, tolerances, element, mesh,
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
            weak_form_kernel_2!(du, u, t, dt, tolerances, element, mesh,
               nonconservative_terms, source_terms, equations,
               dg, cache, 1 - alpha_element)


            fv_kernel!(du, u, dt, volume_integral.reconstruction, mesh,
               nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, alpha_element)
         end
      elseif degree == 3
         # Loop over pure DG elements
         @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
            element = element_ids_dg[idx_element]
            alpha_element = alpha[element]

            # Calculate DG volume integral contribution
            weak_form_kernel_3!(du, u, t, dt, tolerances, element, mesh,
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
            weak_form_kernel_3!(du, u, t, dt, tolerances, element, mesh,
               nonconservative_terms, source_terms, equations,
               dg, cache, 1 - alpha_element)


            fv_kernel!(du, u, dt, volume_integral.reconstruction, mesh,
               nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, alpha_element)
         end
      else
         # Loop over pure DG elements
         @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
            element = element_ids_dg[idx_element]
            alpha_element = alpha[element]

            # Calculate DG volume integral contribution
            weak_form_kernel_4!(du, u, t, dt, tolerances, element, mesh,
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
            weak_form_kernel_4!(du, u, t, dt, tolerances, element, mesh,
               nonconservative_terms, source_terms, equations,
               dg, cache, 1 - alpha_element)

            fv_kernel!(du, u, dt, volume_integral.reconstruction, mesh,
               nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, alpha_element)
         end

         # for element in eachelement(dg, cache)
         #    alpha_element = alpha[element]
         #    # Calculate DG volume integral contribution
         #    weak_form_kernel_4!(du, u, t, dt, tolerances, element, mesh,
         #       nonconservative_terms, source_terms, equations,
         #       dg, cache, 1 - alpha_element)

         #    fv_kernel!(du, u, dt, volume_integral.reconstruction, mesh,
         #       nonconservative_terms, equations, volume_flux_fv,
         #       dg, cache, element, alpha_element)
         # end

      end

      return alpha
   end

   function compute_subcells(::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
      dg::DG, RealT = eltype(dg.basis.weights))
      @unpack nodes, weights = dg.basis
      # subfaces in reference cell
      x_subfaces, y_subfaces = (OffsetArray(Vector{RealT}(undef, nnodes(dg) + 1), OffsetArrays.Origin(0))
                                for _ in 1:2)
      x_subfaces[0] = y_subfaces[0] = -1.0
      for i in eachnode(dg)
         @views x_subfaces[i] = y_subfaces[i] = -1.0 + sum(weights[1:i])
      end

      ξ_extended = OffsetArray(Vector{RealT}(undef, nnodes(dg) + 2), OffsetArrays.Origin(0))
      ξ_extended[0] = -1.0 # TODO - Should this be different for GL points?
      for i in eachnode(dg)
         ξ_extended[i] = nodes[i]
      end
      ξ_extended[end] = 1.0
      return x_subfaces, y_subfaces, ξ_extended
   end

   # Calling in create_cache function of dg_2d.jl
   # Specifically, it is called in
   # https://github.com/trixi-framework/Trixi.jl/blob/e69d066d780d3a6b0cf3184a78a483b4e5777a2c/src/solvers/dgsem_tree/dg_2d.jl#L30
   function create_cache(mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
      equations, ::VolumeIntegralFRShockCapturing,
      dg::DG, uEltype)
      element_ids_dg = Int[]
      element_ids_dgfv = Int[]

      RealT = Float64 # TODO -Do it correctly
      nan_RealT = convert(RealT, NaN)
      # TODO - this create_cache just creates an empty named tuple. Do this the right way!!!
      cache = (;)
      cfl_number = fill(nan_RealT, 1)

      A3dp1_x = Array{uEltype,3}
      A3dp1_y = Array{uEltype,3}

      fstar1_L_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg) + 1, nnodes(dg)) for _ in 1:Threads.nthreads()]
      fstar1_R_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg) + 1, nnodes(dg)) for _ in 1:Threads.nthreads()]
      fstar2_L_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg) + 1) for _ in 1:Threads.nthreads()]
      fstar2_R_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg) + 1) for _ in 1:Threads.nthreads()]

      # NO, DO IT THE SAME WAY AS BEFORE. THAT IS BETTER.
      nvar = nvariables(equations)
      abstract_constructor(tuple_, x, origin) = [OffsetArray(MArray{tuple_,Float64}(x),
         OffsetArrays.Origin(origin))]
      constructor = x -> abstract_constructor(Tuple{nvar,nnodes(dg) + 2,nnodes(dg) + 2}, x, (1, 0, 0))
      uext_threaded = alloc_for_threads(constructor, 1) # u extended by face extrapolation

      uext_threaded = [OffsetArray(zeros(nvar, nnodes(dg) + 2, nnodes(dg) + 2),
                                   OffsetArrays.Origin(1, 0, 0))
                       for _ in 1:Threads.nthreads()]

      mh_face_arrays = [(
                         zeros(nvar, 4, nnodes(dg), nnodes(dg)),
                         zeros(nvar, 4, nnodes(dg), nnodes(dg))
                        )
                        for _ in 1:Threads.nthreads()]
      x_subfaces, y_subfaces, ξ_extended = compute_subcells(mesh, dg)

      return (; cache..., element_ids_dg, element_ids_dgfv,
         fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded,
         uext_threaded, mh_face_arrays, cfl_number,
         x_subfaces, y_subfaces, ξ_extended)
   end

   function compute_alp(
      u_ll, u_rr, primary_element_index, secondary_element_index, Jl, Jr, dt,
      fn, Fn, fn_inner_ll, fn_inner_rr, primary_node_index, equations, dg, volume_integral::VolumeIntegralFR, mesh::TreeMesh)
      return zero(eltype(u_ll))
   end

   function compute_alp(
      u_ll, u_rr, primary_element_index, secondary_element_index, Jl, Jr, dt,
      fn, Fn, fn_inner_ll, fn_inner_rr, primary_node_index, equations, dg, volume_integral::VolumeIntegralFRShockCapturing, mesh::TreeMesh)
      @unpack alpha = volume_integral.indicator.cache
      @unpack weights = dg.basis
      alp = 0.5 * (alpha[primary_element_index] + alpha[secondary_element_index])

      Fn = (1.0 - alp) * Fn_ + alp * fn
      λx, λy = 0.5, 0.5 # blending flux factors (TODO - Do this correctly)
      # u_ll = get_node_vars(ul, equations, dg, nnodes(dg))
      lower_order_update = u_ll - dt * Jl / (weights[nnodes(dg)] * λx) * (Fn - fn_inner_ll)
      if is_admissible(lower_order_update, equations) == false
         return 1.0
      end

      λx, λy = 0.5, 0.5 # blending flux factors (TODO - Do this correctly)
      # u_rr = get_node_vars(ur, equations, dg, 1)
      lower_order_update = u_rr - dt * Jr / (weights[1] * λx) * (fn_inner_rr - Fn)
      if is_admissible(lower_order_update, equations) == false
         return 1.0
      end
      return alp
   end

   function calc_interface_flux!(surface_flux_values, mesh::TreeMesh{2},
      nonconservative_terms::False,
      equations,
      surface_integral, dt, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache)
      @unpack surface_flux = surface_integral
      @unpack u, neighbor_ids, orientations = cache.interfaces
      @unpack U, f, fn_low = cache.interface_cache

      @threaded for interface in eachinterface(dg, cache)
         # Get neighboring elements
         left_id = neighbor_ids[1, interface]
         right_id = neighbor_ids[2, interface]

         # Determine interface direction with respect to elements:
         # orientation = 1: left -> 2, right -> 1
         # orientation = 2: left -> 4, right -> 3
         left_direction = 2 * orientations[interface]
         right_direction = 2 * orientations[interface] - 1

         for i in eachnode(dg)
            # Call pointwise Riemann solver
            U_ll, U_rr = get_surface_node_vars(U, equations, dg, i, interface)
            u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
            f_ll, f_rr = get_surface_node_vars(f, equations, dg, i, interface)
            fn_inner_ll, fn_inner_rr = get_surface_node_vars(fn_low, equations, dg, i, interface)
            Fn_ = surface_flux(f_ll, f_rr, U_ll, U_rr, u_ll, u_rr, orientations[interface], equations)
            fn = surface_flux(u_ll, u_rr, orientations[interface], equations)

            Jl = Jr = cache.interface_cache.inverse_jacobian[i, interface]

            alp = compute_alp(u_ll, u_rr, left_id, right_id, Jl, Jr, dt,
               fn, Fn_, fn_inner_ll, fn_inner_rr, i, equations,
               dg, dg.volume_integral, mesh)
            # Copy flux to left and right element storage
            Fn = alp * fn + (1.0 - alp) * Fn_
            for v in eachvariable(equations)
               surface_flux_values[v, i, left_direction, left_id] = Fn[v]
               surface_flux_values[v, i, right_direction, right_id] = Fn[v]
            end
         end
      end

      return nothing
   end

   @inline function calc_fn_low!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u,
      mesh::Union{TreeMesh{2}},
      nonconservative_terms::False, equations,
      volume_flux_fv, dg::DGSEM, element, cache)
      @unpack fn_low = cache.element_cache
      @unpack weights, derivative_matrix = dg.basis

      # Performance improvement if the metric terms of the subcell FV method are only computed
      # once at the beginning of the simulation, instead of at every Runge-Kutta stage
      fstar1_L[:, 1, :]            .= zero(eltype(fstar1_L))
      fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
      fstar1_R[:, 1, :]            .= zero(eltype(fstar1_R))
      fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

      for j in eachnode(dg)
         for i in (2, nnodes(dg))
            u_ll = get_node_vars(u, equations, dg, i-1, j, element)
            u_rr = get_node_vars(u, equations, dg, i,   j, element)
            flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
            set_node_vars!(fstar1_L, flux, equations, dg, i, j)
            set_node_vars!(fstar1_R, flux, equations, dg, i, j)
         end
      end

      fstar2_L[:, :, 1]            .= zero(eltype(fstar2_L))
      fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
      fstar2_R[:, :, 1]            .= zero(eltype(fstar2_R))
      fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

      for i in eachnode(dg)
         for j in (2, nnodes(dg))
            u_ll = get_node_vars(u, equations, dg, i, j-1, element)
            u_rr = get_node_vars(u, equations, dg, i, j,   element)
            flux = volume_flux_fv(u_ll, u_rr, 2, equations) # orientation 2: y direction
            set_node_vars!(fstar2_L, flux, equations, dg, i, j)
            set_node_vars!(fstar2_R, flux, equations, dg, i, j)
         end
      end

      load_fn_low!(fn_low, mesh, dg, nonconservative_terms,
         fstar1_L, fstar2_L, fstar1_R, fstar2_R, element)

      return nothing
   end

end # muladd macro