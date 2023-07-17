using Trixi: AbstractEquationsParabolic, reset_du!, have_constant_speed,
             calc_viscous_fluxes!, transform_variables!, have_nonconservative_terms,
             prolong2interfaces!, calc_surface_integral!, apply_jacobian!, timer,
             calc_gradient!, get_node_coords, eachinterface, get_surface_node_vars,
             eachboundary, get_unsigned_normal_vector_2d

import Trixi: create_cache, calc_volume_integral!, prolong2boundaries!, calc_boundary_flux!
import Trixi

using MuladdMacro

@muladd begin

# TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?
function rhs!(du, u, t, mesh::Union{TreeMesh{2},P4estMesh{2}}, equations,
   equations_parabolic::AbstractEquationsParabolic, initial_condition,
   boundary_conditions, boundary_conditions_parabolic, source_terms,
   dg::DG, parabolic_scheme, time_discretization::AbstractLWTimeDiscretization, cache,
   cache_parabolic, tolerances::NamedTuple)

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
      flux_viscous, gradients, u_transformed, mesh, equations_parabolic, dg, cache,
      cache_parabolic)

   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

   # Calculate volume integral
   @trixi_timeit timer() "volume integral" calc_volume_integral!(
      du, flux_viscous, gradients, u_transformed, u, t, dt, tolerances, mesh,
      have_nonconservative_terms(equations), source_terms, equations, equations_parabolic,
      dg.volume_integral, time_discretization, dg, cache, cache_parabolic)

   # # Prolong solution to interfaces
   # TODO - This seems unnecessary because the next function also does this prolongation
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
   # Unlike the previous prolongation, here we do prolongation of advective
   # and viscous parts separately and even add their contribution to surface_flux_values
   # separately. First the advective part is handled and then the viscous part.

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

function my_calc_surface_integral!(du, u,
                                mesh::P4estMesh{2},
                                equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGSEM, cache)
    @unpack boundary_interpolation = dg.basis
    @unpack surface_flux_values = cache.elements

    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # Access the factors only once before beginning the loop to increase performance.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor_1 = boundary_interpolation[1, 1]
    factor_2 = boundary_interpolation[nnodes(dg), 2]
    @threaded for element in eachelement(dg, cache)
        for l in eachnode(dg)
            for v in eachvariable(equations)
                # surface at -x
                du[v, 1, l, element] = (du[v, 1, l, element] +
                                        surface_flux_values[v, l, 1, element] *
                                        factor_1)

                # surface at +x
                du[v, nnodes(dg), l, element] = (du[v, nnodes(dg), l, element] +
                                                 surface_flux_values[v, l, 2, element] *
                                                 factor_2)

                # surface at -y
                du[v, l, 1, element] = (du[v, l, 1, element] +
                                        surface_flux_values[v, l, 3, element] *
                                        factor_1)

                # surface at +y
                du[v, l, nnodes(dg), element] = (du[v, l, nnodes(dg), element] +
                                                 surface_flux_values[v, l, 4, element] *
                                                 factor_2)
            end
        end
    end

    return nothing
end

# Parabolic cache
# TODO - Merge with hyperbolic cache
function create_cache(mesh::Union{TreeMesh{2}, P4estMesh{2}},
   equations::AbstractEquationsParabolic,
   time_discretization::AbstractLWTimeDiscretization, dg, RealT, uEltype, cache)
   nan_RealT = convert(RealT, NaN)
   nan_uEltype = convert(uEltype, NaN)

   n_nodes = nnodes(dg)
   n_variables = nvariables(equations)
   n_elements = nelements(dg, cache)
   n_interfaces = ninterfaces(dg, cache)

   Fv = fill(nan_uEltype, (n_variables, 2, n_nodes, n_nodes, n_elements))
   Fb = fill(nan_uEltype, (2, n_variables, n_nodes, n_interfaces))

   nt = Threads.nthreads()
   cell_array_sizes = Dict(1 => 12, 2 => 14, 3 => 20, 4 => 22)
   # big_eval_data_sizes = Dict(1 => 12, 2 => 32, 3 => 40, 4 => 56)
   # small_eval_data_sizes = Dict(1 => 4, 2 => 4, 3 => 4, 4 => 4)
   # if bflux_ind ==  extrapolate
   degree = polydeg(dg)
   degree = min(4, degree)
   cell_array_size = cell_array_sizes[degree]
   big_eval_data_size = 2
   small_eval_data_size = 2
   # elseif bflux_ind == evaluate
   #    cell_array_size = cell_array_sizes[degree]
   #    big_eval_data_size = big_eval_data_sizes[degree]
   #    small_eval_data_size = small_eval_data_sizes[degree]
   # else
   #    @assert false "Incorrect bflux"
   # end

   # Construct `cache_size` number of objects with `constructor`
   # and store them in an SVector
   function alloc(constructor, cache_size)
      SVector{cache_size}(constructor(undef) for _ in Base.OneTo(cache_size))
   end

   # Create the result of `alloc` for each thread. Basically,
   # for each thread, construct `cache_size` number of objects with
   # `constructor` and store them in an SVector
   function alloc_for_threads(constructor, cache_size)
      nt = Threads.nthreads()
      SVector{nt}([alloc(constructor, cache_size) for _ in Base.OneTo(nt)])
   end

   MArr = MArray{Tuple{n_variables, n_nodes, n_nodes}, Float64}
   cell_arrays = alloc_for_threads(MArr, cell_array_size)

   MEval = MArray{Tuple{n_variables, n_nodes},Float64}
   eval_data_big = alloc_for_threads(MEval, big_eval_data_size)

   MEval_small = MArray{Tuple{n_variables, 1}, Float64}
   eval_data_small = alloc_for_threads(MEval_small, small_eval_data_size)

   eval_data = (; eval_data_big, eval_data_small)
   lw_res_cache = (; cell_arrays, eval_data)
   if isa(get_time_discretization(dg), MDRK)
      Fv2 = copy(Fv)
      mdrk_cache = (; Fv1 = Fv, Fv2)
   else
      mdrk_cache = (;)
   end

   cache = (; lw_res_cache, Fv, Fb, mdrk_cache)
end

fluxes(u, grad_u::Tuple{<:Any,<:Any}, equations_parabolic::AbstractEquationsParabolic{2}) = (
   Trixi.flux(u, grad_u, 1, equations_parabolic), Trixi.flux(u, grad_u, 2, equations_parabolic))

function calc_volume_integral!(
   du, flux_viscous, gradients, u_transformed, u, t, dt, tolerances::NamedTuple,
   mesh::Union{TreeMesh{2}, P4estMesh{2}},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic)

   degree = polydeg(dg)

   if degree == 1
      @threaded for element in eachelement(dg, cache)
         weak_form_kernel_1!(du, flux_viscous, gradients, u_transformed, u, t, dt,
            tolerances, mesh,
            have_nonconservative_terms, source_terms,
            equations, equations_parabolic,
            volume_integral, time_discretization,
            dg, cache, cache_parabolic, element)
      end
   elseif degree == 2
      @threaded for element in eachelement(dg, cache)
         weak_form_kernel_2!(du, flux_viscous, gradients, u_transformed, u, t, dt,
            tolerances, mesh,
            have_nonconservative_terms, source_terms,
            equations, equations_parabolic,
            volume_integral, time_discretization,
            dg, cache, cache_parabolic, element)
      end
   elseif degree == 3
      @threaded for element in eachelement(dg, cache)
         weak_form_kernel_3!(du, flux_viscous, gradients, u_transformed, u, t, dt,
            tolerances, mesh,
            have_nonconservative_terms, source_terms,
            equations, equations_parabolic,
            volume_integral, time_discretization,
            dg, cache, cache_parabolic, element)
      end
   else
      @threaded for element in eachelement(dg, cache)
         weak_form_kernel_4!(du, flux_viscous, gradients, u_transformed, u, t, dt,
            tolerances, mesh,
            have_nonconservative_terms, source_terms,
            equations, equations_parabolic,
            volume_integral, time_discretization,
            dg, cache, cache_parabolic, element)
      end
   end

end

fluxes(u, grad_u, equations::AbstractEquations{2}, equations_parabolic::AbstractEquationsParabolic{2}
) = (Trixi.flux(u, 1, equations),
Trixi.flux(u, 2, equations)),
(Trixi.flux(u, grad_u, 1, equations_parabolic),
Trixi.flux(u, grad_u, 2, equations_parabolic))

function weak_form_kernel_1!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::TreeMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache = cache
   @unpack cell_arrays = lw_res_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   id = Threads.threadid()

   Fa, Ga, ut, U, up, um, ft, gt, S = cell_arrays[id]

   utx, uty, upx, upy, umx, umy, f_visc_t, g_visc_t, Fv, Gv,
     u_np1, unp1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(arr)))

   refresh!.((ut, ft, gt, f_visc_t, g_visc_t, utx, uty))

   # Calculate volume terms in one element

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
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
      end

      set_node_vars!(Fa, flux_adv_1, equations, dg, i, j)
      set_node_vars!(Ga, flux_adv_2, equations, dg, i, j)
      set_node_vars!(Fv, flux_visc_1, equations, dg, i, j)
      set_node_vars!(Gv, flux_visc_2, equations, dg, i, j)

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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      fta = 0.5 * (fpa - fma)
      gta = 0.5 * (gpa - gma)
      multiply_add_to_node_vars!(Fa, 0.5, fta, equations, dg, i, j)
      multiply_add_to_node_vars!(Ga, 0.5, gta, equations, dg, i, j)

      ftv = 0.5 * (fpv - fmv)
      gtv = 0.5 * (gpv - gmv)
      multiply_add_to_node_vars!(Fv, 0.5, ftv, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv, 0.5, gtv, equations, dg, i, j)

      Fa_node = get_node_vars(Fa, equations, dg, i, j)
      Fv_node = get_node_vars(Fv, equations, dg, i, j)

      F = Fa_node - Fv_node

      Ga_node = get_node_vars(Ga, equations, dg, i, j)
      Gv_node = get_node_vars(Gv, equations, dg, i, j)

      G = Ga_node - Gv_node
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F, equations,
            dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G, equations,
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
      set_node_vars!(cache.element_cache.U, U_node, equations, dg, i, j, element)
      set_node_vars!(cache.element_cache.F, Fa_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Fv_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache.element_cache.F, Ga_node, equations, dg, 2, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Gv_node, equations, dg, 2, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      # inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
   end

   return nothing
end

function weak_form_kernel_2!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::TreeMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache = cache
   @unpack cell_arrays = lw_res_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   id = Threads.threadid()

   fa, ga, Fa, Ga, ut, utt, U, up, um, S = cell_arrays[id]

   utx, uty, uttx, utty, upx, upy, umx, umy, fv, gv,
   Fv, Gv, u_np1, u_np1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(u)))

   refresh!.((ut, utx, uty, utt, uttx, utty))

   # Calculate volume terms in one element

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
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
      end

      set_node_vars!(fa, flux_adv_1, equations, dg, i, j)
      set_node_vars!(ga, flux_adv_2, equations, dg, i, j)

      set_node_vars!(fv, flux_visc_1, equations, dg, i, j)
      set_node_vars!(gv, flux_visc_2, equations, dg, i, j)

      set_node_vars!(Fa, flux_adv_1, equations, dg, i, j)
      set_node_vars!(Ga, flux_adv_2, equations, dg, i, j)
      set_node_vars!(Fv, flux_visc_1, equations, dg, i, j)
      set_node_vars!(Gv, flux_visc_2, equations, dg, i, j)

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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      fta = 0.5 * (fpa - fma)
      gta = 0.5 * (gpa - gma)
      multiply_add_to_node_vars!(Fa, 0.5, fta, equations, dg, i, j)
      multiply_add_to_node_vars!(Ga, 0.5, gta, equations, dg, i, j)

      ftv = 0.5 * (fpv - fmv)
      gtv = 0.5 * (gpv - gmv)
      multiply_add_to_node_vars!(Fv, 0.5, ftv, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv, 0.5, gtv, equations, dg, i, j)

      ft = fta - ftv
      gt = gta - gtv

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utt, derivative_matrix[ii, i], ft, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utt, derivative_matrix[jj, j], gt, equations,
            dg, i, jj)
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

   # Compute ∇u_tt
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
      # inv_jacobian = inverse_jacobian[i, j, element]
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

      multiply_add_to_node_vars!(upx, 0.5, uttx_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umx, 0.5, uttx_node, equations, dg, i, j)

      multiply_add_to_node_vars!(upy, 0.5, utty_node, equations, dg, i, j)
      multiply_add_to_node_vars!(umy, 0.5, utty_node, equations, dg, i, j)

      fa_node = get_node_vars(fa, equations, dg, i, j)
      ga_node = get_node_vars(ga, equations, dg, i, j)
      fv_node = get_node_vars(fv, equations, dg, i, j)
      gv_node = get_node_vars(gv, equations, dg, i, j)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      um_node = get_node_vars(um, equations, dg, i, j)
      up_node = get_node_vars(up, equations, dg, i, j)

      umx_node = get_node_vars(umx, equations, dg, i, j)
      upx_node = get_node_vars(upx, equations, dg, i, j)
      umy_node = get_node_vars(umy, equations, dg, i, j)
      upy_node = get_node_vars(upy, equations, dg, i, j)

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)
      ftta, gtta = fpa - 2.0 * fa_node + fma, gpa - 2.0 * ga_node + gma
      fttv, gttv = fpv - 2.0 * fv_node + fmv, gpv - 2.0 * gv_node + gmv

      multiply_add_to_node_vars!(Fa, 1.0 / 6.0, ftta, equations, dg, i, j)
      multiply_add_to_node_vars!(Fv, 1.0 / 6.0, fttv, equations, dg, i, j)

      multiply_add_to_node_vars!(Ga, 1.0 / 6.0, gtta, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv, 1.0 / 6.0, gttv, equations, dg, i, j)

      Fa_node = get_node_vars(Fa, equations, dg, i, j)
      Fv_node = get_node_vars(Fv, equations, dg, i, j)

      F = Fa_node - Fv_node

      Ga_node = get_node_vars(Ga, equations, dg, i, j)
      Gv_node = get_node_vars(Gv, equations, dg, i, j)

      G = Ga_node - Gv_node
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, derivative_dhat[ii, i], F, equations,
            dg, ii, j, element)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G, equations,
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
      set_node_vars!(cache.element_cache.U, U_node, equations, dg, i, j, element)

      set_node_vars!(cache.element_cache.F, Fa_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Fv_node, equations, dg, 1, i, j, element)

      set_node_vars!(cache.element_cache.F, Ga_node, equations, dg, 2, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Gv_node, equations, dg, 2, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      # inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
   end

   return nothing
end

function weak_form_kernel_3!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::TreeMesh{2},
   have_nonconservative_terms, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache = cache
   @unpack cell_arrays = lw_res_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   id = Threads.threadid()

   fa, ga, Fa, Ga, ut, utt, uttt, U, up, upp, um, umm, S = cell_arrays[id]

   utx, uty, uttx, utty, utttx, uttty, upx, upy, umx, umy, uppx, uppy,
   ummx, ummy, fv, gv, Fv, Gv, u_np1, u_np1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(u)))

   refresh!.((ut, utx, uty, utt, uttx, utty, uttt, utttx, uttty))

   # Calculate volume terms in one element

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
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
      end

      set_node_vars!(fa, flux_adv_1, equations, dg, i, j)
      set_node_vars!(ga, flux_adv_2, equations, dg, i, j)

      set_node_vars!(fv, flux_visc_1, equations, dg, i, j)
      set_node_vars!(gv, flux_visc_2, equations, dg, i, j)

      set_node_vars!(Fa, flux_adv_1, equations, dg, i, j)
      set_node_vars!(Ga, flux_adv_2, equations, dg, i, j)
      set_node_vars!(Fv, flux_visc_1, equations, dg, i, j)
      set_node_vars!(Gv, flux_visc_2, equations, dg, i, j)

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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm_node, (ummx_node, ummy_node), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp_node, (uppx_node, uppy_node), equations, equations_parabolic)

      fta = 1.0 / 12.0 * (-fppa + 8.0 * fpa - 8.0 * fma + fmma)
      gta = 1.0 / 12.0 * (-gppa + 8.0 * gpa - 8.0 * gma + gmma)
      multiply_add_to_node_vars!(Fa, 0.5, fta, equations, dg, i, j)
      multiply_add_to_node_vars!(Ga, 0.5, gta, equations, dg, i, j)

      ftv = 1.0 / 12.0 * (-fppv + 8.0 * fpv - 8.0 * fmv + fmmv)
      gtv = 1.0 / 12.0 * (-gppv + 8.0 * gpv - 8.0 * gmv + gmmv)
      multiply_add_to_node_vars!(Fv, 0.5, ftv, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv, 0.5, gtv, equations, dg, i, j)

      ft = fta - ftv
      gt = gta - gtv

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utt, derivative_matrix[ii, i], ft, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utt, derivative_matrix[jj, j], gt, equations,
            dg, i, jj)
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

   # Compute ∇u_tt
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
      # inv_jacobian = inverse_jacobian[i, j, element]
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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)
      ftta, gtta = fpa - 2.0 * fa_node + fma, gpa - 2.0 * ga_node + gma
      fttv, gttv = fpv - 2.0 * fv_node + fmv, gpv - 2.0 * gv_node + gmv

      multiply_add_to_node_vars!(Fa, 1.0 / 6.0, ftta, equations, dg, i, j)
      multiply_add_to_node_vars!(Fv, 1.0 / 6.0, fttv, equations, dg, i, j)

      multiply_add_to_node_vars!(Ga, 1.0 / 6.0, gtta, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv, 1.0 / 6.0, gttv, equations, dg, i, j)

      ftt = ftta - fttv
      gtt = gtta - gttv

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], ftt, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], gtt, equations,
            dg, i, jj)
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

   # Compute ∇u_ttt
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

   # Scale ∇u_ttt
   for j in eachnode(dg), i in eachnode(dg)
      # inv_jacobian = inverse_jacobian[i, j, element]
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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm_node, (ummx_node, ummy_node), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp_node, (uppx_node, uppy_node), equations, equations_parabolic)

      fttta = 0.5 * (fppa - 2.0 * fpa + 2.0 * fma - fmma)
      multiply_add_to_node_vars!(Fa, 1.0 / 24.0, fttta, equations, dg, i, j)
      ftttv = 0.5 * (fppv - 2.0 * fpv + 2.0 * fmv - fmmv)
      multiply_add_to_node_vars!(Fv, 1.0 / 24.0, ftttv, equations, dg, i, j)

      gttta = 0.5 * (gppa - 2.0 * gpa + 2.0 * gma - gmma)
      multiply_add_to_node_vars!(Ga, 1.0 / 24.0, gttta, equations, dg, i, j)
      gtttv = 0.5 * (gppv - 2.0 * gpv + 2.0 * gmv - gmmv)
      multiply_add_to_node_vars!(Gv, 1.0 / 24.0, gtttv, equations, dg, i, j)

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
      set_node_vars!(cache.element_cache.U, U_node, equations, dg, i, j, element)

      set_node_vars!(cache.element_cache.F, Fa_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Fv_node, equations, dg, 1, i, j, element)

      set_node_vars!(cache.element_cache.F, Ga_node, equations, dg, 2, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Gv_node, equations, dg, 2, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      # inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
         i, j, element)
   end

   return nothing
end

function weak_form_kernel_4!(
   du, flux_viscous, gradients, u_transformed, u, t, dt,
   tolerances, mesh::TreeMesh{2},
   have_nonconservative_terms::False, source_terms,
   equations, equations_parabolic::AbstractEquationsParabolic,
   volume_integral::VolumeIntegralFR, time_discretization::AbstractLWTimeDiscretization,
   dg::DGSEM, cache, cache_parabolic, element)

   gradients_x, gradients_y = gradients
   flux_viscous_x, flux_viscous_y = flux_viscous # viscous fluxes computed by correction

   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack node_coordinates = cache.elements

   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache

   inv_jacobian = cache.elements.inverse_jacobian[element]

   id = Threads.threadid()

   fa, ga, Fa, Ga, ut, utt, uttt, utttt, U, up, upp, um, umm, S = cell_arrays[id]

   utx, uty, uttx, utty, utttx, uttty, uttttx, utttty, upx, upy, umx, umy, uppx, uppy,
   ummx, ummy, fv, gv, Fv, Gv, u_np1, u_np1_low = cache_parabolic.lw_res_cache.cell_arrays[id]

   refresh!(arr) = fill!(arr, zero(eltype(u)))

   refresh!.((ut, utx, uty, utt, uttx, utty, uttt, utttx, uttty, utttt, uttttx, utttty))

   # Calculate volume terms in one element

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
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], flux1, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], flux2, equations, dg, i, jj)
      end

      set_node_vars!(fa, flux_adv_1, equations, dg, i, j)
      set_node_vars!(ga, flux_adv_2, equations, dg, i, j)

      set_node_vars!(fv, flux_visc_1, equations, dg, i, j)
      set_node_vars!(gv, flux_visc_2, equations, dg, i, j)

      set_node_vars!(Fa, flux_adv_1, equations, dg, i, j)
      set_node_vars!(Ga, flux_adv_2, equations, dg, i, j)
      set_node_vars!(Fv, flux_visc_1, equations, dg, i, j)
      set_node_vars!(Gv, flux_visc_2, equations, dg, i, j)

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
      ut_node  = get_node_vars(ut,  equations, dg, i, j)
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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm_node, (ummx_node, ummy_node), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp_node, (uppx_node, uppy_node), equations, equations_parabolic)

      fta = 1.0 / 12.0 * (-fppa + 8.0 * fpa - 8.0 * fma + fmma)
      multiply_add_to_node_vars!(Fa, 0.5, fta, equations, dg, i, j)
      gta = 1.0 / 12.0 * (-gppa + 8.0 * gpa - 8.0 * gma + gmma)
      multiply_add_to_node_vars!(Ga, 0.5, gta, equations, dg, i, j)

      ftv = 1.0 / 12.0 * (-fppv + 8.0 * fpv - 8.0 * fmv + fmmv)
      multiply_add_to_node_vars!(Fv, 0.5, ftv, equations, dg, i, j)
      gtv = 1.0 / 12.0 * (-gppv + 8.0 * gpv - 8.0 * gmv + gmmv)
      multiply_add_to_node_vars!(Gv, 0.5, gtv, equations, dg, i, j)

      ft = fta - ftv
      gt = gta - gtv

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utt, derivative_matrix[ii, i], ft, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(utt, derivative_matrix[jj, j], gt, equations,
            dg, i, jj)
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

   # Compute ∇u_tt
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
      # inv_jacobian = inverse_jacobian[i, j, element]
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
      u_node = get_node_vars(u, equations, dg, i, j, element)

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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm_node, (ummx_node, ummy_node), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp_node, (uppx_node, uppy_node), equations, equations_parabolic)

      ftta = (1.0 / 12.0) * (-fppa + 16.0 * fpa - 30.0 * fa_node + 16.0 * fma - fmma)
      fttv = (1.0 / 12.0) * (-fppv + 16.0 * fpv - 30.0 * fv_node + 16.0 * fmv - fmmv)

      gtta = (1.0 / 12.0) * (-gppa + 16.0 * gpa - 30.0 * ga_node + 16.0 * gma - gmma)
      gttv = (1.0 / 12.0) * (-gppv + 16.0 * gpv - 30.0 * gv_node + 16.0 * gmv - gmmv)

      multiply_add_to_node_vars!(Fa, 1.0 / 6.0, ftta, equations, dg, i, j)
      multiply_add_to_node_vars!(Fv, 1.0 / 6.0, fttv, equations, dg, i, j)

      multiply_add_to_node_vars!(Ga, 1.0 / 6.0, gtta, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv, 1.0 / 6.0, gttv, equations, dg, i, j)

      ftt = ftta - fttv
      gtt = gtta - gttv

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], ftt, equations,
            dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], gtt, equations,
            dg, i, jj)
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
      umm_node = get_node_vars(umm, equations, dg, i, j)
      upp_node = get_node_vars(upp, equations, dg, i, j)

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stt = calc_source_tt_N4(u_node, up_node, upp_node, um_node, umm_node, x, t, dt, source_terms,
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

   # Scale ∇u_ttt
   for j in eachnode(dg), i in eachnode(dg)
      # inv_jacobian = inverse_jacobian[i, j, element]
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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm_node, (ummx_node, ummy_node), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp_node, (uppx_node, uppy_node), equations, equations_parabolic)

      fttta = 0.5 * (fppa - 2.0 * fpa + 2.0 * fma - fmma)
      multiply_add_to_node_vars!(Fa, 1.0 / 24.0, fttta, equations, dg, i, j)
      ftttv = 0.5 * (fppv - 2.0 * fpv + 2.0 * fmv - fmmv)
      multiply_add_to_node_vars!(Fv, 1.0 / 24.0, ftttv, equations, dg, i, j)
      fttt = fttta - ftttv

      gttta = 0.5 * (gppa - 2.0 * gpa + 2.0 * gma - gmma)
      multiply_add_to_node_vars!(Ga, 1.0 / 24.0, gttta, equations, dg, i, j)
      gtttv = 0.5 * (gppv - 2.0 * gpv + 2.0 * gmv - gmmv)
      multiply_add_to_node_vars!(Gv, 1.0 / 24.0, gtttv, equations, dg, i, j)
      gttt = gttta - gtttv

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[ii, i], fttt, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
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

   # Scale ∇u_tttt
   for j in eachnode(dg), i in eachnode(dg)
      # inv_jacobian = inverse_jacobian[i, j, element]
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

      (fma, gma), (fmv, gmv) = fluxes(um_node, (umx_node, umy_node), equations, equations_parabolic)
      (fpa, gpa), (fpv, gpv) = fluxes(up_node, (upx_node, upy_node), equations, equations_parabolic)

      (fmma, gmma), (fmmv, gmmv) = fluxes(umm_node, (ummx_node, ummy_node), equations, equations_parabolic)
      (fppa, gppa), (fppv, gppv) = fluxes(upp_node, (uppx_node, uppy_node), equations, equations_parabolic)

      ftttta = 0.5 * (fppa - 4.0 * fpa + 6.0 * fa_node - 4.0 * fma + fmma)
      fttttv = 0.5 * (fppv - 4.0 * fpv + 6.0 * fv_node - 4.0 * fmv + fmmv)

      # UPDATING u_np1_low HERE!!!

      Fa_node_ = get_node_vars(Fa, equations, dg, i, j)
      Fv_node_ = get_node_vars(Fv, equations, dg, i, j)

      F_ = Fa_node_ - Fv_node_

      Ga_node_ = get_node_vars(Ga, equations, dg, i, j)
      Gv_node_ = get_node_vars(Gv, equations, dg, i, j)

      G_ = Ga_node_ - Gv_node_

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(u_np1_low, -dt*inv_jacobian*derivative_matrix[ii, i],
                                          F_, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(u_np1_low, -dt*inv_jacobian*derivative_matrix[jj, j],
                                          G_, equations, dg, i, jj)
      end

      # UPDATING u_np1_low ENDS!!!


      multiply_add_to_node_vars!(Fa, 1.0 / 120.0, ftttta, equations, dg, i, j)
      multiply_add_to_node_vars!(Fv, 1.0 / 120.0, fttttv, equations, dg, i, j)

      gtttta = 0.5 * (gppa - 4.0 * gpa + 6.0 * ga_node - 4.0 * gma + gmma)
      gttttv = 0.5 * (gppv - 4.0 * gpv + 6.0 * gv_node - 4.0 * gmv + gmmv)
      multiply_add_to_node_vars!(Ga, 1.0 / 120.0, gtttta, equations, dg, i, j)
      multiply_add_to_node_vars!(Gv, 1.0 / 120.0, gttttv, equations, dg, i, j)

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

         multiply_add_to_node_vars!(u_np1, -dt*inv_jacobian*derivative_matrix[ii, i],
                                          F, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, derivative_dhat[jj, j], G, equations, dg, i, jj, element)

         multiply_add_to_node_vars!(u_np1, -dt*inv_jacobian*derivative_matrix[jj, j],
                                          G, equations, dg, i, jj)
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

      set_node_vars!(cache.element_cache.F, Fa_node, equations, dg, 1, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Fv_node, equations, dg, 1, i, j, element)

      set_node_vars!(cache.element_cache.F, Ga_node, equations, dg, 2, i, j, element)
      set_node_vars!(cache_parabolic.Fv, Gv_node, equations, dg, 2, i, j, element)

      set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = get_node_vars(S, equations, dg, i, j)
      # inv_jacobian = inverse_jacobian[i, j, element]
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
         i, j, element)
   end

   @unpack temporal_errors = cache
   @unpack abstol,  reltol = tolerances
   temporal_errors[element] = zero(dt)
   for j in eachnode(dg), i in eachnode(dg)
      u_np1_node     = get_node_vars(u_np1, equations, dg, i, j)
      u_np1_low_node = get_node_vars(u_np1_low, equations, dg, i, j)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      for v in eachvariable(equations)
         temporal_errors[element] += (
            ( u_np1_node[v] - u_np1_low_node[v] )
            /
            ( abstol + reltol * max(abs(u_np1_node[v]), abs(u_np1_low_node[v])) )
            )^2
      end
   end

   # TODO - Add source term contribution too

   return nothing
end

# This does ALL the prolongation at once.
# The previous prolong2interfaces! function can be ignored
function prolong2interfaces_lw_parabolic!(cache, cache_parabolic, u,
   mesh::TreeMesh{2}, equations, surface_integral, dg::DG)

   @unpack interfaces, element_cache, interface_cache = cache
   @unpack U, F = cache.element_cache
   @unpack Fv = cache_parabolic
   @unpack orientations = interfaces

   @threaded for interface in eachinterface(dg, cache)
      left_element = interfaces.neighbor_ids[1, interface]
      right_element = interfaces.neighbor_ids[2, interface]

      if orientations[interface] == 1
         # interface in x-direction
         for j in eachnode(dg), v in eachvariable(equations)
            interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j, left_element]
            interfaces.u[2, v, j, interface] = u[v, 1, j, right_element]
            interface_cache.U[1, v, j, interface] = U[v, nnodes(dg), j, left_element]
            interface_cache.U[2, v, j, interface] = U[v, 1, j, right_element]

            # Fluxes
            interface_cache.f[1, v, j, interface] = F[v, 1, nnodes(dg), j, left_element]
            interface_cache.f[2, v, j, interface] = F[v, 1, 1, j, right_element]

            cache_parabolic.Fb[1, v, j, interface] = Fv[v, 1, nnodes(dg), j, left_element]
            cache_parabolic.Fb[2, v, j, interface] = Fv[v, 1, 1, j, right_element]
         end
      else # if orientations[interface] == 2
         # interface in y-direction
         for i in eachnode(dg), v in eachvariable(equations)
            interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), left_element]
            interfaces.u[2, v, i, interface] = u[v, i, 1, right_element]

            interface_cache.U[1, v, i, interface] = U[v, i, nnodes(dg), left_element]
            interface_cache.U[2, v, i, interface] = U[v, i, 1, right_element]

            # Fluxes
            interface_cache.f[1, v, i, interface] = F[v, 2, i, nnodes(dg), left_element]
            interface_cache.f[2, v, i, interface] = F[v, 2, i, 1, right_element]

            cache_parabolic.Fb[1, v, i, interface] = Fv[v, 2, i, nnodes(dg), left_element]
            cache_parabolic.Fb[2, v, i, interface] = Fv[v, 2, i, 1, right_element]
         end
      end
   end

   return nothing
end

function calc_interface_flux_hyperbolic_parabolic!(surface_flux_values, mesh::TreeMesh{2},
   # nonconservative_terms::Val{false},
   equations,
   equations_parabolic,
   surface_integral, dg::DG, cache, cache_parabolic)
   @unpack interface_cache = cache
   @unpack surface_flux = surface_integral
   @unpack u, neighbor_ids, orientations = cache.interfaces

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
         U_ll, U_rr = get_surface_node_vars(interface_cache.U, equations, dg, i, interface)
         u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
         f_ll, f_rr = get_surface_node_vars(interface_cache.f, equations, dg, i, interface)
         flux_hyperbolic = surface_flux(f_ll, f_rr, U_ll, U_rr, u_ll, u_rr, orientations[interface], equations)
         f_visc_ll, f_visc_rr = get_surface_node_vars(cache_parabolic.Fb, equations, dg, i, interface)
         flux_parabolic = 0.5 * (f_visc_ll + f_visc_rr)
         # Copy flux to left and right element storage
         for v in eachvariable(equations)
            surface_flux_values[v, i, left_direction, left_id] = flux_hyperbolic[v] - flux_parabolic[v]
            surface_flux_values[v, i, right_direction, right_id] = flux_hyperbolic[v] - flux_parabolic[v]
         end
      end
   end

   return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
function prolong2boundaries_visc_lw!(cache_parabolic, flux_viscous,
   mesh::TreeMesh{2},
   equations_parabolic::AbstractEquationsParabolic,
   surface_integral, dg::DG, cache)
   @unpack boundaries, Fv = cache_parabolic
   @unpack orientations, neighbor_sides = boundaries
   flux_viscous_x, flux_viscous_y = flux_viscous

   @threaded for boundary in eachboundary(dg, cache_parabolic)
      element = boundaries.neighbor_ids[boundary]

      if orientations[boundary] == 1
         # boundary in x-direction
         if neighbor_sides[boundary] == 1
            # element in -x direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
               # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
               # boundaries.u[1, v, l, boundary] = flux_viscous_x[v, nnodes(dg), l, element]
               boundaries.u[1, v, l, boundary] = Fv[v, 1, nnodes(dg), l, element]
            end
         else # Element in +x direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
               # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
               # boundaries.u[2, v, l, boundary] = flux_viscous_x[v, 1,          l, element]
               boundaries.u[2, v, l, boundary] = Fv[v, 1, 1, l, element]
            end
         end
      else # if orientations[boundary] == 2
         # boundary in y-direction
         if neighbor_sides[boundary] == 1
            # element in -y direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
               # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
               # boundaries.u[1, v, l, boundary] = flux_viscous_y[v, l, nnodes(dg), element]
               boundaries.u[1, v, l, boundary] = Fv[v, 2, l, nnodes(dg), element]
            end
         else
            # element in +y direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
               # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
               # boundaries.u[2, v, l, boundary] = flux_viscous_y[v, l, 1,          element]
               boundaries.u[2, v, l, boundary] = Fv[v, 2, l, 1, element]
            end
         end
      end
   end

   return nothing
end

function calc_boundary_flux_divergence_lw!(
   cache_parabolic, cache_hyperbolic, t, boundary_conditions_parabolic::BoundaryConditionPeriodic, mesh::TreeMesh{2},
   equations_parabolic::AbstractEquationsParabolic, surface_integral, dg::DG, scaling_factor = 1)
   return nothing
end

function calc_boundary_flux_divergence_lw!(cache, cache_hyperbolic, t, boundary_conditions_parabolic::NamedTuple,
   mesh::TreeMesh{2}, equations_parabolic::AbstractEquationsParabolic,
   surface_integral, dg::DG, scaling_factor = 1)
   @unpack surface_flux_values = cache_hyperbolic.elements
   @unpack n_boundaries_per_direction = cache.boundaries

   # Calculate indices
   lasts = accumulate(+, n_boundaries_per_direction)
   firsts = lasts - n_boundaries_per_direction .+ 1

   # Calc boundary fluxes in each direction
   calc_boundary_flux_by_direction_divergence_lw!(surface_flux_values, t, boundary_conditions_parabolic[1],
      equations_parabolic, surface_integral, dg, cache,
      1, firsts[1], lasts[1], scaling_factor)
   calc_boundary_flux_by_direction_divergence_lw!(surface_flux_values, t,
      boundary_conditions_parabolic[2],
      equations_parabolic, surface_integral, dg, cache,
      2, firsts[2], lasts[2], scaling_factor)
   calc_boundary_flux_by_direction_divergence_lw!(surface_flux_values, t,
      boundary_conditions_parabolic[3],
      equations_parabolic, surface_integral, dg, cache,
      3, firsts[3], lasts[3], scaling_factor)
   calc_boundary_flux_by_direction_divergence_lw!(surface_flux_values, t, boundary_conditions_parabolic[4],
      equations_parabolic, surface_integral, dg, cache,
      4, firsts[4], lasts[4], scaling_factor)
end

function calc_boundary_flux_by_direction_divergence_lw!(surface_flux_values::AbstractArray{<:Any,4}, t,
   boundary_condition,
   equations_parabolic::AbstractEquationsParabolic,
   surface_integral, dg::DG, cache,
   direction, first_boundary, last_boundary, scaling_factor)
   @unpack surface_flux = surface_integral

   # Note: cache.boundaries.u contains the unsigned normal component (using "orientation", not "direction")
   # of the viscous flux, as computed in `prolong2boundaries!`
   @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

   @threaded for boundary in first_boundary:last_boundary
      # Get neighboring element
      neighbor = neighbor_ids[boundary]

      for i in eachnode(dg)
         # Get viscous boundary fluxes
         flux_ll, flux_rr = get_surface_node_vars(u, equations_parabolic, dg, i, boundary)
         if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
            flux_inner = flux_ll
         else # Element is on the right, boundary on the left
            flux_inner = flux_rr
         end

         x = get_node_coords(node_coordinates, equations_parabolic, dg, i, boundary)

         # TODO: add a field in `cache.boundaries` for gradient information.
         # Here, we pass in `u_inner = nothing` since we overwrite cache.boundaries.u with gradient information.
         # This currently works with Dirichlet/Neuman boundary conditions for LaplaceDiffusion2D and
         # NoSlipWall/Adiabatic boundary conditions for CompressibleNavierStokesDiffusion2D as of 2022-6-27.
         # It will not work with implementations which utilize `u_inner` to impose boundary conditions.
         flux = boundary_condition(flux_inner, nothing, get_unsigned_normal_vector_2d(direction),
            x, t, Divergence(), equations_parabolic, get_time_discretization(dg), scaling_factor)

         # Copy flux to left and right element storage
         for v in eachvariable(equations_parabolic)
            surface_flux_values[v, i, direction, neighbor] -= flux[v]
         end
      end
   end

   return nothing
end

end # muladd macro