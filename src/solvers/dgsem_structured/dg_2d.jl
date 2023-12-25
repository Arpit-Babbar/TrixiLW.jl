# import Trixi: create_cache, calc_volume_integral!, calc_interface_flux!
using Trixi: TreeMesh, P4estMesh, BoundaryConditionPeriodic,
   prolong2mortars!, calc_mortar_flux!,
   calc_surface_integral!, apply_jacobian!, reset_du!,
   max_dt,
   StructuredMesh, UnstructuredMesh2D,
   DG, DGSEM, nnodes, nelements, False,
   get_node_vars, set_node_vars!

import Trixi: calc_interface_flux!, calc_boundary_flux!
using MuladdMacro
using LoopVectorization: @turbo
using Trixi: @threaded

@muladd begin

function rhs!(du, u, t, dt,
   mesh::StructuredMesh{2}, equations,
   initial_condition, boundary_conditions, source_terms,
   dg::DG, time_discretization::AbstractLWTimeDiscretization, cache, tolerances::NamedTuple)

   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

   # Calculate volume integral
   alpha = @trixi_timeit timer() "volume integral" calc_volume_integral!(
      du, u,
      t, dt, tolerances, mesh,
      have_nonconservative_terms(equations), source_terms, equations,
      dg.volume_integral, time_discretization,
      dg, cache)

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

@inline function calc_fn_low_kernel!(
   du, u,
   mesh::Union{TreeMesh{2},StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms, equations,
   volume_flux_fv, dg::DGSEM, cache, element, alpha=true)

   @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache
   @unpack inverse_weights = dg.basis

   # Calculate FV two-point fluxes
   fstar1_L = fstar1_L_threaded[Threads.threadid()]
   fstar2_L = fstar2_L_threaded[Threads.threadid()]
   fstar1_R = fstar1_R_threaded[Threads.threadid()]
   fstar2_R = fstar2_R_threaded[Threads.threadid()]
   calc_fn_low!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
      nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

   return nothing
end

function finite_differences(h1, h2, ul, u, ur)
   back_diff = (u - ul) / h1
   fwd_diff = (ur - u) / h2
   a, b, c = -(h2 / (h1 * (h1 + h2))), (h2 - h1) / (h1 * h2), (h1 / (h2 * (h1 + h2)))
   cent_diff = a * ul + b * u + c * ur
   return back_diff, cent_diff, fwd_diff
end

function minmod(a, b, c, beta, Mdx2=1e-10)
   if abs(b) < Mdx2
      return b
   end
   slope = min(beta * abs(a), abs(b), beta * abs(c))
   s1, s2, s3 = sign(a), sign(b), sign(c)
   if (s1 != s2) || (s2 != s3)
      return zero(a)
   else
      slope = s1 * slope
      return slope
   end
end

function minmod(a, b, Mdx2=1e-10)
   if abs(b) < Mdx2 && abs(a) < Mdx2
      return b
   end
   slope = min(abs(a), abs(b))
   s1, s2 = sign(a), sign(b)
   if (s1 != s2)
      return zero(a)
   else
      slope = s1 * slope
      return slope
   end
end

@inline function calc_fn_low!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u,
   mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms::False, equations,
   volume_flux_fv, dg::DGSEM, element, cache)
   @unpack contravariant_vectors = cache.elements
   @unpack fn_low = cache.element_cache
   @unpack weights, derivative_matrix = dg.basis

   # Performance improvement if the metric terms of the subcell FV method are only computed
   # once at the beginning of the simulation, instead of at every Runge-Kutta stage
   fstar1_L[:, 1, :] .= zero(eltype(fstar1_L))
   fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
   fstar1_R[:, 1, :] .= zero(eltype(fstar1_R))
   fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

   for j in eachnode(dg)
      normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j, element)

      for i in (2, nnodes(dg))
         u_ll = Trixi.get_node_vars(u, equations, dg, i - 1, j, element)
         u_rr = Trixi.get_node_vars(u, equations, dg, i, j, element)

         for m in 1:nnodes(dg)
            normal_direction += weights[i-1] * derivative_matrix[i-1, m] * get_contravariant_vector(1, contravariant_vectors, m, j, element)
         end

         # Compute the contravariant flux
         contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

         Trixi.set_node_vars!(fstar1_L, contravariant_flux, equations, dg, i, j)
         Trixi.set_node_vars!(fstar1_R, contravariant_flux, equations, dg, i, j)
      end
   end

   fstar2_L[:, :, 1] .= zero(eltype(fstar2_L))
   fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
   fstar2_R[:, :, 1] .= zero(eltype(fstar2_R))
   fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

   for i in eachnode(dg)
      normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1, element)

      for j in (2, nnodes(dg))
         u_ll = Trixi.get_node_vars(u, equations, dg, i, j - 1, element)
         u_rr = Trixi.get_node_vars(u, equations, dg, i, j, element)

         for m in 1:nnodes(dg)
            normal_direction += weights[j-1] * derivative_matrix[j-1, m] * get_contravariant_vector(2, contravariant_vectors, i, m, element)
         end

         # Compute the contravariant flux by taking the scalar product of the
         # normal vector and the flux vector
         contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

         Trixi.set_node_vars!(fstar2_L, contravariant_flux, equations, dg, i, j)
         Trixi.set_node_vars!(fstar2_R, contravariant_flux, equations, dg, i, j)
      end
   end

   load_fn_low!(fn_low, mesh, dg, nonconservative_terms,
      fstar1_L, fstar2_L, fstar1_R, fstar2_R, element)

   return nothing
end

function calcflux_muscl!(fstar1_L, fstar1_R, fstar2_L, fstar2_R,
   uf, uext, alpha, u, Δt,
   mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms::False, equations,
   volume_flux_fv, dg::DGSEM, element, cache)

   # TODO - Create a boundary neighbour element and re-use it
   for boundary in eachboundary(dg, cache)
      nbrs = cache.boundaries.neighbor_ids[boundary]
      if element in nbrs
         # If element neighbours boundary, reduce to first order
         calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
            nonconservative_terms, equations, volume_flux_fv, dg, element, cache)
         return nothing
      end
   end

   @unpack x_subfaces, y_subfaces, ξ_extended = cache
   @unpack contravariant_vectors, inverse_jacobian = cache.elements
   @unpack weights, derivative_matrix = dg.basis
   nvar = nvariables(equations)

   nd = nnodes(dg)
   @turbo @views begin
      uext[:, 1:nd, 1:nd] .= u[:, :, :, element]
      uext[:, 0, 1:nd] .= uext[:, 1, 1:nd]
      uext[:, nd+1, 1:nd] .= uext[:, nd, 1:nd]
      uext[:, 1:nd, 0] .= uext[:, 1:nd, 1]
      uext[:, 1:nd, nd+1] .= uext[:, 1:nd, nd]
   end

   # Loop over subcells to extrapolate to faces
   for j in eachnode(dg), i in eachnode(dg)
      u_ = Trixi.get_node_vars(uext, equations, dg, i, j)
      ul = Trixi.get_node_vars(uext, equations, dg, i - 1, j)
      ur = Trixi.get_node_vars(uext, equations, dg, i + 1, j)
      ud = Trixi.get_node_vars(uext, equations, dg, i, j - 1)
      uu = Trixi.get_node_vars(uext, equations, dg, i, j + 1)

      Δx1, Δx2 = ξ_extended[i] - ξ_extended[i-1], ξ_extended[i+1] - ξ_extended[i]
      Δy1, Δy2 = ξ_extended[j] - ξ_extended[j-1], ξ_extended[j+1] - ξ_extended[j]

      back_x, cent_x, fwd_x = finite_differences(Δx1, Δx2, ul, u_, ur)
      back_y, cent_y, fwd_y = finite_differences(Δy1, Δy2, ud, u_, uu)

      beta1, beta2 = 2.0 - alpha, 2.0 - alpha # Unfortunate way to fix type instability

      # slope_tuple_x = (minmod(back_x[n], cent_x[n], fwd_x[n], beta1, 0.0)
      #                  for n in eachvariable(equations))
      slope_tuple_x = (minmod(back_x[n], fwd_x[n], 0.0) for n in eachvariable(equations))
      slope_x = SVector{nvar}(slope_tuple_x)

      # slope_tuple_y = (minmod(back_y[n], cent_y[n], fwd_y[n], beta2, 0.0)
      #                  for n in eachvariable(equations))
      slope_tuple_y = (minmod(back_y[n], fwd_y[n], 0.0) for n in eachvariable(equations))
      slope_y = SVector{nvar}(slope_tuple_y)

      ufl = u_ + slope_x * (x_subfaces[i-1] - ξ_extended[i]) # left face value u_{i-1/2,j}
      ufr = u_ + slope_x * (x_subfaces[i] - ξ_extended[i]) # right face value u_{i+1/2,j}

      ufd = u_ + slope_y * (y_subfaces[j-1] - ξ_extended[j]) # lower face value u_{i, j-1/2}
      ufu = u_ + slope_y * (y_subfaces[j] - ξ_extended[j]) # upper face value u_{i, j+1/2}

      u_star_ll = u_ + 2.0 * slope_x * (x_subfaces[i-1] - ξ_extended[i]) # left face value u_{i-1/2,j}
      u_star_rr = u_ + 2.0 * slope_x * (x_subfaces[i] - ξ_extended[i]) # right face value u_{i+1/2,j}

      u_star_d = u_ + 2.0 * slope_y * (y_subfaces[j-1] - ξ_extended[j]) # lower face value u_{i,j-1/2}
      u_star_u = u_ + 2.0 * slope_y * (y_subfaces[j] - ξ_extended[j]) # upper face value u_{i,j+1/2}

      ufl, ufr = limit_slope(equations, slope_x, ufl, u_star_ll, ufr, u_star_rr, u_,
         x_subfaces[i-1] - ξ_extended[i], x_subfaces[i] - ξ_extended[i])

      ufd, ufu = limit_slope(equations, slope_y, ufd, u_star_d, ufu, u_star_u, u_,
         y_subfaces[j-1] - ξ_extended[j], y_subfaces[j] - ξ_extended[j])

      @turbo @views begin
         uf[:, 1, i, j] .= ufl
         uf[:, 2, i, j] .= ufr
         uf[:, 3, i, j] .= ufd
         uf[:, 4, i, j] .= ufu
      end
   end

   # Performance improvement if the metric terms of the subcell FV method are only computed
   # once at the beginning of the simulation, instead of at every Runge-Kutta stage
   @turbo fstar1_L[:, 1, :] .= zero(eltype(fstar1_L))
   @turbo fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
   @turbo fstar1_R[:, 1, :] .= zero(eltype(fstar1_R))
   @turbo fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

   for j in eachnode(dg)
      normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j, element)

      for i in 2:nnodes(dg)
         u_ll = Trixi.get_node_vars(uf, equations, dg, 2, i - 1, j)
         u_rr = Trixi.get_node_vars(uf, equations, dg, 1, i, j)

         for m in 1:nnodes(dg)
            normal_direction += weights[i-1] * derivative_matrix[i-1, m] * get_contravariant_vector(1, contravariant_vectors, m, j, element)
         end

         # Compute the contravariant flux
         contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

         Trixi.set_node_vars!(fstar1_L, contravariant_flux, equations, dg, i, j)
         Trixi.set_node_vars!(fstar1_R, contravariant_flux, equations, dg, i, j)
      end
   end

   @turbo fstar2_L[:, :, 1] .= zero(eltype(fstar2_L))
   @turbo fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
   @turbo fstar2_R[:, :, 1] .= zero(eltype(fstar2_R))
   @turbo fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

   for i in eachnode(dg)
      normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1, element)

      for j in 2:nnodes(dg)
         u_ll = Trixi.get_node_vars(uf, equations, dg, 4, i, j - 1)
         u_rr = Trixi.get_node_vars(uf, equations, dg, 3, i, j)

         for m in 1:nnodes(dg)
            normal_direction += weights[j-1] * derivative_matrix[j-1, m] * get_contravariant_vector(2, contravariant_vectors, i, m, element)
         end

         # Compute the contravariant flux by taking the scalar product of the
         # normal vector and the flux vector
         contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

         Trixi.set_node_vars!(fstar2_L, contravariant_flux, equations, dg, i, j)
         Trixi.set_node_vars!(fstar2_R, contravariant_flux, equations, dg, i, j)
      end
   end

   return nothing
end

function calcflux_mh!(fstar1_L, fstar1_R, fstar2_L, fstar2_R,
   unph, uf, uext, alpha, u, Δt,
   mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms::False, equations,
   volume_flux_fv, dg::DGSEM, element, cache)

   # TODO - Create a boundary neighbour element and re-use it
   for boundary in eachboundary(dg, cache)
      if element in cache.boundaries.neighbor_ids[boundary]
         # If element neighbours boundary, reduce to first order
         calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
            nonconservative_terms, equations, volume_flux_fv, dg, element, cache)
         return nothing
      end
   end

   @unpack x_subfaces, y_subfaces, ξ_extended = cache
   @unpack contravariant_vectors, inverse_jacobian = cache.elements
   @unpack weights, derivative_matrix = dg.basis
   nvar = nvariables(equations)

   unph .= zero(eltype(unph))
   nd = nnodes(dg)

   nd = nnodes(dg)
   @turbo @views begin
      uext[:, 1:nd, 1:nd] .= u[:, :, :, element]
      uext[:, 0, 1:nd] .= uext[:, 1, 1:nd]
      uext[:, nd+1, 1:nd] .= uext[:, nd, 1:nd]
      uext[:, 1:nd, 0] .= uext[:, 1:nd, 1]
      uext[:, 1:nd, nd+1] .= uext[:, 1:nd, nd]
   end

   # Loop over subcells to extrapolate to faces
   for j in eachnode(dg), i in eachnode(dg)
      u_ = Trixi.get_node_vars(uext, equations, dg, i, j)
      ul = Trixi.get_node_vars(uext, equations, dg, i - 1, j)
      ur = Trixi.get_node_vars(uext, equations, dg, i + 1, j)
      ud = Trixi.get_node_vars(uext, equations, dg, i, j - 1)
      uu = Trixi.get_node_vars(uext, equations, dg, i, j + 1)

      Δx1, Δx2 = ξ_extended[i] - ξ_extended[i-1], ξ_extended[i+1] - ξ_extended[i]
      Δy1, Δy2 = ξ_extended[j] - ξ_extended[j-1], ξ_extended[j+1] - ξ_extended[j]

      back_x, cent_x, fwd_x = finite_differences(Δx1, Δx2, ul, u_, ur)
      back_y, cent_y, fwd_y = finite_differences(Δy1, Δy2, ud, u_, uu)

      beta1, beta2 = 2.0 - alpha, 2.0 - alpha # Unfortunate way to fix type instability

      slope_tuple_x = (minmod(back_x[n], cent_x[n], fwd_x[n], beta1, 0.0)
                        for n in eachvariable(equations))
      slope_x = SVector{nvar}(slope_tuple_x)

      slope_tuple_y = (minmod(back_y[n], cent_y[n], fwd_y[n], beta2, 0.0)
                        for n in eachvariable(equations))
      slope_y = SVector{nvar}(slope_tuple_y)

      ufl = u_ + slope_x * (x_subfaces[i-1] - ξ_extended[i]) # left face value u_{i-1/2,j}
      ufr = u_ + slope_x * (x_subfaces[i] - ξ_extended[i]) # right face value u_{i+1/2,j}

      ufd = u_ + slope_y * (y_subfaces[j-1] - ξ_extended[j]) # lower face value u_{i, j-1/2}
      ufu = u_ + slope_y * (y_subfaces[j] - ξ_extended[j]) # upper face value u_{i, j+1/2}

      u_star_ll = u_ + 2.0 * slope_x * (x_subfaces[i-1] - ξ_extended[i]) # left face value u_{i-1/2,j}
      u_star_rr = u_ + 2.0 * slope_x * (x_subfaces[i] - ξ_extended[i]) # right face value u_{i+1/2,j}

      u_star_d = u_ + 2.0 * slope_y * (y_subfaces[j-1] - ξ_extended[j]) # lower face value u_{i,j-1/2}
      u_star_u = u_ + 2.0 * slope_y * (y_subfaces[j] - ξ_extended[j]) # upper face value u_{i,j+1/2}

      ufl, ufr = limit_slope(equations, slope_x, ufl, u_star_ll, ufr, u_star_rr, u_,
         x_subfaces[i-1] - ξ_extended[i], x_subfaces[i] - ξ_extended[i])

      ufd, ufu = limit_slope(equations, slope_y, ufd, u_star_d, ufu, u_star_u, u_,
         y_subfaces[j-1] - ξ_extended[j], y_subfaces[j] - ξ_extended[j])

      @turbo @views begin
         unph[:, 1, i, j] .= uf[:, 1, i, j] .= ufl
         unph[:, 2, i, j] .= uf[:, 2, i, j] .= ufr
         unph[:, 3, i, j] .= uf[:, 3, i, j] .= ufd
         unph[:, 4, i, j] .= uf[:, 4, i, j] .= ufu
      end
   end

   for j in eachnode(dg)
      normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j, element)

      for i in 2:nnodes(dg)
         # u_ll = get_node_vars(u, equations, dg, i-1, j, element)
         # u_rr = get_node_vars(u, equations, dg, i,   j, element)

         for m in 1:nnodes(dg)
            normal_direction += weights[i-1] * derivative_matrix[i-1, m] * get_contravariant_vector(1, contravariant_vectors, m, j, element)
         end

         Δxi = x_subfaces[i] - x_subfaces[i-1]

         u_ll = get_node_vars(uf, equations, dg, 2, i - 1, j)
         f_ll = flux(u_ll, normal_direction, equations)
         res_ll = -0.5 * Δt * inverse_jacobian[i-1, j, element] / Δxi * f_ll
         for n in eachvariable(equations) # Loop somehow needed to avoid allocations
            unph[n, 1, i-1, j] += res_ll[n]
            unph[n, 2, i-1, j] += res_ll[n]

            # Multi-D
            unph[n, 3, i-1, j] += res_ll[n]
            unph[n, 4, i-1, j] += res_ll[n]
         end

         if i < nnodes(dg) # TODO - Find a better fix!
            # (1, i, j) means left trace value at (i,j) cell.
            # For i = nnodes(dg), there is no left trace value at (i,j) cell
            # so quantities here are not needed in that case
            u_rr = get_node_vars(uf, equations, dg, 1, i, j)
            f_rr = flux(u_rr, normal_direction, equations)
            Δxip1 = x_subfaces[i+1] - x_subfaces[i]
            res_rr = 0.5 * Δt * inverse_jacobian[i, j, element] / Δxip1 * f_rr
            for n in eachvariable(equations) # Loop somehow needed to avoid allocations
               unph[n, 1, i, j] += res_rr[n]
               unph[n, 2, i, j] += res_rr[n]

               # Multi-D
               unph[n, 3, i, j] += res_rr[n]
               unph[n, 4, i, j] += res_rr[n]
            end
         end
      end
   end

   for i in eachnode(dg)
      normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1, element)

      for j in 2:nnodes(dg)
         for m in 1:nnodes(dg)
            normal_direction += weights[j-1] * derivative_matrix[j-1, m] * get_contravariant_vector(2, contravariant_vectors, i, m, element)
         end

         Δyi = y_subfaces[j] - y_subfaces[j-1]

         u_dd = get_node_vars(uf, equations, dg, 4, i, j - 1)
         f_dd = flux(u_dd, normal_direction, equations)

         res_dd = -0.5 * Δt * inverse_jacobian[i, j, element] / Δyi * f_dd

         # Multi-D
         for n in eachvariable(equations) # Loop somehow needed to avoid allocations
            unph[n, 3, i, j-1] += res_dd[n]
            unph[n, 4, i, j-1] += res_dd[n]

            # Multi-D
            unph[n, 1, i, j-1] += res_dd[n]
            unph[n, 2, i, j-1] += res_dd[n]
         end
         if j < nnodes(dg)
            # (3, i, j) means bottom trace value at (i,j) cell.
            # For j = nnodes(dg), there is no bottom trace value of (i,j) cell
            # so quantities here are not needed in that case
            u_uu = get_node_vars(uf, equations, dg, 3, i, j)
            f_uu = flux(u_uu, normal_direction, equations)
            Δyip1 = y_subfaces[j+1] - y_subfaces[j]
            res_uu = 0.5 * Δt * inverse_jacobian[i, j+1, element] / Δyip1 * f_uu
            for n in eachvariable(equations) # Loop somehow needed to avoid allocations
               unph[n, 3, i, j] += res_uu[n]
               unph[n, 4, i, j] += res_uu[n]

               # Multi-D
               unph[n, 1, i, j] += res_uu[n]
               unph[n, 2, i, j] += res_uu[n]
            end
         end
      end
   end

   # Performance improvement if the metric terms of the subcell FV method are only computed
   # once at the beginning of the simulation, instead of at every Runge-Kutta stage
   fstar1_L[:, 1, :] .= zero(eltype(fstar1_L))
   fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
   fstar1_R[:, 1, :] .= zero(eltype(fstar1_R))
   fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

   for j in eachnode(dg)
      normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j, element)

      for i in 2:nnodes(dg)
         u_ll = get_node_vars(unph, equations, dg, 2, i - 1, j)
         u_rr = get_node_vars(unph, equations, dg, 1, i, j)

         for m in 1:nnodes(dg)
            normal_direction += weights[i-1] * derivative_matrix[i-1, m] * get_contravariant_vector(1, contravariant_vectors, m, j, element)
         end

         # Compute the contravariant flux
         contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

         set_node_vars!(fstar1_L, contravariant_flux, equations, dg, i, j)
         set_node_vars!(fstar1_R, contravariant_flux, equations, dg, i, j)
      end
   end

   fstar2_L[:, :, 1] .= zero(eltype(fstar2_L))
   fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
   fstar2_R[:, :, 1] .= zero(eltype(fstar2_R))
   fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

   for i in eachnode(dg)
      normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1, element)

      for j in 2:nnodes(dg)

         u_ll = get_node_vars(unph, equations, dg, 4, i, j - 1)
         u_rr = get_node_vars(unph, equations, dg, 3, i, j)

         for m in 1:nnodes(dg)
            normal_direction += weights[j-1] * derivative_matrix[j-1, m] * get_contravariant_vector(2, contravariant_vectors, i, m, element)
         end

         # Compute the contravariant flux by taking the scalar product of the
         # normal vector and the flux vector
         contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

         set_node_vars!(fstar2_L, contravariant_flux, equations, dg, i, j)
         set_node_vars!(fstar2_R, contravariant_flux, equations, dg, i, j)
      end
   end

   return nothing
end


function contravariant_flux(u, i, j, element, contravariant_vectors, eq::AbstractEquations{2})
   Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
   Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
   flux1, flux2 = Trixi.flux(u, 1, eq), Trixi.flux(u, 2, eq)
   contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
   contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
   return flux1, flux2, contravariant_flux1, contravariant_flux2
end

function contravariant_flux(u, Ja, eq::AbstractEquations{2})
   (Ja11, Ja12), (Ja21, Ja22) = Ja
   flux1, flux2 = Trixi.flux(u, 1, eq), Trixi.flux(u, 2, eq)
   contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
   contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
   return flux1, flux2, contravariant_flux1, contravariant_flux2
end

@inline function compute_temporal_errors!(cache, cell_arrays, tolerances, dt, equations, dg,
                                          element)
   @unpack u_np1, u_np1_low = cell_arrays
   @unpack temporal_errors = cache
   @unpack abstol, reltol = tolerances
   temporal_errors[element] = zero(dt)
   for j in eachnode(dg), i in eachnode(dg)
      u_np1_node = Trixi.get_node_vars(u_np1, equations, dg, i, j)
      u_np1_low_node = Trixi.get_node_vars(u_np1_low, equations, dg, i, j)
      # u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      for v in eachvariable(equations)
         temporal_errors[element] += (
            (u_np1_node[v] - u_np1_low_node[v])
            /
            (abstol + reltol * max(abs(u_np1_node[v]), abs(u_np1_low_node[v])))
         )^2
      end
   end
end

@inline function compute_f_s_ut!(cell_arrays, t, dt, u, source_terms, equations, dg,
   cache, element)
   @unpack derivative_matrix = dg.basis
   @unpack lw_res_cache, element_cache = cache
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack f, g, ftilde, gtilde, Ftilde, Gtilde, ut, S = cell_arrays
   for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      flux1, flux2, cv_flux1, cv_flux2 = contravariant_flux(u_node, Ja, equations)

      Trixi.set_node_vars!(element_cache.F, flux1, equations, dg, 1, i, j, element)
      Trixi.set_node_vars!(element_cache.F, flux2, equations, dg, 2, i, j, element)
      Trixi.set_node_vars!(f, flux1, equations, dg, i, j)
      Trixi.set_node_vars!(g, flux2, equations, dg, i, j)

      Trixi.set_node_vars!(Ftilde, cv_flux1, equations, dg, i, j)
      Trixi.set_node_vars!(ftilde, cv_flux1, equations, dg, i, j)
      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1, equations, dg, ii, j)
      end

      Trixi.set_node_vars!(Gtilde, cv_flux2, equations, dg, i, j)
      Trixi.set_node_vars!(gtilde, cv_flux2, equations, dg, i, j)
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end
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
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      s_node = calc_source(u_node, x, t, source_terms, equations, dg, cache)
      Trixi.set_node_vars!(S, s_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end
end

@inline function compute_ft_st_utt!(cell_arrays, t, dt, u, source_terms, equations, dg,
   cache, element)
   @unpack derivative_matrix = dg.basis
   @unpack lw_res_cache, element_cache = cache
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack f, g, ftilde, gtilde, Ftilde, Gtilde, ut, utt, uttt, utttt, U,
   up, um, upp, umm, S, u_np1, u_np1_low = cell_arrays
   for j in eachnode(dg), i in eachnode(dg)
      ut_node = Trixi.get_node_vars(ut, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, -2.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0, ut_node, equations, dg, i, j)

      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)

      f_t = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
      g_t = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)
      ftilde_t = 1.0 / 12.0 * (-cv_fpp + 8.0 * cv_fp - 8.0 * cv_fm + cv_fmm)
      gtilde_t = 1.0 / 12.0 * (-cv_gpp + 8.0 * cv_gp - 8.0 * cv_gm + cv_gmm)

      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, f_t, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, g_t, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 0.5, ftilde_t, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Gtilde, 0.5, gtilde_t, equations, dg, i, j)
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], ftilde_t, equations, dg, ii, j)
      end
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], gtilde_t, equations, dg, i, jj)
      end
   end

   # Apply Jacobian to utt
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
end

@inline function compute_ftt_stt_uttt!(cell_arrays, t, dt, u, source_terms, equations, dg,
   cache, element)
   @unpack derivative_matrix = dg.basis
   @unpack lw_res_cache, element_cache = cache
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack f, g, ftilde, gtilde, Ftilde, Gtilde, ut, utt, uttt, utttt, U,
   up, um, upp, umm, S, u_np1, u_np1_low = cell_arrays

   for j in eachnode(dg), i in eachnode(dg)
      utt_node = Trixi.get_node_vars(utt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, 2.0, utt_node, equations, dg, i, j)

      f_node, g_node = Trixi.get_node_vars(f, equations, dg, i, j), Trixi.get_node_vars(g, equations, dg, i, j)
      ftilde_node = Trixi.get_node_vars(ftilde, equations, dg, i, j)
      gtilde_node = Trixi.get_node_vars(gtilde, equations, dg, i, j)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)
      f_tt = (1.0 / 12.0) * (-fpp + 16.0 * fp - 30.0 * f_node + 16.0 * fm - fmm)
      g_tt = (1.0 / 12.0) * (-gpp + 16.0 * gp - 30.0 * g_node + 16.0 * gm - gmm)
      ftilde_tt = (1.0 / 12.0) * (-cv_fpp + 16.0 * cv_fp - 30.0 * ftilde_node + 16.0 * cv_fm - cv_fmm)
      gtilde_tt = (1.0 / 12.0) * (-cv_gpp + 16.0 * cv_gp - 30.0 * gtilde_node + 16.0 * cv_gm - cv_gmm)

      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 6.0, ftilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, f_tt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 6.0, gtilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, g_tt, equations, dg, 2, i, j,
                                       element)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], ftilde_tt,
                                          equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], gtilde_tt,
                                          equations, dg, i, jj)
      end
   end

   # Apply Jacobian to uttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         uttt[v, i, j] *= inv_jacobian
      end
   end

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
end

@inline function compute_fttt_sttt_utttt!(cell_arrays, t, dt, u, source_terms, equations, dg,
   cache, element)
   @unpack derivative_matrix = dg.basis
   @unpack lw_res_cache, element_cache = cache
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack f, g, ftilde, gtilde, Ftilde, Gtilde, ut, utt, uttt, utttt, U,
   up, um, upp, umm, S, u_np1, u_np1_low = cell_arrays
   for j in eachnode(dg), i in eachnode(dg)
      uttt_node = Trixi.get_node_vars(uttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, equations, dg, i, j)

      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)
      fttt = 0.5 * (fpp - 2.0 * fp + 2.0 * fm - fmm)
      ftilde_ttt = 0.5 * (cv_fpp - 2.0 * cv_fp + 2.0 * cv_fm - cv_fmm)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, fttt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 24.0, ftilde_ttt, equations, dg, i, j)
      gttt = 0.5 * (gpp - 2.0 * gp + 2.0 * gm - gmm)
      gtilde_ttt = 0.5 * (cv_gpp - 2.0 * cv_gp + 2.0 * cv_gm - cv_gmm)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, gttt, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 24.0, gtilde_ttt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * ft for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[ii, i], fttt, equations, dg, ii, j)
      end
      for jj in eachnode(dg)
         # C += -lam*gt*Dm' for each variable
         # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[jj, j], gttt, equations, dg, i, jj)
      end
   end

   # Apply jacobian on utttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utttt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to utttt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      sttt = calc_source_ttt_N34(u_node, up_node, upp_node, um_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(utttt, dt, sttt, equations, dg, i, j) # has no jacobian factor
   end
end

@inline function compute_ftttt_stttt_du!(du, cell_arrays, t, dt, u, source_terms, equations, dg,
   cache, element, alpha)
   @unpack derivative_matrix, derivative_dhat = dg.basis
   @unpack lw_res_cache, element_cache = cache
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack f, g, ftilde, gtilde, Ftilde, Gtilde, ut, utt, uttt, utttt, U,
   up, um, upp, umm, S, u_np1, u_np1_low = cell_arrays
   for j in eachnode(dg), i in eachnode(dg)
      utttt_node = Trixi.get_node_vars(utttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, 1.0 / 24.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0 / 24.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, 2.0 / 3.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0 / 3.0, utttt_node, equations, dg, i, j)

      f_node = Trixi.get_node_vars(f, equations, dg, i, j)
      g_node = Trixi.get_node_vars(g, equations, dg, i, j)
      ftilde_node = Trixi.get_node_vars(ftilde, equations, dg, i, j)
      gtilde_node = Trixi.get_node_vars(gtilde, equations, dg, i, j)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)

      ftttt = 0.5 * (fpp - 4.0 * fp + 6.0 * f_node - 4.0 * fm + fmm)
      gtttt = 0.5 * (gpp - 4.0 * gp + 6.0 * g_node - 4.0 * gm + gmm)
      ftilde_tttt = 0.5 * (cv_fpp - 4.0 * cv_fp + 6.0 * ftilde_node - 4.0 * cv_fm + cv_fmm)
      gtilde_tttt = 0.5 * (cv_gpp - 4.0 * cv_gp + 6.0 * gtilde_node - 4.0 * cv_gm + cv_gmm)

      # Updating u_np1_low here
      F_ = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      G_ = Trixi.get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         inv_jacobian = inverse_jacobian[ii, j, element]
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[ii, i],
            F_, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         inv_jacobian = inverse_jacobian[i, jj, element]
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[jj, j],
            G_, equations, dg, i, jj)
      end

      # TODO - Check the source term contribution
      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(u_np1_low, 1.0, S_node, equations, dg, i, j)

      # UPDATING u_np1_low ENDS!!!

      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 120.0, ftttt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 120.0, gtttt, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 120.0, ftilde_tttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 120.0, gtilde_tttt, equations, dg, i, j)

      Ftilde_node = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      Gtilde_node = Trixi.get_node_vars(Gtilde, equations, dg, i, j)
      for ii in eachnode(dg)
         inv_jacobian = inverse_jacobian[ii, j, element]
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], Ftilde_node,
            equations, dg, ii, j, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[ii, i],
            Ftilde_node, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         inv_jacobian = inverse_jacobian[i, jj, element]
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], Gtilde_node,
            equations, dg, i, jj, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[jj, j],
            Gtilde_node, equations, dg, i, jj)
      end

      # TODO - Add source term contribution to u_np1, u_np1_low too

      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stttt = calc_source_tttt_N4(u_node, up_node, um_node, upp_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 120.0, stttt, equations, dg, i, j)

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = Trixi.get_node_vars(U, equations, dg, i, j)

      # Ub = UT * V
      # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
      Trixi.set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      Trixi.multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
         i, j, element)
      Trixi.multiply_add_to_node_vars!(u_np1, 1.0, S_node, equations, dg, i, j)
   end
end

function lw_volume_kernel_1!(du, u, t, dt, tolerances,
   element, mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms::False, source_terms, equations,
   dg::DGSEM, cache, alpha=true)
   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache
   @unpack elements = cache # To access cache.U and cache.F
   refresh!(arr) = fill!(arr, zero(eltype(u)))
   id = Threads.threadid()
   ftilde, gtilde, Ftilde, Gtilde, ut, U, up, um,
   ftildet, gtildet, S, u_np1, u_np1_low = cell_arrays[id]
   refresh!.((ut, ftildet, gtildet))
   for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      flux1, flux2, cv_flux1, cv_flux2 = contravariant_flux(u_node, Ja, equations)

      set_node_vars!(element_cache.F, flux1, equations, dg, 1, i, j, element)
      set_node_vars!(element_cache.F, flux2, equations, dg, 2, i, j, element)

      set_node_vars!(Ftilde, cv_flux1, equations, dg, i, j)
      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1,
            equations, dg, ii, j)
      end

      set_node_vars!(Gtilde, cv_flux2, equations, dg, i, j)
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2,
            equations, dg, i, jj)
      end

      Trixi.set_node_vars!(u_np1, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

      Trixi.set_node_vars!(um, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(up, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(U, u_node, equations, dg, i, j)
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
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      s_node = calc_source(u_node, x, t, source_terms, equations, dg, cache)
      Trixi.set_node_vars!(S, s_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = Trixi.get_node_vars(ut, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)

      ft = 0.5 * (fp - fm)
      gt = 0.5 * (gp - gm)

      # Updating u_np1_low here
      F_ = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      G_ = Trixi.get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         inv_jacobian = inverse_jacobian[ii, j, element]
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[ii, i],
            F_, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         inv_jacobian = inverse_jacobian[i, jj, element]
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[jj, j],
            G_, equations, dg, i, jj)
      end

      # TODO - Check the source term contribution
      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(u_np1_low, 1.0, S_node, equations, dg, i, j)

      # UPDATING u_np1_low ENDS!!!

      multiply_add_to_node_vars!(ftildet, 0.5, cv_fp, -0.5, cv_fm, equations, i, j)
      multiply_add_to_node_vars!(gtildet, 0.5, cv_gp, -0.5, cv_gm, equations, i, j)

      multiply_add_to_node_vars!(element_cache.F, 0.5, ft, equations, 1, i, j, element)
      multiply_add_to_node_vars!(element_cache.F, 0.5, gt, equations, 2, i, j, element)
      ftildet_node = Trixi.get_node_vars(ftildet, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Ftilde,
         0.5, ftildet_node,
         equations, dg, i, j)
      gtildet_node = Trixi.get_node_vars(gtildet, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Gtilde,
         0.5, gtildet_node,
         equations, dg, i, j)
      Ftilde_node = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      Gtilde_node = Trixi.get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], Ftilde_node, equations, dg, ii, j, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[ii, i],
            Ftilde_node, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], Gtilde_node, equations, dg, i, jj, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[jj, j],
            Gtilde_node, equations, dg, i, jj)
      end

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N12(up_node, um_node, x, t, dt, source_terms, equations,
         dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = Trixi.get_node_vars(U, equations, dg, i, j)

      # Ub = UT * V
      # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
      Trixi.set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      Trixi.multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
      Trixi.multiply_add_to_node_vars!(u_np1, 1.0, S_node, equations, dg, i, j)
   end

   @unpack temporal_errors = cache
   @unpack abstol, reltol = tolerances
   temporal_errors[element] = zero(dt)
   for j in eachnode(dg), i in eachnode(dg)
      u_np1_node = Trixi.get_node_vars(u_np1, equations, dg, i, j)
      u_np1_low_node = Trixi.get_node_vars(u_np1_low, equations, dg, i, j)
      # u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      for v in eachvariable(equations)
         temporal_errors[element] += (
            (u_np1_node[v] - u_np1_low_node[v])
            /
            (abstol + reltol * max(abs(u_np1_node[v]), abs(u_np1_low_node[v])))
         )^2
         @show temporal_errors[element]
      end
   end

   return nothing
end

function lw_volume_kernel_2!(du, u, t, dt, tolerances,
   element, mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms::False, source_terms, equations,
   dg::DGSEM, cache, alpha=true)
   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache
   @unpack elements = cache # To access cache.U and cache.F
   refresh!(arr) = fill!(arr, zero(eltype(u)))
   id = Threads.threadid()
   f, g, ftilde, gtilde, Ftilde, Gtilde, ut, utt, U,
   up, um, S, u_np1, u_np1_low = cell_arrays[id]
   refresh!.((ut, utt))
   for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      flux1, flux2, cv_flux1, cv_flux2 = contravariant_flux(u_node, Ja, equations)

      Trixi.set_node_vars!(element_cache.F, flux1, equations, dg, 1, i, j, element)
      Trixi.set_node_vars!(element_cache.F, flux2, equations, dg, 2, i, j, element)
      Trixi.set_node_vars!(f, flux1, equations, dg, i, j)
      Trixi.set_node_vars!(g, flux2, equations, dg, i, j)

      Trixi.set_node_vars!(Ftilde, cv_flux1, equations, dg, i, j)
      Trixi.set_node_vars!(ftilde, cv_flux1, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1, equations, dg, ii, j)
      end
      Trixi.set_node_vars!(Gtilde, cv_flux2, equations, dg, i, j)
      Trixi.set_node_vars!(gtilde, cv_flux2, equations, dg, i, j)
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end

      Trixi.set_node_vars!(u_np1, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

      Trixi.set_node_vars!(um, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(up, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(U, u_node, equations, dg, i, j)
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
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      s_node = calc_source(u_node, x, t, source_terms, equations, dg, cache)
      Trixi.set_node_vars!(S, s_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = Trixi.get_node_vars(ut, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U,
         0.5, ut_node,
         equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)

      f_t = 0.5 * (fp - fm)
      g_t = 0.5 * (gp - gm)
      ftilde_t = 0.5 * (cv_fp - cv_fm)
      gtilde_t = 0.5 * (cv_gp - cv_gm)

      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, f_t, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, g_t, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 0.5, ftilde_t, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Gtilde, 0.5, gtilde_t, equations, dg, i, j)
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], ftilde_t, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], gtilde_t, equations, dg, i, jj)
      end
   end

   # Apply Jacobian to utt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to utt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      st = calc_source_t_N12(up_node, um_node, x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 0.5, st, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(utt, dt, st, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      utt_node = Trixi.get_node_vars(utt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)

      f_node, g_node = Trixi.get_node_vars(f, equations, dg, i, j), Trixi.get_node_vars(g, equations, dg, i, j)
      ftilde_node = Trixi.get_node_vars(ftilde, equations, dg, i, j)
      gtilde_node = Trixi.get_node_vars(gtilde, equations, dg, i, j)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      f_tt, g_tt = fp - 2.0 * f_node + fm, gp - 2.0 * g_node + gm
      ftilde_tt, gtilde_tt = cv_fp - 2.0 * ftilde_node + cv_fm, cv_gp - 2.0 * gtilde_node + cv_gm

      # Updating u_np1_low here
      F_ = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      G_ = Trixi.get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         inv_jacobian = inverse_jacobian[ii, j, element]
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[ii, i],
            F_, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         inv_jacobian = inverse_jacobian[i, jj, element]
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[jj, j],
            G_, equations, dg, i, jj)
      end

      # TODO - Check the source term contribution
      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(u_np1_low, 1.0, S_node, equations, dg, i, j)
      # UPDATING u_np1_low ENDS!!!

      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 6.0, ftilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, f_tt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 6.0, gtilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, g_tt, equations, dg, 2, i, j, element)

      Ftilde_node = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      Gtilde_node = Trixi.get_node_vars(Gtilde, equations, dg, i, j)
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], Ftilde_node, equations, dg, ii, j, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[ii, i],
            Ftilde_node, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], Gtilde_node, equations, dg, i, jj, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[jj, j],
            Gtilde_node, equations, dg, i, jj)
      end

      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stt = calc_source_tt_N23(u_node, up_node, um_node, x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, equations, dg, i, j)

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = Trixi.get_node_vars(U, equations, dg, i, j)

      # Ub = UT * V
      # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
      Trixi.set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      Trixi.multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg, i, j, element)
      Trixi.multiply_add_to_node_vars!(u_np1, 1.0, S_node, equations, dg, i, j)
   end

   @unpack temporal_errors = cache
   @unpack abstol, reltol = tolerances
   temporal_errors[element] = zero(dt)
   for j in eachnode(dg), i in eachnode(dg)
      u_np1_node = Trixi.get_node_vars(u_np1, equations, dg, i, j)
      u_np1_low_node = Trixi.get_node_vars(u_np1_low, equations, dg, i, j)
      # u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
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

function lw_volume_kernel_3!(du, u, t, dt, tolerances,
   element, mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms::False, source_terms, equations,
   dg::DGSEM, cache, alpha=true)
   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack element_cache, lw_res_cache = cache
   @unpack cell_arrays = lw_res_cache
   @unpack elements = cache # To access cache.U and cache.F
   refresh!(arr) = fill!(arr, zero(eltype(u)))
   id = Threads.threadid()
   f, g, ftilde, gtilde, Ftilde, Gtilde, ut, utt, uttt, U,
   up, um, upp, umm, S, u_np1, u_np1_low = cell_arrays[id]
   refresh!.((ut, utt, uttt))
   for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      flux1, flux2, cv_flux1, cv_flux2 = contravariant_flux(u_node, Ja, equations)

      Trixi.set_node_vars!(element_cache.F, flux1, equations, dg, 1, i, j, element)
      Trixi.set_node_vars!(element_cache.F, flux2, equations, dg, 2, i, j, element)
      Trixi.set_node_vars!(f, flux1, equations, dg, i, j)
      Trixi.set_node_vars!(g, flux2, equations, dg, i, j)

      Trixi.set_node_vars!(Ftilde, cv_flux1, equations, dg, i, j)
      Trixi.set_node_vars!(ftilde, cv_flux1, equations, dg, i, j)
      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1,
            equations, dg, ii, j)
      end

      Trixi.set_node_vars!(Gtilde, cv_flux2, equations, dg, i, j)
      Trixi.set_node_vars!(gtilde, cv_flux2, equations, dg, i, j)
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end

      Trixi.set_node_vars!(u_np1, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(u_np1_low, u_node, equations, dg, i, j)

      Trixi.set_node_vars!(um, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(up, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(umm, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(upp, u_node, equations, dg, i, j)
      Trixi.set_node_vars!(U, u_node, equations, dg, i, j)
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
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      s_node = calc_source(u_node, x, t, source_terms, equations, dg, cache)
      Trixi.set_node_vars!(S, s_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = Trixi.get_node_vars(ut, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, -2.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0, ut_node, equations, dg, i, j)

      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)

      f_t = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
      g_t = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)
      ftilde_t = 1.0 / 12.0 * (-cv_fpp + 8.0 * cv_fp - 8.0 * cv_fm + cv_fmm)
      gtilde_t = 1.0 / 12.0 * (-cv_gpp + 8.0 * cv_gp - 8.0 * cv_gm + cv_gmm)

      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, f_t, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, g_t, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 0.5, ftilde_t, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Gtilde, 0.5, gtilde_t, equations, dg, i, j)
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], ftilde_t, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], gtilde_t, equations, dg, i, jj)
      end
   end

   # Apply Jacobian to utt
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

   for j in eachnode(dg), i in eachnode(dg)
      utt_node = Trixi.get_node_vars(utt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, 2.0, utt_node, equations, dg, i, j)

      f_node, g_node = Trixi.get_node_vars(f, equations, dg, i, j), Trixi.get_node_vars(g, equations, dg, i, j)
      ftilde_node = Trixi.get_node_vars(ftilde, equations, dg, i, j)
      gtilde_node = Trixi.get_node_vars(gtilde, equations, dg, i, j)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      f_tt, g_tt = fp - 2.0 * f_node + fm, gp - 2.0 * g_node + gm
      ftilde_tt, gtilde_tt = cv_fp - 2.0 * ftilde_node + cv_fm, cv_gp - 2.0 * gtilde_node + cv_gm

      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 6.0, ftilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, f_tt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 6.0, gtilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, g_tt, equations, dg, 2, i, j, element)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], ftilde_tt, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], gtilde_tt, equations, dg, i, jj)
      end
   end

   # Apply Jacobian to uttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         uttt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to uttt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stt = calc_source_tt_N23(u_node, up_node, um_node, x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 6.0, stt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(uttt, dt, stt, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      uttt_node = Trixi.get_node_vars(uttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, equations, dg, i, j)

      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)
      fttt = 0.5 * (fpp - 2.0 * fp + 2.0 * fm - fmm)
      ftilde_ttt = 0.5 * (cv_fpp - 2.0 * cv_fp + 2.0 * cv_fm - cv_fmm)

      # Updating u_np1_low here
      F_ = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      G_ = Trixi.get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         inv_jacobian = inverse_jacobian[ii, j, element]
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[ii, i],
            F_, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         inv_jacobian = inverse_jacobian[i, jj, element]
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[jj, j],
            G_, equations, dg, i, jj)
      end


      # TODO - Check the source term contribution
      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(u_np1_low, 1.0, S_node, equations, dg, i, j)
      # UPDATING u_np1_low ENDS!!!

      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, fttt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 24.0, ftilde_ttt, equations, dg, i, j)
      gttt = 0.5 * (gpp - 2.0 * gp + 2.0 * gm - gmm)
      gtilde_ttt = 0.5 * (cv_gpp - 2.0 * cv_gp + 2.0 * cv_gm - cv_gmm)
      multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, gttt, equations, dg, 2, i, j, element)
      multiply_add_to_node_vars!(Gtilde, 1.0 / 24.0, gtilde_ttt, equations, dg, i, j)

      Ftilde_node = get_node_vars(Ftilde, equations, dg, i, j)
      Gtilde_node = get_node_vars(Gtilde, equations, dg, i, j)
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], Ftilde_node, equations, dg, ii, j, element)

         multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[ii, i],
            Ftilde_node, equations, dg, ii, j)
      end
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], Gtilde_node, equations, dg, i, jj, element)

         multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[jj, j],
            Gtilde_node, equations, dg, i, jj)
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
      multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
         i, j, element)
      multiply_add_to_node_vars!(u_np1, 1.0, S_node, equations, dg, i, j)
   end

   @unpack temporal_errors = cache
   @unpack abstol, reltol = tolerances
   temporal_errors[element] = zero(dt)
   for j in eachnode(dg), i in eachnode(dg)
      u_np1_node = Trixi.get_node_vars(u_np1, equations, dg, i, j)
      u_np1_low_node = Trixi.get_node_vars(u_np1_low, equations, dg, i, j)
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

function lw_volume_kernel_4!(du, u, t, dt, tolerances,
   element, mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
   nonconservative_terms::False, source_terms, equations, dg::DGSEM, cache, alpha=true)


   # true * [some floating point value] == [exactly the same floating point value]
   # This can (hopefully) be optimized away due to constant propagation.
   @unpack derivative_dhat, derivative_matrix = dg.basis
   @unpack contravariant_vectors, inverse_jacobian, node_coordinates = cache.elements
   @unpack lw_res_cache, element_cache = cache
   @unpack cell_arrays = lw_res_cache
   @unpack elements = cache # To access cache.U and cache.F
   refresh!(arr) = fill!(arr, zero(eltype(u)))
   id = Threads.threadid()
   f, g, ftilde, gtilde, Ftilde, Gtilde, ut, utt, uttt, utttt, U,
   up, um, upp, umm, S, u_np1, u_np1_low = cell_arrays[id]
   refresh!.((ut, utt, uttt, utttt))
   u_element = @view u[:,:,:,element]
   @.. begin
   u_np1     = u_element
   u_np1_low = u_element
   um        = u_element
   up        = u_element
   umm       = u_element
   upp       = u_element
   U         = u_element
   end
   for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      flux1, flux2, cv_flux1, cv_flux2 = contravariant_flux(u_node, Ja, equations)

      Trixi.set_node_vars!(element_cache.F, flux1, equations, dg, 1, i, j, element)
      Trixi.set_node_vars!(element_cache.F, flux2, equations, dg, 2, i, j, element)
      Trixi.set_node_vars!(f, flux1, equations, dg, i, j)
      Trixi.set_node_vars!(g, flux2, equations, dg, i, j)

      Trixi.set_node_vars!(Ftilde, cv_flux1, equations, dg, i, j)
      Trixi.set_node_vars!(ftilde, cv_flux1, equations, dg, i, j)
      for ii in eachnode(dg)
         # ut              += -lam * D * f for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] f[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[ii, i], cv_flux1, equations, dg, ii, j)
      end

      Trixi.set_node_vars!(Gtilde, cv_flux2, equations, dg, i, j)
      Trixi.set_node_vars!(gtilde, cv_flux2, equations, dg, i, j)
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(ut, -dt * derivative_matrix[jj, j], cv_flux2, equations, dg, i, jj)
      end
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
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      s_node = calc_source(u_node, x, t, source_terms, equations, dg, cache)
      Trixi.set_node_vars!(S, s_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(ut, dt, s_node, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      ut_node = Trixi.get_node_vars(ut, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 0.5, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, -2.0, ut_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0, ut_node, equations, dg, i, j)

      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)

      f_t = 1.0 / 12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
      g_t = 1.0 / 12.0 * (-gpp + 8.0 * gp - 8.0 * gm + gmm)
      ftilde_t = 1.0 / 12.0 * (-cv_fpp + 8.0 * cv_fp - 8.0 * cv_fm + cv_fmm)
      gtilde_t = 1.0 / 12.0 * (-cv_gpp + 8.0 * cv_gp - 8.0 * cv_gm + cv_gmm)

      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, f_t, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 0.5, g_t, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 0.5, ftilde_t, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Gtilde, 0.5, gtilde_t, equations, dg, i, j)
      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[ii, i], ftilde_t, equations, dg, ii, j)
      end
      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(utt, -dt * derivative_matrix[jj, j], gtilde_t, equations, dg, i, jj)
      end
   end

   # Apply Jacobian to utt
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

   for j in eachnode(dg), i in eachnode(dg)
      utt_node = Trixi.get_node_vars(utt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 6.0, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 0.5, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, 0.5, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0, utt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, 2.0, utt_node, equations, dg, i, j)

      f_node, g_node = Trixi.get_node_vars(f, equations, dg, i, j), Trixi.get_node_vars(g, equations, dg, i, j)
      ftilde_node = Trixi.get_node_vars(ftilde, equations, dg, i, j)
      gtilde_node = Trixi.get_node_vars(gtilde, equations, dg, i, j)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)
      f_tt = (1.0 / 12.0) * (-fpp + 16.0 * fp - 30.0 * f_node + 16.0 * fm - fmm)
      g_tt = (1.0 / 12.0) * (-gpp + 16.0 * gp - 30.0 * g_node + 16.0 * gm - gmm)
      ftilde_tt = (1.0 / 12.0) * (-cv_fpp + 16.0 * cv_fp - 30.0 * ftilde_node + 16.0 * cv_fm - cv_fmm)
      gtilde_tt = (1.0 / 12.0) * (-cv_gpp + 16.0 * cv_gp - 30.0 * gtilde_node + 16.0 * cv_gm - cv_gmm)

      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 6.0, ftilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, f_tt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 6.0, gtilde_tt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 6.0, g_tt, equations, dg, 2, i, j,
                                       element)

      for ii in eachnode(dg)
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[ii, i], ftilde_tt,
                                          equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(uttt, -dt * derivative_matrix[jj, j], gtilde_tt,
                                          equations, dg, i, jj)
      end
   end

   # Apply Jacobian to uttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         uttt[v, i, j] *= inv_jacobian
      end
   end

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

   for j in eachnode(dg), i in eachnode(dg)
      uttt_node = Trixi.get_node_vars(uttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 24.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, -1.0 / 6.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0 / 6.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, -4.0 / 3.0, uttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 4.0 / 3.0, uttt_node, equations, dg, i, j)

      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)

      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)

      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)
      fttt = 0.5 * (fpp - 2.0 * fp + 2.0 * fm - fmm)
      ftilde_ttt = 0.5 * (cv_fpp - 2.0 * cv_fp + 2.0 * cv_fm - cv_fmm)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, fttt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 24.0, ftilde_ttt, equations, dg, i, j)
      gttt = 0.5 * (gpp - 2.0 * gp + 2.0 * gm - gmm)
      gtilde_ttt = 0.5 * (cv_gpp - 2.0 * cv_gp + 2.0 * cv_gm - cv_gmm)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 24.0, gttt, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 24.0, gtilde_ttt, equations, dg, i, j)

      for ii in eachnode(dg)
         # ut              += -lam * D * ft for each variable
         # i.e.,  ut[ii,j] += -lam * Dm[ii,i] ft[i,j] (sum over i)
         Trixi.multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[ii, i], fttt, equations, dg, ii, j)
      end
      for jj in eachnode(dg)
         # C += -lam*gt*Dm' for each variable
         # C[i,jj] += -lam*gt[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(utttt, -dt * derivative_matrix[jj, j], gttt, equations, dg, i, jj)
      end
   end

   # Apply jacobian on utttt
   for j in eachnode(dg), i in eachnode(dg)
      inv_jacobian = inverse_jacobian[i, j, element]
      for v in eachvariable(equations)
         utttt[v, i, j] *= inv_jacobian
      end
   end

   # Add source term contribution to utttt and some to S
   for j in eachnode(dg), i in eachnode(dg)
      # Add source term contribution to ut
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      sttt = calc_source_ttt_N34(u_node, up_node, upp_node, um_node, umm_node,
         x, t, dt, source_terms,
         equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 24.0, sttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(utttt, dt, sttt, equations, dg, i, j) # has no jacobian factor
   end

   for j in eachnode(dg), i in eachnode(dg)
      utttt_node = Trixi.get_node_vars(utttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(U, 1.0 / 120.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(um, 1.0 / 24.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(up, 1.0 / 24.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(umm, 2.0 / 3.0, utttt_node, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(upp, 2.0 / 3.0, utttt_node, equations, dg, i, j)

      f_node = Trixi.get_node_vars(f, equations, dg, i, j)
      g_node = Trixi.get_node_vars(g, equations, dg, i, j)
      ftilde_node = Trixi.get_node_vars(ftilde, equations, dg, i, j)
      gtilde_node = Trixi.get_node_vars(gtilde, equations, dg, i, j)
      um_node = Trixi.get_node_vars(um, equations, dg, i, j)
      up_node = Trixi.get_node_vars(up, equations, dg, i, j)
      umm_node = Trixi.get_node_vars(umm, equations, dg, i, j)
      upp_node = Trixi.get_node_vars(upp, equations, dg, i, j)
      Ja = get_contravariant_matrix(contravariant_vectors, i, j, element)
      fm, gm, cv_fm, cv_gm = contravariant_flux(um_node, Ja, equations)
      fp, gp, cv_fp, cv_gp = contravariant_flux(up_node, Ja, equations)
      fmm, gmm, cv_fmm, cv_gmm = contravariant_flux(umm_node, Ja, equations)
      fpp, gpp, cv_fpp, cv_gpp = contravariant_flux(upp_node, Ja, equations)

      ftttt = 0.5 * (fpp - 4.0 * fp + 6.0 * f_node - 4.0 * fm + fmm)
      gtttt = 0.5 * (gpp - 4.0 * gp + 6.0 * g_node - 4.0 * gm + gmm)
      ftilde_tttt = 0.5 * (cv_fpp - 4.0 * cv_fp + 6.0 * ftilde_node - 4.0 * cv_fm + cv_fmm)
      gtilde_tttt = 0.5 * (cv_gpp - 4.0 * cv_gp + 6.0 * gtilde_node - 4.0 * cv_gm + cv_gmm)

      # Updating u_np1_low here
      F_ = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      G_ = Trixi.get_node_vars(Gtilde, equations, dg, i, j)

      for ii in eachnode(dg)
         inv_jacobian = inverse_jacobian[ii, j, element]
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[ii, i],
            F_, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         inv_jacobian = inverse_jacobian[i, jj, element]
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(u_np1_low, -dt * inv_jacobian * derivative_matrix[jj, j],
            G_, equations, dg, i, jj)
      end

      # TODO - Check the source term contribution
      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(u_np1_low, 1.0, S_node, equations, dg, i, j)

      # UPDATING u_np1_low ENDS!!!

      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 120.0, ftttt, equations, dg, 1, i, j, element)
      Trixi.multiply_add_to_node_vars!(element_cache.F, 1.0 / 120.0, gtttt, equations, dg, 2, i, j, element)
      Trixi.multiply_add_to_node_vars!(Ftilde, 1.0 / 120.0, ftilde_tttt, equations, dg, i, j)
      Trixi.multiply_add_to_node_vars!(Gtilde, 1.0 / 120.0, gtilde_tttt, equations, dg, i, j)

      Ftilde_node = Trixi.get_node_vars(Ftilde, equations, dg, i, j)
      Gtilde_node = Trixi.get_node_vars(Gtilde, equations, dg, i, j)
      for ii in eachnode(dg)
         inv_jacobian = inverse_jacobian[ii, j, element]
         # res              += -lam * D * F for each variable
         # i.e.,  res[ii,j] += -lam * Dm[ii,i] F[i,j] (sum over i)U_node
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], Ftilde_node,
            equations, dg, ii, j, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[ii, i],
            Ftilde_node, equations, dg, ii, j)
      end

      for jj in eachnode(dg)
         inv_jacobian = inverse_jacobian[i, jj, element]
         # C += -lam*g*Dm' for each variable
         # C[i,jj] += -lam*g[i,j]*Dm[jj,j] (sum over j)
         Trixi.multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], Gtilde_node,
            equations, dg, i, jj, element)

         Trixi.multiply_add_to_node_vars!(u_np1, -dt * inv_jacobian * derivative_matrix[jj, j],
            Gtilde_node, equations, dg, i, jj)
      end

      # TODO - Add source term contribution to u_np1, u_np1_low too

      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      stttt = calc_source_tttt_N4(u_node, up_node, um_node, upp_node, umm_node,
         x, t, dt, source_terms, equations, dg, cache)
      Trixi.multiply_add_to_node_vars!(S, 1.0 / 120.0, stttt, equations, dg, i, j)

      # TODO - update to v1.8 and call with @inline
      # Give u1_ or U depending on dissipation model
      U_node = Trixi.get_node_vars(U, equations, dg, i, j)

      # Ub = UT * V
      # Ub[j] += ∑_i UT[j,i] * V[i] = ∑_i U[i,j] * V[i]
      Trixi.set_node_vars!(element_cache.U, U_node, equations, dg, i, j, element)

      S_node = Trixi.get_node_vars(S, equations, dg, i, j)
      inv_jacobian = inverse_jacobian[i, j, element]
      Trixi.multiply_add_to_node_vars!(du, -1.0 / inv_jacobian, S_node, equations, dg,
         i, j, element)
      Trixi.multiply_add_to_node_vars!(u_np1, 1.0, S_node, equations, dg, i, j)
   end

   @unpack temporal_errors = cache
   @unpack abstol, reltol = tolerances
   temporal_errors[element] = zero(dt)
   for j in eachnode(dg), i in eachnode(dg)
      u_np1_node = Trixi.get_node_vars(u_np1, equations, dg, i, j)
      u_np1_low_node = Trixi.get_node_vars(u_np1_low, equations, dg, i, j)
      # u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      for v in eachvariable(equations)
         temporal_errors[element] += (
            (u_np1_node[v] - u_np1_low_node[v])
            /
            (abstol + reltol * max(abs(u_np1_node[v]), abs(u_np1_low_node[v])))
         )^2
      end
   end

   return nothing

   # cell_arrays = cache.lw_res_cache.cell_arrays[Threads.threadid()]
   # refresh!(arr) = fill!(arr, zero(eltype(u)))
   # @unpack ut, utt, uttt, utttt, U, up, um, upp, umm, S, u_np1, u_np1_low = cell_arrays
   # refresh!.((ut, utt, uttt, utttt))
   # u_element = @view u[:,:,:,element]
   # @turbo u_np1 .=  u_np1_low .=  um .= up .= umm .= upp .= U .= u_element # TODO - Is this the problem?

   # # Compute flux f and source term s and use them to compute ut
   # compute_f_s_ut!(cell_arrays, t, dt, u, source_terms, equations, dg, cache, element)

   # # Compute ft, st and use them to compute utt
   # compute_ft_st_utt!(cell_arrays, t, dt, u, source_terms, equations, dg, cache, element)

   # # Compute ftt, stt and use them to compute uttt
   # compute_ftt_stt_uttt!(cell_arrays, t, dt, u, source_terms, equations, dg, cache, element)

   # # Compute ftt, stt and use them to compute uttt
   # compute_fttt_sttt_utttt!(cell_arrays, t, dt, u, source_terms, equations, dg, cache, element)

   # # Compute ftt, stt and use them to compute uttt
   # compute_ftttt_stttt_du!(du, cell_arrays, t, dt, u, source_terms, equations, dg, cache, element, alpha)

   # # Compute temporal_errors
   # compute_temporal_errors!(cache, cell_arrays, tolerances, dt, equations, dg, element)

   # return nothing
end

function calc_interface_flux!(cache, u, dt,
   mesh::StructuredMesh{2},
   nonconservative_terms, # can be Val{true}/False
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, alpha, dg::DG)
   @unpack elements = cache

   @threaded for element in eachelement(dg, cache)
      # Interfaces in negative directions
      # Faster version of "for orientation in (1, 2)"

      # Interfaces in x-direction (`orientation` = 1)
      calc_interface_flux!(elements.surface_flux_values,
         elements.left_neighbors[1, element],
         element, 1, u, dt, mesh,
         nonconservative_terms, equations,
         surface_integral, time_discretization, alpha, dg, cache)

      # Interfaces in y-direction (`orientation` = 2)
      calc_interface_flux!(elements.surface_flux_values,
         elements.left_neighbors[2, element],
         element, 2, u, dt, mesh,
         nonconservative_terms, equations,
         surface_integral, time_discretization, alpha, dg, cache)
   end

   return nothing
end

function calc_interface_flux!(cache, u, dt,
   mesh::StructuredMesh{2},
   nonconservative_terms, # can be Val{true}/False
   equations, surface_integral, time_discretization, alpha, dg::DG)
   @unpack elements = cache

   @threaded for element in eachelement(dg, cache)
      # Interfaces in negative directions
      # Faster version of "for orientation in (1, 2)"

      # Interfaces in x-direction (`orientation` = 1)
      calc_interface_flux!(elements.surface_flux_values,
         elements.left_neighbors[1, element],
         element, 1, u, dt, mesh,
         nonconservative_terms, equations,
         surface_integral, time_discretization, alpha, dg, cache)

      # Interfaces in y-direction (`orientation` = 2)
      calc_interface_flux!(elements.surface_flux_values,
         elements.left_neighbors[2, element],
         element, 2, u, dt, mesh,
         nonconservative_terms, equations,
         surface_integral, time_discretization, alpha, dg, cache)
   end

   return nothing
end

function compute_alp(
   u_ll, u_rr, primary_element_index, secondary_element_index, Jl, Jr, dt,
   fn, Fn, fn_inner_ll, fn_inner_rr, primary_node_index, equations, dg, volume_integral::VolumeIntegralFR, mesh::Union{StructuredMesh,UnstructuredMesh2D})
   return zero(eltype(u_ll))
end

function compute_alp(u_ll, u_rr, left_element, right_element, Jl, Jr, dt, fn, Fn_,
   fn_inner_ll, fn_inner_rr, i,
   equations, dg, volume_integral::VolumeIntegralFRShockCapturing,
   mesh::Union{StructuredMesh,UnstructuredMesh2D})
   @unpack weights = dg.basis
   @unpack alpha = volume_integral.indicator.cache
   alp = 0.5 * (alpha[left_element] + alpha[right_element])

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

@inline function calc_interface_flux!(surface_flux_values,
   left_element, right_element,
   orientation, u, dt,
   mesh::StructuredMesh{2},
   nonconservative_terms::False,
   equations,
   surface_integral, time_discretization, alpha, dg::DG, cache)
   # This is slow for LSA, but for some reason faster for Euler (see #519)
   @unpack elements = cache
   @unpack U, F, fn_low = cache.element_cache

   if left_element <= 0 # left_element = 0 at boundaries
      return nothing
   end

   @unpack surface_flux = surface_integral
   @unpack contravariant_vectors, inverse_jacobian = cache.elements

   right_direction = 2 * orientation
   left_direction = right_direction - 1

   for i in eachnode(dg)
      if orientation == 1
         U_ll = Trixi.get_node_vars(U, equations, dg, nnodes(dg), i, left_element)
         f_ll_ = get_flux_vars(F, equations, dg, nnodes(dg), i, left_element)
         U_rr = Trixi.get_node_vars(U, equations, dg, 1, i, right_element)
         f_rr_ = get_flux_vars(F, equations, dg, 1, i, right_element)

         # For blending
         # ulow_ll  = Trixi.get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
         # ulow_rr  = Trixi.get_node_vars(u, equations, dg, 1,          i, right_element)
         # ulow_ll   = @view u[:,:,i,left_element]
         # ulow_rr   = @view u[:,:,i,right_element]
         u_ll = Trixi.get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
         u_rr = Trixi.get_node_vars(u, equations, dg, 1, i, right_element)
         Jl = inverse_jacobian[nnodes(dg), i, left_element]
         Jr = inverse_jacobian[1, i, right_element]
         fn_inner_ll = Trixi.get_node_vars(fn_low, equations, dg, i, 2, left_element)
         fn_inner_rr = Trixi.get_node_vars(fn_low, equations, dg, i, 1, right_element)

         # If the mapping is orientation-reversing, the contravariant vectors' orientation
         # is reversed as well. The normal vector must be oriented in the direction
         # from `left_element` to `right_element`, or the numerical flux will be computed
         # incorrectly (downwind direction).
         sign_jacobian = sign(inverse_jacobian[1, i, right_element])

         # First contravariant vector Ja^1 as SVector
         normal_direction = sign_jacobian * get_contravariant_vector(1, contravariant_vectors,
            1, i, right_element)
      else # orientation == 2
         U_ll = Trixi.get_node_vars(U, equations, dg, i, nnodes(dg), left_element)
         f_ll_ = get_flux_vars(F, equations, dg, i, nnodes(dg), left_element)
         U_rr = Trixi.get_node_vars(U, equations, dg, i, 1, right_element)
         f_rr_ = get_flux_vars(F, equations, dg, i, 1, right_element)

         # For blending
         # ulow_ll  = Trixi.get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
         # ulow_rr  = Trixi.get_node_vars(u, equations, dg, i, 1,          right_element)
         u_ll = Trixi.get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
         u_rr = Trixi.get_node_vars(u, equations, dg, i, 1, right_element)
         Jl = inverse_jacobian[i, nnodes(dg), left_element]
         Jr = inverse_jacobian[i, 1, right_element]
         fn_inner_ll = Trixi.get_node_vars(fn_low, equations, dg, i, 4, left_element)
         fn_inner_rr = Trixi.get_node_vars(fn_low, equations, dg, i, 3, right_element)

         # See above
         sign_jacobian = sign(inverse_jacobian[i, 1, right_element])

         # Second contravariant vector Ja^2 as SVector
         normal_direction = sign_jacobian * get_contravariant_vector(2, contravariant_vectors,
            i, 1, right_element)
      end

      f_ll = normal_product(f_ll_, equations, normal_direction)
      f_rr = normal_product(f_rr_, equations, normal_direction)
      # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
      # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
      Fn = sign_jacobian * surface_flux(f_ll, f_rr,
         u_ll, u_rr,
         U_ll, U_rr,
         normal_direction, equations)
      fn = compute_low_order_flux(u_ll, u_rr, equations, dg,
         normal_direction, surface_flux, sign_jacobian)

      alp = compute_alp(u_ll, u_rr, left_element, right_element, Jl, Jr, dt,
         fn, Fn, fn_inner_ll, fn_inner_rr, i,
         equations, dg, dg.volume_integral, mesh)

      for v in eachvariable(equations)
         surface_flux_values[v, i, right_direction, left_element] = alp * fn[v] + (1.0 - alp) * Fn[v]
         surface_flux_values[v, i, left_direction, right_element] = alp * fn[v] + (1.0 - alp) * Fn[v]
      end
   end

   return nothing
end

function calc_boundary_flux!(cache, u, t, dt, boundary_condition::BoundaryConditionPeriodic,
   mesh::StructuredMesh{2}, equations, surface_integral,
   time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @assert Trixi.isperiodic(mesh)
end

function calc_boundary_flux!(cache, u, t, dt, boundary_conditions::NamedTuple,
   mesh::StructuredMesh{2}, equations, surface_integral,
   time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @unpack surface_flux_values = cache.elements
   linear_indices = LinearIndices(size(mesh))

   for cell_y in axes(mesh, 2)
      # Negative x-direction
      direction = 1
      element = linear_indices[begin, cell_y]

      for j in eachnode(dg)
         calc_boundary_flux_by_direction!(surface_flux_values, u, t, dt, 1,
            boundary_conditions[direction],
            mesh, equations, surface_integral, time_discretization, dg, cache,
            direction, (1, j), (j,), element)
      end

      # Positive x-direction
      direction = 2
      element = linear_indices[end, cell_y]

      for j in eachnode(dg)
         calc_boundary_flux_by_direction!(surface_flux_values, u, t, dt, 1,
            boundary_conditions[direction],
            mesh, equations, surface_integral, time_discretization, dg, cache,
            direction, (nnodes(dg), j), (j,), element)
      end
   end

   for cell_x in axes(mesh, 1)
      # Negative y-direction
      direction = 3
      element = linear_indices[cell_x, begin]

      for i in eachnode(dg)
         calc_boundary_flux_by_direction!(surface_flux_values, u, t, dt, 2,
            boundary_conditions[direction],
            mesh, equations, surface_integral, time_discretization, dg, cache,
            direction, (i, 1), (i,), element)
      end

      # Positive y-direction
      direction = 4
      element = linear_indices[cell_x, end]

      for i in eachnode(dg)
         calc_boundary_flux_by_direction!(surface_flux_values, u, t, dt, 2,
            boundary_conditions[direction],
            mesh, equations, surface_integral, time_discretization, dg, cache,
            direction, (i, nnodes(dg)), (i,), element)
      end
   end
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t, dt, orientation,
   boundary_condition,
   mesh::StructuredMesh, equations,
   surface_integral, time_discretization::AbstractLWTimeDiscretization,
   dg::DG, cache,
   direction, node_indices, surface_node_indices, element)
   @unpack boundary_cache, elements = cache
   @unpack U, F = cache.element_cache
   @unpack outer_cache = boundary_cache
   @unpack node_coordinates, contravariant_vectors, inverse_jacobian = elements
   @unpack surface_flux = surface_integral

   u_inner = Trixi.get_node_vars(u, equations, dg, node_indices..., element)
   U_inner = Trixi.get_node_vars(U, equations, dg, node_indices..., element)
   f_inner_ = get_flux_vars(F, equations, dg, node_indices..., element)
   x = get_node_coords(node_coordinates, equations, dg, node_indices..., element)

   # If the mapping is orientation-reversing, the contravariant vectors' orientation
   # is reversed as well. The normal vector must be oriented in the direction
   # from `left_element` to `right_element`, or the numerical flux will be computed
   # incorrectly (downwind direction).
   sign_jacobian = sign(inverse_jacobian[node_indices..., element])

   # Contravariant vector Ja^i is the normal vector
   normal = sign_jacobian * get_contravariant_vector(orientation, contravariant_vectors,
      node_indices..., element)

   f_inner = normal_product(f_inner_, equations, normal)
   # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
   # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
   # TODO - First U_inner should be u_inner
   flux = sign_jacobian * boundary_condition(U_inner, f_inner, u_inner, outer_cache, normal,
      direction, x, t,
      dt, surface_flux, equations, dg,
      time_discretization)

   for v in eachvariable(equations)
      surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
   end
end

end # muladd macro
