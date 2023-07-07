using Trixi: DG, indices2direction, P4estMesh, eachnode, get_normal_direction,
   False, eachboundary

import Trixi
import Trixi: prolong2interfaces!, calc_interface_flux!, calc_boundary_flux!,
   prolong2mortars!, calc_mortar_flux!

function prolong2interfaces!(cache, u,
   mesh::P4estMesh{2},
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @unpack interfaces, elements, interface_cache, element_cache = cache
   @unpack U, F, fn_low = element_cache
   @unpack contravariant_vectors = elements
   index_range = eachnode(dg)

   @threaded for interface in eachinterface(dg, cache)
      # Copy solution data from the primary element using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.
      # Note that in the current implementation, the interface will be
      # "aligned at the primary element", i.e., the index of the primary side
      # will always run forwards.
      primary_element = interfaces.neighbor_ids[1, interface]
      primary_indices = interfaces.node_indices[1, interface]
      primary_direction = indices2direction(primary_indices) # for normal, fn_low (bottom, right, top, left)

      i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1], index_range)
      j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2], index_range)

      i_primary = i_primary_start
      j_primary = j_primary_start
      for i in eachnode(dg)
         f = get_flux_vars(F, equations, dg, i_primary, j_primary, primary_element)
         normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
            i_primary, j_primary, primary_element)
         fn_node = normal_product(f, equations, normal_direction)

         for v in eachvariable(equations)
            interface_cache.u[1, v, i, interface] = u[v, i_primary, j_primary, primary_element]
            interface_cache.U[1, v, i, interface] = U[v, i_primary, j_primary, primary_element]
            interface_cache.f[1, v, i, interface] = fn_node[v]
         end
         i_primary += i_primary_step
         j_primary += j_primary_step
      end

      # Copy solution data from the secondary element using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.
      secondary_element = interfaces.neighbor_ids[2, interface]
      secondary_indices = interfaces.node_indices[2, interface]
      secondary_direction = indices2direction(secondary_indices) # for normal, fn_low

      i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1], index_range)
      j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2], index_range)

      i_secondary = i_secondary_start
      j_secondary = j_secondary_start
      for i in eachnode(dg)
         f = get_flux_vars(F, equations, dg, i_secondary, j_secondary, secondary_element)
         normal_direction = -get_normal_direction(secondary_direction, contravariant_vectors,
            i_secondary, j_secondary, secondary_element)
         fn_node = normal_product(f, equations, normal_direction)

         for v in eachvariable(equations)
            interface_cache.u[2, v, i, interface] = u[v, i_secondary, j_secondary, secondary_element]
            interface_cache.U[2, v, i, interface] = U[v, i_secondary, j_secondary, secondary_element]
            interface_cache.f[2, v, i, interface] = fn_node[v]
         end
         i_secondary += i_secondary_step
         j_secondary += j_secondary_step
      end

      for v in eachvariable(equations), i in eachnode(dg)
         interface_cache.fn_low[1, v, i, interface] = fn_low[v, i, primary_direction, primary_element]
         interface_cache.fn_low[2, v, i, interface] = fn_low[v, i, secondary_direction, secondary_element]
      end
   end

   return nothing
end

function calc_interface_flux!(surface_flux_values,
   mesh::P4estMesh{2},
   nonconservative_terms,
   equations, surface_integral, dt, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache)
   @unpack neighbor_ids, node_indices = cache.interfaces
   @unpack contravariant_vectors = cache.elements
   index_range = eachnode(dg)
   index_end = last(index_range)

   @threaded for interface in eachinterface(dg, cache)
      # Get element and side index information on the primary element
      primary_element = neighbor_ids[1, interface]
      primary_indices = node_indices[1, interface]
      primary_direction = indices2direction(primary_indices)

      # Create the local i,j indexing on the primary element used to pull normal direction information
      i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1], index_range)
      j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2], index_range)

      i_primary = i_primary_start
      j_primary = j_primary_start

      # Get element and side index information on the secondary element
      secondary_element = neighbor_ids[2, interface]
      secondary_indices = node_indices[2, interface]
      secondary_direction = indices2direction(secondary_indices)

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
         # Get the normal direction on the primary element.
         # Contravariant vectors at interfaces in negative coordinate direction
         # are pointing inwards. This is handled by `get_normal_direction`.
         normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
            i_primary, j_primary, primary_element)

         calc_interface_flux!(surface_flux_values, mesh, nonconservative_terms, equations,
            surface_integral, dt, time_discretization, dg, cache,
            interface, normal_direction,
            node, primary_direction, primary_element,
            node_secondary, secondary_direction, secondary_element)

         # Increment primary element indices to pull the normal direction
         i_primary += i_primary_step
         j_primary += j_primary_step
         # Increment the surface node index along the secondary element
         node_secondary += node_secondary_step
      end
   end

   return nothing
end

# Inlined version of the interface flux computation for conservation laws
@inline function calc_interface_flux!(surface_flux_values,
   mesh::P4estMesh{2},
   nonconservative_terms::False, equations,
   surface_integral, dt, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache,
   interface_index, normal_direction,
   primary_node_index, primary_direction_index, primary_element_index,
   secondary_node_index, secondary_direction_index, secondary_element_index)
   @unpack u, f, fn_low, U = cache.interface_cache
   @unpack surface_flux = surface_integral

   U_ll, U_rr = get_surface_node_vars(U, equations, dg, primary_node_index, interface_index)
   u_ll, u_rr = get_surface_node_vars(u, equations, dg, primary_node_index, interface_index)
   f_ll, f_rr = get_surface_node_vars(f, equations, dg, primary_node_index, interface_index)
   fn_inner_ll, fn_inner_rr = get_surface_node_vars(fn_low, equations, dg, primary_node_index, interface_index)

   # flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)
   Fn = surface_flux(f_ll, f_rr, u_ll, u_rr, U_ll, U_rr, normal_direction,
      equations)

   fn = surface_flux(u_ll, u_rr, normal_direction, equations)

   Jl = Jr = cache.interface_cache.inverse_jacobian[primary_node_index, interface_index]
   alp = compute_alp(u_ll, u_rr, primary_element_index, secondary_element_index,
      Jl, Jr, dt,
      fn, Fn, fn_inner_ll, fn_inner_rr, primary_node_index,
      equations, dg, dg.volume_integral)

   for v in eachvariable(equations)
      surface_flux_values[v, primary_node_index, primary_direction_index, primary_element_index] = (
         alp * fn[v] + (1.0 - alp) * Fn[v]
         # Fn[v]
      )
      surface_flux_values[v, secondary_node_index, secondary_direction_index, secondary_element_index] = -(
         alp * fn[v] + (1.0 - alp) * Fn[v]
         # Fn[v]
      )
   end
end

function compute_alp(
   u_ll, u_rr, primary_element_index, secondary_element_index, Jl, Jr, dt,
   fn, Fn, fn_inner_ll, fn_inner_rr, primary_node_index, equations, dg, volume_integral::VolumeIntegralFR)
   return zero(eltype(u_ll))
end

function compute_alp(
   u_ll, u_rr, primary_element_index, secondary_element_index, Jl, Jr, dt,
   fn, Fn_, fn_inner_ll, fn_inner_rr, primary_node_index, equations, dg,
   volume_integral::VolumeIntegralFRShockCapturing)
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

function prolong2boundaries!(cache, u,
   mesh::P4estMesh{2},
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @unpack boundaries, boundary_cache, elements, element_cache = cache
   @unpack U, F = element_cache
   @unpack contravariant_vectors = elements
   index_range = eachnode(dg)

   @threaded for boundary in eachboundary(dg, cache)
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
         f = get_flux_vars(F, equations, dg, i_node, j_node, element)
         normal_direction = get_normal_direction(direction, contravariant_vectors,
            i_node, j_node, element)
         fn_node = normal_product(f, equations, normal_direction)
         for v in eachvariable(equations)
            boundary_cache.u[v, i, boundary] = u[v, i_node, j_node, element]
            boundary_cache.U[v, i, boundary] = U[v, i_node, j_node, element]
            boundary_cache.f[v, i, boundary] = fn_node[v]
         end
         i_node += i_node_step
         j_node += j_node_step
      end
   end

   return nothing
end

function calc_boundary_flux!(cache, t, dt, boundary_condition, boundary_indexing::Vector,
   mesh::P4estMesh{2},
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG,
   scaling_factor)
   @unpack boundaries = cache
   @unpack surface_flux_values = cache.elements
   index_range = eachnode(dg)

   @threaded for local_index in eachindex(boundary_indexing)
      # Use the local index to get the global boundary index from the pre-sorted list
      boundary = boundary_indexing[local_index]

      # Get information on the adjacent element, compute the surface fluxes,
      # and store them
      element = boundaries.neighbor_ids[boundary]
      node_indices = boundaries.node_indices[boundary]
      direction = indices2direction(node_indices)

      i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
      j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

      i_node = i_node_start
      j_node = j_node_start
      for node in eachnode(dg)
         calc_boundary_flux!(surface_flux_values, t, dt, boundary_condition,
            mesh, have_nonconservative_terms(equations),
            equations, surface_integral, time_discretization, dg, cache,
            i_node, j_node,
            node, direction, element, boundary, scaling_factor)

         i_node += i_node_step
         j_node += j_node_step
      end
   end
end


# inlined version of the boundary flux calculation along a physical interface
@inline function calc_boundary_flux!(surface_flux_values, t, dt, boundary_condition,
   mesh::P4estMesh{2},
   nonconservative_terms::False, equations,
   surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache,
   i_index, j_index,
   node_index, direction_index, element_index, boundary_index, scaling_factor)
   @unpack boundaries, boundary_cache = cache
   @unpack outer_cache = boundary_cache
   @unpack node_coordinates, contravariant_vectors = cache.elements
   @unpack surface_flux = surface_integral

   # Extract solution data from boundary container
   u_inner = Trixi.get_node_vars(boundary_cache.u, equations, dg, node_index, boundary_index)
   U_inner = Trixi.get_node_vars(boundary_cache.U, equations, dg, node_index, boundary_index)
   f_inner = Trixi.get_node_vars(boundary_cache.f, equations, dg, node_index, boundary_index)

   # Outward-pointing normal direction (not normalized)
   normal_direction = get_normal_direction(direction_index, contravariant_vectors,
      i_index, j_index, element_index)

   # Coordinates at boundary node
   x = get_node_coords(node_coordinates, equations, dg, i_index, j_index, element_index)

   # flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

   flux_ = boundary_condition(U_inner, f_inner, u_inner, outer_cache, normal_direction, x, t, dt,
      surface_flux, equations, dg, time_discretization, scaling_factor)

   # Copy flux to element storage in the correct orientation
   for v in eachvariable(equations)
      surface_flux_values[v, node_index, direction_index, element_index] = flux_[v]
   end
end

function prolong2mortars!(cache, u,
   mesh::P4estMesh{2}, equations,
   mortar_l2::LobattoLegendreMortarL2,
   surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DGSEM)
   @unpack neighbor_ids, node_indices = cache.mortars
   @unpack U, F, fn_low = cache.element_cache
   @unpack contravariant_vectors = cache.elements
   @unpack mortars, lw_mortars = cache
   index_range = eachnode(dg)

   @threaded for mortar in eachmortar(dg, cache)
      # Copy solution data from the small elements using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.

      small_indices = node_indices[1, mortar]
      small_direction = indices2direction(small_indices)

      i_small_start, i_small_step = index_to_start_step_2d(small_indices[1], index_range)
      j_small_start, j_small_step = index_to_start_step_2d(small_indices[2], index_range)

      for position in 1:2
         i_small = i_small_start
         j_small = j_small_start
         element = neighbor_ids[position, mortar]
         for i in eachnode(dg)
            f = get_flux_vars(F, equations, dg, i_small, j_small, element)
            normal_direction = get_normal_direction(small_direction, contravariant_vectors,
               i_small, j_small, element)
            fn_node = normal_product(f, equations, normal_direction)
            for v in eachvariable(equations)
               cache.mortars.u[1, v, position, i, mortar] = u[v, i_small, j_small, element]
               lw_mortars.U[1, v, position, i, mortar] = U[v, i_small, j_small, element]
               lw_mortars.F[1, v, position, i, mortar] = fn_node[v]
            end
            i_small += i_small_step
            j_small += j_small_step
         end

         for v in eachvariable(equations), i in eachnode(dg)
            lw_mortars.fn_low[1, v, position, i, mortar] = fn_low[v, i, small_direction, element]
         end

      end

      # Buffer to copy solution values of the large element in the correct orientation
      # before interpolating
      u_buffer = cache.u_threaded[Threads.threadid()]
      U_buffer = lw_mortars.tmp.U_threaded[Threads.threadid()]
      F_buffer = lw_mortars.tmp.F_threaded[Threads.threadid()]
      fn_low_buffer = lw_mortars.tmp.fn_low_threaded[Threads.threadid()]

      # Copy solution of large element face to buffer in the
      # correct orientation
      large_indices = node_indices[2, mortar]
      large_direction = indices2direction(large_indices)

      i_large_start, i_large_step = index_to_start_step_2d(large_indices[1], index_range)
      j_large_start, j_large_step = index_to_start_step_2d(large_indices[2], index_range)

      i_large = i_large_start
      j_large = j_large_start
      element = neighbor_ids[3, mortar]
      for i in eachnode(dg)
         f = get_flux_vars(F, equations, dg, i_large, j_large, element)
         # TODO - Why this minus? Why this 0.5? Is it correct? Does it work on curved meshes?
         normal_direction = -0.5 * get_normal_direction(large_direction, contravariant_vectors,
            i_large, j_large, element)
         u_node = Trixi.get_node_vars(u, equations, dg, i_large, j_large, element)
         fn_node = normal_product(f, equations, normal_direction)
         fn_node = Trixi.flux(u_node, normal_direction, equations)
         for v in eachvariable(equations)
            u_buffer[v, i] = u[v, i_large, j_large, element]
            U_buffer[v, i] = U[v, i_large, j_large, element]
            F_buffer[v, i] = fn_node[v]
            # TODO - Should this have a 0.5 factor?
            fn_low_buffer[v, i] = fn_low[v, i, large_direction, element]
         end
         i_large += i_large_step
         j_large += j_large_step
      end

      # Interpolate large element face data from buffer to small face locations
      multiply_dimensionwise!(view(cache.mortars.u, 2, :, 1, :, mortar),
         mortar_l2.forward_lower,
         u_buffer)
      multiply_dimensionwise!(view(cache.mortars.u, 2, :, 2, :, mortar),
         mortar_l2.forward_upper,
         u_buffer)

      multiply_dimensionwise!(view(cache.lw_mortars.U, 2, :, 1, :, mortar),
         mortar_l2.forward_lower,
         U_buffer)
      multiply_dimensionwise!(view(cache.lw_mortars.U, 2, :, 2, :, mortar),
         mortar_l2.forward_upper,
         U_buffer)

      multiply_dimensionwise!(view(cache.lw_mortars.F, 2, :, 1, :, mortar),
         mortar_l2.forward_lower,
         F_buffer)
      multiply_dimensionwise!(view(cache.lw_mortars.F, 2, :, 2, :, mortar),
         mortar_l2.forward_upper,
         F_buffer)

      multiply_dimensionwise!(view(cache.lw_mortars.fn_low, 2, :, 1, :, mortar),
         mortar_l2.forward_lower,
         fn_low_buffer)
      multiply_dimensionwise!(view(cache.lw_mortars.fn_low, 2, :, 2, :, mortar),
         mortar_l2.forward_upper,
         fn_low_buffer)
   end

   return nothing
end


function calc_mortar_flux!(surface_flux_values,
   mesh::P4estMesh{2},
   nonconservative_terms, equations,
   mortar_l2::LobattoLegendreMortarL2,
   surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache)
   @unpack neighbor_ids, node_indices = cache.mortars
   @unpack contravariant_vectors = cache.elements
   @unpack fstar_upper_threaded, fstar_lower_threaded = cache
   index_range = eachnode(dg)
   dt = cache.dt[1]

   @threaded for mortar in eachmortar(dg, cache)
      # Choose thread-specific pre-allocated container
      fstar = (fstar_lower_threaded[Threads.threadid()],
         fstar_upper_threaded[Threads.threadid()])

      # Get index information on the small elements
      small_indices = node_indices[1, mortar]
      small_direction = indices2direction(small_indices)

      i_small_start, i_small_step = index_to_start_step_2d(small_indices[1], index_range)
      j_small_start, j_small_step = index_to_start_step_2d(small_indices[2], index_range)

      for position in 1:2
         i_small = i_small_start
         j_small = j_small_start
         element = neighbor_ids[position, mortar]
         for node in eachnode(dg)
            # Get the normal direction on the small element.
            # Note, contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(small_direction, contravariant_vectors,
               i_small, j_small, element)

            element_large = neighbor_ids[3, mortar]

            calc_mortar_flux!(fstar, mesh, nonconservative_terms, equations, dt,
               surface_integral, time_discretization, dg, cache,
               mortar, position, element, element_large, normal_direction,
               node)

            i_small += i_small_step
            j_small += j_small_step
         end
      end

      # Buffer to interpolate flux values of the large element to before
      # copying in the correct orientation
      u_buffer = cache.u_threaded[Threads.threadid()]

      mortar_fluxes_to_elements!(surface_flux_values,
         mesh, equations, mortar_l2, dg, cache,
         mortar, fstar, u_buffer)
   end

   return nothing
end


# Inlined version of the mortar flux computation on small elements for conservation laws
@inline function calc_mortar_flux!(fstar,
   mesh::P4estMesh{2},
   nonconservative_terms::False, equations, dt,
   surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache,
   mortar_index, position_index, element_small, element_large,
   normal_direction, node_index)
   @unpack u = cache.mortars
   @unpack U, F, fn_low = cache.lw_mortars
   @unpack surface_flux = surface_integral

   u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, node_index, mortar_index)
   U_ll, U_rr = get_surface_node_vars(U, equations, dg, position_index, node_index, mortar_index)
   f_ll, f_rr = get_surface_node_vars(F, equations, dg, position_index, node_index, mortar_index)
   fn_inner_ll, fn_inner_rr = get_surface_node_vars(fn_low, equations, dg, position_index, node_index, mortar_index)

   fn = surface_flux(u_ll, u_rr, normal_direction, equations)

   Jl = Jr = cache.lw_mortars.inverse_jacobian[node_index, mortar_index]

   Fn = surface_flux(f_ll, f_rr, u_ll, u_rr, U_ll, U_rr, normal_direction, equations)

   alp = compute_alp(u_ll, u_rr, element_small, element_large,
      Jl, Jr, dt, fn, Fn, fn_inner_ll, fn_inner_rr, node_index, equations, dg, dg.volume_integral)

   # Copy flux to buffer
   Trixi.set_node_vars!(fstar[position_index], alp * fn + (1 - alp) * Fn,
      equations, dg, node_index)
end

