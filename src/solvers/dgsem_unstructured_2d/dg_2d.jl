using Trixi: UnstructuredMesh2D, P4estMesh, calc_surface_integral!, apply_jacobian!,
   reset_du!, get_surface_normal, get_one_sided_surface_node_vars, False

using TrixiLW: calc_volume_integral! # defined in dgsem_structured


import Trixi: prolong2interfaces!, calc_interface_flux!,
   prolong2boundaries!, calc_boundary_flux!

using MuladdMacro

@muladd begin

function rhs!(du, u, t,
   mesh::UnstructuredMesh2D, equations,
   initial_condition, boundary_conditions, source_terms,
   dg::DG, time_discretization::AbstractLWTimeDiscretization, cache, tolerances::NamedTuple)
   # Reset du
   @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

   dt = cache.dt[1]

   # Calculate volume integral
   alpha = @trixi_timeit timer() "volume integral" calc_volume_integral!(
      du, u, t, dt, tolerances, mesh,
      have_nonconservative_terms(equations), source_terms, equations,
      dg.volume_integral, time_discretization,
      dg, cache)

   # Prolong solution to interfaces
   @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
      cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)

   # Calculate interface fluxes
   @trixi_timeit timer() "interface flux" calc_interface_flux!(
      cache.elements.surface_flux_values, dt, mesh,
      have_nonconservative_terms(equations), equations,
      dg.surface_integral, time_discretization, alpha, dg, cache)

   # Prolong solution to boundaries
   @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
      cache, u, mesh, equations, dg.surface_integral,
      time_discretization, dg)

   # Calculate boundary fluxes
   @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
      cache, t, boundary_conditions, mesh, equations, dg.surface_integral,
      time_discretization, dg)

   # Calculate surface integrals
   @trixi_timeit timer() "surface integral" calc_surface_integral!(
      du, u, mesh, equations, dg.surface_integral, dg, cache)

   # Apply Jacobian from mapping to reference element
   #  Note! this routine is reused from dg_curved/dg_2d.jl
   @trixi_timeit timer() "Jacobian" apply_jacobian!(
      du, mesh, equations, dg, cache)

   return nothing
end

function prolong2interfaces!(cache, u,
   mesh::UnstructuredMesh2D,
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @unpack interfaces, interface_cache = cache
   @unpack U, F, fn_low = cache.element_cache
   @unpack start_index, index_increment = interfaces
   @unpack normal_directions = cache.elements

   @threaded for interface in eachinterface(dg, cache)
      primary_element   = interfaces.element_ids[1, interface]
      secondary_element = interfaces.element_ids[2, interface]

      primary_side   = interfaces.element_side_ids[1, interface]
      secondary_side = interfaces.element_side_ids[2, interface]

      for v in eachvariable(equations), i in eachnode(dg)
         interface_cache.fn_low[1, v, i, interface] = fn_low[v, i, primary_side, primary_element]
         interface_cache.fn_low[2, v, i, interface] = fn_low[v, i, secondary_side, secondary_element]
      end

      if primary_side == 1
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, i, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, i, 1, primary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, i, 1, primary_element)
            f = get_flux_vars(F, equations, dg, i, 1, primary_element)
            fn_node = normal_product(f, equations, outward_direction)
            # f_node = Trixi.flux(u_node, outward_direction, equations)
            # @assert fn_node ≈ f_node f_node,fn_node
            # fn_node = Trixi.flux(u_node, outward_direction, equations)
            for v in eachvariable(equations)
               interface_cache.u[1, v, i, interface] = u_node[v]
               interface_cache.U[1, v, i, interface] = U_node[v]
               interface_cache.f[1, v, i, interface] = fn_node[v]
            end
         end
      elseif primary_side == 2
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, i, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, nnodes(dg), i, primary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, nnodes(dg), i, primary_element)
            f = get_flux_vars(F, equations, dg, nnodes(dg), i, primary_element)
            fn_node = normal_product(f, equations, outward_direction)
            # f_node = Trixi.flux(u_node, outward_direction, equations)
            # @assert fn_node ≈ f_node f_node,fn_node
            # fn_node = Trixi.flux(u_node, outward_direction, equations)
            for v in eachvariable(equations)
               interface_cache.u[1, v, i, interface] = u_node[v]
               interface_cache.U[1, v, i, interface] = U_node[v]
               interface_cache.f[1, v, i, interface] = fn_node[v]
            end
         end
      elseif primary_side == 3
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, i, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, i, nnodes(dg), primary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, i, nnodes(dg), primary_element)
            f = get_flux_vars(F, equations, dg, i, nnodes(dg), primary_element)
            fn_node = normal_product(f, equations, outward_direction)
            for v in eachvariable(equations)
               interface_cache.u[1, v, i, interface] = u_node[v]
               interface_cache.U[1, v, i, interface] = U_node[v]
               interface_cache.f[1, v, i, interface] = fn_node[v]
            end
         end
      else # primary_side == 4
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, i, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, 1, i, primary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, 1, i, primary_element)
            f = get_flux_vars(F, equations, dg, 1, i, primary_element)
            fn_node = normal_product(f, equations, outward_direction)
            for v in eachvariable(equations)
               interface_cache.u[1, v, i, interface] = u_node[v]
               interface_cache.U[1, v, i, interface] = U_node[v]
               interface_cache.f[1, v, i, interface] = fn_node[v]
            end
         end
      end

      secondary_index = start_index[interface]
      if secondary_side == 1
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, secondary_index, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, i, 1, secondary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, i, 1, secondary_element)
            f = get_flux_vars(F, equations, dg, i, 1, secondary_element)
            fn_node = normal_product(f, equations, outward_direction)
            for v in eachvariable(equations)
               interface_cache.u[2, v, i, interface] = u_node[v]
               interface_cache.U[2, v, i, interface] = U_node[v]
               interface_cache.f[2, v, i, interface] = fn_node[v]
            end
            secondary_index += index_increment[interface]
         end
      elseif secondary_side == 2
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, secondary_index, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, nnodes(dg), i, secondary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, nnodes(dg), i, secondary_element)
            f = get_flux_vars(F, equations, dg, nnodes(dg), i, secondary_element)
            fn_node = normal_product(f, equations, outward_direction)
            for v in eachvariable(equations)
               interface_cache.u[2, v, i, interface] = u_node[v]
               interface_cache.U[2, v, i, interface] = U_node[v]
               interface_cache.f[2, v, i, interface] = fn_node[v]
            end
            secondary_index += index_increment[interface]
         end
      elseif secondary_side == 3
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, secondary_index, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, i, nnodes(dg), secondary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, i, nnodes(dg), secondary_element)
            f = get_flux_vars(F, equations, dg, i, nnodes(dg), secondary_element)
            fn_node = normal_product(f, equations, outward_direction)
            for v in eachvariable(equations)
               interface_cache.u[2, v, i, interface] = u_node[v]
               interface_cache.U[2, v, i, interface] = U_node[v]
               interface_cache.f[2, v, i, interface] = fn_node[v]
            end
            secondary_index += index_increment[interface]
         end
      else # secondary_side == 4
         for i in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, secondary_index, primary_side,
               primary_element)
            u_node = Trixi.get_node_vars(u, equations, dg, 1, i, secondary_element)
            U_node = Trixi.get_node_vars(U, equations, dg, 1, i, secondary_element)
            f = get_flux_vars(F, equations, dg, 1, i, secondary_element)
            fn_node = normal_product(f, equations, outward_direction)
            for v in eachvariable(equations)
               interface_cache.u[2, v, i, interface] = u_node[v]
               interface_cache.U[2, v, i, interface] = U_node[v]
               interface_cache.f[2, v, i, interface] = fn_node[v]
            end
            secondary_index += index_increment[interface]
         end
      end
   end

   return nothing
end

function calc_interface_flux!(surface_flux_values, dt,
   mesh::UnstructuredMesh2D,
   nonconservative_terms::False, equations,
   surface_integral, time_discretization::AbstractLWTimeDiscretization, alpha, dg::DG, cache)
   @unpack surface_flux = surface_integral
   @unpack interface_cache = cache
   @unpack u, U, f, fn_low = interface_cache
   @unpack start_index, index_increment, element_ids, element_side_ids = cache.interfaces
   @unpack normal_directions = cache.elements

   @threaded for interface in eachinterface(dg, cache)
      # Get neighboring elements
      primary_element = element_ids[1, interface]
      secondary_element = element_ids[2, interface]

      # Get the local side id on which to compute the flux
      primary_side = element_side_ids[1, interface]
      secondary_side = element_side_ids[2, interface]

      # initial index for the coordinate system on the secondary element
      secondary_index = start_index[interface]

      # loop through the primary element coordinate system and compute the interface coupling
      for primary_index in eachnode(dg)
         # pull the primary and secondary states from the boundary u values
         f_ll = get_one_sided_surface_node_vars(f, equations, dg, 1, primary_index, interface)
         ual = get_one_sided_surface_node_vars(u, equations, dg, 1, primary_index, interface)
         u_ll = get_one_sided_surface_node_vars(U, equations, dg, 1, primary_index, interface)
         fn_inner_ll = get_one_sided_surface_node_vars(fn_low, equations, dg, 1, primary_index, interface)

         f_rr = get_one_sided_surface_node_vars(f, equations, dg, 2, secondary_index, interface)
         uar = get_one_sided_surface_node_vars(u, equations, dg, 2, secondary_index, interface)
         u_rr = get_one_sided_surface_node_vars(U, equations, dg, 2, secondary_index, interface)
         fn_inner_rr = get_one_sided_surface_node_vars(fn_low, equations, dg, 2, secondary_index, interface)

         # pull the outward pointing (normal) directional vector
         #   Note! this assumes a conforming approximation, more must be done in terms of the normals
         #         for hanging nodes and other non-conforming approximation spaces
         outward_direction = get_surface_normal(normal_directions, primary_index, primary_side,
            primary_element)

         # Call pointwise numerical flux with rotation. Direction is normalized inside this function
         # flux = surface_flux(f_ll, f_rr, u_ll, u_rr, outward_direction, equations)
         Fn = surface_flux(f_ll, f_rr, ual, uar, u_ll, u_rr, outward_direction,
            equations)

         fn = surface_flux(ual, uar, outward_direction, equations)

         Jl = Jr = interface_cache.inverse_jacobian[primary_index, interface]
         alp = compute_alp(u_ll, u_rr, primary_element, secondary_element,
            Jl, Jr, dt,
            fn, Fn, fn_inner_ll, fn_inner_rr, primary_index,
            equations, dg, dg.volume_integral, mesh)
         # Copy flux back to primary/secondary element storage
         # Note the sign change for the normal flux in the secondary element!
         for v in eachvariable(equations)
            surface_flux_values[v, primary_index, primary_side, primary_element] = (
               alp * fn[v] + (1.0 - alp) * Fn[v])
            surface_flux_values[v, secondary_index, secondary_side, secondary_element] = -(
               alp * fn[v] + (1.0 - alp) * Fn[v])
         end

         # increment the index of the coordinate system in the secondary element
         secondary_index += index_increment[interface]
      end
   end

   return nothing
end

# move the approximate solution onto physical boundaries within a "right-handed" element
function prolong2boundaries!(cache, u,
   mesh::UnstructuredMesh2D,
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @unpack boundaries = cache
   @unpack normal_directions = cache.elements
   @unpack U, F = cache.element_cache

   @threaded for boundary in eachboundary(dg, cache)
      element = boundaries.element_id[boundary]
      side = boundaries.element_side_id[boundary]

      if side == 1
         for node_index in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, node_index,
               side, element)
            u_node = get_node_vars(U, equations, dg, node_index, 1, element)
            f_node = get_flux_vars(F, equations, dg, node_index, 1, element)
            fn_node = normal_product(f_node, equations, outward_direction)
            # f_node = Trixi.flux(u_node, outward_direction, equations)
            for v in eachvariable(equations)
               boundaries.u[v, node_index, boundary] = u_node[v]
               boundaries.f[v, node_index, boundary] = fn_node[v]
            end
         end
      elseif side == 2
         for node_index in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, node_index,
               side, element)
            u_node = get_node_vars(U, equations, dg, nnodes(dg), node_index, element)
            f_node = get_flux_vars(F, equations, dg, nnodes(dg), node_index, element)
            fn_node = normal_product(f_node, equations, outward_direction)
            # f_node = Trixi.flux(u_node, outward_direction, equations)
            for v in eachvariable(equations)
               boundaries.u[v, node_index, boundary] = u_node[v]
               boundaries.f[v, node_index, boundary] = fn_node[v]
            end
         end
      elseif side == 3
         for node_index in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, node_index,
               side, element)
            u_node = get_node_vars(U, equations, dg, node_index, nnodes(dg), element)
            f_node = get_flux_vars(F, equations, dg, node_index, nnodes(dg), element)
            fn_node = normal_product(f_node, equations, outward_direction)
            # f_node = Trixi.flux(u_node, outward_direction, equations)
            for v in eachvariable(equations)
               boundaries.u[v, node_index, boundary] = u_node[v]
               boundaries.f[v, node_index, boundary] = fn_node[v]
            end
         end
      else # side == 4
         for node_index in eachnode(dg)
            outward_direction = get_surface_normal(normal_directions, node_index,
               side, element)
            u_node = get_node_vars(U, equations, dg, 1, node_index, element)
            f_node = get_flux_vars(F, equations, dg, 1, node_index, element)
            fn_node = normal_product(f_node, equations, outward_direction)
            # f_node = Trixi.flux(u_node, outward_direction, equations)
            for v in eachvariable(equations)
               boundaries.u[v, node_index, boundary] = u_node[v]
               boundaries.f[v, node_index, boundary] = fn_node[v]
            end
         end
      end
   end

   return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
   mesh::Union{UnstructuredMesh2D,P4estMesh},
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @assert isempty(eachboundary(dg, cache))
end


# Function barrier for type stability
function calc_boundary_flux!(cache, t, boundary_conditions,
   mesh::Union{UnstructuredMesh2D,P4estMesh},
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @unpack boundary_condition_types, boundary_indices = boundary_conditions

   calc_boundary_flux_by_type!(cache, t, boundary_condition_types, boundary_indices,
      mesh, equations, surface_integral, time_discretization, dg)
   return nothing
end


# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type!(cache, t, BCs::NTuple{N,Any},
   BC_indices::NTuple{N,Vector{Int}},
   mesh::Union{UnstructuredMesh2D,P4estMesh},
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG) where {N}
   # Extract the boundary condition type and index vector
   boundary_condition = first(BCs)
   boundary_condition_indices = first(BC_indices)
   # Extract the remaining types and indices to be processed later
   remaining_boundary_conditions = Base.tail(BCs)
   remaining_boundary_condition_indices = Base.tail(BC_indices)

   # process the first boundary condition type
   calc_boundary_flux!(cache, t, boundary_condition, boundary_condition_indices,
      mesh, equations, surface_integral, time_discretization, dg)

   # recursively call this method with the unprocessed boundary types
   calc_boundary_flux_by_type!(cache, t, remaining_boundary_conditions,
      remaining_boundary_condition_indices,
      mesh, equations, surface_integral, time_discretization, dg)

   return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(cache, t, BCs::Tuple{}, BC_indices::Tuple{},
   mesh::Union{UnstructuredMesh2D,P4estMesh},
   equations, surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   nothing
end


function calc_boundary_flux!(cache, t, boundary_condition, boundary_indexing,
   mesh::UnstructuredMesh2D, equations,
   surface_integral, time_discretization::AbstractLWTimeDiscretization, dg::DG)
   @unpack surface_flux_values = cache.elements
   @unpack element_id, element_side_id = cache.boundaries

   @threaded for local_index in eachindex(boundary_indexing)
      # use the local index to get the global boundary index from the pre-sorted list
      boundary = boundary_indexing[local_index]

      # get the element and side IDs on the boundary element
      element = element_id[boundary]
      side = element_side_id[boundary]

      # calc boundary flux on the current boundary interface
      for node in eachnode(dg)
         calc_boundary_flux!(surface_flux_values, t, boundary_condition,
            mesh, have_nonconservative_terms(equations),
            equations, surface_integral, time_discretization, dg, cache,
            node, side, element, boundary)
      end
   end
end

# inlined version of the boundary flux calculation along a physical interface where the
# boundary flux values are set according to a particular `boundary_condition` function
@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
   mesh::UnstructuredMesh2D,
   nonconservative_terms::False, equations,
   surface_integral, dt, time_discretization::AbstractLWTimeDiscretization, dg::DG, cache,
   node_index, side_index, element_index, boundary_index)
   @unpack normal_directions = cache.elements
   @unpack u, f, node_coordinates = cache.boundaries
   @unpack surface_flux = surface_integral

   # pull the inner solution state from the boundary u values on the boundary element
   u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

   # pull the outward pointing (normal) directional vector
   outward_direction = get_surface_normal(normal_directions, node_index, side_index, element_index)
   # f_inner = Trixi.flux(u_inner, outward_direction, equations)
   f_inner = get_node_vars(f, equations, dg, node_index, boundary_index)

   # get the external solution values from the prescribed external state
   x = get_node_coords(node_coordinates, equations, dg, node_index, boundary_index)

   # Call pointwise numerical flux function in the normal direction on the boundary
   flux = boundary_condition(u_inner, f_inner, outward_direction, x, t, dt,
      surface_flux, equations, dg, dg.time_discretization)

   for v in eachvariable(equations)
      surface_flux_values[v, node_index, side_index, element_index] = flux[v]
   end
end

end # muladd