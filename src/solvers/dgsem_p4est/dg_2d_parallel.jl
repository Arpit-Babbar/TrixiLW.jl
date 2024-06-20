using Trixi: indices2direction, index_to_start_step_2d, eachmpiinterface, eachnode, get_normal_direction,
             eachvariable, ParallelP4estMesh, get_surface_node_vars

import Trixi: prolong2mpiinterfaces!, calc_mpi_interface_flux!
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function prolong2mpiinterfaces!(cache, u,
                                mesh::ParallelP4estMesh{2},
                                equations, surface_integral,
                                time_discretization::AbstractLWTimeDiscretization,
                                dg::DG)
    @unpack mpi_interfaceslw, elements = cache
    @unpack U, F = cache.element_cache
    @unpack contravariant_vectors = elements
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Copy solution data from the local element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_side = mpi_interfaceslw.local_sides[interface]
        local_element = mpi_interfaceslw.local_neighbor_ids[interface]
        local_indices = mpi_interfaceslw.node_indices[interface]

        local_direction = indices2direction(local_indices)

        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start
        for i in eachnode(dg)
            # Get the normal direction on the local element. The above `local_indices`
            # will take care of giving us the outward unit normal. The main point is that
            # this is the normal_direction used in `calc_mpi_interface_flux!` function.
            normal_direction = get_normal_direction(local_direction, contravariant_vectors,
            i_element, j_element, local_element)
            f = get_flux_vars(F, equations, dg, i_element, j_element, local_element)

            # The flux should be in the same normal direction when the surface flux function is
            # called. This involves flipping signs to be combatible with the second call of
            # `calc_mpi_interface_flux!``
            if local_side == 1
                f_normal = normal_product(f, equations, normal_direction)
            else # local_side == 2
                f_normal = normal_product(f, equations, -normal_direction)
            end

            for v in eachvariable(equations)
                mpi_interfaceslw.mpi_interfaces_.u[local_side, v, i, interface] = u[v, i_element,
                                                                                  j_element,
                                                                                  local_element]
                mpi_interfaceslw.U[local_side, v, i, interface] = U[v, i_element,
                                                                  j_element,
                                                                  local_element]
                mpi_interfaceslw.F[local_side, v, i, interface] = f_normal[v]
            end
            i_element += i_element_step
            j_element += j_element_step
        end
    end

    return nothing
end

function calc_mpi_interface_flux!(surface_flux_values, mesh::ParallelP4estMesh,
                                  nonconservative_terms, equations, surface_integral,
                                  time_discretization::AbstractLWTimeDiscretization,
                                  dg::DG, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaceslw
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)
    index_end = last(index_range)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Get element and side index information on the local element
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        # Create the local i,j indexing on the local element used to pull normal direction information
        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start
        # Initiate the node index to be used in the surface for loop,
        # the surface flux storage must be indexed in alignment with the local element indexing
        if :i_backward in local_indices
            surface_node = index_end
            surface_node_step = -1
        else
            surface_node = 1
            surface_node_step = 1
        end

        for node in eachnode(dg)
            # Get the normal direction on the local element
            # Contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(local_direction,
                                                    contravariant_vectors,
                                                    i_element, j_element, local_element)

            calc_mpi_interface_flux!(surface_flux_values, mesh, nonconservative_terms,
                                     equations, surface_integral,
                                     time_discretization, dg, cache,
                                     interface, normal_direction,
                                     node, local_side,
                                     surface_node, local_direction, local_element)

            # Increment local element indices to pull the normal direction
            i_element += i_element_step
            j_element += j_element_step

            # Increment the surface node index along the local element
            surface_node += surface_node_step
        end
    end

    return nothing
end

# Inlined version of the interface flux computation for conservation laws
@inline function calc_mpi_interface_flux!(surface_flux_values,
                                          mesh::ParallelP4estMesh{2},
                                          nonconservative_terms::False, equations,
                                          surface_integral,
                                          time_discretization::AbstractLWTimeDiscretization,
                                          dg::DG, cache,
                                          interface_index, normal_direction,
                                          interface_node_index, local_side,
                                          surface_node_index, local_direction_index,
                                          local_element_index)
    @unpack u = cache.mpi_interfaceslw.mpi_interfaces_
    @unpack U, F = cache.mpi_interfaceslw
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, interface_node_index,
                                       interface_index)
    U_ll, U_rr = get_surface_node_vars(U, equations, dg, interface_node_index,
                                       interface_index)
    F_ll, F_rr = get_surface_node_vars(F, equations, dg, interface_node_index,
                                       interface_index)

    if local_side == 1
        flux_ = surface_flux(F_ll, F_rr, u_ll, u_rr, U_ll, U_rr, normal_direction, equations)
    else # local_side == 2
        flux_ = -surface_flux(F_ll, F_rr, u_ll, u_rr, U_ll, U_rr, -normal_direction, equations)
    end

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_index, local_direction_index, local_element_index] = flux_[v]
    end
end

# TODO: code `prolong2mpimortars!`
# TODO: code `calc_mpi_mortar_flux!`
end #muladd
