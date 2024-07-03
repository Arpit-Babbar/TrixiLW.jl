using Trixi: indices2direction, index_to_start_step_2d, eachmpiinterface, eachnode, get_normal_direction,
             eachvariable, ParallelP4estMesh, get_surface_node_vars, set_node_vars!, mpi_mortar_fluxes_to_elements!

import Trixi: prolong2mpiinterfaces!, calc_mpi_interface_flux!, calc_mpi_mortar_flux!, prolong2mpimortars!
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
    @unpack mpi_interfaces, mpi_interfaces_lw, elements = cache
    @unpack U, F = cache.element_cache
    @unpack contravariant_vectors = elements
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Copy solution data from the local element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_side = mpi_interfaces.local_sides[interface]
        local_element = mpi_interfaces.local_neighbor_ids[interface]
        local_indices = mpi_interfaces.node_indices[interface]

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
            # called. This involves flipping signs to be compatible with the second call of
            # `calc_mpi_interface_flux!``
            if local_side == 1
                f_normal = normal_product(f, equations, normal_direction)
            else # local_side == 2
                f_normal = normal_product(f, equations, -normal_direction)
            end

            for v in eachvariable(equations)
                mpi_interfaces.u[local_side, v, i, interface] = u[v, i_element,
                                                                  j_element,
                                                                  local_element]
                mpi_interfaces_lw.U[local_side, v, i, interface] = U[v, i_element,
                                                                  j_element,
                                                                  local_element]
                mpi_interfaces_lw.F[local_side, v, i, interface] = f_normal[v]
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
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
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
    @unpack u = cache.mpi_interfaces
    @unpack U, F = cache.mpi_interfaces_lw
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

function prolong2mpimortars!(cache, u,
                             mesh::ParallelP4estMesh{2},
                             equations,
                             mortar_l2::LobattoLegendreMortarL2,
                             surface_integral,
                             time_discretization::AbstractLWTimeDiscretization,
                             dg::DGSEM)
    @unpack node_indices, neighbor_ids = cache.mpi_mortars
    @unpack U, F = cache.mpi_mortars_lw
    @unpack contravariant_vectors = cache.elements
    # @unpack mortars, lw_mortars = cache
    @unpack mpi_mortars_lw = cache
    index_range = eachnode(dg)
    @threaded for mortar in eachmpimortar(dg, cache)
        local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

        # Get start value and step size for indices on both sides to get the correct face
        # and orientation
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 3 # -> large element
                # Buffer to copy solution values of the large element in the correct orientation
                # before interpolating
                u_buffer = cache.u_threaded[Threads.threadid()]
                U_buffer = mpi_mortars_lw.tmp.U_threaded[Threads.threadid()]
                F_buffer = mpi_mortars_lw.tmp.F_threaded[Threads.threadid()]
                i_large = i_large_start
                j_large = j_large_start
                for i in eachnode(dg)
                    f = get_flux_vars(F, equations, dg, i_large, j_large, element)
                    # For getting the normal for interfaces, we can multiply 0.5 to normal of
                    # mortar and then flip the direction i.e. multiply by -0.5
                    normal_direction = -0.5 * get_normal_direction(large_direction, contravariant_vectors,
                                                                   i_large, j_large, element)
                    u_node = Trixi.get_node_vars(u, equations, dg, i_large, j_large, element)
                    fn_node = normal_product(f, equations, normal_direction)
                    fn_node = Trixi.flux(u_node, normal_direction, equations)
                    for v in eachvariable(equations)
                        u_buffer[v, i] = u[v, i_large, j_large, element]
                        U_buffer[v, i] = U[v, i_large, j_large, element]
                        F_buffer[v, i] = fn_node[v]            
                    end
                    i_large += i_large_step
                    j_large += j_large_step
                end

                # Interpolate large element face data from buffer to small face locations
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 1, :, mortar),
                                        mortar_l2.forward_lower,
                                        u_buffer)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 2, :, mortar),
                                        mortar_l2.forward_upper,
                                        u_buffer)

                multiply_dimensionwise!(view(cache.mpi_mortars_lw.U, 2, :, 1, :, mortar),
                                        mortar_l2.forward_lower,
                                        U_buffer)
                multiply_dimensionwise!(view(cache.mpi_mortars_lw.U, 2, :, 2, :, mortar),
                                        mortar_l2.forward_upper,
                                        U_buffer)

                multiply_dimensionwise!(view(cache.mpi_mortars_lw.F, 2, :, 1, :, mortar),
                                        mortar_l2.forward_lower,
                                        F_buffer)
                multiply_dimensionwise!(view(cache.mpi_mortars_lw.F, 2, :, 2, :, mortar),
                                        mortar_l2.forward_upper,
                                        F_buffer)
            else # position in (1, 2) -> small element
                # Copy solution data from the small elements
                i_small = i_small_start
                j_small = j_small_start
                for i in eachnode(dg)
                    f = get_flux_vars(F, equations, dg, i_small, j_small, element)
                    normal_direction = get_normal_direction(small_direction, contravariant_vectors,
                                                            i_small, j_small, element)
                    fn_node = normal_product(f, equations, normal_direction)

                    for v in eachvariable(equations)
                        cache.mpi_mortars.u[1, v, position, i, mortar] = u[v, i_small,
                                                                           j_small,
                                                                           element]
                        mpi_mortars_lw.U[1, v, position, i, mortar] = U[v, i_small,
                                                                    j_small,
                                                                    element]
                        mpi_mortars_lw.F[1, v, position, i, mortar] = fn_node[v]
                    end
                    i_small += i_small_step
                    j_small += j_small_step
                end
            end
        end
    end

    return nothing

end

function calc_mpi_mortar_flux!(surface_flux_values,
                               mesh::ParallelP4estMesh{2},
                               nonconservative_terms, equations,
                               mortar_l2::LobattoLegendreMortarL2,
                               surface_integral,
                               time_discretization::AbstractLWTimeDiscretization,
                               dg::DG, cache)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_upper_threaded, fstar_lower_threaded = cache
    index_range = eachnode(dg)

    @threaded for mortar in eachmpimortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar = (fstar_lower_threaded[Threads.threadid()],
                 fstar_upper_threaded[Threads.threadid()])

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            for node in eachnode(dg)
                # Get the normal direction on the small element.
                normal_direction = get_normal_direction(cache.mpi_mortars, node,
                                                        position, mortar)

                calc_mpi_mortar_flux!(fstar, mesh, nonconservative_terms, equations,
                                      surface_integral, time_discretization, dg, cache,
                                      mortar, position, normal_direction,
                                      node)

                i_small += i_small_step
                j_small += j_small_step
            end
        end

        # Buffer to interpolate flux values of the large element to before
        # copying in the correct orientation
        u_buffer = cache.u_threaded[Threads.threadid()]
        U_buffer = cache.mpi_mortars_lw.tmp.U_threaded[Threads.threadid()]
        F_buffer = cache.mpi_mortars_lw.tmp.F_threaded[Threads.threadid()]

        mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                       mesh, equations, mortar_l2,
                                       time_discretization, dg, cache,
                                       mortar, fstar, u_buffer, U_buffer,
                                       F_buffer)
    end

    return nothing
end

# Inlined version of the mortar flux computation on small elements for conservation laws
@inline function calc_mpi_mortar_flux!(fstar,
                                       mesh::ParallelP4estMesh{2},
                                       nonconservative_terms::False, equations,
                                       surface_integral,
                                       time_discretization::AbstractLWTimeDiscretization,
                                       dg::DG, cache,
                                       mortar_index, position_index, normal_direction,
                                       node_index)
    @unpack u = cache.mpi_mortars
    @unpack U, F = cache.mpi_mortars_lw
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, node_index,
                                       mortar_index)
    U_ll, U_rr = get_surface_node_vars(U, equations, dg, position_index, node_index,
                                       mortar_index)
    F_ll, F_rr = get_surface_node_vars(F, equations, dg, position_index, node_index,
                                       mortar_index)

    flux = surface_flux(F_ll, F_rr, u_ll, u_rr, U_ll, U_rr, normal_direction, equations)

    # Copy flux to buffer
    set_node_vars!(fstar[position_index], flux, equations, dg, node_index)
end

end #muladd
