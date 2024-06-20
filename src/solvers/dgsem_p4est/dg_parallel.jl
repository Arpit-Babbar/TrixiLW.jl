using Trixi: balance!, partition!, update_ghost_layer!, init_elements,
             mpi_nranks, mpi_rank, nmpiinterfaces, init_mpi_neighbor_connectivity,
             InitNeighborRankConnectivityIterFaceUserData, exchange_normal_directions!,
             init_interfaces, init_boundaries, init_mpi_mortars, init_mortars,
             finish_mpi_send!, start_mpi_receive!, start_mpi_send!

import Trixi: init_mpi_cache, init_mpi_cache!, init_mpi_interfaces, init_mpi_cache,
              start_mpi_send!, finish_mpi_receive!, create_cache
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function start_mpi_send!(mpi_cache::P4estMPICache, mesh, equations,
                         time_discretization::AbstractLWTimeDiscretization,
                         dg, cache)
    # @assert false "start_mpi_send!"
    lw_data_size_factor = 3 # LW requires thrice the amount of data transfer than RK
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    for d in 1:length(mpi_cache.mpi_neighbor_ranks)
        send_buffer = mpi_cache.mpi_send_buffers[d]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
            first1 = lw_data_size_factor * (index - 1) * data_size + 1        # for u
            last1 = first1 + data_size - 1

            first2 = last1 + 1      # for U
            last2 =  first2 + data_size - 1

            first3 = last2 + 1      # for F
            last3 = first3 + data_size - 1

            local_side = cache.mpi_interfaces.local_sides[interface]
            @views send_buffer[first1:last1] .= vec(cache.mpi_interfaceslw.mpi_interfaces_.u[local_side, ..,
                                                                         interface])
            @views send_buffer[first2:last2] .= vec(cache.mpi_interfaceslw.U[local_side, ..,
                                                                         interface])
            @views send_buffer[first3:last3] .= vec(cache.mpi_interfaceslw.F[local_side, ..,
                                                                         interface])
        end
        # TODO: Mortar code here
    end
    # Start sending
    for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_send_requests[index] = MPI.Isend(mpi_cache.mpi_send_buffers[index],
                                                       d, mpi_rank(), mpi_comm())
    end

    return nothing
end

function finish_mpi_receive!(mpi_cache::P4estMPICache, mesh, equations,
                             time_discretization::AbstractLWTimeDiscretization,
                             dg, cache)
    # @assert false "Finish"
    lw_data_size_factor = 3 # LW requires thrice the amount of data transfer than RK
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    # Start receiving and unpack received data until all communication is finished
    d = MPI.Waitany(mpi_cache.mpi_recv_requests)
    while d !== nothing
        recv_buffer = mpi_cache.mpi_recv_buffers[d]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
            first1 = lw_data_size_factor * (index - 1) * data_size + 1        # for u
            last1 = first1 + data_size - 1

            first2 = last1 + 1      # for U
            last2 =  first2 + data_size - 1

            first3 = last2 + 1      # for F
            last3 = first3 + data_size - 1

            if cache.mpi_interfaces.local_sides[interface] == 1 # local element on primary side
                @views vec(cache.mpi_interfaceslw.mpi_interfaces_.u[2, .., interface]) .= recv_buffer[first1:last1]
                @views vec(cache.mpi_interfaceslw.U[2, .., interface]) .= recv_buffer[first2:last2]
                @views vec(cache.mpi_interfaceslw.F[2, .., interface]) .= recv_buffer[first3:last3]
            else # local element at secondary mpi_interfaceslw.side
                @views vec(cache.mpi_interfaceslw.mpi_interfaces_.u[1, .., interface]) .= recv_buffer[first1:last1]
                @views vec(cache.mpi_interfaceslw.U[1, .., interface]) .= recv_buffer[first2:last2]
                @views vec(cache.mpi_interfaceslw.F[1, .., interface]) .= recv_buffer[first3:last3]
            end
        end
        d = MPI.Waitany(mpi_cache.mpi_recv_requests)
    end

    return nothing
end


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::ParallelP4estMesh, equations,
                      time_discretization::AbstractLWTimeDiscretization, dg::DG,
                      RealT, ::Type{uEltype}, cache) where {uEltype <: Real}
    # Make sure to balance and partition the p4est and create a new ghost layer before creating any
    # containers in case someone has tampered with the p4est after creating the mesh
    cache = create_cache_serial(mesh, equations, time_discretization, dg, RealT, uEltype, cache)
    # @assert false "I am called9887"
    balance!(mesh)
    partition!(mesh)
    update_ghost_layer!(mesh)

    elements = init_elements(mesh, equations, dg.basis, uEltype)

    mpi_interfaceslw = init_mpi_interfaces(mesh, equations, dg.basis, time_discretization, elements)
    mpi_mortars = init_mpi_mortars(mesh, equations, dg.basis, elements)
    # @assert false mpi_interfaces
    mpi_cache = init_mpi_cache(mesh, mpi_interfaceslw, mpi_mortars,
                               nvariables(equations), nnodes(dg),
                               time_discretization, uEltype)

    exchange_normal_directions!(mpi_mortars, mpi_cache, mesh, nnodes(dg))

    interfaces = init_interfaces(mesh, equations, dg.basis, elements)
    boundaries = init_boundaries(mesh, equations, dg.basis, elements)
    mortars = init_mortars(mesh, equations, dg.basis, elements)

    cache = (; cache..., elements, interfaces, mpi_interfaceslw, boundaries, mortars, mpi_mortars,
             mpi_cache)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, time_discretization, dg, uEltype)...)
    cache = (; cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

    return cache
end

function init_mpi_cache(mesh::ParallelP4estMesh, mpi_interfaceslw, mpi_mortars, nvars,
                        nnodes, time_discretization::AbstractLWTimeDiscretization,
                        uEltype)
    mpi_cache = P4estMPICache(uEltype)
    init_mpi_cache!(mpi_cache, mesh, mpi_interfaceslw, mpi_mortars, nvars, nnodes, time_discretization,
                    uEltype)

    return mpi_cache
end

function init_mpi_cache!(mpi_cache::P4estMPICache, mesh::ParallelP4estMesh,
                         mpi_interfaceslw, mpi_mortars, nvars, n_nodes,
                         time_discretization::AbstractLWTimeDiscretization,
                         uEltype)
    mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars = init_mpi_neighbor_connectivity(mpi_interfaceslw,
                                                                                                       mpi_mortars,
                                                                                                       mesh)

    mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests = init_mpi_data_structures(mpi_neighbor_interfaces,
                                                                                                        mpi_neighbor_mortars,
                                                                                                        ndims(mesh),
                                                                                                        nvars,
                                                                                                        n_nodes,
                                                                                                        uEltype,
                                                                                                        time_discretization)

    # Determine local and total number of elements
    n_elements_global = Int(mesh.p4est.global_num_quadrants[])
    n_elements_by_rank = vcat(Int.(unsafe_wrap(Array, mesh.p4est.global_first_quadrant,
                                               mpi_nranks())),
                              n_elements_global) |> diff # diff sufficient due to 0-based quad indices
    n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))
    # Account for 1-based indexing in Julia
    first_element_global_id = Int(mesh.p4est.global_first_quadrant[mpi_rank() + 1]) + 1
    @assert n_elements_global==sum(n_elements_by_rank) "error in total number of elements"

    # TODO reuse existing structures
    @pack! mpi_cache = mpi_neighbor_ranks, mpi_neighbor_interfaces,
                       mpi_neighbor_mortars,
                       mpi_send_buffers, mpi_recv_buffers,
                       mpi_send_requests, mpi_recv_requests,
                       n_elements_by_rank, n_elements_global,
                       first_element_global_id
end

# # Function barrier for type stability
# function init_neighbor_rank_connectivity_iter_face_inner(info, user_data)
#     @unpack interfaces, interface_id, global_interface_ids, neighbor_ranks_interface,
#     mortars, mortar_id, global_mortar_ids, neighbor_ranks_mortar, mesh = user_data

#     info_pw = PointerWrapper(info)
#     # Get the global interface/mortar ids and neighbor rank if current face belongs to an MPI
#     # interface/mortar
#     if info_pw.sides.elem_count[] == 2 # MPI interfaces/mortars have two neighboring elements
#         # Extract surface data
#         sides_pw = (load_pointerwrapper_side(info_pw, 1),
#                     load_pointerwrapper_side(info_pw, 2))

#         if sides_pw[1].is_hanging[] == false && sides_pw[2].is_hanging[] == false # No hanging nodes for MPI interfaces
#             if sides_pw[1].is.full.is_ghost[] == true
#                 remote_side = 1
#                 local_side = 2
#             elseif sides_pw[2].is.full.is_ghost[] == true
#                 remote_side = 2
#                 local_side = 1
#             else # both sides are on this rank -> skip since it's a regular interface
#                 return nothing
#             end

#             # Sanity check, current face should belong to current MPI interface
#             local_tree_pw = load_pointerwrapper_tree(mesh.p4est,
#                                                      sides_pw[local_side].treeid[] + 1) # one-based indexing
#             local_quad_id = local_tree_pw.quadrants_offset[] +
#                             sides_pw[local_side].is.full.quadid[]
#             @assert interfaces.local_neighbor_ids[interface_id] == local_quad_id + 1 # one-based indexing

#             # Get neighbor ID from ghost layer
#             proc_offsets = unsafe_wrap(Array,
#                                        info_pw.ghost_layer.proc_offsets,
#                                        mpi_nranks() + 1)
#             ghost_id = sides_pw[remote_side].is.full.quadid[] # indexes the ghost layer, 0-based
#             neighbor_rank = findfirst(r -> proc_offsets[r] <= ghost_id <
#                                            proc_offsets[r + 1],
#                                       1:mpi_nranks()) - 1 # MPI ranks are 0-based
#             neighbor_ranks_interface[interface_id] = neighbor_rank

#             # Global interface id is the globally unique quadrant id of the quadrant on the primary
#             # side (1) multiplied by the number of faces per quadrant plus face
#             if local_side == 1
#                 offset = mesh.p4est.global_first_quadrant[mpi_rank() + 1] # one-based indexing
#                 primary_quad_id = offset + local_quad_id
#             else
#                 offset = mesh.p4est.global_first_quadrant[neighbor_rank + 1] # one-based indexing
#                 primary_quad_id = offset + sides_pw[1].is.full.quad.p.piggy3.local_num[]
#             end
#             global_interface_id = 2 * ndims(mesh) * primary_quad_id + sides_pw[1].face[]
#             global_interface_ids[interface_id] = global_interface_id

#             user_data.interface_id += 1
#         else # hanging node
#             if sides_pw[1].is_hanging[] == true
#                 hanging_side = 1
#                 full_side = 2
#             else
#                 hanging_side = 2
#                 full_side = 1
#             end
#             # Verify before accessing is.full / is.hanging
#             @assert sides_pw[hanging_side].is_hanging[] == true &&
#                     sides_pw[full_side].is_hanging[] == false

#             # If all quadrants are locally available, this is a regular mortar -> skip
#             if sides_pw[full_side].is.full.is_ghost[] == false &&
#                all(sides_pw[hanging_side].is.hanging.is_ghost[] .== false)
#                 return nothing
#             end

#             trees_pw = (load_pointerwrapper_tree(mesh.p4est, sides_pw[1].treeid[] + 1),
#                         load_pointerwrapper_tree(mesh.p4est, sides_pw[2].treeid[] + 1))

#             # Find small quads that are remote and determine which rank owns them
#             remote_small_quad_positions = findall(sides_pw[hanging_side].is.hanging.is_ghost[] .==
#                                                   true)
#             proc_offsets = unsafe_wrap(Array,
#                                        info_pw.ghost_layer.proc_offsets,
#                                        mpi_nranks() + 1)
#             # indices of small remote quads inside the ghost layer, 0-based
#             ghost_ids = map(pos -> sides_pw[hanging_side].is.hanging.quadid[][pos],
#                             remote_small_quad_positions)
#             neighbor_ranks = map(ghost_ids) do ghost_id
#                 return findfirst(r -> proc_offsets[r] <= ghost_id < proc_offsets[r + 1],
#                                  1:mpi_nranks()) - 1 # MPI ranks are 0-based
#             end
#             # Determine global quad id of large element to determine global MPI mortar id
#             # Furthermore, if large element is ghost, add its owner rank to neighbor_ranks
#             if sides_pw[full_side].is.full.is_ghost[] == true
#                 ghost_id = sides_pw[full_side].is.full.quadid[]
#                 large_quad_owner_rank = findfirst(r -> proc_offsets[r] <= ghost_id <
#                                                        proc_offsets[r + 1],
#                                                   1:mpi_nranks()) - 1 # MPI ranks are 0-based
#                 push!(neighbor_ranks, large_quad_owner_rank)

#                 offset = mesh.p4est.global_first_quadrant[large_quad_owner_rank + 1] # one-based indexing
#                 large_quad_id = offset +
#                                 sides_pw[full_side].is.full.quad.p.piggy3.local_num[]
#             else
#                 offset = mesh.p4est.global_first_quadrant[mpi_rank() + 1] # one-based indexing
#                 large_quad_id = offset + trees_pw[full_side].quadrants_offset[] +
#                                 sides_pw[full_side].is.full.quadid[]
#             end
#             neighbor_ranks_mortar[mortar_id] = neighbor_ranks
#             # Global mortar id is the globally unique quadrant id of the large quadrant multiplied by the
#             # number of faces per quadrant plus face
#             global_mortar_ids[mortar_id] = 2 * ndims(mesh) * large_quad_id +
#                                            sides_pw[full_side].face[]

#             user_data.mortar_id += 1
#         end
#     end

#     return nothing
# end


end #muladd