using Trixi: balance!, partition!, update_ghost_layer!, init_elements,
             mpi_nranks, mpi_rank, nmpiinterfaces, init_mpi_neighbor_connectivity,
             InitNeighborRankConnectivityIterFaceUserData, exchange_normal_directions!,
             init_interfaces, init_boundaries, init_mpi_mortars, init_mortars,
             finish_mpi_send!, start_mpi_receive!

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
            else # local element at secondary side
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
    balance!(mesh)
    partition!(mesh)
    update_ghost_layer!(mesh)

    elements = init_elements(mesh, equations, dg.basis, uEltype)

    mpi_interfaceslw = init_mpi_interfaces(mesh, equations, dg.basis, time_discretization, elements)
    mpi_mortars = init_mpi_mortars(mesh, equations, dg.basis, elements)
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

end #muladd