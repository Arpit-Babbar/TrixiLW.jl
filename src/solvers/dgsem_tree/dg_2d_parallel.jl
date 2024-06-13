# dg::DG contains info about the solver such as basis(GL nodes), weights etc.
using Trixi: prolong2mpimortars!, start_mpi_receive!, MPICache, init_elements, local_leaf_cells,
             init_interfaces, init_boundaries, init_mortars, init_mpi_mortars,
             init_mpi_neighbor_connectivity, nmpiinterfaces, reset_du!, get_surface_node_vars,
             finish_mpi_send!, calc_mpi_mortar_flux!, mpi_mortar_fluxes_to_elements!,
             ParallelTreeMesh, ParallelP4estMesh, eachmpiinterface, mpi_nranks, mpi_rank, mpi_comm,
             mpi_isroot, calc_sources!, calc_surface_integral!

import Trixi: init_mpi_cache, init_mpi_cache!, start_mpi_send!, finish_mpi_receive!, prolong2mpiinterfaces!,
calc_mpi_interface_flux!
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

# everything related to a DG semidiscretization in 2D using MPI,
# currently limited to Lobatto-Legendre nodes

using MuladdMacro
@muladd begin
#! format: noindent

function start_mpi_send!(mpi_cache::MPICache, mesh, equations,
                         time_discretization::AbstractLWTimeDiscretization,
                         dg, cache)
    lw_extras = 3
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    for d in 1:length(mpi_cache.mpi_neighbor_ranks)
        send_buffer = mpi_cache.mpi_send_buffers[d]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
            first1 = lw_extras * (index - 1) * data_size + 1        # for u
            last1 = first1 + data_size - 1

            first2 = last1 + 1      # for U
            last2 =  first2 + data_size - 1

            first3 = last2 + 1      # for F
            last3 = first3 + data_size - 1

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views send_buffer[first1:last1] .= vec(cache.mpi_interfaces.u[2, :, :,
                                                                             interface])
                @views send_buffer[first2:last2] .= vec(cache.mpi_interfaces.U[2, :, :,
                                                                             interface])
                @views send_buffer[first3:last3] .= vec(cache.mpi_interfaces.F[2, :, :,
                                                                             interface])
            else # local element in negative direction
                @views send_buffer[first1:last1] .= vec(cache.mpi_interfaces.u[1, :, :,
                                                                             interface])
                @views send_buffer[first2:last2] .= vec(cache.mpi_interfaces.U[1, :, :,
                                                                             interface])
                @views send_buffer[first3:last3] .= vec(cache.mpi_interfaces.F[1, :, :,
                                                                             interface])
            end
        end

        # mortar code here
        # mortar code needs to be change according to the first1:last1; first2:last2 etc. indices
        # since U, F will be extra so indices needs to managed as done in normal element case
    end
        # Start sending
    for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_send_requests[index] = MPI.Isend(mpi_cache.mpi_send_buffers[index],
                                                       d, mpi_rank(), mpi_comm())
    end

    return nothing

end

function finish_mpi_receive!(mpi_cache::MPICache, mesh, equations,
                             time_discretization::AbstractLWTimeDiscretization,
                             dg, cache)
    lw_extras = 3
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    # Start receiving and unpack received data until all communication is finished
    d = MPI.Waitany(mpi_cache.mpi_recv_requests)

    while d !== nothing
        recv_buffer = mpi_cache.mpi_recv_buffers[d]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
            first1 = lw_extras * (index - 1) * data_size + 1        # for u
            last1 = first1 + data_size - 1

            first2 = last1 + 1      # for U
            last2 =  first2 + data_size - 1

            first3 = last2 + 1      # for F
            last3 = first3 + data_size - 1

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views vec(cache.mpi_interfaces.u[1, :, :, interface]) .= recv_buffer[first1:last1]
                @views vec(cache.mpi_interfaces.U[1, :, :, interface]) .= recv_buffer[first2:last2]
                @views vec(cache.mpi_interfaces.F[1, :, :, interface]) .= recv_buffer[first3:last3]
            else # local element in negative direction
                @views vec(cache.mpi_interfaces.u[2, :, :, interface]) .= recv_buffer[first1:last1]
                @views vec(cache.mpi_interfaces.U[2, :, :, interface]) .= recv_buffer[first2:last2]
                @views vec(cache.mpi_interfaces.F[2, :, :, interface]) .= recv_buffer[first3:last3]
            end
        end
        # mortar code here
        # mortar code needs to be change according to the first1:last1; first2:last2 etc. indices
        # since U, F will be extra so indices needs to managed as done in normal element case

        d = MPI.Waitany(mpi_cache.mpi_recv_requests)
    end
    return nothing
end


function create_cache(mesh::ParallelTreeMesh{2}, equations,
                      time_discretization::AbstractLWTimeDiscretization,
                      dg::DG, RealT, ::Type{uEltype}, cache) where {uEltype <: Real}


    # TODO - fix this repetition
    # Serial LW part begin

    nan_RealT   = convert(RealT, NaN)
    nan_uEltype = convert(uEltype, NaN)

    n_variables = nvariables(equations)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)

    temporal_errors = fill(nan_uEltype, n_elements)

    NDIMS = ndims(equations)

    # TODO - Put U, F, fn_low in cell_cache named tuple
    # TODO - Don't pass so many things, only pass cache, equations etc.
    element_cache = create_element_cache(mesh, nan_uEltype, NDIMS, n_variables,
       n_nodes, n_elements, time_discretization)

    interface_cache = create_interface_cache(mesh, equations, dg, uEltype, RealT,
       cache, time_discretization)

    # Create the result of `alloc` for each thread. Basically,
    # for each thread, construct `cache_size` number of objects with
    # `constructor` and store them in an SVector

    MOuter = MArray{Tuple{n_variables},Float64}
    outer_cache = alloc_for_threads(MOuter, 2)
    boundary_cache = create_boundary_cache(mesh, equations, dg, uEltype, RealT,
       cache, outer_cache, time_discretization)

    lw_mortars = create_mortar_cache(mesh, equations, dg, uEltype, RealT,
       cache, time_discretization)
    cfl_number = fill(nan_RealT, 1)
    dt = fill(nan_RealT, 1)

    cell_array_sizes = Dict(1 => 12, 2 => 14, 3 => 17, 4 => 18)
    degree = n_nodes - 1
    cell_array_size = cell_array_sizes[min(4, degree)]

    MArr = MArray{Tuple{n_variables, n_nodes, n_nodes},Float64}
    N4() = NamedTuple{(:f, :g, :ftilde, :gtilde, :Ftilde, :Gtilde, :ut, :utt, :uttt, :utttt,
                       :U, :up, :um, :upp, :umm, :S, :u_np1, :u_np1_low)}((MArr(undef) for _=1:18))

    if degree >= 4
       nt = Threads.nthreads()
       cell_arrays = SVector{Threads.nthreads()}([N4() for _ in 1:Threads.nthreads()])
    else
       cell_arrays = alloc_for_threads(MArr, cell_array_size)
    end

    lw_res_cache = (; cell_arrays)

    cache = (; cache..., element_cache, lw_res_cache, cfl_number, dt,
       temporal_errors, interface_cache, boundary_cache, lw_mortars)

    # Serial LW part end

    leaf_cell_ids = local_leaf_cells(mesh.tree)             # Extracting all leaf cells to create element

    # All these are taken from Trixi
    # Create elements, record their coordinates, maps GL nodes to coordinate axis of each element
    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    # Generate interfaces to store info about of adjacent elements
    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    mpi_interfaces = init_mpi_interfaces(leaf_cell_ids, mesh, time_discretization, elements)

    # records elements which contains boundaries
    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    mpi_mortars = init_mpi_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    mpi_cache = init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars,
                                nvariables(equations), nnodes(dg), uEltype, time_discretization)

    cache = (; cache..., elements, interfaces, mpi_interfaces, boundaries, mortars, mpi_mortars,
            mpi_cache)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache..., create_cache(mesh, equations, dg.volume_integral, time_discretization, dg, uEltype)...)
    cache = (; cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

    return cache
end

function init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars, nvars, nnodes,
                        uEltype, time_discretization::AbstractLWTimeDiscretization)
    mpi_cache = MPICache(uEltype)

    init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars, nvars,
                    nnodes, uEltype, time_discretization)
    return mpi_cache
end

function init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars, nvars,
                         nnodes, uEltype, time_discretization::AbstractLWTimeDiscretization)
    mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars = init_mpi_neighbor_connectivity(elements,
                                                                                                       mpi_interfaces,
                                                                                                       mpi_mortars,
                                                                                                       mesh)

    mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests = init_mpi_data_structures(mpi_neighbor_interfaces,
                                                                                                        mpi_neighbor_mortars,
                                                                                                        ndims(mesh),
                                                                                                        nvars,
                                                                                                        nnodes,
                                                                                                        uEltype,
                                                                                                        time_discretization)
    # Define local and total number of elements
    n_elements_by_rank = Vector{Int}(undef, mpi_nranks())   # vector of length `size`
    n_elements_by_rank[mpi_rank() + 1] = nelements(elements)    # number of elements each rank has.

    # This will create a buffer on each rank of the needed size as described in the array n_elements_by_rank then gather
    # all those buffers and make a single buf which is then bcast to all ranks.
    # MPI.UBuffer(::Array, ::DataType)- create fixed size separate buffer on each rank.
    MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())

    n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))  # Overwriting same array and index changing
    n_elements_global = MPI.Allreduce(nelements(elements), +, mpi_comm())   # total number of elements
    @assert n_elements_global == sum(n_elements_by_rank) "error in total number of elements"

    # Determine the global element id of the first element
    # MPI.Exscan()-> partial reduction(here sum) but exclude process's own cotribution
    # MPI.Scan()-> include process's own contribution
    first_element_global_id = MPI.Exscan(nelements(elements), +, mpi_comm())
    if mpi_isroot()
        # With Exscan, the result on the first rank is undefined
        first_element_global_id = 1
    else
        # on all other ranks, +1 since julia is 1-based indexing
        first_element_global_id += 1
    end

    @pack! mpi_cache = mpi_neighbor_ranks, mpi_neighbor_interfaces,
                       mpi_neighbor_mortars,
                       mpi_send_buffers, mpi_recv_buffers,
                       mpi_send_requests, mpi_recv_requests,
                       n_elements_by_rank, n_elements_global,
                       first_element_global_id
end

function rhs!(du, u, t, dt, mesh::Union{ParallelTreeMesh{2}, ParallelP4estMesh{2}},
            equations, initial_condition, boundary_conditions, source_terms::Source, dg::DG,
            time_discretization::AbstractLWTimeDiscretization,
            cache, tolerances::NamedTuple) where {Source}
    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, t, dt, tolerances, mesh,
                              have_nonconservative_terms(equations), source_terms,
                              equations, dg.volume_integral, time_discretization,
                              dg, cache)
    end

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(cache, u, mesh, equations, dg.surface_integral,
                               time_discretization, dg)
    end

    # Prolong solution to MPI mortars
    # TODO: code `prolong2mpimortars!()`
    @trixi_timeit timer() "prolong2mpimortars" begin
        prolong2mpimortars!(cache, u, mesh, equations,
                            dg.mortar, dg.surface_integral, dg)
    end

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(cache.mpi_cache, mesh, equations, time_discretization,
                        dg, cache)
    end

    # Prolong solution to interfaces
    # TODO: Taal decide order of arguments, consistent vs. modified cache first?
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations, dg.surface_integral,
                            time_discretization, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dt, time_discretization,
                             dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u, mesh, equations, dg.surface_integral,
                            time_discretization, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, dt, boundary_conditions, mesh, equations,
                            dg.surface_integral, time_discretization, dg)
    end

    # Prolong solution to mortars
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u, mesh, equations, dg.mortar, dg.surface_integral,
                         time_discretization, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, time_discretization,
                          dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(cache.mpi_cache, mesh, equations, time_discretization,
                            dg, cache)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, time_discretization,
                                 dg, cache)
    end

    # Calculate MPI mortar fluxes
    @trixi_timeit timer() "MPI mortar flux" begin
        calc_mpi_mortar_flux!(cache.elements.surface_flux_values, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.mortar, dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)


    # Finish to send MPI data
    @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

    return nothing

end

# U, F are extra in here because it is using LW instead of RK
function prolong2mpiinterfaces!(cache, u, mesh::ParallelTreeMesh{2},
                                equations, surface_integral,
                                time_discretization::AbstractLWTimeDiscretization,
                                dg::DG)
    @unpack mpi_interfaces = cache
    @unpack U, F = cache.element_cache

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = mpi_interfaces.local_neighbor_ids[interface]

        if mpi_interfaces.orientations[interface] == 1 # interface in x direction
            if mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                for j in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[2, v, j, interface] = u[v, 1, j, local_element]
                    mpi_interfaces.U[2, v, j, interface] = U[v, 1, j, local_element]
                    mpi_interfaces.F[2, v, j, interface] = F[v, 1, 1, j, local_element]
                end

            else # local element in negative x-direction
                for j in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j,
                                                            local_element]
                    mpi_interfaces.U[1, v, j, interface] = U[v, nnodes(dg), j,
                                                            local_element]
                    mpi_interfaces.F[1, v, j, interface] = F[v, 1, nnodes(dg), j,
                                                            local_element]
                end
            end
        else # interface in y-direction
            if mpi_interfaces.remote_sides[interface] == 1 # local element in positive y direction
                for i in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[2, v, i, interface] = u[v, i, 1, local_element]
                    mpi_interfaces.U[2, v, i, interface] = U[v, i, 1, local_element]
                    mpi_interfaces.F[2, v, i, interface] = F[v, 2, i, 1, local_element]
                end
            else # local element in negative y-direction
                for i in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg),
                                                            local_element]
                    mpi_interfaces.U[1, v, i, interface] = U[v, i, nnodes(dg),
                                                             local_element]
                    mpi_interfaces.F[1, v, i, interface] = F[v, 2, i, nnodes(dg),
                                                             local_element]
                end
            end
        end
    end

    return nothing
end

function calc_mpi_interface_flux!(surface_flux_values,
                                  mesh::ParallelTreeMesh{2},
                                  nonconservative_terms::False, equations,
                                  surface_integral, time_discretization::AbstractLWTimeDiscretization,
                                  dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u, U, F, local_neighbor_ids, orientations, remote_sides = cache.mpi_interfaces

    @threaded for interface in eachmpiinterface(dg, cache)
        # Get local neighboring element
        element = local_neighbor_ids[interface]

        # Determine interface direction with respect to element:
        if orientations[interface] == 1 # interface in x-direction
            if remote_sides[interface] == 1 # local element in positive direction
                direction = 1
            else # local element in negative direction
                direction = 2
            end
        else # interface in y-direction
            if remote_sides[interface] == 1 # local element in positive direction
                direction = 3
            else # local element in negative direction
                direction = 4
            end
        end

        for i in eachnode(dg)
            # Call pointwise Riemann solver
            u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
            U_ll, U_rr = get_surface_node_vars(U, equations, dg, i, interface)
            F_ll, F_rr = get_surface_node_vars(F, equations, dg, i, interface)

            flux = surface_flux(F_ll, F_rr, U_ll, U_rr, u_ll, u_rr, orientations[interface], equations)

            # Copy flux to local element storage
            for v in eachvariable(equations)
                surface_flux_values[v, i, direction, element] = flux[v]
            end
        end
    end

    return nothing
end
end # @muladd