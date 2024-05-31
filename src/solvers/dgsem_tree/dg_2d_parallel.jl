# dg::DG contains info about the solver such as basis(GL nodes), weights etc.
#!
using MuladdMacro
@muladd begin

mutable struct MPICache{uEltype <: Real}
    mpi_neighbor_ranks::Vector{Int}
    mpi_neighbor_interfaces::Vector{Vector{Int}}
    mpi_neighbor_mortars::Vector{Vector{Int}}
    mpi_send_buffers::Vector{Vector{uEltype}}
    mpi_recv_buffers::Vector{Vector{uEltype}}
    mpi_send_requests::Vector{MPI.Request}
    mpi_recv_requests::Vector{MPI.Request}
    n_elements_by_rank::OffsetArray{Int, 1, Array{Int, 1}}
    n_elements_global::Int
    first_element_global_id::Int
end

function MPICache(uEltype)      # Outer constructor
    # MPI communication "just works" for bitstypes only(Int, Float, bool)
    # complex types such as struct etc. can not be transferred.
    if !isbitstype(uEltype)
        throw(ArgumentError("MPICache only supports bitstypes, $uEltype is not a bittype."))
    end
    mpi_neighbor_ranks = Vector{Int}(undef, 0)  # vector of 0 length; later will be filled
    mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, 0)
    mpi_neighbor_mortars = Vector{Vector{Int}}(undef, 0)
    mpi_send_buffers = Vector{Vector{uEltype}}(undef, 0)
    mpi_recv_buffers = Vector{Vector{uEltype}}(undef, 0)
    mpi_send_requests = Vector{MPI.Request}(undef, 0)
    mpi_recv_requests = Vector{MPI.Request}(undef, 0)
    n_elements_by_rank = offsetArray(Vector{Int}(undef, 0), 0:-1)
    n_elements_by_global = 0
    first_element_global_id = 0

    # Initializing the MPICache struct (outer constructor approach)
    # Creating a instance of MPICache and return the resulting instance
    MPICache{uEltype}(mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars,
                      mpi_send_buffers, mpi_recv_buffers,
                      mpi_send_requests, mpi_recv_requests,
                      n_elements_by_rank, n_elements_by_global,
                      first_element_global_id)
end
# Overloading Base.eltype function to tell type of elements of instances of MPICache 
@inline Base.eltype(::MPICache{uEltype}) where {uEltype} = uEltype

# TODO: what are mpi_neighbor_ranks? How are they distributing the elements?
# find out this info in Trixi.jl.

# mpi_neighbor_ranks -> array of rank of the neighbor element
function start_mpi_receive!(mpi_cache::MPICache)
    for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_recv_requests[index] = MPI.Irecv!(mpi_cache.mpi_recv_buffers[index],
                                                        d, d, mpi_comm())
    end

    return nothing
end

function start_mpi_send!(mpi_cache::MPICache, mesh, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    for d in 1:length(mpi_cache.mpi_neighbor_ranks)
        send_buffer = mpi_cache.mpi_send_buffers[d]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[2, :, :, interface])

            else # local element in negative direction
                @views send_buffer[first:last] .= vec(cache.mpi_interface.u[1, :, :, interface])

            end
        end

        # Mortar code here
    end

    # start sending
    for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_send_requests[index] = MPI.Isend(mpi_cache.mpi_send_buffers[index],
                                                        d, mpi_rank(), mpi_comm())
    end
    return nothing
end

function finish_mpi_send!(mpi_cache::MPICache)
    MPI.Waitall(mpi_cache.mpi_send_requests, MPI.Status)
end

function finish_mpi_receive!(mpi_cache::MPICache, mesh, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)
    d = MPI.Waitany(mpi_cache.mpi_recv_requests)

    while d !== nothing
        recv_buffer = mpi_cache.mpi_recv_buffers[d]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views vec(cache.mpi_interface.u[1, :, :, interface]) .= recv_buffer[first:last]
            else # local element in negative direction
                @views vec(cache.mpi_interface.u[2, :, :, interface]) .= recv_buffer[first:last]
            end
        end
        # mortar code here

        d = MPI.Waitany(mpi_cache.mpi_recv_requests)
    end
    return nothing
end


function create_cache(mesh::ParallelTreeMesh{2}, equations,
                    dg::DG, RealT, ::Type{uEltype}) where {uEltype <: Real}
    leaf_cell_ids = local_leaf_cells(mesh.tree) # Extracting all leaf cells to create element

    # TODO: create init_elements()
    # Create elements, record their coordinates, maps GL nodes to coordinate axis of each element
    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    # TODO: create init_interfaces()
    # Generate interfaces to store info about of adjacent elements
    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    # TODO: init_mpi_interfaces() in container_2d.jl
    mpi_interfaces = init_mpi_interfaces(leaf_cell_ids, mesh, elements)

    # TODO: create init_boundaries()
    # records elements which contains boundaries
    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    # TODO: create init_mortars()
    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    # TODO: create init_mpi_mortars()
    mpi_mortars = init_mpi_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    # TODO: create init_mpi_cache()
    mpi_cache = init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars,
                                nvariables(equations), nnodes(dg), uEltype)

    cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars, mpi_mortars,
            mpi_cache)

    # Add specialized parts of the cache required to compute the volume integral etc.
    # TODO: create this. (dg.volume implementation is in dg_2d_subcell_limiters.jl of Trixi.jl)
    cache = (; cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
    # TODO:create this. (dg.mortar implementation is in dg_2d.jl of Trixi.jl)
    cache = (; cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

    return cache    
end

function init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars, nvars, nnodes,
                        uEltype)
    mpi_cache = MPICache(uEltype)

    init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars, nvars,
                    nnodes, uEltype)

    return mpi_cache
end

function init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars, nvars,
                         nnodes, uEltype)
    mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars = init_mpi_neighbor_connectivity(elements,
                                                                                                        mpi_interfaces,
                                                                                                        mpi_mortars,
                                                                                                        mesh)
    mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests = init_mpi_data_structures(mpi_neighbor_interfaces,
                                                                                                        mpi_neighbor_mortars,
                                                                                                        ndims(mesh),
                                                                                                        nvars,
                                                                                                        nnodes,
                                                                                                        uEltype)
    # Define local and total number of elements
    n_elements_by_rank = Vector{Int}(undef, mpi_nranks())   # vector of length `size`
    n_elements_by_rank[mpi_rank() + 1] = nelements(elements)    # number of elements each rank has.
    MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())  # MPI.UBuffer()-> create buffer without datatype
    n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))  # Overwriting same array and index changing
    n_elements_global = MPI.Allreduce(nelements(elements), +, mpi_comm())   # total number of elements
    @assert n_elements_global == sum(n_elements_by_rank) "error in total number of elements"

    # Determine the global element id of the first element
    # MPI.Exscan()-> partial reduction(here sum) but exclude process's own cotribution
    # Complement-> MPI.Scan()-> include process's own contribution 
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
                       n_element_by_rank, n_element_global,
                       first_element_global_id
end

# Initialize connectivity between MPI neighbor ranks
function init_mpi_neighbor_connectivity(element, mpi_interfaces, mpi_mortars,
                                        mesh::TreeMesh2D)
    tree = mesh.tree

    # Determine neighbor ranks and sides for MPI interfaces
    neighbor_ranks_interfaces = fill(-1, nmpiinterfaces(mpi_interfaces))
    # 



# TODO: Add functionality for time averaged solution and fluxes
# These functions are extra in LW because it is using LW instead of RK?(ASK)
function prolong2mpiinterfaces!(cache, u, mesh::ParallelTreeMesh{2},
                                equations, surface_integral, 
                                time_discretization::AbstractLWTimeDiscretization, dg::DG)
    @unpack mpi_interfaces = cache

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = mpi_interface.local_neighbor_ids[interface]
        
        # TODO: create fluxes anad time averaged solution part too. 
        #TODO: create struct in containers_2d.jl that contains `u`, `remote_sides` etc
        # check similar struct in container_2d.jl in Trixi.jl
        if mpi_interfaces.orientations[interface] == 1 # interface in x direction
            if mpi_interface.remote_sides[interface] == 1 # local element in positive direction 
                for j in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[2, v, j, interface] = u[v, 1, j, local_element]
                end

            else # local element in negative x-direction
                for j in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j,
                                                            local_element]
                end
            end
        else # interface in y-direction
            if mpi_interfaces.remote_sides[interface] == 1 # local element in positive y direction
                for i in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg),
                                                            local_element]
                end
            else # local element in negative y-direction
                for i in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg),
                                                            local_element]
                end
            end
        end
    end

    return nothing
end


end # @muladd