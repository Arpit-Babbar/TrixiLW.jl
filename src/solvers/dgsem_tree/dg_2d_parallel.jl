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

    # This will create a buffer on each rank of the needed size as described in the array n_elements_by_rank then gather
    # all those buffers and make a single buf which is then bcast to all ranks.
    # MPI.UBuffer(::Array, ::DataType)- create separate buffer on each rank.
    MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())
    
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
    # The global interface id is the smaller of the (globally unique) neighbor cell ids, 
    # multiplied by number of directions (2 * ndims; for 2D 4 directions) plus direction minus one
    global_interface_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
    for interface_id in 1:nmpiinterfaces(mpi_interfaces)
        orientation = mpi_interfaces.orientations[interface_id]
        remote_side = mpi_interfaces.remote_sides[interface_id]
        # Direction is from local cell to remote cell
        if orientation == 1 # MPI interface is in X-direction
            if remote_side == 1 # remote cell is on the "left" of MPI Interface  
                direction = 1
            else # remote cell is on the "right" of MPI Interface
                direction = 2
            end
        else # MPI interface is in y-direction
            if remote_side == 1 # remote cell is on the "left" of MPI Interface
                direction = 3
            else # remote cell is on the "right" of MPI Interface
                direction = 4
            end
        end

        local_neighbor_id = mpi_interfaces.local_neighbor_ids[interface_id]
        local_cell_id = elements.cell_ids[local_neighbor_id]
        remote_cell_id = tree.neighbor_ids[direction, local_cell_id]
        neighbor_ranks_interface[interface_id] = tree.mpi_ranks[remote_cell_id]

        if local_cell_id < remote_cell_id
            global_interface_ids[interface_id] = 2 * ndims(tree) * local_cell_id + direction - 1
        else
            global_interface_ids[interface_id] = (2 * ndims(tree) * remote_cell_id + opposite_direction(direction) - 1)
        end
    end

    # Determine neighbor ranks for MPI mortars
    # TODO: mortar code here

    # Get sorted, unique neighbor ranks
    # |> - pipe operator (transfers output of left fxn to input of right fxn)
    # vcat() - vertical concatenation
    # sort - sort the array
    # unique - remove duplicates
    mpi_neighbor_ranks = vcat(neighbor_ranks_interface, neighbor_raks_mortar...) |> sort |> unique

    # Sort interfaces by global interface id
    p = sortperm(global_interface_ids)      # sortperm() - returns indices for which, sorted array can be obtained
    # p contains sorted indices
    neighbor_ranks_interface .= neighbor_ranks_interface[p] # array will get sorted here
    interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

    # Sort mortars by global mortar id
    # TODO: mortar code here

    # For each neighbor rank, init connectivity data structures
    mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    mpi_neighbor_mortars = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    for (index, d) in enumerate(mpi_neighbor_ranks)
        # findall(predicate, collection) - returns indices of those elements for which predicate is true
        mpi-neighbor_ranks_interfaces[index] = interface_ids[findall(x -> (x == d)), neighbor_ranks_interface]
        mpi_neighbor_mortars[index] = mortar_ids[findall(x -> (x == d)), neighbor_raks_mortar]
    end

    # Check that we counted all interfaces exactly once
    @assert sum(length(v) for v in mpi_neighbor_interfaces) == nmpiinterfaces(mpi_interfaces)

    return mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars

end

# TODO: complete rhs!()
function rhs!(du, u, t, mesh::Union{ParallelTreeMesh{2}, ParallelP4estMesh{2}, ParallelT8codeMesh{2}}, equations,
              initial_condition, boundary_conditions, source_terms::Source, dg::DG, time_discretization::AbstractLWTimeDiscretization,
              cache, tolerances::NamedTuple) where {Source}
    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(cache, u, mesh, equations, dg.surface_integral, time_discretization, dg)
    end

    # Prolong solution to MPI mortars
    # TODO: code `prolong2mpimortars!()`
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(cache.mpi_cache, mesh, equations, dg, cache)
    end


end

# TODO: Add functionality for time averaged solution and fluxes
# These functions are extra in LW because it is using LW instead of RK?(ASK)
function prolong2mpiinterfaces!(cache, u, mesh::ParallelTreeMesh{2},
                                equations, surface_integral, 
                                time_discretization::AbstractLWTimeDiscretization, dg::DG)
    @unpack mpi_interfaces = cache

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = mpi_interface.local_neighbor_ids[interface]
        
        # TODO: create fluxes anad time averaged solution part too. 
        # TODO: create struct in containers_2d.jl that contains `u`, `remote_sides` etc
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