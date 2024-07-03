import Trixi: init_mpi_interfaces, init_mpi_interface_node_indices!, nmpiinterfaces, nmpimortars, init_mpi_mortars

using Trixi: count_required_surfaces, init_surfaces!, init_mpi_interfaces!, P4estMPICache, ParallelP4estMesh,
             P4estMPIInterfaceContainer, init_mpi_mortars!, P4estMPIMortarContainer,index_to_start_step_2d

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct P4estMPIInterfaceContainerLW{NDIMS, uEltype <: Real, NDIMSP2} <:
               AbstractContainer
    u::Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]
    U::Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]
    F::Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]

    local_neighbor_ids::Vector{Int}                     # [interface]
    node_indices::Vector{NTuple{NDIMS, Symbol}}         # [interface]
    local_sides::Vector{Int}                            # [interface]

    # internal `resize!`able storage
    _u::Vector{uEltype}
    _U::Vector{uEltype}
    _F::Vector{uEltype}
end

@inline function nmpiinterfaces(interfaces::P4estMPIInterfaceContainerLW)
    length(interfaces.local_sides)
end

@inline Base.ndims(::P4estMPIInterfaceContainerLW{NDIMS}) where {NDIMS} = NDIMS

# For AMR, to be tested
function Base.resize!(mpi_interfaces::P4estMPIInterfaceContainerLW, capacity)
    @unpack _u, _U, _F, local_neighbor_ids, node_indices, local_sides = mpi_interfaces

    n_dims = ndims(mpi_interfaces)
    n_nodes = size(mpi_interfaces.u, 3)
    n_variables = size(mpi_interfaces.u, 2)

    resize!(_u, 2 * n_variables * n_nodes^(n_dims - 1) * capacity)
    resize!(_U, 2 * n_variables * n_nodes^(n_dims - 1) * capacity)
    resize!(_F, 2 * n_variables * n_nodes^(n_dims - 1) * capacity)

    mpi_interfaces.u = unsafe_wrap(Array, pointer(_u),
                                   (2, n_variables, ntuple(_ -> n_nodes, n_dims - 1)...,
                                    capacity))
    mpi_interfaces.U = unsafe_wrap(Array, pointer(_U),
                                   (2, n_variables, ntuple(_ -> n_nodes, n_dims - 1)...,
                                    capacity))
    mpi_interfaces.F = unsafe_wrap(Array, pointer(_F),
                                   (2, n_variables, ntuple(_ -> n_nodes, n_dims - 1)...,
                                    capacity))

    resize!(local_neighbor_ids, capacity)

    resize!(node_indices, capacity)

    resize!(local_sides, capacity)

    return nothing
end

# Create MPI interface container and initialize interface data
function init_mpi_interfaces(mesh::ParallelP4estMesh,
                             equations, basis, time_discretization::AbstractLWTimeDiscretization,
                             elements)
    NDIMS = ndims(elements)
    uEltype = eltype(elements)

    # Initialize container
    n_mpi_interfaces = count_required_surfaces(mesh).mpi_interfaces

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * nnodes(basis)^(NDIMS - 1) *
                         n_mpi_interfaces)
    _U = Vector{uEltype}(undef,
                         2 * nvariables(equations) * nnodes(basis)^(NDIMS - 1) *
                         n_mpi_interfaces)
    _F = Vector{uEltype}(undef,
                         2 * nvariables(equations) * nnodes(basis)^(NDIMS - 1) *
                         n_mpi_interfaces)

    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                     n_mpi_interfaces))
    U = unsafe_wrap(Array, pointer(_U),
                    (2, nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                     n_mpi_interfaces))
    F = unsafe_wrap(Array, pointer(_F),
                    (2, nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                     n_mpi_interfaces))

    local_neighbor_ids = Vector{Int}(undef, n_mpi_interfaces)

    node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, n_mpi_interfaces)

    local_sides = Vector{Int}(undef, n_mpi_interfaces)

    # This is the container from Trixi, we only use it because the `init_mpi_interfaces!`
    # function from trixi initializes some connectivity information for which we want to
    # use the Trixi function without duplication
    mpi_interfaces = P4estMPIInterfaceContainer{NDIMS, uEltype, NDIMS + 2}(u,
                                                                           local_neighbor_ids,
                                                                           node_indices,
                                                                           local_sides,
                                                                           _u)
    init_mpi_interfaces!(mpi_interfaces, mesh)

    # Now we just move what we get from the Trixi function into our own container
    mpi_interfaces_lw = P4estMPIInterfaceContainerLW{NDIMS, uEltype, NDIMS + 2}(mpi_interfaces.u,
                                                                               U, F,
                                                                               mpi_interfaces.local_neighbor_ids,
                                                                               mpi_interfaces.node_indices,
                                                                               mpi_interfaces.local_sides,
                                                                               mpi_interfaces._u,
                                                                               _U, _F)

    return mpi_interfaces, mpi_interfaces_lw
end

# Container data structure (structure-of-arrays style) for DG L2 mortars
#
# Similar to `P4estMortarContainer`. The field `neighbor_ids` has been split up into
# `local_neighbor_ids` and `local_neighbor_positions` to describe the ids and positions of the locally
# available elements belonging to a particular MPI mortar. Furthermore, `normal_directions` holds
# the normal vectors on the surface of the small elements for each mortar.
mutable struct P4estMPIMortarContainerLW{NDIMS, uEltype <: Real, RealT <: Real, NDIMSP1,
                                       NDIMSP2, NDIMSP3} <: AbstractContainer
    u::Array{uEltype, NDIMSP3} # [small/large side, variable, position, i, j, mortar]
    U::Array{uEltype, NDIMSP3} # [small/large side, variable, position, i, j, mortar]
    F::Array{uEltype, NDIMSP3} # [small/large side, variable, position, i, j, mortar]

    local_neighbor_ids::Vector{Vector{Int}} # [mortar]
    local_neighbor_positions::Vector{Vector{Int}} # [mortar]
    node_indices::Matrix{NTuple{NDIMS, Symbol}} # [small/large, mortar]
    normal_directions::Array{RealT, NDIMSP2} # [dimension, i, j, position, mortar]

    # internal `resize!`able storage
    _u::Vector{uEltype}
    _U::Vector{uEltype}
    _F::Vector{uEltype}
    _node_indices::Vector{NTuple{NDIMS, Symbol}}
    _normal_directions::Vector{RealT}
    tmp::NamedTuple
end

@inline function nmpimortars(mpi_mortars::P4estMPIMortarContainerLW)
    length(mpi_mortars.local_neighbor_ids)
end

@inline Base.ndims(::P4estMPIMortarContainerLW{NDIMS}) where {NDIMS} = NDIMS

function Base.resize!(mpi_mortars::P4estMPIMortarContainerLW, capacity)
    @unpack _u, _U, _F, _node_indices, _normal_directions = mpi_mortars

    n_dims = ndims(mpi_mortars)
    n_nodes = size(mpi_mortars.u, 4)
    n_variables = size(mpi_mortars.u, 2)

    resize!(_u, 2 * n_variables * 2^(n_dims - 1) * n_nodes^(n_dims - 1) * capacity)
    mpi_mortars.u = unsafe_wrap(Array, pointer(_u),
                                (2, n_variables, 2^(n_dims - 1),
                                 ntuple(_ -> n_nodes, n_dims - 1)..., capacity))

    resize!(_U, 2 * n_variables * 2^(n_dims - 1) * n_nodes^(n_dims - 1) * capacity)
    mpi_mortars.U = unsafe_wrap(Array, pointer(_u),
                                (2, n_variables, 2^(n_dims - 1),
                                 ntuple(_ -> n_nodes, n_dims - 1)..., capacity))

    resize!(_F, 2 * n_variables * 2^(n_dims - 1) * n_nodes^(n_dims - 1) * capacity)
    mpi_mortars.F = unsafe_wrap(Array, pointer(_u),
                                (2, n_variables, 2^(n_dims - 1),
                                 ntuple(_ -> n_nodes, n_dims - 1)..., capacity))

    resize!(mpi_mortars.local_neighbor_ids, capacity)
    resize!(mpi_mortars.local_neighbor_positions, capacity)

    resize!(_node_indices, 2 * capacity)
    mpi_mortars.node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, capacity))

    resize!(_normal_directions,
            n_dims * n_nodes^(n_dims - 1) * 2^(n_dims - 1) * capacity)
    mpi_mortars.normal_directions = unsafe_wrap(Array, pointer(_normal_directions),
                                                (n_dims,
                                                 ntuple(_ -> n_nodes, n_dims - 1)...,
                                                 2^(n_dims - 1), capacity))

    return nothing
end

# Create MPI mortar container and initialize MPI mortar data
function init_mpi_mortars(mesh::ParallelP4estMesh, equations,
                          basis, elements, cache)
    NDIMS = ndims(mesh)
    RealT = real(mesh)
    uEltype = eltype(elements)

    # Initialize container
    n_mpi_mortars = count_required_surfaces(mesh).mpi_mortars

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * 2^(NDIMS - 1) *
                         nnodes(basis)^(NDIMS - 1) * n_mpi_mortars)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), 2^(NDIMS - 1),
                     ntuple(_ -> nnodes(basis), NDIMS - 1)..., n_mpi_mortars))

    _U = Vector{uEltype}(undef,
                         2 * nvariables(equations) * 2^(NDIMS - 1) *
                         nnodes(basis)^(NDIMS - 1) * n_mpi_mortars)
    U = unsafe_wrap(Array, pointer(_U),
                    (2, nvariables(equations), 2^(NDIMS - 1),
                     ntuple(_ -> nnodes(basis), NDIMS - 1)..., n_mpi_mortars))

    _F = Vector{uEltype}(undef,
                         2 * nvariables(equations) * 2^(NDIMS - 1) *
                         nnodes(basis)^(NDIMS - 1) * n_mpi_mortars)
    F = unsafe_wrap(Array, pointer(_F),
                    (2, nvariables(equations), 2^(NDIMS - 1),
                     ntuple(_ -> nnodes(basis), NDIMS - 1)..., n_mpi_mortars))

    local_neighbor_ids = fill(Vector{Int}(), n_mpi_mortars)
    local_neighbor_positions = fill(Vector{Int}(), n_mpi_mortars)

    _node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, 2 * n_mpi_mortars)
    node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, n_mpi_mortars))

    _normal_directions = Vector{RealT}(undef,
                                       NDIMS * nnodes(basis)^(NDIMS - 1) *
                                       2^(NDIMS - 1) * n_mpi_mortars)
    normal_directions = unsafe_wrap(Array, pointer(_normal_directions),
                                    (NDIMS, ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                                     2^(NDIMS - 1), n_mpi_mortars))

    # This is the container from Trixi, we only use it because the `init_mpi_mortars!`
    # function from trixi initializes some connectivity information for which we want to
    # use the Trixi function without duplication
    mpi_mortars = P4estMPIMortarContainer{NDIMS, uEltype, RealT, NDIMS + 1, NDIMS + 2,
                                          NDIMS + 3}(u, local_neighbor_ids,
                                                     local_neighbor_positions,
                                                     node_indices, normal_directions,
                                                     _u, _node_indices,
                                                     _normal_directions)
    if n_mpi_mortars > 0
        init_mpi_mortars!(mpi_mortars, mesh, basis, elements)
    end
    # For immutable objects, always use deep copy
    U_threaded, F_threaded = (deepcopy(cache.u_threaded) for _ in 1:2)

    # Now we just move what we get from the Trixi function into our own container
    mpi_mortars_lw = P4estMPIMortarContainerLW{NDIMS, uEltype, RealT, NDIMS + 1, NDIMS + 2,
                                          NDIMS + 3}(mpi_mortars.u,
                                                     U, F,
                                                     mpi_mortars.local_neighbor_ids,
                                                     mpi_mortars.local_neighbor_positions,
                                                     mpi_mortars.node_indices,
                                                     mpi_mortars.normal_directions,
                                                     mpi_mortars._u,
                                                     _U, _F,
                                                     mpi_mortars._node_indices,
                                                     mpi_mortars._normal_directions,
                                                     (; U_threaded, F_threaded))
    return mpi_mortars, mpi_mortars_lw
end

end # muladd
