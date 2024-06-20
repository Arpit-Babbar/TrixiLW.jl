import Trixi: init_mpi_interfaces, init_mpi_interface_node_indices!, nmpiinterfaces

using Trixi: count_required_surfaces, init_surfaces!, init_mpi_interfaces!, P4estMPICache,
             ParallelP4estMesh, P4estMPIInterfaceContainer

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct P4estMPIInterfaceContainerLW{NDIMS, uEltype <: Real, NDIMSP2} <:
               AbstractContainer
    mpi_interfaces_::P4estMPIInterfaceContainer
    U::Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]
    F::Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]

    local_neighbor_ids::Vector{Int}                     # [interface]
    node_indices::Vector{NTuple{NDIMS, Symbol}}         # [interface]
    local_sides::Vector{Int}                            # [interface]

    # internal `resize!`able storage
    _U::Vector{uEltype}
    _F::Vector{uEltype}
end

@inline function nmpiinterfaces(interfaces::P4estMPIInterfaceContainerLW)
    length(interfaces.local_sides)
end

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

    mpi_interfaces = P4estMPIInterfaceContainer{NDIMS, uEltype, NDIMS + 2}(u,
                                                                           local_neighbor_ids,
                                                                           node_indices,
                                                                           local_sides,
                                                                           _u)
    mpi_interfaceslw = P4estMPIInterfaceContainerLW{NDIMS, uEltype, NDIMS + 2}(mpi_interfaces,
                                                                               U, F,
                                                                               local_neighbor_ids,
                                                                               node_indices,
                                                                               local_sides,
                                                                               _U, _F)

    init_mpi_interfaces!(mpi_interfaceslw, mesh)

    return mpi_interfaceslw
end

# Initialize node_indices of MPI interface container
@inline function init_mpi_interface_node_indices!(mpi_interfaceslw::P4estMPIInterfaceContainerLW{2},
                                                  faces, local_side, orientation,
                                                  mpi_interface_id)
    # Align interface in positive coordinate direction of primary element.
    # For orientation == 1, the secondary element needs to be indexed backwards
    # relative to the interface.
    if local_side == 1 || orientation == 0
        # Forward indexing
        i = :i_forward
    else
        # Backward indexing
        i = :i_backward
    end
    # @show mpi_interfaceslw.node_indices

    if faces[local_side] == 0
        # Index face in negative x-direction
        mpi_interfaceslw.node_indices[mpi_interface_id] = (:begin, i)
    elseif faces[local_side] == 1
        # Index face in positive x-direction
        mpi_interfaceslw.node_indices[mpi_interface_id] = (:end, i)
    elseif faces[local_side] == 2
        # Index face in negative y-direction
        mpi_interfaceslw.node_indices[mpi_interface_id] = (i, :begin)
    else # faces[local_side] == 3
        # Index face in positive y-direction
        mpi_interfaceslw.node_indices[mpi_interface_id] = (i, :end)
    end

    return mpi_interfaceslw
end

# TODO: mortar code

end # muladd