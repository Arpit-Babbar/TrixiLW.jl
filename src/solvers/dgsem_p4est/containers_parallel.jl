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
    mpi_interfaces_ = P4estMPIInterfaceContainer{NDIMS, uEltype, NDIMS + 2}(u,
                                                                           local_neighbor_ids,
                                                                           node_indices,
                                                                           local_sides,
                                                                           _u)
    init_mpi_interfaces!(mpi_interfaces_, mesh)

    # Now we just move what we get from the Trixi function into our own container
    mpi_interfaces = P4estMPIInterfaceContainerLW{NDIMS, uEltype, NDIMS + 2}(mpi_interfaces_.u,
                                                                               U, F,
                                                                               mpi_interfaces_.local_neighbor_ids,
                                                                               mpi_interfaces_.node_indices,
                                                                               mpi_interfaces_.local_sides,
                                                                               mpi_interfaces_._u,
                                                                               _U, _F)

    return mpi_interfaces
end

# TODO: mortar code

end # muladd
