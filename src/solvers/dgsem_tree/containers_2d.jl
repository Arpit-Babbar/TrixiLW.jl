import Trixi: init_mpi_interfaces, nmpiinterfaces, nvariables, nnodes
using Trixi: eachinterface, count_required_mpi_interfaces, ElementContainer2D, ParallelTreeMesh, init_mpi_interfaces!

function create_interface_cache(mesh::Union{TreeMesh{2},UnstructuredMesh2D,P4estMesh{2}}, equations, dg,
   uEltype, RealT, cache, time_discretization)
   n_interfaces = ninterfaces(mesh, dg, cache, time_discretization)
   n_variables = nvariables(equations)
   n_nodes = nnodes(dg)
   NDIMS = ndims(equations)
   nan_uEltype = convert(uEltype, NaN)
   nan_RealT = convert(RealT, NaN)
   _fn_low = fill(nan_uEltype, 2 * n_variables * n_nodes * n_interfaces)
   _u = fill(nan_uEltype, 2 * n_variables * n_nodes * n_interfaces)
   _U = fill(nan_uEltype, 2 * n_variables * n_nodes * n_interfaces)
   _f = fill(nan_uEltype, 2 * n_variables * n_nodes * n_interfaces)
   _inverse_jacobian = fill(nan_RealT, n_nodes * n_interfaces)
   wrap_(u) = unsafe_wrap(Array{uEltype,NDIMS + 2}, pointer(u),
      (2, n_variables, n_nodes, n_interfaces))
   u, U, f, fn_low = wrap_.((_u, _U, _f, _fn_low))
   inverse_jacobian = unsafe_wrap(Array{RealT,NDIMS}, pointer(_inverse_jacobian),
      (n_nodes, n_interfaces))
   load_inverse_jacobian!(inverse_jacobian, mesh, eachinterface(dg, cache), dg, cache)
   return LWInterfaceContainer(u, U, f, fn_low, inverse_jacobian,
      _u, _U, _f, _fn_low, _inverse_jacobian)
end

function load_inverse_jacobian!(inverse_jacobian, mesh::TreeMesh, interface_range, dg, cache)
   inverse_jacobian .= cache.elements.inverse_jacobian[1] # It is constant for tree mesh
end

function create_boundary_cache(mesh::Union{TreeMesh{2}}, equations, dg, uEltype, RealT,
   cache, outer_cache, time_discretization)
   n_boundaries = nboundaries(mesh, dg, cache, time_discretization)
   n_variables = nvariables(equations)
   n_nodes = nnodes(dg)
   nan_uEltype = convert(uEltype, NaN)
   _U, _u, _f = (fill(nan_uEltype, 2 * n_variables * n_nodes * n_boundaries) for _ in 1:3)
   wrap_(u) = unsafe_wrap(Array, pointer(u), (2, n_variables, n_nodes, n_boundaries))
   U, u, f = wrap_.((_U, _u, _f))
   return LWBoundariesContainer(U, u, f, _U, _u, _f, outer_cache)
end

# Container data structure (structure-of-arrays style) for DG MPI interfaces
mutable struct MPIInterfaceContainerLW2D{uEltype <: Real} <: AbstractContainer
   u::Array{uEltype, 4}            # [leftright, variables, i, interfaces]
   U::Array{uEltype, 4}
   F::Array{uEltype, 4}
   local_neighbor_ids::Vector{Int} # [interfaces]
   orientations::Vector{Int}       # [interfaces]
   remote_sides::Vector{Int}       # [interfaces]
   # internal `resize!`able storage
   _u::Vector{uEltype}
   _U::Vector{uEltype}
   _F::Vector{uEltype}
end

nvariables(mpi_interfaces::MPIInterfaceContainerLW2D) = size(mpi_interfaces.u, 2)
nnodes(mpi_interfaces::MPIInterfaceContainerLW2D) = size(mpi_interfaces.u, 3)
Base.eltype(mpi_interfaces::MPIInterfaceContainerLW2D) = eltype(mpi_interfaces.u)

# See explanation of Base.resize! for the element container
# For AMR, to be tested
function Base.resize!(mpi_interfaces::MPIInterfaceContainerLW2D, capacity)
   n_nodes = nnodes(mpi_interfaces)
   n_variables = nvariables(mpi_interfaces)
   @unpack _u, _U, _F, local_neighbor_ids, orientations, remote_sides = mpi_interfaces

   resize!(_u, 2 * n_variables * n_nodes * capacity)
   resize!(_U, 2 * n_variables * n_nodes * capacity)
   resize!(_F, 2 * n_variables * n_nodes * capacity)

   mpi_interfaces.u = unsafe_wrap(Array, pointer(_u),
                                  (2, n_variables, n_nodes, capacity))
   mpi_interfaces.u = unsafe_wrap(Array, pointer(_U),
                                  (2, n_variables, n_nodes, capacity))
   mpi_interfaces.u = unsafe_wrap(Array, pointer(_F),
                                  (2, n_variables, n_nodes, capacity))

   resize!(local_neighbor_ids, capacity)

   resize!(orientations, capacity)

   resize!(remote_sides, capacity)

   return nothing
end

function MPIInterfaceContainerLW2D{uEltype}(capacity::Integer, n_variables,
                                            n_nodes) where {uEltype <: Real}
   nan = convert(uEltype, NaN)

   # Initialize fields with defaults
   _u = fill(nan, 2 * n_variables * n_nodes * capacity)
   _U = fill(nan, 2 * n_variables * n_nodes * capacity)
   _F = fill(nan, 2 * n_variables * n_nodes * capacity)
   u = unsafe_wrap(Array, pointer(_u), (2, n_variables, n_nodes, capacity))
   U = unsafe_wrap(Array, pointer(_U), (2, n_variables, n_nodes, capacity))
   F = unsafe_wrap(Array, pointer(_F), (2, n_variables, n_nodes, capacity))

   local_neighbor_ids = fill(typemin(Int), capacity)

   orientations = fill(typemin(Int), capacity)

   remote_sides = fill(typemin(Int), capacity)

   return MPIInterfaceContainerLW2D{uEltype}(u, U, F, local_neighbor_ids, orientations,
                                             remote_sides,
                                             _u, _U, _F)
end

# Return number of interfaces
function nmpiinterfaces(mpi_interfaces::MPIInterfaceContainerLW2D)
   length(mpi_interfaces.orientations)
end


# Create MPI interface container and initialize MPI interface data in `elements`.
function init_mpi_interfaces(cell_ids, mesh::ParallelTreeMesh, time_discretization::AbstractLWTimeDiscretization,
                             elements::ElementContainer2D)
   # Initialize container
   n_mpi_interfaces = count_required_mpi_interfaces(mesh, cell_ids)
   mpi_interfaces = MPIInterfaceContainerLW2D{eltype(elements)}(n_mpi_interfaces,
                                                                nvariables(elements),
                                                                nnodes(elements))

   # Connect elements with interfaces
   init_mpi_interfaces!(mpi_interfaces, elements, mesh)
   return mpi_interfaces
end