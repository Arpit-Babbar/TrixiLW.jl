import Trixi: ninterfaces
using StaticArrays

function ninterfaces(mesh::StructuredMesh, dg, cache, ::AbstractLWTimeDiscretization)
   return 2 # Not to be used
end

function ninterfaces(::Union{LW}, cache)
   return 2 # Not to be used
end


function nboundaries(mesh::StructuredMesh, dg, cache, ::AbstractLWTimeDiscretization)
   return 2 # Not to be used
end

# function create_element_cache(::Union{StructuredMesh,UnstructuredMesh2D,P4estMesh},
#    nan_uEltype, NDIMS, n_variables, n_nodes, n_elements, n_interfaces)
#    U = fill(nan_uEltype, (n_variables, n_nodes, n_nodes, n_elements))
#    F = fill(nan_uEltype, (n_variables, NDIMS, n_nodes, n_nodes, n_elements))
#    # TODO - Do we really need interface_cache
#    Ub = fill(nan_uEltype, (NDIMS, n_variables, n_nodes, n_interfaces))
#    Fb = fill(nan_uEltype, (NDIMS, n_variables, n_nodes, n_interfaces))
#    fn_low = fill(nan_uEltype, (n_variables, n_nodes, 2 * NDIMS, n_elements))

#    return U, F, Ub, Fb, fn_low
# end


function create_interface_cache(mesh::Union{StructuredMesh{2}}, equations, dg,
   uEltype, RealT,
   cache, time_discretization)
   return nothing
end

function create_boundary_cache(mesh::Union{StructuredMesh{2}}, equations, dg,
   uEltype, RealT, cache, outer_cache, time_discretization)
   MOuter = MArray{Tuple{nvariables(equations)},Float64}
   outer_cache = alloc_for_threads(MOuter, 2)
   return (; outer_cache)
end

function create_mortar_cache(mesh::Union{StructuredMesh,UnstructuredMesh2D}, equations, dg,
   uEltype, RealT, cache, time_discretization::AbstractLWTimeDiscretization)
   return nothing
end
