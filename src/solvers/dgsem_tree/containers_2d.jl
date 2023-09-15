import Trixi: eachinterface

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
   _U_low = fill(nan_uEltype, 2 * n_variables * n_nodes * n_interfaces)
   _f = fill(nan_uEltype, 2 * n_variables * n_nodes * n_interfaces)
   _f_low = fill(nan_uEltype, 2 * n_variables * n_nodes * n_interfaces)
   _inverse_jacobian = fill(nan_RealT, n_nodes * n_interfaces)
   wrap_(u) = unsafe_wrap(Array{uEltype,NDIMS + 2}, pointer(u),
      (2, n_variables, n_nodes, n_interfaces))
   u, U, U_low, f, f_low, fn_low = wrap_.((_u, _U, _U_low, _f, _f_low, _fn_low))
   inverse_jacobian = unsafe_wrap(Array{RealT,NDIMS}, pointer(_inverse_jacobian),
      (n_nodes, n_interfaces))
   load_inverse_jacobian!(inverse_jacobian, mesh, eachinterface(dg, cache), dg, cache)
   return LWInterfaceContainer(u, U, U_low, f, f_low, fn_low, inverse_jacobian,
      _u, _U, _U_low, _f, _f_low, _fn_low, _inverse_jacobian)
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
   _U, _U_low, _u, _f, _f_low = (fill(nan_uEltype, 2 * n_variables * n_nodes * n_boundaries) for _ in 1:5)
   wrap_(u) = unsafe_wrap(Array, pointer(u), (2, n_variables, n_nodes, n_boundaries))
   U, U_low, u, f, f_low = wrap_.((_U, _U_low, _u, _f, _f_low))
   return LWBoundariesContainer(U, U_low, u, f, f_low, _U, _U_low, _u, _f, _f_low, outer_cache)
end
