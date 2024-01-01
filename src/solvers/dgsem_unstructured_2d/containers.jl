function load_inverse_jacobian!(inverse_jacobian, mesh::UnstructuredMesh2D, interface_range,
   dg, cache)
   n_nodes = nnodes(dg)
   @unpack element_ids, element_side_ids = cache.interfaces
   @unpack elements = cache
   for interface in eachinterface(dg, cache)
      primary_element = element_ids[1, interface]
      primary_side = element_side_ids[1, interface]
      for i in 1:n_nodes
         if primary_side == 1
            inverse_jacobian[i, interface] = elements.inverse_jacobian[i, 1, primary_element]
         elseif primary_side == 2
            inverse_jacobian[i, interface] = elements.inverse_jacobian[n_nodes, i, primary_element]
         elseif primary_side == 3
            inverse_jacobian[i, interface] = elements.inverse_jacobian[i, n_nodes, primary_element]
         elseif primary_side == 4
            inverse_jacobian[i, interface] = elements.inverse_jacobian[1, i, primary_element]
         else
            @assert false "Wrong primary side"
         end
      end
   end
end

function create_boundary_cache(mesh::Union{UnstructuredMesh2D,P4estMesh{2}}, equations, dg, uEltype, RealT,
   cache, outer_cache, time_discretization)
   n_boundaries = nboundaries(mesh, dg, cache, time_discretization)
   n_variables = nvariables(equations)
   n_nodes = nnodes(dg)
   nan_uEltype = convert(uEltype, NaN)
   _U, _u, _f, _fn_low = (fill(nan_uEltype, n_variables * n_nodes * n_boundaries) for _ in 1:4)
   wrap_(u) = unsafe_wrap(Array, pointer(u), (n_variables, n_nodes, n_boundaries))
   U, u, f, fn_low = wrap_.((_U, _u, _f, _fn_low))
   # TODO - Also keep u_inner_big here?
   return LWBoundariesContainer(U, u, f, fn_low, _U, _u, _f, _fn_low, outer_cache)
end