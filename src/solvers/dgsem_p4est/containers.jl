function load_inverse_jacobian!(inverse_jacobian, mesh::P4estMesh{2}, interface_range, dg, cache)
   n_nodes = nnodes(dg)
   @unpack interfaces = cache
   @unpack elements = cache
   for interface in interface_range
      primary_element = interfaces.neighbor_ids[1, interface]
      primary_indices = interfaces.node_indices[1, interface]
      index_range = Base.OneTo(n_nodes)
      i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1], index_range)
      j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2], index_range)

      i_primary = i_primary_start
      j_primary = j_primary_start

      for i in 1:n_nodes
         inverse_jacobian[i, interface] = elements.inverse_jacobian[i_primary, j_primary, primary_element]
         i_primary += i_primary_step
         j_primary += j_primary_step
      end
   end
end

mutable struct L2MortarContainer_lw_P4est{RealT,uEltype<:Real,NDIMS,NDIMSP3,Temp} <: AbstractContainer
   U::Array{uEltype,NDIMSP3}       # [leftright, variables, updown, i, mortars]
   F::Array{uEltype,NDIMSP3}            # [leftright, variables, updown, i, mortars]
   fn_low::Array{uEltype,NDIMSP3}       # [leftright, variables, updown, i, mortars]
   inverse_jacobian::Array{RealT,NDIMS}
   _U::Vector{uEltype}
   _F::Vector{uEltype}
   _fn_low::Vector{uEltype}
   _inverse_jacobian::Vector{RealT}
   tmp::Temp
end

function create_mortar_cache(mesh::P4estMesh, equations, dg, uEltype, RealT, cache, time_discretization::AbstractLWTimeDiscretization)
   @unpack mortars, u_threaded = cache
   n_mortars = nmortars(dg, cache)
   n_nodes = nnodes(dg)

   # Create arrays of sizes (leftright, n_variables, updown, n_nodes, n_mortars)
   @unpack _u, u = mortars

   # We will prolong into all these quantities
   _U, _F, _fn_low = (copy(_u) for _ in 1:3)

   # For immutable objects, always use deep copy
   U_threaded, F_threaded, fn_low_threaded = (deepcopy(u_threaded) for _ in 1:3)

   wrap_(v) = unsafe_wrap(Array, pointer(v), size(u))
   U, F, fn_low = wrap_.((_U, _F, _fn_low))

   NDIMS = ndims(equations)
   _inverse_jacobian = zeros(n_nodes * n_mortars)
   inverse_jacobian = unsafe_wrap(Array{RealT,NDIMS}, pointer(_inverse_jacobian),
      (n_nodes, n_mortars))

   L2MortarContainer_lw_P4est(
      U, F, fn_low, inverse_jacobian,
      _U, _F, _fn_low, _inverse_jacobian,
      (; U_threaded, F_threaded, fn_low_threaded))
end

function create_mortar_cache(mesh::P4estMesh, equations::AbstractEquationsParabolic, dg, uEltype, RealT, cache, time_discretization::AbstractLWTimeDiscretization)
   return (;)
end
