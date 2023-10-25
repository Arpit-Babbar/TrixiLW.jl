# Called in AMR callback
function DiffEqBase.resize!(integrator::LWIntegrator, i::Int)
   for c in integrator.cache
      c !== nothing && Base.resize!(c, i)
   end
   Base.resize!(integrator.uprev, i)

   semi = integrator.p
   @unpack mesh, equations, solver, cache = semi
   n_elements = nelements(semi.solver, semi.cache)
   n_interfaces = ninterfaces(semi.solver, semi.cache)
   n_boundaries = nboundaries(semi.solver, semi.cache)

   resize!(cache.temporal_errors, n_elements)

   resize_element_cache!(mesh, equations, solver, cache)
   resize_interface_cache!(mesh, equations, solver, cache)
   resize_mortar_cache!(mesh, equations, solver, cache)
   resize_boundary_cache!(mesh, equations, solver, cache)

end

resize_element_cache!(
   mesh::Union{StructuredMesh,UnstructuredMesh2D}, equations, solver, cache) = nothing

function resize_element_cache!(mesh::Union{TreeMesh,P4estMesh}, equations, solver, cache)
   @unpack element_cache = cache
   @unpack _U, _F, _fn_low = element_cache

   n_variables = nvariables(equations)
   n_nodes = nnodes(solver)
   n_elements = nelements(solver, cache)
   NDIMS = ndims(equations)

   resize!(_U, n_variables * n_nodes^NDIMS * n_elements)
   resize!(_F, n_variables * NDIMS * n_nodes^NDIMS * n_elements)
   resize!(_fn_low, n_variables * n_nodes^(NDIMS - 1) * 2^NDIMS * n_elements)

   element_cache.U = unsafe_wrap(
      Array, pointer(_U), (n_variables, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
   element_cache.F = unsafe_wrap(
      Array, pointer(_F), (n_variables, NDIMS, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
   element_cache.fn_low = unsafe_wrap(
      Array, pointer(_fn_low),
      (n_variables, ntuple(_ -> n_nodes, NDIMS - 1)..., 2^NDIMS, n_elements))


   if isa(get_time_discretization(solver), MDRK) # TODO - Multiple dispatch instead?
      @unpack _us, _u_low, _U2, _S2, _F2 = element_cache.mdrk_cache

      resize!(_us, n_variables * n_nodes^NDIMS * n_elements)
      resize!(_u_low, n_variables * n_nodes^NDIMS * n_elements)
      resize!(_U2, n_variables * n_nodes^NDIMS * n_elements)
      resize!(_S2, n_variables * n_nodes^NDIMS * n_elements)
      resize!(_F2, n_variables * NDIMS * n_nodes^NDIMS * n_elements)

      element_cache.mdrk_cache.us = unsafe_wrap(
         Array, pointer(_us), (n_variables, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
      element_cache.mdrk_cache.u_low = unsafe_wrap(
         Array, pointer(_u_low), (n_variables, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
      element_cache.mdrk_cache.U2 = unsafe_wrap(
            Array, pointer(_U2), (n_variables, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
      element_cache.mdrk_cache.S2 = unsafe_wrap(
               Array, pointer(_S2), (n_variables, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
      element_cache.mdrk_cache.F2 = unsafe_wrap(
         Array, pointer(_F2), (n_variables, NDIMS, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
   end
   return nothing
end

function resize_interface_cache!(mesh::Union{TreeMesh, P4estMesh}, equations, dg, cache)
   @unpack interface_cache = cache
   @unpack _U, _u, _f, _fn_low, _inverse_jacobian = interface_cache

   n_variables = nvariables(equations)
   n_nodes = nnodes(dg)
   n_interfaces = ninterfaces(dg, cache)
   NDIMS = ndims(equations)

   resize_!(u) = resize!(u, 2 * n_variables * n_nodes * n_interfaces)

   resize_!.((_U, _u, _f, _fn_low))

   wrap_(u) = unsafe_wrap(Array, pointer(u), (2, n_variables, n_nodes, n_interfaces))

   resize!(_inverse_jacobian, n_nodes * n_interfaces)

   RealT = eltype(_inverse_jacobian)

   interface_cache.inverse_jacobian = unsafe_wrap(Array{RealT, NDIMS},
                                                   pointer(_inverse_jacobian),
                                                   (n_nodes, n_interfaces))

   load_inverse_jacobian!(interface_cache.inverse_jacobian, mesh, eachinterface(dg, cache),
                           dg, cache)

   interface_cache.U, interface_cache.u, interface_cache.f,
   interface_cache.fn_low = wrap_.((_U, _u, _f, _fn_low))

   return nothing
end

function resize_boundary_cache!(mesh::Union{StructuredMesh,UnstructuredMesh2D},
   equations, solver, cache)
   return nothing
end

function resize_boundary_cache!(mesh::TreeMesh, equations, solver, cache)
   @unpack boundary_cache = cache
   @unpack _U, _u, _f = boundary_cache

   n_variables = nvariables(equations)
   n_nodes = nnodes(solver)
   n_boundaries = nboundaries(solver, cache)

   resize_boundary_array!(u) = resize!(u, 2 * n_variables * n_nodes * n_boundaries)

   resize_boundary_array!.((_U, _u, _f))

   wrap_(u) = unsafe_wrap(Array, pointer(u), (2, n_variables, n_nodes, n_boundaries))

   boundary_cache.U, boundary_cache.u, boundary_cache.f = wrap_.((_U, _u, _f))

   return nothing
end

function resize_boundary_cache!(mesh::P4estMesh, equations, solver, cache)
   @unpack boundary_cache = cache
   @unpack _U, _u, _f = boundary_cache

   n_variables = nvariables(equations)
   n_nodes = nnodes(solver)
   n_boundaries = nboundaries(solver, cache)

   resize_boundary_array!(u) = resize!(u, n_variables * n_nodes * n_boundaries)

   resize_boundary_array!.((_U, _u, _f))

   wrap_(u) = unsafe_wrap(Array, pointer(u), (n_variables, n_nodes, n_boundaries))

   boundary_cache.U, boundary_cache.u, boundary_cache.f = wrap_.((_U, _u, _f))

   return nothing
end

resize_mortar_cache!(
   mesh::Union{StructuredMesh,UnstructuredMesh2D}, equations, dg, cache) = nothing

function resize_mortar_cache!(mesh::TreeMesh, equations, dg, cache)
   @unpack mortars, lw_mortars = cache
   n_mortars = nmortars(dg, cache)
   n_nodes = nnodes(dg)
   n_var = nvariables(equations)

   @unpack _u_upper, u_upper = mortars
   @unpack _U_upper, _U_lower, _F_upper, _F_lower, _fn_low_upper, _fn_low_lower,
   U_upper, U_lower, F_upper, F_lower, fn_low_upper, fn_low_lower = lw_mortars

   resize_mortar_array!(u) = resize!(u, 2 * n_var * n_nodes * n_mortars)

   resize_mortar_array!.((_U_upper, _U_lower, _F_upper, _F_lower, _fn_low_upper, _fn_low_lower))

   @assert length(_U_upper) == length(_u_upper) length(_U_upper), length(_u_upper)

   wrap_(u) = unsafe_wrap(Array, pointer(u), size(u_upper))

   lw_mortars.U_upper, lw_mortars.U_lower, lw_mortars.F_upper,
   lw_mortars.F_lower, lw_mortars.fn_low_upper,
   lw_mortars.fn_low_lower = wrap_.(
      (_U_upper, _U_lower, _F_upper, _F_lower, _fn_low_upper, _fn_low_lower)
   )

   return nothing
end

function resize_mortar_cache!(mesh::P4estMesh{2}, equations, dg, cache)
   @unpack mortars, lw_mortars = cache
   n_mortars = nmortars(dg, cache)
   n_nodes = nnodes(dg)
   n_var = nvariables(equations)

   @unpack _u, u = mortars
   @unpack _U, _F, _fn_low, _inverse_jacobian = lw_mortars

   resize_mortar_array!(u) = resize!(u, 2 * n_var * 2 * n_nodes * n_mortars)

   resize_mortar_array!.((_U, _F, _fn_low))

   @assert length(_U) == length(_u)

   wrap_(v) = unsafe_wrap(Array, pointer(v), size(u))

   resize!(_inverse_jacobian, n_nodes * n_mortars)

   RealT = eltype(_inverse_jacobian)
   NDIMS = ndims(equations)

   lw_mortars.inverse_jacobian = unsafe_wrap(Array{RealT, NDIMS}, pointer(_inverse_jacobian),
                                             (n_nodes, n_mortars))

   load_inverse_jacobian!(lw_mortars.inverse_jacobian, mesh, eachmortar(dg, cache), dg, cache)

   lw_mortars.U, lw_mortars.F, lw_mortars.fn_low = wrap_.( (_U, _F, _fn_low) )

   return nothing
end
