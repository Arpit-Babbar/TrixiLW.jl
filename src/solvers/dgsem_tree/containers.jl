import Trixi: ninterfaces, nboundaries
using StaticArrays

# Called within the create_cache function of dg_2d.jl
function create_cache(mesh::Union{TreeMesh,StructuredMesh,UnstructuredMesh2D,P4estMesh},
   equations::AbstractEquations, time_discretization::LW,
   dg, RealT, uEltype, cache)

   nan_RealT   = convert(RealT, NaN)
   nan_uEltype = convert(uEltype, NaN)

   n_variables = nvariables(equations)
   n_nodes = nnodes(dg)
   n_elements = nelements(dg, cache)

   temporal_errors = fill(nan_uEltype, n_elements)

   NDIMS = ndims(equations)

   # TODO - Put U, F, fn_low in cell_cache named tuple
   # TODO - Don't pass so many things, only pass cache, equations etc.
   element_cache = create_element_cache(mesh, nan_uEltype, NDIMS, n_variables,
      n_nodes, n_elements)

   interface_cache = create_interface_cache(mesh, equations, dg, uEltype, RealT,
      cache, time_discretization)

   function alloc_for_threads(constructor, cache_size)
      nt = Threads.nthreads()
      SVector{nt}([alloc(constructor, cache_size) for _ in Base.OneTo(nt)])
   end

   # Construct `cache_size` number of objects with `constructor`
   # and store them in an SVector
   function alloc(constructor, cache_size)
      SVector{cache_size}(constructor(undef) for _ in Base.OneTo(cache_size))
   end

   # Create the result of `alloc` for each thread. Basically,
   # for each thread, construct `cache_size` number of objects with
   # `constructor` and store them in an SVector

   MOuter = MArray{Tuple{n_variables},Float64}
   outer_cache = alloc_for_threads(MOuter, 2)
   boundary_cache = create_boundary_cache(mesh, equations, dg, uEltype, RealT,
      cache, outer_cache, time_discretization)

   lw_mortars = create_mortar_cache(mesh, equations, dg, uEltype, RealT,
      cache, time_discretization)
   cfl_number = fill(nan_RealT, 1)
   dt = fill(nan_RealT, 1)

   cell_array_sizes = Dict(1 => 13, 2 => 14, 3 => 17, 4 => 18)
   degree = n_nodes - 1
   cell_array_size = cell_array_sizes[min(4, degree)]

   MArr = MArray{Tuple{n_variables,n_nodes,n_nodes},Float64}
   cell_arrays = alloc_for_threads(MArr, cell_array_size)

   lw_res_cache = (; cell_arrays)

   cache = (; element_cache, lw_res_cache, cfl_number, dt,
      temporal_errors, interface_cache, boundary_cache, lw_mortars)
   return cache
end

mutable struct LWElementContainer{uEltype<:Real,NDIMSP2,NDIMSP3}
   U::Array{uEltype,NDIMSP2}      # [variables, i, j, k, element]
   F::Array{uEltype,NDIMSP3}      # [variables, coordinate, i, j, k, element]
   fn_low::Array{uEltype,NDIMSP2} # [variable, i, j, left/right/bottom/top, elements]
   _U::Vector{uEltype}
   _F::Vector{uEltype}
   _fn_low::Vector{uEltype}
end

mutable struct LWInterfaceContainer{RealT,uEltype,NDIMS,NDIMSP2}
   u::Array{uEltype,NDIMSP2}      # [left/right, variables, i, j, k, interface]
   U::Array{uEltype,NDIMSP2}      # [left/right, variables, i, j, k, interface]
   f::Array{uEltype,NDIMSP2}      # [left/right, variables, i, j, k, interface]
   fn_low::Array{uEltype,NDIMSP2} # [left/right, variables, i, j, k, interface]
   inverse_jacobian::Array{RealT,NDIMS} # [i, j, interface]
   _u::Vector{uEltype}
   _U::Vector{uEltype}
   _f::Vector{uEltype}
   _fn_low::Vector{uEltype}
   _inverse_jacobian::Vector{RealT}
end

mutable struct LWBoundariesContainer{uEltype<:Real,D,OuterCache}
   U::Array{uEltype,D}  # [variables, i, boundaries]
   u::Array{uEltype,D}  # [variables, i, boundaries]
   f::Array{uEltype,D}  # [variables, i, boundaries]
   _U::Vector{uEltype}
   _u::Vector{uEltype}
   _f::Vector{uEltype}
   outer_cache::OuterCache
end

function create_element_cache(::Union{TreeMesh,StructuredMesh,UnstructuredMesh2D,P4estMesh},
   nan_uEltype, NDIMS, n_variables, n_nodes, n_elements)

   _U = fill(nan_uEltype, n_variables * n_nodes^NDIMS * n_elements)
   _F = fill(nan_uEltype, n_variables * NDIMS * n_nodes^NDIMS * n_elements)
   _fn_low = fill(nan_uEltype, n_variables * n_nodes^(NDIMS - 1) * 2^NDIMS * n_elements)

   uEltype = typeof(nan_uEltype)
   U = unsafe_wrap(Array{uEltype,NDIMS + 2}, pointer(_U),
                   (n_variables, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
   F = unsafe_wrap(Array{uEltype,NDIMS + 3}, pointer(_F),
                   (n_variables, NDIMS, ntuple(_ -> n_nodes, NDIMS)..., n_elements))

   fn_low = unsafe_wrap(Array{uEltype,NDIMS + 2}, pointer(_fn_low),
                        (n_variables, ntuple(_ -> n_nodes, NDIMS - 1)..., 2^NDIMS, n_elements))
   return LWElementContainer(U, F, fn_low, _U, _F, _fn_low)
end

mutable struct L2MortarContainer_lw_Tree{uEltype<:Real,Temp} <: AbstractContainer
   U_upper::Array{uEltype,4}  # [leftright, variables, i, mortars]
   U_lower::Array{uEltype,4}  # [leftright, variables, i, mortars]
   F_upper::Array{uEltype,4}  # [leftright, variables, i, mortars]
   F_lower::Array{uEltype,4}  # [leftright, variables, i, mortars]
   fn_low_upper::Array{uEltype,4}  # [leftright, variables, i, mortars]
   fn_low_lower::Array{uEltype,4}  # [leftright, variables, i, mortars]
   _U_upper::Vector{uEltype}
   _U_lower::Vector{uEltype}
   _F_upper::Vector{uEltype}
   _F_lower::Vector{uEltype}
   _fn_low_upper::Vector{uEltype}
   _fn_low_lower::Vector{uEltype}
   tmp::Temp
end

function create_mortar_cache(mesh::TreeMesh, equations, dg, uEltype, RealT,
   cache, time_discretization::LW)
   @unpack mortars = cache

   # Create arrays of sizes (2, n_variables, n_nodes, n_mortars)
   @unpack _u_upper, u_upper = mortars

   # We will prolong into all these quantities
   _U_upper, _U_lower, _F_upper, _F_lower, _fn_low_upper, _fn_low_lower = (
      copy(_u_upper) for _ in 1:6)

   wrap_(u) = unsafe_wrap(Array, pointer(u), size(u_upper))
   U_upper, U_lower, F_upper, F_lower, fn_low_upper, fn_low_lower = wrap_.((
      _U_upper, _U_lower, _F_upper, _F_lower, _fn_low_upper, _fn_low_lower))

   L2MortarContainer_lw_Tree(
      U_upper, U_lower, F_upper, F_lower, fn_low_upper, fn_low_lower,
      _U_upper, _U_lower, _F_upper, _F_lower, _fn_low_upper, _fn_low_lower,
      (;))
end

function ninterfaces(mesh::Union{TreeMesh,UnstructuredMesh2D,P4estMesh}, dg, cache, ::Union{LW}) # Add RK Type here
   return ninterfaces(dg, cache)
end

function nboundaries(mesh::Union{TreeMesh,UnstructuredMesh2D,P4estMesh}, dg, cache, ::Union{LW}) # Add RK Type here
   return nboundaries(dg, cache)
end

include(solvers_dir() * "/dgsem_tree/containers_2d.jl")