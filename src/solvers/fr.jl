using Trixi: AbstractVolumeIntegral, DG, summary_header, summary_line,
   increment_indent, summary_footer,
   TreeMesh, StructuredMesh, UnstructuredMesh2D, P4estMesh,
   AbstractIndicator, False
import Trixi: create_cache, isfinished, get_element_variables!
import Trixi
import DiffEqBase
using DiffEqBase: @..
using OffsetArrays

using MuladdMacro

@muladd begin

struct VolumeIntegralFR{TimeDiscretization<:AbstractTimeDiscretization} <: AbstractVolumeIntegral
   time_discretization::TimeDiscretization
end

get_time_discretization(volume_integral::VolumeIntegralFR) = volume_integral.time_discretization

# TODO - When is this called again?
create_cache(mesh, equations, ::VolumeIntegralFR, dg, uEltype) = (;)

Base.show(io::IO, ::LW) = print(io, "Lax-Wendroff")
Base.show(io::IO, ::MDRK) = print(io, "Multi-Derivative Runge-Kutta")
Base.show(io::IO, ::RK) = print(io, "Runge-Kutta")

function Base.show(io::IO, dg::DG{<:Any,<:Any,<:Any,<:VolumeIntegralFR})
   @nospecialize dg # reduce precompilation time

   print(io, "DG{", real(dg), "}(")
   print(io, dg.basis)
   print(io, ", ", dg.mortar)
   print(io, ", ", dg.surface_integral)
   print(io, ", ", dg.volume_integral)
   print(io, ", ", get_time_discretization(dg))
   print(io, ")")
end


function Base.show(io::IO, mime::MIME"text/plain", dg::DG{<:Any,<:Any,<:Any,<:VolumeIntegralFR})
   @nospecialize dg # reduce precompilation time

   if get(io, :compact, false)
      show(io, dg)
   else
      summary_header(io, "DG{" * string(real(dg)) * "}")
      summary_line(io, "basis", dg.basis)
      summary_line(io, "mortar", dg.mortar)
      summary_line(io, "surface integral", dg.surface_integral |> typeof |> nameof)
      show(increment_indent(io), mime, dg.surface_integral)
      summary_line(io, "volume integral", dg.volume_integral |> typeof |> nameof)
      summary_line(io, "time discretization", get_time_discretization(dg))
      summary_footer(io)
   end
end


struct VolumeIntegralFRShockCapturing{TimeDiscretization,VolumeFluxFV,Indicator,Reconstruction} <: AbstractVolumeIntegral
   volume_integralFR::VolumeIntegralFR{TimeDiscretization} # Keeping the previous struct
   volume_flux_fv::VolumeFluxFV                            # Typically Rusanov's flux
   indicator::Indicator                                    # HG's indicator with chosen parameters or something else
   reconstruction::Reconstruction                          # First Order / MUSCL-Hancock
end



get_time_discretization(volume_integral::VolumeIntegralFRShockCapturing) =
   volume_integral.volume_integralFR.time_discretization

get_time_discretization(solver::DG) = get_time_discretization(solver.volume_integral)

# Reconstruction types
struct FirstOrderReconstruction end
struct MUSCLHancockReconstruction end
struct MUSCLReconstruction end

function VolumeIntegralFRShockCapturing(indicator; volume_integralFR=VolumeIntegralFR(LW()),
   volume_flux_fv=flux_lax_friedrichs,
   reconstruction=FirstOrderReconstruction())

   return VolumeIntegralFRShockCapturing{
      typeof(get_time_discretization(volume_integralFR)), typeof(volume_flux_fv),
      typeof(indicator), typeof(reconstruction)}(
      volume_integralFR, volume_flux_fv, indicator, reconstruction)
end

# Initializes cache
create_cache(mesh, equations, ::VolumeIntegralFRShockCapturing, dg, uEltype) = (;)

function Base.show(io::IO, mime::MIME"text/plain",
   integral::VolumeIntegralFRShockCapturing)
   @nospecialize integral # reduce precompilation time

   if get(io, :compact, false)
      show(io, integral)
   else
      summary_header(io, "VolumeIntegralFRShockCapturing")
      summary_line(io, "volume flux FV", integral.volume_flux_fv)
      summary_line(io, "indicator", integral.indicator |> typeof |> nameof)
      show(increment_indent(io), mime, integral.indicator)
      summary_footer(io)
   end
end

function get_element_variables!(element_variables, u, mesh, equations,
   volume_integral::VolumeIntegralFRShockCapturing, dg, cache)
   # call the indicator to get up-to-date values for IO
   volume_integral.indicator(u, mesh, equations, dg, cache)
   get_element_variables!(element_variables, volume_integral.indicator, volume_integral)
end

# TODO - Where should this be?
function get_element_variables!(element_variables, indicator::AbstractIndicator,
   ::VolumeIntegralFRShockCapturing)
   element_variables[:indicator_shock_capturing] = indicator.cache.alpha
   return nothing
end

# Extract contravariant vector Ja^i (i = index) as SVector
@inline function get_contravariant_vector(index, contravariant_vectors, indices...)
   SVector(ntuple(@inline(dim -> contravariant_vectors[dim, index, indices...]), Val(ndims(contravariant_vectors) - 3)))
end

@inline function get_contravariant_matrix(contravariant_vectors::AbstractArray, indices...)
   Ja1 = SVector(ntuple(@inline(dim -> contravariant_vectors[dim, 1, indices...]), Val(ndims(contravariant_vectors) - 3)))
   Ja2 = SVector(ntuple(@inline(dim -> contravariant_vectors[dim, 2, indices...]), Val(ndims(contravariant_vectors) - 3)))
   return Ja1, Ja2
end

@inline function get_flux_vars(u, equations, solver::DG, indices...)
   # There is a cut-off at `n == 10` inside of the method
   # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
   # in Julia `v1.5`, leading to type instabilities if
   # more than ten variables are used. That's why we use
   # `Val(...)` below.
   # We use `@inline` to make sure that the `getindex` calls are
   # really inlined, which might be the default choice of the Julia
   # compiler for standard `Array`s but not necessarily for more
   # advanced array types such as `PtrArray`s, cf.
   # https://github.com/JuliaSIMD/VectorizationBase.jl/issues/55
   dims = Val(ndims(equations))
   ntuple(@inline(d -> Trixi.get_node_vars(u, equations, solver, d, indices...)), dims)
end

function normal_product(F, equations, normal)
   # Fn = SVector(ntuple(@inline(v -> F[v,1]*normal[1] + F[v,2]*normal[2]), Val(nvariables(equations))))
   Fn = SVector(ntuple(@inline(v -> F[1][v] * normal[1] + F[2][v] * normal[2]), Val(nvariables(equations))))
   return Fn
end

function compute_low_order_flux(u_ll, u_rr, equations, dg, normal_direction,
   surface_flux, sign_jacobian)
   fn = sign_jacobian * surface_flux(u_ll, u_rr, normal_direction, equations)
   return fn
end

end # muladd