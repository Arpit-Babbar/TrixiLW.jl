using Trixi: AbstractSemidiscretization, mesh_equations_solver_cache,
             PositivityPreservingLimiterZhangShu
using TrixiLW: LWIntegrator
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

function (limiter!::PositivityPreservingLimiterZhangShu)(
   u_ode, integrator::LWIntegrator, semi::AbstractSemidiscretization, t)
   u = wrap_array(u_ode, semi)
   @trixi_timeit timer() "positivity-preserving limiter" my_limiter_zhang_shu!(
      u, limiter!.thresholds, limiter!.variables, mesh_equations_solver_cache(semi)...)
end


# Iterate over tuples in a type-stable way using "lispy tuple programming",
# similar to https://stackoverflow.com/a/55849398:
# Iterating over tuples of different functions isn't type-stable in general
# but accessing the first element of a tuple is type-stable. Hence, it's good
# to process one element at a time and replace iteration by recursion here.
# Note that you shouldn't use this with too many elements per tuple since the
# compile times can increase otherwise - but a handful of elements per tuple
# is definitely fine.
function my_limiter_zhang_shu!(u, thresholds::NTuple{N,<:Real}, variables::NTuple{N,Any},
   mesh, equations, solver, cache) where {N}
   threshold = first(thresholds)
   remaining_thresholds = Base.tail(thresholds)
   variable = first(variables)
   remaining_variables = Base.tail(variables)

   my_limiter_zhang_shu!(u, threshold, variable, mesh, equations, solver, cache)
   my_limiter_zhang_shu!(u, remaining_thresholds, remaining_variables, mesh, equations, solver, cache)
   return nothing
end

# terminate the type-stable iteration over tuples
function my_limiter_zhang_shu!(u, thresholds::Tuple{}, variables::Tuple{},
   mesh, equations, solver, cache)
   nothing
end


include("positivity_zhang_shu_dg2d.jl")


end # @muladd
