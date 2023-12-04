using Trixi: SemidiscretizationHyperbolic, SemidiscretizationHyperbolicParabolic, initialize!, SummaryCallback, timer, have_constant_speed
import DiffEqBase

# Contains all that is needed to perform LW update
struct LWUpdate{RHS,uType,SolutionArrays,SemiDiscretization}
   f::RHS
   u0::uType
   soln_arrays::SolutionArrays
   tspan::Tuple{Float64,Float64}
   p::SemiDiscretization
end

struct LWSolution{uType, tType, Problem<:LWUpdate}
   u::uType
   prob::Problem
   t::tType
   stats::DiffEqBase.Stats
end

mutable struct LWOptions{tType,CallBacks}
   adaptive::Bool
   dtmax::tType
   tolerances::NamedTuple{(:abstol, :reltol),NTuple{2,tType}} # relative, absolute tolerances
   abstol::tType
   reltol::tType
   controller::NamedTuple{(:β1, :β2, :β3),NTuple{3,tType}}
   callback::CallBacks
end

# To enable usage with Trixi's callbacks
mutable struct LWIntegrator{
   SemiDiscretization,
   uType,
   Solution <: LWSolution,
   CacheType,
   tType,
   RHS,
   CallBacks,
   LWTimeDiscretization <: AbstractLWTimeDiscretization
   }
   p::SemiDiscretization # p = parameters
   sol::Solution
   u::uType
   uprev::uType
   cache::CacheType # "universal" cache in Trixi.jl
   iter::Int
   t::tType
   tspan::Tuple{tType,tType}
   dt::tType
   f::RHS
   dtpropose::tType
   dtcache::tType
   stats::DiffEqBase.Stats # number of accepted, rejected time steps
   epsilon::OffsetVector{tType, Vector{tType}} # error estimates
   opts::LWOptions{tType, CallBacks} # isadaptive, tolerances, controller, etc
   alg::LWTimeDiscretization # time_discretization, used by summary callback
end
# Remark - It may seem natural to use the same Integrator struct that Trixi.jl uses. And, if needed
# Use its a subfield of LWIntegrator and make additions as needed, following
# https://github.com/JuliaLang/julia/issues/4935#issuecomment-877302452.
# There are two problems with it
# (a) Trixi.jl uses the ODEIntegrator from OrdinaryDiffEq.jl, that is a heavy Library
# (b) ODEIntegrator requires the algorithm type to be <: Union{OrdinaryDiffEqAlgorithm, DAEAlgorithm},
# that is unnatural to put here.
# Thus, some code repetition is currently happening, even though it could be improved at a later stage

# For some callbacks from Trixi
DiffEqBase.get_tmp_cache(integrator::LWIntegrator) = integrator.cache

function update_soln!(integrator, u, uprev, du)
   @.. u = uprev + integrator.dt * du
   return nothing
end

# sets dt in integrator, cache. Decreases dt if we are stepping over Tf
function set_dt!(integrator::LWIntegrator, dt)
   t = integrator.t
   Tf = last(integrator.tspan)
   if t + dt > Tf
      dt = Tf - t
   end

   semi = integrator.p
   # Update the dt everywhere
   semi.cache.dt[1] = dt
   integrator.dtcache = dt
   integrator.dtpropose = dt
   integrator.dt = dt
   return dt
end

function set_t_and_iter!(integrator::LWIntegrator, dt) # TODO - Remove dt from arguments
   t = integrator.t
   Tf = last(integrator.tspan)
   if t + dt > Tf
      dt = Tf - t
   end
   integrator.t += dt
   integrator.sol.t[1] += dt
   integrator.iter += 1
   integrator.stats.naccept += 1
   return nothing
end

function initialize_callbacks!(callbacks::NTuple{N, Any}, integrator::LWIntegrator) where {N}
   # Type stable iteration over tuples learnt from Trixi.jl
   callback = first(callbacks)
   remaining_callbacks = Base.tail(callbacks)
   callback.initialize(callback, integrator.u, 0.0, integrator)
   initialize_callbacks!(remaining_callbacks, integrator)
end

function initialize_callbacks!(callbacks::Tuple{}, integrator::LWIntegrator)
   nothing
end

function apply_callbacks!(callbacks::NTuple{N, Any}, integrator::LWIntegrator) where {N}
   # Type stable iteration over tuples learnt from Trixi.jl
   callback = first(callbacks)
   remaining_callbacks = Base.tail(callbacks)
   if callback.condition(integrator.u, integrator.t, integrator)
      callback.affect!(integrator)
   end
   apply_callbacks!(remaining_callbacks, integrator)
end

function apply_callbacks!(callbacks::Tuple{}, integrator::LWIntegrator)
   nothing
end

function apply_limiters!(limiters, integrator::LWIntegrator, u_ode = integrator.u)
   semi  = integrator.p
   t     = integrator.t
   for limiter! in limiters
      limiter!(u_ode, integrator, semi, t)
   end
end

# TODO - It is currently wrong. The ! means it should mutate
#        Is this inconsequential?
function DiffEqBase.u_modified!(integrator::LWIntegrator, bool::Bool)
   # integrator.u_modified = bool
   return bool
end

function isfinished(integrator::LWIntegrator)
   return integrator.t ≈ last(integrator.tspan)
end

function LWIntegrator(lw_update::LWUpdate, time_discretization, sol, callbacks, tolerances,
   ; time_step_computation, tType=Float64)
   @unpack soln_arrays, tspan = lw_update
   @unpack u0_ode, du_ode = soln_arrays # Previous time level solution and residual
   integrator_cache = (du_ode,) # Don't know what else to put here.
   u = sol.u[1]
   stats = sol.stats
   iter = 0
   t = first(tspan)
   tType = typeof(t)
   dt = dtpropose = dtcache = zero(tType)

   controller = (; β1=0.6, β2=-0.2, β3=0.0)
   @unpack abstol, reltol = tolerances
   opts = LWOptions(isadaptive(time_step_computation), dtpropose,
      tolerances, abstol, reltol, controller,
      callbacks # TODO - Is this really the right way to pass callbacks? Why are they needed here?
   )
   epsilon = OffsetArray(ones(tType, 3), OffsetArrays.Origin(-1))
   f = lw_update.f # TODO - Trixi.jl wants it to be more generally chosen by the user
   LWIntegrator(lw_update.p, sol, u, u0_ode, integrator_cache, iter, t, tspan, dt, f,
      dtpropose, dtcache, stats, epsilon,
      opts,
      time_discretization)
end

function compute_dt(semi::SemidiscretizationHyperbolic,
   mesh::Union{TreeMesh,StructuredMesh,UnstructuredMesh2D,P4estMesh},
   time_step_computation::CFLBased, integrator)
   t = integrator.t
   u_ode = integrator.u
   @unpack equations, solver, cache = semi
   @unpack cfl_number = time_step_computation
   u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)

   dt = Trixi.@trixi_timeit timer() "calculate dt" cfl_number * max_dt(
      u, t, mesh, have_constant_speed(equations), equations, solver, cache)
   return dt
end

function compute_dt(semi::SemidiscretizationHyperbolicParabolic,
   mesh::Union{TreeMesh, P4estMesh}, time_step_computation::CFLBased,
   integrator)
   t = integrator.t
   u_ode = integrator.u
   @unpack equations, solver, cache, equations_parabolic = semi
   @unpack cfl_number = time_step_computation
   u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)

   dt_adv = Trixi.@trixi_timeit timer() "calculate dt" cfl_number * max_dt(
      u, t, mesh, have_constant_speed(equations), equations, solver, cache)

   dt_visc = Trixi.@trixi_timeit timer() "calculate dt" cfl_number * max_dt(
      u, t, mesh, equations_parabolic, solver, cache)

   dt = min(dt_adv, dt_visc)

   return dt
end

function perform_step!(integrator, limiters, callbacks, lw_update,
                       time_step_computation::CFLBased, ::LW)
   semi = integrator.p
   @unpack mesh = semi
   dt = compute_dt(semi, mesh, time_step_computation, integrator)
   dt = set_dt!(integrator, dt)
   @unpack u, uprev, epsilon = integrator
   @unpack soln_arrays = lw_update
   rhs! = lw_update.f
   @unpack du_ode, u0_ode = soln_arrays         # Vectors form for compability with callbacks
   @.. uprev = u

   rhs!(du_ode, u, semi, integrator.t)        # Compute du = u^{n+1}-dt*u^n
   update_soln!(integrator, u, uprev, du_ode) # u += dt * du
   apply_limiters!(limiters, integrator)
   set_t_and_iter!(integrator, dt)

   return nothing
end

# This will allow both LW and FR. For debugging, we plant to rewrite the RKFR from Trixi.
function solve_lwfr(lw_update, callbacks, dt_initial, tolerances;
   time_step_computation=CFLBased(), limiters=(;))
   @unpack soln_arrays, tspan = lw_update
   semi = lw_update.p
   @unpack du_ode, u0_ode = soln_arrays            # Vectors form for compability with callbacks
   u = compute_coefficients(first(tspan), semi)    # u satisfying initial condition
   sol = LWSolution([u], lw_update, [0.0], DiffEqBase.Stats(0))
   @unpack solver = semi
   time_discretization = get_time_discretization(solver)
   integrator = LWIntegrator(lw_update, time_discretization, sol, callbacks, tolerances,
      time_step_computation=time_step_computation)

   initialize_callbacks!(callbacks, integrator)

   # Initialize the dt in integrator, semi.cache
   set_dt!(integrator, dt_initial)
   apply_limiters!(limiters, integrator)
   apply_callbacks!(callbacks, integrator) # stepsize, analysis callbacks
   while !(isfinished(integrator)) # Check t < final_time
      perform_step!(integrator, limiters, callbacks, lw_update, time_step_computation,
                    time_discretization)
      apply_callbacks!(callbacks, integrator)
      apply_limiters!(limiters, integrator)
   end

   println("Total failed time steps = ", integrator.stats.nreject)

   return sol
end

include(solvers_dir() * "/adaptive.jl")
include(solvers_dir() * "/mdrk.jl")
