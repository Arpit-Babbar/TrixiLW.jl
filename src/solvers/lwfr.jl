using Trixi: SemidiscretizationHyperbolic, SemidiscretizationHyperbolicParabolic, initialize!, SummaryCallback, timer, have_constant_speed

struct LWSolution{uType, tType, Problem}
   u::uType
   prob::Problem
   t::tType
end

struct LWProblem{F,uType,tType,SemiDiscretization}
   f::F
   u0::uType
   p::SemiDiscretization
   tspan::Tuple{tType,tType}
end

# Contains all that is needed to perform LW update
struct LWUpdate{RHS,SolutionArrays,SemiDiscretization}
   rhs!::RHS
   soln_arrays::SolutionArrays
   tspan::Tuple{Float64,Float64}
   semi::SemiDiscretization
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
   SolType,
   Solution,
   CacheType,
   tType,
   F,
   Algorithm,
   CallBacks
   }
   p::SemiDiscretization # Prolly p = parameters in ODE integrators
   sol::Solution
   u::SolType
   uprev::SolType
   cache::CacheType
   iter::Int
   t::tType
   tspan::Tuple{tType,tType}
   dt::tType
   f::F
   dtpropose::tType
   dtcache::tType
   stats::DiffEqBase.DEStats
   epsilon::OffsetVector{tType, Vector{tType}}
   opts::LWOptions{tType, CallBacks} # OrdinaryDiffEq.DEOptions originally
   old_nelements::Int
   old_ninterfaces::Int
   old_nboundaries::Int
   old_nmortars::Int
   alg::Algorithm
end

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
   integrator.iter += 1
   integrator.stats.naccept += 1
   return nothing
end

function initialize_callbacks!(callbacks, integrator::LWIntegrator)
   # TODO - iterating over tuple of functions is not type stable
   for callback in callbacks
      initialize!(callback, integrator.u, 0.0, integrator)
   end
end

function apply_callbacks!(callbacks, integrator::LWIntegrator)
   # TODO - iterating over tuple of functions is not type stable
   for callback in callbacks # run callbacks
      if callback.condition(integrator.u, integrator.t, integrator)
         callback.affect!(integrator)
      end
   end
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
   @unpack soln_arrays, semi, tspan = lw_update
   @unpack u0_ode, du_ode = soln_arrays # Previous time level solution and residual
   integrator_cache = (du_ode,) # Don't know what else to put here.
   u = sol.u
   iter = 0
   t = first(tspan)
   tType = typeof(t)
   dt = dtpropose = dtcache = zero(tType)
   stats = DiffEqBase.DEStats(0)

   controller = (; β1=0.6, β2=-0.2, β3=0.0)
   @unpack abstol, reltol = tolerances
   opts = LWOptions(isadaptive(time_step_computation), dtpropose,
      tolerances, abstol, reltol, controller,
      callbacks # Is this really the right way to pass callbacks?
   )
   epsilon = OffsetArray(ones(tType, 3), OffsetArrays.Origin(-1))
   f = (u, v, w, t) -> 0.0 # TODO - What is this supposed to be?
   n_elements = nelements(semi.solver, semi.cache)
   n_interfaces = ninterfaces(semi.mesh, semi.solver, semi.cache, time_discretization)
   n_boundaries = nboundaries(semi.mesh, semi.solver, semi.cache, time_discretization)
   n_mortars = 1
   @show time_discretization
   LWIntegrator(semi, sol, u, u0_ode, integrator_cache, iter, t, tspan, dt, f,
      dtpropose, dtcache, stats, epsilon,
      opts, n_elements, n_interfaces, n_boundaries, n_mortars, time_discretization)
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
   mesh::TreeMesh, time_step_computation::CFLBased,
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
   @unpack rhs!, soln_arrays = lw_update
   @unpack du_ode, u0_ode = soln_arrays         # Vectors form for compability with callbacks
   @.. uprev = u

   rhs!(du_ode, u, semi, integrator.t)        # Compute du = u^{n+1}-dt*u^n
   update_soln!(integrator, u, uprev, du_ode) # u += dt * du
   set_t_and_iter!(integrator, dt)

   return nothing
end

# This will allow both LW and FR. For debugging, we rewrite the RKFR from Trixi.

function solve_lwfr(lw_update, callbacks, dt_initial, tolerances;
   time_step_computation=CFLBased(), limiters=(;))
   @unpack rhs!, soln_arrays, tspan, semi = lw_update
   @unpack du_ode, u0_ode = soln_arrays            # Vectors form for compability with callbacks
   prob = LWProblem(rhs!, u0_ode, semi, tspan)     # Would be an ODE problem in Trixi
   u = compute_coefficients(first(tspan), semi)    # u satisfying initial condition
   sol = LWSolution(u, prob, 0.0)
   @unpack solver = semi
   time_discretization = get_time_discretization(solver)
   integrator = LWIntegrator(lw_update, time_discretization, sol, callbacks, tolerances,
      time_step_computation=time_step_computation)

   # Initialize callbacks. e.g. resetting summary_callback
   summary_callback = SummaryCallback()
   Trixi.initialize_summary_callback(summary_callback, u, 0.0, integrator);

   initialize_callbacks!(callbacks, integrator)

   # Initialize the dt in integrator, semi.cache
   set_dt!(integrator, dt_initial)
   apply_limiters!(limiters, integrator)
   apply_callbacks!(callbacks, integrator) # stepsize, analysis callbacks
   while !(isfinished(integrator)) # Check t < final_time
      perform_step!(integrator, limiters, callbacks, lw_update, time_step_computation,
                    time_discretization)
      apply_limiters!(limiters, integrator)
      apply_callbacks!(callbacks, integrator)
   end

   return sol, summary_callback
end

include(solvers_dir() * "/adaptive.jl")
include(solvers_dir() * "/mdrk.jl")
