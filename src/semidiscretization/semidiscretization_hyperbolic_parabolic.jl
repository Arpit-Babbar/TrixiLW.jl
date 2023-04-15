using Trixi: compute_coefficients, wrap_array, default_parabolic_solver,
             boundary_condition_periodic, real, digest_boundary_conditions,
             create_cache_parabolic, @trixi_timeit, timer

import Trixi: SemidiscretizationHyperbolicParabolic

# This name is terrible!
function semidiscretize(semi::SemidiscretizationHyperbolicParabolic,
  time_discretization::AbstractLWTimeDiscretization,
  tspan)
  # Create copies of u_ode here!!
  u0_ode  = compute_coefficients(first(tspan), semi)
  du_ode  = similar(u0_ode)

  soln_arrays = (;u0_ode, du_ode)

  return LWUpdate(rhs!, soln_arrays, tspan, semi)
end

function rhs!(du_ode, u_ode,
   semi::SemidiscretizationHyperbolicParabolic,
   t, tolerances = (;abstol = 0.0, reltol = 0.0))
  @unpack mesh, equations, equations_parabolic, initial_condition, boundary_conditions,
  solver_parabolic, cache, cache_parabolic, boundary_conditions_parabolic, source_terms, solver = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations, equations_parabolic,
    initial_condition, boundary_conditions,
    boundary_conditions_parabolic, source_terms, solver,
    solver_parabolic,
    get_time_discretization(solver), cache,
    cache_parabolic,
    tolerances)
  runtime = time_ns() - time_start
  # This is a struct in Trixi with two elements for counting
  # times of parabolic and hyperbolic part separately.
  # For Lax-Wendroff there needs to be only one because
  # we handle parabolic and hyperbolic part together.
  put!(semi.performance_counter.counters[1], runtime)
  put!(semi.performance_counter.counters[2], runtime)

  return nothing
end

function SemidiscretizationHyperbolicParabolic(mesh, time_discretization::AbstractLWTimeDiscretization,
  equations::Tuple,
  initial_condition, solver;
  solver_parabolic=default_parabolic_solver(),
  source_terms=nothing,
  boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic),
  # `RealT` is used as real type for node locations etc.
  # while `uEltype` is used as element type of solutions etc.
  RealT=real(solver), uEltype=RealT,
  initial_caches=(NamedTuple(), NamedTuple()))

  equations_hyperbolic, equations_parabolic = equations
  boundary_conditions_hyperbolic, boundary_conditions_parabolic = boundary_conditions
  initial_hyperbolic_cache, initial_cache_parabolic = initial_caches


  return SemidiscretizationHyperbolicParabolic(mesh, time_discretization, equations_hyperbolic, equations_parabolic,
    initial_condition, solver; solver_parabolic, source_terms,
    boundary_conditions=boundary_conditions_hyperbolic,
    boundary_conditions_parabolic=boundary_conditions_parabolic,
    RealT, uEltype, initial_cache=initial_hyperbolic_cache,
    initial_cache_parabolic=initial_cache_parabolic)
end

function SemidiscretizationHyperbolicParabolic(mesh, time_discretization::AbstractLWTimeDiscretization,
                                               equations, equations_parabolic, initial_condition,
                                               solver;
                                               solver_parabolic=default_parabolic_solver(),
                                               source_terms=nothing,
                                               boundary_conditions=boundary_condition_periodic,
                                               boundary_conditions_parabolic=boundary_condition_periodic,
                                               # `RealT` is used as real type for node locations etc.
                                               # while `uEltype` is used as element type of solutions etc.
                                               RealT=real(solver), uEltype=RealT,
                                               initial_cache=NamedTuple(),
                                               initial_cache_parabolic=NamedTuple())


  cache = (; create_cache(mesh, equations, solver, RealT, uEltype)..., initial_cache...)

  cache = (; create_cache(mesh, equations, time_discretization, solver, RealT, uEltype, cache)...,
             cache...) # LW additions

  _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver, cache)

  _boundary_conditions_parabolic = digest_boundary_conditions(boundary_conditions_parabolic,
                                                              mesh, solver, cache)


  cache_parabolic = (; create_cache_parabolic(mesh, equations, equations_parabolic,
                                              solver, solver_parabolic, RealT, uEltype)...,
                       initial_cache_parabolic...)

  cache_parabolic = (; create_cache(mesh, equations_parabolic, time_discretization, solver,
                                    RealT, uEltype, cache_parabolic)...,
                       cache_parabolic...) # LW Additions

  SemidiscretizationHyperbolicParabolic{
    typeof(mesh),typeof(equations),typeof(equations_parabolic),
    typeof(initial_condition),typeof(_boundary_conditions),typeof(_boundary_conditions_parabolic),
    typeof(source_terms),typeof(solver),typeof(solver_parabolic),typeof(cache),typeof(cache_parabolic)
    }(
    mesh, equations, equations_parabolic, initial_condition,
    _boundary_conditions, _boundary_conditions_parabolic, source_terms,
    solver, solver_parabolic, cache, cache_parabolic)
end