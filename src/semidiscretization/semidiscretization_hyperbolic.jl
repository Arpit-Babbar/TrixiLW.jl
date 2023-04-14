using Trixi: @trixi_timeit, SemidiscretizationHyperbolic, boundary_condition_periodic,
             digest_boundary_conditions, wrap_array

function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t, tolerances = (;abstol = 0.0, reltol = 0.0), volume_integral = calc_volume_integral!)
   @unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

   u  = wrap_array(u_ode,  mesh, equations, solver, cache)
   du = wrap_array(du_ode, mesh, equations, solver, cache)

   # TODO: Taal decide, do we need to pass the mesh?
   time_start = time_ns()
   @trixi_timeit timer() "rhs!" rhs!(du, u,
   t, mesh, equations, initial_condition, boundary_conditions, source_terms, solver,
   time_discretization(solver),
   cache, tolerances, volume_integral)
   runtime = time_ns() - time_start
   put!(semi.performance_counter, runtime)

   return nothing
 end

 function SemidiscretizationHyperbolic(mesh,
                                       time_discretization::LW,
                                       equations, initial_condition, solver;
                                       source_terms=nothing,
                                       boundary_conditions=boundary_condition_periodic,
                                       # `RealT` is used as real type for node locations etc.
                                       # while `uEltype` is used as element type of solutions etc.
                                       RealT=real(solver), uEltype=RealT,
                                       initial_cache=NamedTuple())

   cache = (; create_cache(mesh, equations, solver, RealT, uEltype)..., initial_cache...)

   cache = (; create_cache(mesh, equations, time_discretization, solver, RealT, uEltype, cache)...,
              cache...) # LW additions

   # TODO - Should I reuse similar constructor from Trixi passing new cache objects
   # to initial_cache?
   _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver, cache)

   semi = SemidiscretizationHyperbolic{typeof(mesh), typeof(equations), typeof(initial_condition), typeof(_boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
     mesh, equations, initial_condition, _boundary_conditions, source_terms, solver, cache)

   return semi

 end