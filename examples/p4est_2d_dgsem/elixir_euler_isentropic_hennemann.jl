using Trixi, TrixiLW
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

Lx() = 0.1
T0() = 300.0
M0() = 0.5
Rgas() = 287.15
u0(equations) = M0()*sqrt(equations.gamma * Rgas() * T0())

function initial_value_vortex_henneman(x_,t,equations)
   β = 0.2 # vortex strength
   xv = yv = 0.5 * Lx()
   Rv = 0.005
   γ = equations.gamma
   Cp = Rgas()*γ/(γ-1.0)
   p0 = 10^5
   ρ0 = p0/(Rgas() * T0())
   x, y   = x_
   r = sqrt( (x-xv)^2 + (y-yv)^2 ) / Rv
   u0_ = u0(equations)
   ux = u0_ * (1.0 - β*(y - yv)/Rv * exp(-0.5*r^2))
   uy = u0_ * β * (x-xv)/Rv * exp(-0.5*r^2)
   T  = T0() - (u0_ * β)^2 / (2.0*Cp) * exp(-r^2)
   ρ  = ρ0 * (T / T0())^(1.0 / (γ-1.0))
   p  = ρ * Rgas() * T
   return prim2cons((ρ, ux, uy, p), equations)
end

initial_condition = initial_value_vortex_henneman

solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=TrixiLW.VolumeIntegralFR(TrixiLW.LW()))

refinement_level = 2
cells_per_dimension = (2^refinement_level * 16, 2^refinement_level * 16)

function mapping_henneman_isentropic(ξ_, η_)
   ξ, η = 0.5*(ξ_+1), 0.5*(η_+1.0)
   Ax = Ay = Lx_ = Ly = Lx()
   x = ξ*Lx_ - Ax*Ly*sinpi(2*η)
   y = η*Ly  + Ay*Lx_*sinpi(2*ξ)
   return (x,y)
end

mesh = P4estMesh(cells_per_dimension, mapping = mapping_henneman_isentropic, polydeg = 4)

cfl_number = 0.25
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
 equations, initial_condition, solver,
#  source_terms = source_terms_convergence_test
 )

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, Lx()/u0(equations))
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

visualization_callback = VisualizationCallback(interval=100,
   save_initial_solution=true,
   save_final_solution=true)

save_restart = SaveRestartCallback(interval=10000,
                                   save_final_restart=true)
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = ( analysis_callback, alive_callback, save_restart,
               save_solution,
               # visualization_callback
            );

###############################################################################
# run the simulation

time_int_tol = 1e-8
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-3;
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                     #  time_step_computation = TrixiLW.Adaptive()
                      time_step_computation = TrixiLW.CFLBased(cfl_number)
                      );
summary_callback() # print the timer summary
