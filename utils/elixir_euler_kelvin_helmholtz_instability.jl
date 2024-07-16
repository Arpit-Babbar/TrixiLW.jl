using Trixi, TrixiLW
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 7.0/5.0
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x_, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  x, y = x_[1], x_[2]
  w0 = 0.1
  sigma = 0.05 / sqrt(2.0)
  p = 2.5
  if 0.25 < y < 0.75
     rho = 2.0
     v1  = 0.5
  else
     rho = 1.0
     v1  = -0.5
  end
  v2  = w0 * sin(4.0*pi*x)
  exp_term  = exp( - ( y - 0.25 )^2 / (2.0 * sigma^2) )
  exp_term += exp( - ( y - 0.75 )^2 / (2.0 * sigma^2) )
  v2 *= exp_term
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

surface_flux = flux_lax_friedrichs
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(indicator_sc;
  volume_integralFR = TrixiLW.VolumeIntegralFR(TrixiLW.LW()),
  volume_flux_fv=surface_flux,
  # reconstruction = TrixiLW.FirstOrderReconstruction(),
  # reconstruction=TrixiLW.MUSCLReconstruction()
  reconstruction=TrixiLW.MUSCLHancockReconstruction()
)
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
  volume_integral=volume_integral)


coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=4, initial_refinement_level=3,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=(true, true))
semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
                                            equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=2000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level=0,
#                                       med_level=3, med_threshold=0.002,
#                                       max_level=5, max_threshold=0.0001)

# amr_callback = AMRCallback(semi, amr_controller,
#                            interval=1,
#                            adapt_initial_condition=true,
#                            adapt_initial_condition_only_refine=true)
# visualization_callback = VisualizationCallback(interval=100,
#    save_initial_solution=true,
#    save_final_solution=true)
callbacks = (;summary_callback, analysis_callback, alive_callback,
              save_solution
              # amr_callback,
              # visualization_callback
              )

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                     variables=(Trixi.density, pressure))

###############################################################################
# run the simulation
time_int_tol = 1e-8
tolerances = (; abstol=time_int_tol, reltol=time_int_tol);
dt_initial = 1e-6;
cfl_number = 0.3
sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
  # time_step_computation = TrixiLW.Adaptive(),
  time_step_computation=TrixiLW.CFLBased(cfl_number),
  limiters=(; stage_limiter!)
);
summary_callback() # print the timer summary
