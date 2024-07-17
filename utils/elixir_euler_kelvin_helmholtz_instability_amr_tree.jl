using TrixiLW
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_kelvin_helmholtz_instability

surface_flux = flux_lax_friedrichs
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
surface_flux=flux_lax_friedrichs
shock_indicator = IndicatorHennemannGassner(equations, basis,
					    alpha_max = 0.002,
					    alpha_min = 0.0001,
					    alpha_smooth = true,
					    variable = density_pressure)
volume_integral = TrixiLW.VolumeIntegralFRShockCapturing(shock_indicator;
volume_flux_fv=surface_flux,
reconstruction=TrixiLW.FirstOrderReconstruction(),
# reconstruction=TrixiLW.MUSCLHancockReconstruction()
)

solver = DGSEM(polydeg=polydeg,surface_flux = surface_flux ,
               volume_integral=volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
tree_per_dimension = (2,2)
# mesh = P4estMesh(tree_per_dimension, polydeg=polydeg, 
# 		 coordinates_min = coordinates_min,
# 		 coordinates_max = coordinates_max,
#                  initial_refinement_level=8)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=9,
                n_cells_max=30_00_000) 


semi = TrixiLW.SemidiscretizationHyperbolic(mesh, get_time_discretization(solver),
 equations, initial_condition, solver,
 )

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
lw_update = TrixiLW.semidiscretize(semi, get_time_discretization(solver), tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# amr_indicator = IndicatorHennemannGassner(semi,
#                                           alpha_max=1.0,
#                                           alpha_min=0.0001,
#                                           alpha_smooth=false,
#                                           variable=Trixi.density)

# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level=4,
#                                       med_level=0, med_threshold=0.0003, # med_level = current level
#                                       max_level=6, max_threshold=0.003)

# amr_callback = AMRCallback(semi, amr_controller,
#                            interval=1,
#                            adapt_initial_condition=true,
#                            adapt_initial_condition_only_refine=true)

callbacks = ( summary_callback,
                        #=analysis_callback,=# alive_callback,
                        save_solution
                        # amr_callback
			)

###############################################################################
# run the simulation


time_int_tol = 1e-6
tolerances = (;abstol = time_int_tol, reltol = time_int_tol);
dt_initial = 1e-8;
cfl_number = 0.3
@time sol = TrixiLW.solve_lwfr(lw_update, callbacks, dt_initial, tolerances,
                      # time_step_computation = TrixiLW.Adaptive(),
                      time_step_computation = TrixiLW.CFLBased(cfl_number),
                      # limiters = (;stage_limiter!)
                      );

summary_callback() # print the timer summary

