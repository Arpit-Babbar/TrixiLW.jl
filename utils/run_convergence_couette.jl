using Trixi, TrixiLW
using DelimitedFiles

n_levels = 5
lw_errors = zeros(n_levels, 5)
for polydeg in 1:4
   for level in 1:n_levels
      # sol_mdrk = trixi_include("$(TrixiLW.examples_dir_trixilw())/p4est_2d_dgsem/elixir_euler_coutte.jl",
      #                          initial_refinement_level = level,
      #                          time_discretization = TrixiLW.MDRK(),
      #                          tspan = (0.,1.0))
      local sol = trixi_include("$(TrixiLW.examples_dir_trixilw())/p4est_2d_dgsem/elixir_euler_couette.jl",
                           initial_refinement_level = level,
                           tspan = (0.,1.0), polydeg = polydeg)
      l2_lw, linf_lw = analysis_callback(sol)
      # mdrk_errors[level] = l2_mdrk[1]
      lw_errors[level, 2:end] .= l2_lw
   end
   lw_errors[:,1] .= [8^2 * 4^level for level in 1:n_levels]
   writedlm(joinpath(TrixiLW.base_dir(), "results/couette_conv_$polydeg.txt"), lw_errors)
end

n_levels_p4est = 3
lw_errors_p4est = zeros(n_levels_p4est, 5)
for polydeg in 3:3
   for level in 1:n_levels_p4est
      # sol_mdrk = trixi_include("$(TrixiLW.examples_dir_trixilw())/p4est_2d_dgsem/elixir_euler_coutte.jl",
      #                          initial_refinement_level = level,
      #                          time_discretization = TrixiLW.MDRK(),
      #                          tspan = (0.,1.0))
      local sol = trixi_include("$(TrixiLW.examples_dir_trixilw())/p4est_2d_dgsem/elixir_euler_couette_p4est.jl",
                           initial_refinement_level = level,
                           tspan = (0.,1.0), polydeg = polydeg)
      l2_lw, linf_lw = analysis_callback(sol)
      # mdrk_errors[level] = l2_mdrk[1]
      lw_errors_p4est[level,1] = size(sol.u[1], 1) / (polydeg+1)^2
      lw_errors_p4est[level, 2:end] .= l2_lw
   end
   writedlm(joinpath(TrixiLW.base_dir(), "results/couette_conv_p4est_$polydeg.txt"), lw_errors_p4est)
end
