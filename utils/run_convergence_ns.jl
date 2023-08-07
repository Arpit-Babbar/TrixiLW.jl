using Trixi, TrixiLW
using DelimitedFiles
trixi_loc = pathof(Trixi)
trimmed_c = findlast("/src/Trixi.jl", trixi_loc)[1]
trixi_loc = trixi_loc[1:trimmed_c]

n_levels = 5
mdrk_errors = zeros(n_levels)
lw_errors = zeros(n_levels)
for level in 1:n_levels
   sol_mdrk = trixi_include("$(TrixiLW.examples_dir_trixilw())/tree_2d_dgsem/elixir_navierstokes_convergence.jl",
                            initial_refinement_level = level,
                            time_discretization = TrixiLW.MDRK(),
                            tspan = (0.,1.0))
   sol_lw = trixi_include("$(TrixiLW.examples_dir_trixilw())/tree_2d_dgsem/elixir_navierstokes_convergence.jl",
                          initial_refinement_level = level,
                          time_discretization = TrixiLW.LW(),
                          tspan = (0.,1.0))
   l2_mdrk, linf_mdrk = analysis_callback(sol_mdrk)
   l2_lw, linf_lw = analysis_callback(sol_lw)
   mdrk_errors[level] = l2_mdrk[1]
   lw_errors[level] = l2_lw[1]
end

nelems = [2^level for level in 1:n_levels]
writedlm(joinpath(TrixiLW.base_dir(), "results/ns_conv_lw.txt"), zip(nelems, lw_errors))
writedlm(joinpath(TrixiLW.base_dir(), "results/ns_conv_mdrk.txt"), zip(nelems, mdrk_errors))
