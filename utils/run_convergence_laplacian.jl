using Trixi, TrixiLW
using DelimitedFiles

n_levels = 5
mdrk_errors = zeros(n_levels)
lw_errors = zeros(n_levels)
advection_velocity = (1.0, 0.0)
elixir_name = "elixir_advection_diffusion_nonperiodic.jl"
for level in 1:n_levels
   # Standard regime
   sol_mdrk = trixi_include("$(TrixiLW.examples_dir_trixilw())/tree_2d_dgsem/$(elixir_name)",
                            initial_refinement_level = level,
                            time_discretization = TrixiLW.MDRK(),
                            advection_velocity = advection_velocity,
                            polydeg = 4,
                            tspan = (0.,1.0))
   sol_lw = trixi_include("$(TrixiLW.examples_dir_trixilw())/tree_2d_dgsem/$(elixir_name)",
                          initial_refinement_level = level,
                          time_discretization = TrixiLW.LW(),
                          advection_velocity = advection_velocity,
                          polydeg = 4,
                          tspan = (0.,1.0))
   l2_mdrk, linf_mdrk = analysis_callback(sol_mdrk)
   l2_lw, linf_lw = analysis_callback(sol_lw)
   mdrk_errors[level] = l2_mdrk[1]
   lw_errors[level] = l2_lw[1]
end

nelems = [2^level for level in 1:n_levels]
if elixir_name == "elixir_advection_diffusion_diff02.jl"
   @assert diffusivity() ≈ 5e-02
   if prod(advection_velocity .≈ (1.5,1.)) == true
      writedlm(joinpath(TrixiLW.base_dir(), "results/adv_diff_lw.txt"), zip(nelems, lw_errors))
      writedlm(joinpath(TrixiLW.base_dir(), "results/adv_diff_mdrk.txt"), zip(nelems, mdrk_errors))
   else
      @assert prod(advection_velocity .≈ (0.0,0.0)) == true
      writedlm(joinpath(TrixiLW.base_dir(), "results/heat_lw.txt"), zip(nelems, lw_errors))
      writedlm(joinpath(TrixiLW.base_dir(), "results/heat_mdrk.txt"), zip(nelems, mdrk_errors))
   end
elseif elixir_name == "elixir_advection_diffusion.jl"
   @assert elixir_name == "elixir_advection_diffusion.jl"
   @assert diffusivity() ≈ 1e-06
   @assert prod(advection_velocity .≈ (1.5,1.)) == true
   writedlm(joinpath(TrixiLW.base_dir(), "results/adv_lw.txt"), zip(nelems, lw_errors))
   writedlm(joinpath(TrixiLW.base_dir(), "results/adv_mdrk.txt"), zip(nelems, mdrk_errors))
else
   @assert elixir_name == "elixir_advection_diffusion_nonperiodic.jl"
   @assert diffusivity() ≈ 5e-02
   @assert prod(advection_velocity .≈ (1.0,0.0)) == true
   writedlm(joinpath(TrixiLW.base_dir(), "results/adv_diff_nonperiodic_lw.txt"), zip(nelems, lw_errors))
   writedlm(joinpath(TrixiLW.base_dir(), "results/adv_diff_nonperiodic_mdrk.txt"), zip(nelems, mdrk_errors))
end
