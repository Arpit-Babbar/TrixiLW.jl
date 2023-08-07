using SSFR
using Trixi: trixi_include
using Plots
# gr()

include("reproduce_base.jl") # defines rep_dir, out_dir, etc.

starting_dir = pwd()
cd(rep_dir())
mkpath(data_dir())

include("plotting.jl")


# composite
composite_dir = joinpath(data_dir(), "composite")
mkpath(composite_dir)
sol_composite_blend_fo = trixi_include("run_rotate_composite.jl", limiter = :blend, blend_type = :FO,
                                    saveto = joinpath(composite_dir, "blend_fo"))
plot2d(sol_composite_blend_fo, "composite", "blend_FO")
sol_composite_blend_mh = trixi_include("run_rotate_composite.jl", limiter = :blend, blend_type = :MH,
                                    saveto = joinpath(composite_dir, "blend_mh"))
plot2d(sol_composite_blend_mh, "composite", "blend_MH")
sol_composite_tvb = trixi_include("run_rotate_composite.jl", limiter = :tvb,
                                  saveto = joinpath(composite_dir, "tvb"))
plot2d(sol_composite_tvb, "composite", "tvb")

# Convergence test_case
let
   dir = joinpath(data_dir(), "isentropic")
   mkpath(dir)
   grid_sizes = [10, 20, 40, 80, 160, 320]
   nsamples = length(grid_sizes)
   l2_errors, l1_errors = (zeros(nsamples) for _ in 1:2)
   for limiter_name in ( :blend, #= :no_limiter,=# )
      for degree in 3:4
         for i in 1:nsamples
            nx = grid_sizes[i]
            global sol_isentropic = trixi_include("run_isentropic.jl", grid_size = [nx, nx],
                                                degree = degree, cfl_safety_factor = 0.95,
                                                limiter = limiter_name)
            l1_errors[i] = sol_isentropic["errors"]["l1_error"]
            l2_errors[i] = sol_isentropic["errors"]["l2_error"]
            writedlm(joinpath(dir, "l1_$(degree)_$(limiter_name).txt"), zip(grid_sizes, l1_errors))
            writedlm(joinpath(dir, "l2_$(degree)_$(limiter_name).txt"), zip(grid_sizes, l2_errors))
         end
      end
   end
end
