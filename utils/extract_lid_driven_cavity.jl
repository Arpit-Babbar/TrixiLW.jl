using Plots, DelimitedFiles
using Trixi
elixir = "../examples/tree_2d_dgsem/elixir_navierstokes_lid_driven_cavity_ghia.jl"

sol_error_lw = trixi_include(elixir,
   time_step_computation = TrixiLW.Adaptive(),
   time_int_tol = 1e-8,
   polydeg = 4)

sol_cfl = trixi_include(elixir,
   time_step_computation = TrixiLW.CFLBased(0.98),
   time_discretization = TrixiLW.LW(),
   polydeg = 4
   )

sol_mdrk = trixi_include(elixir,
   time_discretization = TrixiLW.MDRK(),
   time_step_computation = TrixiLW.Adaptive(),
   time_int_tol = 1e-4,
   polydeg = 4)


pd_y_error = PlotData1D(sol_error_lw.u, semi, slice = :y, point = (0.5,0.5), nvisnodes = 3)
pd_y_cfl = PlotData1D(sol_cfl.u, semi, slice = :y, point = (0.5,0.5), nvisnodes = 3)
pd_y_mdrk = PlotData1D(sol_mdrk.u, semi, slice = :y, point = (0.5,0.5), nvisnodes = 3)

pd_x_error = PlotData1D(sol_error_lw.u, semi, slice = :x, point = (0.5,0.5), nvisnodes = 3)
pd_x_cfl = PlotData1D(sol_cfl.u, semi, slice = :x, point = (0.5,0.5), nvisnodes = 3)
pd_x_mdrk = PlotData1D(sol_mdrk.u, semi, slice = :x, point = (0.5,0.5), nvisnodes = 3)

# Save extracted data

writedlm("data/error_lw_x_vs_u2.txt", zip(pd_y_error.x, pd_y_error.data[:,2]))
writedlm("data/cfl_x_vs_u2.txt", zip(pd_y_cfl.x, pd_y_cfl.data[:,2]))
writedlm("data/mdrk_x_vs_u2.txt", zip(pd_y_mdrk.x, pd_y_mdrk.data[:,2]))

writedlm("data/error_lw_y_vs_u1.txt", zip(pd_x_error.x, pd_x_error.data[:,3]));
writedlm("data/cfl_y_vs_u1.txt", zip(pd_x_cfl.x, pd_x_cfl.data[:,3]));
writedlm("data/mdrk_y_vs_u1.txt", zip(pd_x_mdrk.x, pd_x_mdrk.data[:,3]));

# Load and plot extracted data

error_lw_x = readdlm("data/error_lw_x_vs_u2.txt");
cfl_x = readdlm("data/cfl_x_vs_u2.txt");
mdrk_x = readdlm("data/mdrk_x_vs_u2.txt");

error_lw_y = readdlm("data/error_lw_y_vs_u1.txt");
cfl_y = readdlm("data/cfl_y_vs_u1.txt");
mdrk_y = readdlm("data/mdrk_y_vs_u1.txt");

pyplot()

gr()

p = plot(error_lw_x[:,2], error_lw_x[:,1], width = 3, label = "LW error-based", size = (500,400))
plot!(p, cfl_x[:,2], cfl_x[:,1], width = 3, label = "LW CFL-based", linestyle = :dash)
plot!(p, mdrk_x[:,2], mdrk_x[:,1], width = 3, label = "MDRK error-based", linestyle = :dashdot)
exact_data_y = readdlm("data/lid_driven_cavity_y_vs_u1_r1000.txt")
plot!(p, exact_data_y[:,2], exact_data_y[:,1], seriestype = :scatter, width = 3, label = "Ghia et al.", color = :black)
xlabel!(p, "\$ v_x \$")
ylabel!(p, "\$ y \$")
savefig(p, "data/y_vs_vx.pdf")
display(p)

p = plot(error_lw_y[:,2], error_lw_y[:,1], width = 3, label = "LW error-based", size = (500,400))
plot!(p, cfl_y[:,2], cfl_y[:,1], width = 3, label = "LW CFL-based", linestyle = :dash)
plot!(p, mdrk_y[:,2], mdrk_y[:,1], width = 3, label = "MDRK error-based", linestyle = :dashdot)
exact_data_x = readdlm("data/lid_driven_cavity_x_vs_u2_r1000.txt")
plot!(p, exact_data_x[:,2], exact_data_x[:,1], seriestype = :scatter, color = :black, width = 3,label = "Ghia et al.")
xlabel!(p, "\$ v_y \$")
ylabel!(p, "\$ x \$")
# plot!(flip = true)
savefig(p, "data/x_vs_vy.pdf")
display(p)
