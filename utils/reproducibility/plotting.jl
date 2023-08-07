using TrixiLW: base_dir, utils_dir
import PyPlot as plt
using Plots
using DelimitedFiles
using Plots
using SimpleUnPack: @unpack
using LinearAlgebra
using Printf
# plotlyjs()
gr()
include(joinpath(utils_dir(), "reproducibility/mpl.jl"))

include(joinpath(utils_dir(), "reproducibility/reproduce_base.jl"))

function my_save_fig_python(test_case, figure, name)
   fig_dir = joinpath(rep_dir(), "figures", test_case)
   mkpath(fig_dir)
   figure.savefig(joinpath(fig_dir, name))
   return nothing
end

function my_save_fig_julia(test_case, figure, name)
   fig_dir = joinpath(rep_dir(), "figures", test_case)
   mkpath(fig_dir)
   savefig(figure, joinpath(fig_dir, name))
   return nothing
end

colors = [:green, :blue, :red, :purple]

function format_with_powers(y,_)
   if y > 1e-4
      y = Int64(y)
      return "\$$y\$"
   else
      return @sprintf "%.4E" y
   end
end

function set_ticks!(ax, log_sub, ticks_formatter)
   # Remove scientific notation and set xticks
   # https://stackoverflow.com/a/49306588/3904031

   function anonymous_formatter(y, _)
      if y > 1e-4
         # y_ = parse(Int64, y)
         y_ = Int64(y)
         return "\$$y_^2\$"
      else
         return @sprintf "%.4E" y
      end
   end

   formatter = plt.matplotlib.ticker.FuncFormatter(ticks_formatter)
                                                # (y, _) -> format"{:.4g}".format(int(y)) ) # format"{:.4g}".format(int(y)))
   # https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-plt.matplotlib
   x_major = plt.matplotlib.ticker.LogLocator(base = 2.0, subs=(log_sub,),
                                          numticks = 20) # ticklabels at 2^i*log_sub
   x_minor = plt.matplotlib.ticker.LogLocator(base = 2.0,
                                          subs = LinRange(1.0, 9.0, 9) * 0.1,
                                          numticks = 10)
   #  Used to manipulate tick labels. See help(plt.matplotlib.ticker.LogLocator) for details)
   ax.xaxis.set_major_formatter(anonymous_formatter)
   ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
   ax.xaxis.set_major_locator(x_major)
   ax.xaxis.set_minor_locator(x_minor)
   ax.tick_params(axis="both", which="major")
   ax.tick_params(axis="both", which="minor")
end

function add_theo_factors!(ax, ncells, error, degree, i,
                           theo_factor_even, theo_factor_odd)
   min_y = minimum(error[1:end-1])
   @show error, min_y
   xaxis = ncells[end-1:end]
   slope = degree + 1
   if iseven(slope)
      theo_factor = theo_factor_even
   else
      theo_factor = theo_factor_odd
   end
   y0 = theo_factor*min_y
   y = (1.0 ./ xaxis).^slope * y0 * xaxis[1]^slope
   markers = ["s", "o", "*"]
   # if i == 1
      ax.loglog(xaxis,y,label="\$ O(M^{-$(degree + 1)})\$",linestyle = "--",
               #  marker = markers[i],
                 c = "grey",
                fillstyle = "none")
   # else
      # ax.loglog(xaxis,y, linestyle = "--", c = "grey")
   # end
end

function error_label(error_norm)
   if error_norm in ["l2", "L2"]
      return "\$L^2\$ error"
   elseif error_norm in ["l1","L1"]
      return "\$L^1\$ error"
   end
   @assert false
end

function plot_python_ncells_vs_y(; legend = nothing,
                                    bflux = "",
                                    degree = 3,  show = "yes",
                                    files_labels = ( ("ns_conv_mdrk.txt", "MDRK"), ("ns_conv_lw.txt", "LW-$degree") ),
                                    theo_factor_even = 0.8, theo_factor_odd = 0.8,
                                    title = nothing, log_sub = "1",
                                    saveto_dir = nothing,
                                    outname = nothing,
                                    dir = joinpath(base_dir(), "results"),
                                    error_norm = "l2",
                                    ticks_formatter = format_with_powers,
                                    figsize = (6,7),
                                 )
   # @assert error_type in ["l2","L2"] "Only L2 error for now"
   fig_error, ax_error = plt.subplots(figsize=figsize)
   colors  = ["orange", "royalblue", "green", "m","c","y","k"]
   markers = ["s", "o", "*"]
   n_curves = length(files_labels)
   for i in 1:n_curves
      file, label = files_labels[i]
      error = readdlm(joinpath(dir, file))
      ax_error.loglog(error[:,1], error[:,2], marker = markers[i], c = colors[i], mec = "k",
                        fillstyle = "none", label = label)
   end
   file_for_theo, _ = files_labels[1]
   error = readdlm(joinpath(dir, file_for_theo))
   add_theo_factors!(ax_error, error[:,1], error, degree, degree,
                     theo_factor_even, theo_factor_odd)
   ax_error.set_xlabel("Number of elements \$ (M^2) \$"); ax_error.set_ylabel(error_label(error_norm))

   set_ticks!(ax_error, log_sub, ticks_formatter)

   ax_error.grid(true, linestyle="--")

   if title !== nothing
      ax_error.set_title(title)
   end
   ax_error.legend()

   if saveto_dir !== nothing
      mkpath(saveto_dir)
      @assert outname !== nothing
      full_outname = "$saveto_dir/$(outname)_ndofs.pdf"
      fig_error.savefig(full_outname*".pdf")
      fig_error.savefig(full_outname*".png",bbox_inches="tight", dpi=1200)
      println("Saved to $(full_outname).pdf")
   end

   return fig_error
end

function plot_python_ndofs_vs_y(; legend = nothing,
                                    bflux = "",
                                    degree = 3,  show = "yes",
                                    theo_factor_even = 0.8, theo_factor_odd = 0.8,
                                    files_labels = ( ("ns_conv_mdrk.txt", "MDRK"), ("ns_conv_lw.txt", "LW-$degree") ),
                                    title = nothing, log_sub = 1,
                                    saveto_dir = nothing,
                                    outname = nothing,
                                    dir = joinpath(base_dir(), "results"),
                                    error_norm = "l2",
                                    ticks_formatter = format_with_powers,
                                    figsize = (6,7)
                                 )
   # @assert error_type in ["l2","L2"] "Only L2 error for now"
   fig_error, ax_error = plt.subplots(figsize=figsize)
   colors  = ["orange", "royalblue", "green", "m","c","y","k"]
   markers = ["s", "o", "*"]
   n_curves = length(files_labels)
   for i in 1:n_curves
      file, label = files_labels[i]
      error = readdlm(joinpath(dir, file))
      ax_error.loglog(error[:,1]*(degree+1), error[:,2], marker = markers[i], c = colors[i], mec = "k",
                     fillstyle = "none", label = label)
   end
   file_for_theo, _ = files_labels[1]
   error = readdlm(joinpath(dir, file_for_theo))
   add_theo_factors!(ax_error, (degree+1)*error[:,1], error, degree,degree,
                     theo_factor_even, theo_factor_odd)
   ax_error.set_xlabel("Degrees of freedom \$ (M^2) \$")
   ax_error.set_ylabel(error_label(error_norm))

   set_ticks!(ax_error, log_sub, ticks_formatter)

   ax_error.grid(true, linestyle="--")

   if title !== nothing
      ax_error.set_title(title)
   end
   ax_error.legend()

   if saveto_dir !== nothing
      mkpath(saveto_dir)
      @assert outname !== nothing
      full_outname = "$saveto_dir/$(outname)_ndofs"
      fig_error.savefig(full_outname*".pdf")
      fig_error.savefig(full_outname*".png",bbox_inches="tight", dpi=1200)
      println("Saved to $(full_outname).pdf")
   end

   return fig_error
end

# fig_size_() = (5.5,6.5) # Default size actually
# p = plot_python_ndofs_vs_y(degree = 4, saveto_dir = ".", log_sub = 1, error_norm = "l2",
#                            figsize = fig_size_(),
#                            files_labels = ( ("ns_conv_mdrk.txt", "MDRK"), ("ns_conv_lw.txt", "LW") ),
#                            outname = "ns_conv",
#                            title = "Degree \$ N = 4 \$")
# p = plot_python_ncells_vs_y(degree = 4, saveto_dir = ".", log_sub = 4, error_norm = "l2", figsize = fig_size_(),
#                             files_labels = ( ("ns_conv_mdrk.txt", "MDRK"), ("ns_conv_lw.txt", "LW") ),
#                             outname = "ns_conv",
#                             title = "Degree \$ N = 4 \$")

# p = plot_python_ndofs_vs_y(degree = 4, saveto_dir = ".", log_sub = 1, error_norm = "l2",
#                            figsize = fig_size_(),
#                            files_labels = ( ("adv_diff_mdrk.txt", "MDRK"), ("adv_diff_lw.txt", "LW") ),
#                            outname = "adv_diff",
#                            title = "Degree \$ N = 4 \$")

# p = plot_python_ndofs_vs_y(degree = 4, saveto_dir = ".", log_sub = 1, error_norm = "l2",
#                            figsize = fig_size_(),
#                            files_labels = ( ("adv_mdrk.txt", "MDRK"), ("adv_lw.txt", "LW") ),
#                            outname = "adv",
#                            title = "Degree \$ N = 4 \$")

# p = plot_python_ndofs_vs_y(degree = 4, saveto_dir = ".", log_sub = 1, error_norm = "l2",
#                            figsize = fig_size_(),
#                            files_labels = ( ("heat_mdrk.txt", "MDRK"), ("heat_lw.txt", "LW") ),
#                            outname = "heat",
#                            title = "Degree \$ N = 4 \$")

# p = plot_python_ndofs_vs_y(degree = 4, saveto_dir = ".", log_sub = 1, error_norm = "l2",
#                            figsize = fig_size_(),
#                            files_labels = ( ("adv_diff_nonperiodic_mdrk.txt", "MDRK"), ("adv_diff_nonperiodic_lw.txt", "LW") ),
#                            outname = "adv_diff_nonperiodic",
#                            title = "Degree \$ N = 4 \$")