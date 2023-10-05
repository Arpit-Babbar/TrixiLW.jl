using Plots, DelimitedFiles
using TrixiLW: utils_dir
gr()
include(utils_dir()*"/reproducibility/mpl.jl")
using LaTeXStrings

data = readdlm("avg.txt")
@views npoints = length(data[:,1])
@assert npoints % 2 == 0
rho, vx, vy, pres = data[:,3], data[:,4], data[:,5], data[:,6]
mach = sqrt.(vx.^2+vy.^2) ./ sqrt.(1.4*pres ./ rho)

M = Int64(floor(npoints/2))
origin_y = 40.0
lower_x = zeros(M)
lower_mach = zeros(M)
upper_x = zeros(M)
upper_mach = zeros(M)

it_lower = 1
it_upper = 1
for i in 1:npoints
   y = data[i, 2]
   if y <= origin_y
      if it_lower > M
         push!(lower_x, data[i,1])
         push!(lower_mach, mach[i])
         continue
      end
      lower_x[it_lower] = data[i,1]
      lower_mach[it_lower] = mach[i]
      it_lower += 1
   else
      if it_upper > M
         push!(upper_x, data[i,1])
         push!(upper_mach, mach[i])
         continue
      end
      upper_x[it_upper] = data[i,1]
      upper_mach[it_upper] = mach[i]
      it_upper += 1
   end
end

upper = zeros(length(upper_x),2)
lower = zeros(length(lower_x),2)

upper[:,1] .= upper_x
upper[:,2] .= upper_mach
lower[:,1] .= lower_x
lower[:,2] .= lower_mach

# sort!(upper, by = x -> x[1])
# sort!(lower, dims = 1, by = x -> x[1])

# p = scatter(lower[:,1], lower[:,2], label = L"y > 0")
# scatter!(p, upper[:,1], upper[:,2], label = L"y < 0")
# xlims!(max(minimum(lower[:,1]), minimum(upper[:,1]))-0.1, maximum(lower[:,1])+0.1)

plt.figure()
plt.scatter(lower[:,1] .- 38.0, lower[:,2], label = "\$ y > 0 \$", marker = "s", facecolors = "none", edgecolors = "red")
plt.scatter(upper[:,1] .- 38.0, upper[:,2], s = 70, label = "\$ y < 0 \$", marker = "o", facecolors="none", edgecolors="g")
plt.xlim(max(minimum(lower[:,1]), minimum(upper[:,1])) .- 38.1, maximum(lower[:,1]).- 37.9)
plt.legend()
plt.grid(true)
plt.xlabel("\$ x \$")
plt.ylabel("Mach number")
plt.savefig("cylinder.pdf")
