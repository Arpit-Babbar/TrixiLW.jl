using MPI
using DelimitedFiles
# nprocs = 8
timings = zeros(4)

for np in 0:3
	_np = Int(exp2(np))
	mpiexec() do cmd
		long_cmd = `$cmd -n $_np $(Base.julia_cmd()) --threads=1 --project=@. -e 'using Trixi, TrixiLW;
					trixi_include("../examples/tree_2d_dgsem/elixir_euler_density_wave.jl", initial_refinement_level = 0);
					trixi_include("../examples/tree_2d_dgsem/elixir_euler_density_wave.jl", initial_refinement_level = 8)'`
		time_str = read(pipeline(long_cmd, `tail -1`, `awk '{print $1}'`), String)
		# Removing '\n' character from last and parsing to Float64
		t = parse(Float64, strip(time_str))
		timings[np + 1] = t
		open("scaling_density_wave.txt", "a") do file
			writedlm(file, t, '\n')
		end
	end
end

println(timings)




# A = read(pipeline(`julia --project=../.. elixir_advection_basic.jl`, ` tail -1`, `awk '{print $1}'`), String)
# timings = parse(Float64, A[1:end-1])
