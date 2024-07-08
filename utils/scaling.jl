using MPI
using DelimitedFiles
timings = zeros(5)		# 1, 2, 4, 8, 16 processes

a = 4		# refinement_level
open("utils/scaling_hennemann.txt", "a") do file
	write(file, "\nRefinement level: $a", '\n')
end

# Set the refinement_level in command below, as required.
for np in 0:4
	_np = Int(exp2(np))
	mpiexec() do cmd
		long_cmd = `$cmd -n $_np $(Base.julia_cmd()) --threads=1 --project=@. -e 'using Trixi, TrixiLW;
					trixi_include("utils/elixir_euler_isentropic_hennemann.jl", refinement_level = 0);
					trixi_include("utils/elixir_euler_isentropic_hennemann.jl", refinement_level = 4)'`

		time_str = read(pipeline(long_cmd, `tail -1`, `awk '{print $1}'`), String)

		# Removing '\n' character from last and parsing to Float64
		t = parse(Float64, strip(time_str))

		open("utils/scaling_hennemann.txt", "a") do file
			write(file, "ranks: $_np \t time(sec): $t", '\n')
		end
	end
end



# A = read(pipeline(`julia --project=../.. elixir_advection_basic.jl`, ` tail -1`, `awk '{print $1}'`), String)
# timings = parse(Float64, A[1:end-1])
