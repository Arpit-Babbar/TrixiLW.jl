using MPI
using DelimitedFiles
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
# timings = zeros(5)		# 1, 2, 4, 8, 16 processes

a = 4		# refinement_level
open("utils/scaling_hennemann.txt", "a") do file
	write(file, "\nRefinement level: $a", '\n')
end

# Set the refinement_level in command below, as required.
for np in 4:-1:0
	_np = Int(exp2(np))
	open("utils/scaling_hennemann.txt", "a") do file
		write(file, "\nranks: $_np", '\n')
	end
	mpiexec() do cmd
		long_cmd = `$cmd -n $_np $(Base.julia_cmd()) --threads=1 --project=@. -e 'using Trixi, TrixiLW;
					trixi_include("utils/elixir_euler_isentropic_hennemann.jl", refinement_level = 0);
					trixi_include("utils/elixir_euler_isentropic_hennemann.jl", refinement_level = 4)'`

		time_str = read(pipeline(long_cmd #=,`tail -1`, `awk '{print $1}'`=#), String)
		# println(time_str)
		if rank == 0
			# Find the index of the pattern
			start_idx = findlast("Total failed time steps = 0", time_str)

			# If the pattern is found, read everything after it
			if start_idx !== nothing
				summary = time_str[start_idx[end]+1:end]
			end
			open("utils/scaling_hennemann.txt", "a") do file
				write(file, summary, '\n')
			end
		end
		time_str = nothing
		MPI.Barrier(MPI.COMM_WORLD)






		# summary = read(pipeline(`grep -A END "Total failed time steps = 0" '<<<' $time_str`))
		# println(summary)
		# if occursin("Total failed time steps = 0", time_str)
		# 	position = search(time_str, "Total failed time steps = 0") + length("Total failed time steps = 0")
		# end
		# summary = substring(time_str, position:1:end)
		# println(time_str)
		# println(typeof(time_str))
		# summary = read(pipeline(`grep -A END "Total failed time steps = 0"`))

		# Removing '\n' character from last and parsing to Float64
		# t = parse(Float64, strip(time_str))
	end
end



# A = read(pipeline(`julia --project=../.. elixir_advection_basic.jl`, ` tail -1`, `awk '{print $1}'`), String)
# timings = parse(Float64, A[1:end-1])
