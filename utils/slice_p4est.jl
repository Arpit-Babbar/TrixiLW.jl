using Trixi: cons2prim, nvariables, index_to_start_step_2d, get_node_coords, eachelement,
             nelements
using Trixi
using DelimitedFiles

function slice_at_x(sol, x_loc)
   @unpack cache, equations, mesh, solver = sol.prob.p
   nvar = nvariables(equations)
   dg = solver
   n_nodes = nnodes(dg)
   n_elements = nelements(solver, cache)
   NDIMS = ndims(equations)

   soln_arrays = [Vector{Float64}() for _ in 1:nvar]
   avg_arrays = [Vector{Float64}() for _ in 1:nvar] # TODO - Make this!
   y_vec = Vector{Float64}()
   @unpack node_coordinates = cache.elements
   u = reshape(sol.u[1], (nvar, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
   for element in eachelement(dg, cache)
      for j in eachnode(dg), i in eachnode(dg)
         u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
         node = get_node_coords(node_coordinates, equations, dg, i, j, element)
         prim = cons2prim(u_node, equations)
         x, y = node
         if x ≈ x_loc
            push!(y_vec, y)
            for n in eachvariable(equations)
               push!(soln_arrays[n], prim[n])
            end
         end
      end
   end
   sliced_soln = hcat(y_vec, soln_arrays...)
   return sort!(sliced_soln, dims = 1)
end

sliced_soln = slice_at_x(sol, 2.0)

writedlm("soln.txt", sliced_soln)
# writedlm("avg.txt", avg_array)
