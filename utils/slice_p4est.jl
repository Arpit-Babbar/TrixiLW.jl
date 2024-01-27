using Trixi: cons2prim, nvariables, index_to_start_step_2d, get_node_coords, eachelement,
             nelements
using Trixi
using DelimitedFiles

function slice_at_x(sol, x_loc)
   @unpack cache, equations, mesh, solver = sol.prob.p
   @unpack weights = solver.basis
   nvar = nvariables(equations)
   dg = solver
   n_nodes = nnodes(dg)
   n_elements = nelements(solver, cache)
   NDIMS = ndims(equations)

   soln_arrays = [Vector{Float64}() for _ in 1:nvar]
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
   return sortslices(sliced_soln, dims = 1)
end

# sliced_soln = slice_at_x(sol, 2.0)
# writedlm("soln.txt", sliced_soln)

function slice_at_y(sol, y_loc)
   @unpack cache, equations, mesh, solver = sol.prob.p
   nvar = nvariables(equations)
   dg = solver
   n_nodes = nnodes(dg)
   n_elements = nelements(solver, cache)
   NDIMS = ndims(equations)

   soln_arrays = [Vector{Float64}() for _ in 1:nvar]
   x_vec = Vector{Float64}()
   @unpack node_coordinates = cache.elements
   u = reshape(sol.u[1], (nvar, ntuple(_ -> n_nodes, NDIMS)..., n_elements))
   for element in eachelement(dg, cache)
      for j in eachnode(dg), i in eachnode(dg)
         u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
         node = get_node_coords(node_coordinates, equations, dg, i, j, element)
         prim = cons2prim(u_node, equations)
         x, y = node
         if y ≈ y_loc
            push!(x_vec, x)
            for n in eachvariable(equations)
               push!(soln_arrays[n], prim[n])
            end
         end
      end
   end
   sliced_soln = hcat(x_vec, soln_arrays...)
   # return sliced_soln
   return sortslices(sliced_soln, dims = 1)
end

function convert_slice_to_avg(sol, sliced_sol)
   @unpack cache, equations, mesh, solver = sol.prob.p
   @unpack weights = solver.basis
   nvar = nvariables(equations)
   dg = solver
   n_nodes = nnodes(dg)
   total_points = size(sliced_sol, 1)
   n_elements = Int(total_points / n_nodes)
   node_array = zeros(n_elements)
   avg_array = zeros(nvar, n_elements)
   NDIMS = ndims(equations)

   for e in 1:n_elements
      @views for i in 1:n_nodes
         u_node = sliced_sol[(e-1)*n_nodes + i, :]
         avg_array[:, e] .+= 0.5 * weights[i] * u_node[2:end]
         node_array[e] += 0.5 * weights[i] * u_node[1]
         @show node_array[e]
      end
      # @assert false
   end

   sliced_soln = hcat(node_array, avg_array[1,:])
   return node_array, sortslices(sliced_soln, dims = 1)
end

# sliced_soln = slice_at_y(sol, 1.0)
# writedlm("soln.txt", sliced_soln)

# plot(sliced_soln[:,1], sliced_soln[:,2])

# node_array, avg = convert_slice_to_avg(sol, sliced_soln)

# plot(avg[:,1], avg[:,2])

# writedlm("avg.txt", avg_array)
