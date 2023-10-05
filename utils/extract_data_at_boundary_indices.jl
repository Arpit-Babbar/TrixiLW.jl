using Trixi: cons2prim, nvariables, index_to_start_step_2d, get_node_coords
using Trixi
using DelimitedFiles

@unpack boundary_indices = sol.prob.p.boundary_conditions

airfoil_bc_indices = boundary_indices[2]

function extract_data_boundary_indices(sol, indices)
   @unpack cache, equations, mesh, solver = sol.prob.p
   nvar = nvariables(equations)
   dg = solver
   n_nodes = nnodes(dg)
   n_elements = length(indices)
   dim = 2
   @unpack weights = dg.basis
   avg_array = zeros(n_elements, dim + nvar)
   soln_array = zeros(n_elements*n_nodes, dim + nvar)
   @unpack boundaries, boundary_cache = cache
   @unpack surface_flux_values, node_coordinates = cache.elements
   local it = 1
   local element_it = 1
   index_range = eachnode(dg)
   for local_index in eachindex(indices)
      # Use the local index to get the global boundary index from the pre-sorted list
      boundary = indices[local_index]

      # Get information on the adjacent element, compute the surface fluxes,
      # and store them
      element = boundaries.neighbor_ids[boundary]
      node_indices = boundaries.node_indices[boundary]

      i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
      j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

      i_node = i_node_start
      j_node = j_node_start
      for node_index in eachnode(dg)
         u_node = Trixi.get_node_vars(boundary_cache.u, equations, dg, node_index, boundary)
         x = get_node_coords(node_coordinates, equations, dg, i_node, j_node, element)
         prim = cons2prim(u_node, equations)
         soln_array[it, 1:2  ] .= x
         soln_array[it, 3:end] .= prim
         avg_array[element_it, 1:2  ] .+= x * weights[node_index] / 2.0
         avg_array[element_it, 3:end] .+= prim * weights[node_index] / 2.0
         i_node += i_node_step
         j_node += j_node_step
         it += 1
      end
      element_it += 1
   end
   return soln_array, avg_array
end

soln_array, avg_array = extract_data_boundary_indices(sol, airfoil_bc_indices)

writedlm("soln.txt", soln_array)
writedlm("avg.txt", avg_array)
