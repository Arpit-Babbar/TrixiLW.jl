using Plots
using Trixi: eachinterface, eachmortar, eachboundary, indices2direction,
             polynomial_interpolation_matrix, get_nodes, index_to_start_step_2d,
             get_node_coords
gr()

function is_in_limit(nodes_array, xlimits, ylimits)
   @views begin
   minimum(nodes_array[1,:]) < xlimits[2] + 0.1 &&
   maximum(nodes_array[1,:]) > xlimits[1] - 0.1 &&
   minimum(nodes_array[2,:]) < ylimits[2] + 0.1 &&
   maximum(nodes_array[2,:]) > ylimits[1] - 0.1
   end
end

function plot_mesh(semi, mesh::P4estMesh{2}, dg::DG; nvisnodes = 10,
                  xlimits = (0.0, 4.0), ylimits = (-1.0, 1.0))
   @unpack cache, equations = semi
   @unpack interfaces, elements, mortars, boundaries = cache
   @unpack node_coordinates = cache.elements
   @unpack basis = dg
   index_range = eachnode(dg)

   interpolation_matrix = polynomial_interpolation_matrix(get_nodes(basis), LinRange(-1.0, 1.0, nvisnodes))

   p = plot(xlims = xlimits, ylims = ylimits)
   # xlims!(p, xlimits)
   # ylims!(p, ylimits)

   plotter! = plot!
   linewidth = 1.0

   nodes_array = zeros(2, nnodes(dg))
   for interface in eachinterface(dg, cache)
      primary_element = interfaces.neighbor_ids[1, interface]
      primary_indices = interfaces.node_indices[1, interface]

      i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1], index_range)
      j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2], index_range)

      i_primary = i_primary_start
      j_primary = j_primary_start
      for i in eachnode(dg)
         x = get_node_coords(node_coordinates, equations, dg, i_primary, j_primary, primary_element)
         i_primary += i_primary_step
         j_primary += j_primary_step
         nodes_array[1,i] = x[1]
         nodes_array[2,i] = x[2]
      end
      if is_in_limit(nodes_array, xlimits, ylimits)
         @views plotter!(p,
                         interpolation_matrix * nodes_array[1,:],
                         interpolation_matrix * nodes_array[2,:],
                         color = :black, linewidth = linewidth, label = false)
      end
   end

   for mortar in eachmortar(dg, cache)
      # Copy solution data from the small elements using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.

      small_indices = mortars.node_indices[1, mortar]

      i_small_start, i_small_step = index_to_start_step_2d(small_indices[1], index_range)
      j_small_start, j_small_step = index_to_start_step_2d(small_indices[2], index_range)

      for position in 1:2
         i_small = i_small_start
         j_small = j_small_start
         element = mortars.neighbor_ids[position, mortar]
         for i in eachnode(dg)
            x = get_node_coords(node_coordinates, equations, dg, i_small, j_small, element)
            i_small += i_small_step
            j_small += j_small_step
            nodes_array[1,i] = x[1]
            nodes_array[2,i] = x[2]
         end

         if is_in_limit(nodes_array, xlimits, ylimits)
            @views plotter!(p,
                             interpolation_matrix * nodes_array[1,:],
                             interpolation_matrix * nodes_array[2,:], color = :black, linewidth = linewidth, label = false)
         end
      end
   end

   for boundary in eachboundary(dg, cache)
      # Copy solution data from the element using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.
      element = boundaries.neighbor_ids[boundary]
      node_indices = boundaries.node_indices[boundary]
      direction = indices2direction(node_indices)

      i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
      j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

      i_node = i_node_start
      j_node = j_node_start
      for i in eachnode(dg)
         x = get_node_coords(node_coordinates, equations, dg, i_node, j_node, element)
         i_node += i_node_step
         j_node += j_node_step
         nodes_array[1,i] = x[1]
         nodes_array[2,i] = x[2]
      end
      if is_in_limit(nodes_array, xlimits, ylimits)
         @views plotter!(p, nodes_array[1,:], nodes_array[2,:], color = :black, linewidth = linewidth, label =
         false)
      end
   end

   savefig(p, "test.pdf")
end
