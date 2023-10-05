# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function my_limiter_zhang_shu!(u, threshold::Real, variable,
   mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
   @unpack weights = dg.basis

   for element in eachelement(dg, cache)
      # determine minimum value
      value_min = typemax(eltype(u))
      for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      value_min = min(value_min, variable(u_node, equations))
      end

      # detect if limiting is necessary
      value_min < threshold || continue

      # compute mean value
      u_mean = zero(Trixi.get_node_vars(u, equations, dg, 1, 1, element))
      for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      u_mean += u_node * weights[i] * weights[j]
      end
      # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
      u_mean = u_mean / 2^ndims(mesh)

      # We compute the value directly with the mean values, as we assume that
      # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
      value_mean = variable(u_mean, equations)
      theta = (value_mean - threshold) / (value_mean - value_min)
      for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      Trixi.set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
         equations, dg, i, j, element)
      end
   end

   limit_mortars!(u, threshold, variable, mesh, equations, dg, cache)
   return nothing
end

function limit_mortars!(u, threshold::Real, variable,
   mesh::TreeMesh{2}, equations, dg::DGSEM, cache)
   mortar_l2 = dg.mortar
   @unpack weights = dg.basis

   for mortar in eachmortar(dg, cache)
      # Find the bigger element neighbour of mortar
      # Get trace values from that neighbour to mortar
      # Find mean of that bigger neighbour
      # Blend with the mean so that the mortar values are admissible

      large_element = cache.mortars.neighbor_ids[3, mortar] # neighbouring large element
      upper_element = cache.mortars.neighbor_ids[2, mortar]
      lower_element = cache.mortars.neighbor_ids[1, mortar]

      # Copy solution small to small
      if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
      if cache.mortars.orientations[mortar] == 1
         # L2 mortars in x-direction
         for l in eachnode(dg)
            for v in eachvariable(equations)
            cache.mortars.u_upper[2, v, l, mortar] = u[v, 1, l,
               upper_element]
            cache.mortars.u_lower[2, v, l, mortar] = u[v, 1, l,
               lower_element]
            end
         end
      else
         # L2 mortars in y-direction
         for l in eachnode(dg)
            for v in eachvariable(equations)
            cache.mortars.u_upper[2, v, l, mortar] = u[v, l, 1,
               upper_element]
            cache.mortars.u_lower[2, v, l, mortar] = u[v, l, 1,
               lower_element]
            end
         end
      end
      else # large_sides[mortar] == 2 -> small elements on left side
      if cache.mortars.orientations[mortar] == 1
         # L2 mortars in x-direction
         for l in eachnode(dg)
            for v in eachvariable(equations)
            cache.mortars.u_upper[1, v, l, mortar] = u[v, nnodes(dg), l,
               upper_element]
            cache.mortars.u_lower[1, v, l, mortar] = u[v, nnodes(dg), l,
               lower_element]
            end
         end
      else
         # L2 mortars in y-direction
         for l in eachnode(dg)
            for v in eachvariable(equations)
            cache.mortars.u_upper[1, v, l, mortar] = u[v, l, nnodes(dg),
               upper_element]
            cache.mortars.u_lower[1, v, l, mortar] = u[v, l, nnodes(dg),
               lower_element]
            end
         end
      end
      end

      # compute mean value
      u_mean_large = zero(get_node_vars(u, equations, dg, 1, 1, large_element))
      u_mean_upper = zero(u_mean_large)
      u_mean_lower = zero(u_mean_large)
      for j in eachnode(dg), i in eachnode(dg)
      u_node_large = get_node_vars(u, equations, dg, i, j, large_element)
      u_node_upper = get_node_vars(u, equations, dg, i, j, upper_element)
      u_node_lower = get_node_vars(u, equations, dg, i, j, lower_element)
      u_mean_large += u_node_large * weights[i] * weights[j]
      u_mean_upper += u_node_upper * weights[i] * weights[j]
      u_mean_lower += u_node_lower * weights[i] * weights[j]
      end
      # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
      u_mean_large = u_mean_large / 2^ndims(mesh)

      # We compute the value directly with the mean values, as we assume that
      # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
      value_mean = variable(u_mean_large, equations)
      local leftright
      if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
      leftright = 1
      if cache.mortars.orientations[mortar] == 1
         # L2 mortars in x-direction
         u_large = view(u, :, nnodes(dg), :, large_element)
         element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
            mortar, u_large)
      else
         # L2 mortars in y-direction
         u_large = view(u, :, :, nnodes(dg), large_element)
         element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
            mortar, u_large)
      end
      else # large_sides[mortar] == 2 -> large element on right side
      leftright = 2
      if cache.mortars.orientations[mortar] == 1
         # L2 mortars in x-direction
         u_large = view(u, :, 1, :, large_element)
         element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
            mortar, u_large)
      else
         # L2 mortars in y-direction
         u_large = view(u, :, :, 1, large_element)
         element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
            mortar, u_large)
      end
      end

      # determine minimum value
      value_min = typemax(eltype(u))
      for j in eachnode(dg)
      u_upper_node = @view cache.mortars.u_upper[leftright, :, j, mortar]
      u_lower_node = @view cache.mortars.u_lower[leftright, :, j, mortar]
      value_min = min(value_min, variable(u_upper_node, equations),
         variable(u_lower_node, equations))
      end

      # Correct u so that mortar values will be positive
      theta = min(abs((value_mean - threshold) / (value_mean - value_min)), 1)
      # if theta < 1.0
      @show theta, leftright
      # end
      for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, large_element)
      set_node_vars!(u, theta * u_node + (1 - theta) * u_mean_large,
         equations, dg, i, j, large_element)
      end
   end
end

function limit_mortars!(u, threshold::Real, variable,
   mesh::P4estMesh{2}, equations, dg::DGSEM, cache)

   mortar_l2 = dg.mortar

   @unpack neighbor_ids, node_indices = cache.mortars
   @unpack U, F, fn_low = cache.element_cache
   @unpack contravariant_vectors = cache.elements
   @unpack mortars, lw_mortars = cache
   @unpack weights = dg.basis
   index_range = eachnode(dg)

   for mortar in eachmortar(dg, cache)
      # Copy solution data from the small elements using "delayed indexing" with
      # a start value and a step size to get the correct face and orientation.

      small_indices = node_indices[1, mortar]
      small_direction = indices2direction(small_indices)


      # Buffer to copy solution values of the large element in the correct orientation
      # before interpolating
      u_buffer = cache.u_threaded[Threads.threadid()]

      # Copy solution of large element face to buffer in the
      # correct orientation
      large_indices = node_indices[2, mortar]
      large_direction = indices2direction(large_indices)

      i_large_start, i_large_step = index_to_start_step_2d(large_indices[1], index_range)
      j_large_start, j_large_step = index_to_start_step_2d(large_indices[2], index_range)

      i_large = i_large_start
      j_large = j_large_start
      element = neighbor_ids[3, mortar] # large element
      for i in eachnode(dg)
      for v in eachvariable(equations)
         u_buffer[v, i] = u[v, i_large, j_large, element]
      end
      i_large += i_large_step
      j_large += j_large_step
      end

      # compute mean value
      u_mean = zero(Trixi.get_node_vars(u, equations, dg, 1, 1, element))
      for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      u_mean += u_node * weights[i] * weights[j]
      end
      # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
      u_mean = u_mean / 2^ndims(mesh)

      # We compute the value directly with the mean values, as we assume that
      # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
      value_mean = variable(u_mean, equations)

      # Interpolate large element face data from buffer to small face locations
      multiply_dimensionwise!(view(cache.mortars.u, 2, :, 1, :, mortar),
      mortar_l2.forward_lower,
      u_buffer)
      multiply_dimensionwise!(view(cache.mortars.u, 2, :, 2, :, mortar),
      mortar_l2.forward_upper,
      u_buffer)

      value_min = typemax(eltype(u))
      for i = 1:4
      u_node1 = @view cache.mortars.u[2, :, 1, i, mortar]
      u_node2 = @view cache.mortars.u[2, :, 2, i, mortar]
      value_min = min(value_min, variable(u_node1, equations),
         variable(u_node2, equations))
      end

      # Correct u so that mortar values will be positive
      theta = min(abs((value_mean - threshold) / (value_mean - value_min)), 1)
      for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
      Trixi.set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
         equations, dg, i, j, element)
      end
   end
end


 end # @muladd
