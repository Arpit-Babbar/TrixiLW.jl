using Trixi

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function prolong2mpiinterfaces!(cache, u,
                                mesh::ParallelP4estMesh{2},
                                equations, surface_integral,
                                time_discretization::AbstractLWTimeDiscretization,
                                dg::DG)
    @unpack mpi_interfaces = cache
    @unpack U, F = cache.element_cache
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Copy solution data from the local element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_side = mpi_interfaces.local_sides[interface]
        local_element = mpi_interfaces.local_neighbor_ids[interface]
        local_indices = mpi_interfaces.node_indices[interface]

        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start
        for i in eachnode(dg)
            for v in eachvariable(equations)
                mpi_interfaces.u[local_side, v, i, interface] = u[v, i_element,
                                                                  j_element,
                                                                  local_element]
                mpi_interfaces.U[local_side, v, i, interface] = U[v, i_element,
                                                                  j_element,
                                                                  local_element]
                # Will F here have two components??
                mpi_interfaces.F[local_side, v, i, interface] = F[v, i_element,
                                                                  j_element,
                                                                  local_element]
            end
            i_element += i_element_step
            j_element += j_element_step
        end
    end

    return nothing
end


end # muladd