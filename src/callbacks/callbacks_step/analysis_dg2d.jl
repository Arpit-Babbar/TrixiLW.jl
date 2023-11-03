using Trixi: integrate_via_indices, norm
import Trixi: analyze, pretty_form_ascii, pretty_form_utf

struct RhoRes end
struct RhoV1Res end
struct RhoV2Res end
struct ERes end
struct CFLComputation end
struct CFLComputationMax end

struct AnalysisSurfaceIntegral{Indices, Variable}
    indices::Indices
    variable::Variable
end

function lift_force(u, normal_direction, equations::CompressibleEulerEquations2D)
    p = pressure(u, equations)
    return p * normal_direction[2]
end

function drag_force(u, normal_direction, equations::CompressibleEulerEquations2D)
    p = pressure(u, equations)
    return p * normal_direction[1]
end

function analyze(lift_computation::AnalysisSurfaceIntegral, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    @unpack boundaries, boundary_cache = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = lift_computation
    indices_ = indices(cache)

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
       # Use the local index to get the global boundary index from the pre-sorted list
       boundary = indices_[local_index]

       # Get information on the adjacent element, compute the surface fluxes,
       # and store them
       element = boundaries.neighbor_ids[boundary]
       node_indices = boundaries.node_indices[boundary]
       direction = indices2direction(node_indices)

       i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
       j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

       i_node = i_node_start
       j_node = j_node_start
       for node_index in eachnode(dg)
          u_node = Trixi.get_node_vars(boundary_cache.u, equations, dg, node_index, boundary)
          normal_direction = get_normal_direction(direction, contravariant_vectors, i_node, j_node,
                                                  element)

          # L2 norm of normal direction is the surface element
          dA = 0.5 * weights[node_index] * norm(normal_direction)
          surface_integral += variable(u_node, normal_direction, equations) * dA

          i_node += i_node_step
          j_node += j_node_step
       end
    end
    return surface_integral
end

pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, typeof(lift_force)}) = "Lift"
pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, typeof(lift_force)}) = "Lift"
pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, typeof(drag_force)}) = "Drag"
pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, typeof(drag_force)}) = "Drag"

function analyze(::CFLComputation, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    dt = cache.dt[1]
    max_scaled_speed = nextfloat(zero(t))
    min_cfl = 1.0e20
    max_cfl = -1.0e20

    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    for element in eachelement(dg, cache)
        max_lambda1 = max_lambda2 = zero(max_scaled_speed)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            lambda1, lambda2 = max_abs_speeds(u_node, equations)

            # Local speeds transformed to the reference element
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            lambda1_transformed = abs(Ja11 * lambda1 + Ja12 * lambda2)
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)
            lambda2_transformed = abs(Ja21 * lambda1 + Ja22 * lambda2)

            inv_jacobian = abs(inverse_jacobian[i, j, element])

            max_lambda1 = max(max_lambda1, lambda1_transformed * inv_jacobian)
            max_lambda2 = max(max_lambda2, lambda2_transformed * inv_jacobian)
        end

        scaled_speed = max_lambda1 + max_lambda2
        dt_local = 2 / (nnodes(dg) * scaled_speed)
        cfl = dt / dt_local
        min_cfl = min(min_cfl, cfl)
        max_cfl = min(max_cfl, cfl)
    end

    return min_cfl
end

function analyze(::CFLComputationMax, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    dt = cache.dt[1]
    max_scaled_speed = nextfloat(zero(t))
    min_cfl = 1.0e20
    max_cfl = -1.0e20

    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    for element in eachelement(dg, cache)
        max_lambda1 = max_lambda2 = zero(max_scaled_speed)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            lambda1, lambda2 = max_abs_speeds(u_node, equations)

            # Local speeds transformed to the reference element
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            lambda1_transformed = abs(Ja11 * lambda1 + Ja12 * lambda2)
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)
            lambda2_transformed = abs(Ja21 * lambda1 + Ja22 * lambda2)

            inv_jacobian = abs(inverse_jacobian[i, j, element])

            max_lambda1 = max(max_lambda1, lambda1_transformed * inv_jacobian)
            max_lambda2 = max(max_lambda2, lambda2_transformed * inv_jacobian)
        end

        scaled_speed = max_lambda1 + max_lambda2
        dt_local = 2 / (nnodes(dg) * scaled_speed)
        cfl = dt / dt_local
        min_cfl = min(min_cfl, cfl)
        max_cfl = max(max_cfl, cfl)
    end

    return max_cfl
end

function analyze(::RhoRes, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    # in the below do syntax, the anonymous function is
    # (u_, i, j, element, equations, dg, cache, derivative_matrix) -> u_[1,i,j,element]^2
    # The integrate_via_indices function is calling it with u_=du
    integrate_via_indices(du, mesh, equations, dg, cache, cache,
                dg.basis.derivative_matrix) do u_, i, j, element, equations,
                                                dg, cache, derivative_matrix
        res = u_[1,i,j,element]^2
    end |> sqrt
end

function analyze(::RhoV1Res, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    integrate_via_indices(du, mesh, equations, dg, cache, cache,
                dg.basis.derivative_matrix) do u_, i, j, element, equations,
                                                dg, cache, derivative_matrix
        res = u_[2,i,j,element]^2
    end |> sqrt
end

function analyze(::RhoV2Res, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    integrate_via_indices(du, mesh, equations, dg, cache, cache,
                dg.basis.derivative_matrix) do u_, i, j, element, equations,
                                                dg, cache, derivative_matrix
        res = u_[3,i,j,element]^2
    end |> sqrt
end

function analyze(::ERes, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    integrate_via_indices(du, mesh, equations, dg, cache, cache,
                dg.basis.derivative_matrix) do u_, i, j, element, equations,
                                                dg, cache, derivative_matrix
        res = u_[4,i,j,element]^2
    end |> sqrt
end
