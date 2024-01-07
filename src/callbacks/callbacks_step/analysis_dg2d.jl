using Trixi: integrate_via_indices, norm
using DelimitedFiles
import Trixi: analyze, pretty_form_ascii, pretty_form_utf

struct RhoRes end
struct RhoV1Res end
struct RhoV2Res end
struct ERes end
struct CFLComputation end
struct CFLComputationMax end

abstract type SurfaceQuantitiyViscous end

struct SaveSurfacePrimitives{Indices}
    indices::Indices
end

struct AnalysisSurfaceIntegral{Indices, Variable}
    indices::Indices
    variable::Variable
end

struct AnalysisSurfaceFrictionCoefficient{Indices} <: SurfaceQuantitiyViscous
    indices::Indices
end

struct AnalysisSurfaceIntegralViscous{Indices, Variable} <: SurfaceQuantitiyViscous
    indices::Indices
    variable::Variable
end

# WARNING - This must be done before AnalysisSurfaceIntegralViscous as
# AnalysisSurfaceIntegralViscous will overwrite the gradient!
struct AnalysisSurfaceIntegralViscousCorrectedGrad{Indices, Variable} <: SurfaceQuantitiyViscous
    indices::Indices
    variable::Variable
end


function lift_force(u, normal_direction, equations::CompressibleEulerEquations2D)
    p = pressure(u, equations)
    return p * normal_direction[2] / norm(normal_direction)
end

struct ForceState{RealT <: Real}
    Ψl::Tuple{RealT, RealT}
    rhoinf::RealT
    uinf::RealT
    linf::RealT
end

struct LiftForcePressure{RealT <: Real}
    force_state::ForceState{RealT}
end

struct LiftForceViscous{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragForcePressure{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragForceViscous{RealT <: Real}
    force_state::ForceState{RealT}
end

function LiftForcePressure(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    Ψl = (-sin(aoa), cos(aoa))
    force_state = ForceState(Ψl, rhoinf, uinf, linf)
    return LiftForcePressure(force_state)
end

function DragForcePressure(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    Ψd = (cos(aoa), sin(aoa))
    return DragForcePressure(ForceState(Ψd, rhoinf, uinf, linf))
end

function LiftForceViscous(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    Ψl = (-sin(aoa), cos(aoa))
    force_state = ForceState(Ψl, rhoinf, uinf, linf)
    return LiftForceViscous(force_state)
end

function DragForceViscous(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    Ψd = (cos(aoa), sin(aoa))
    return DragForceViscous(ForceState(Ψd, rhoinf, uinf, linf))
end

function (lift_force::LiftForcePressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψl, rhoinf, uinf, linf = lift_force.force_state
    n = dot(normal_direction, Ψl) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

# TODO - Have only one function. Don't name it lift/drag. Varying the alpha allows you
# to choose between lift or drag in the elixir file.
function (lift_force_viscous::LiftForceViscous)(u, gradients, normal_direction, equations)
    @unpack Ψl, rhoinf, uinf, linf = lift_force_viscous.force_state
    @unpack mu = equations

    _, dv1dx, dv2dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
    _, dv1dy, dv2dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)

    # Components of viscous stress tensor

    # (4/3 * (v1)_x - 2/3 * (v2)_y)
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
    # ((v1)_y + (v2)_x)
    # stress tensor is symmetric
    tau_12 = dv1dy + dv2dx # = tau_21
    tau_21 = tau_12 # For readability
    # (4/3 * (v2)_y - 2/3 * (v1)_x)
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    n = normal_direction / norm(normal_direction)
    force = tau_11*n[1]*Ψl[1] + tau_12*n[2]*Ψl[1] + tau_21*n[1]*Ψl[2] + tau_22*n[2]*Ψl[2]
    force *= mu
    factor = 0.5 * rhoinf * uinf^2 * linf
    return force / factor
end

function surface_skin_friction(u, gradients, normal_direction, equations)
    @unpack rhoinf, uinf, linf = surface_skin_coefficient.force_state
    @unpack mu = equations

    _, dv1dx, dv2dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
    _, dv1dy, dv2dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)

    # Components of viscous stress tensor

    # (4/3 * (v1)_x - 2/3 * (v2)_y)
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
    # ((v1)_y + (v2)_x)
    # stress tensor is symmetric
    tau_12 = dv1dy + dv2dx # = tau_21
    tau_21 = tau_12 # For readability
    # (4/3 * (v2)_y - 2/3 * (v1)_x)
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    n = normal_direction / norm(normal_direction)
    n_perp = (-n[2], n[1])
    Cf = (  tau_11*n[1]*n_perp[1] + tau_12*n[2]*n_perp[1]
          + tau_21*n[1]*n_perp[2] + tau_22*n[2]*n_perp[2])
    Cf *= mu
    factor = 0.5 * rhoinf * uinf^2 * linf
    return Cf / factor
end


function (drag_force_viscous::DragForceViscous)(u, gradients, normal_direction, equations)
    @unpack Ψl, rhoinf, uinf, linf = drag_force_viscous.force_state
    mu = equations.mu

    _, dv1dx, dv2dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
    _, dv1dy, dv2dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)

    # Components of viscous stress tensor

    # (4/3 * (v1)_x - 2/3 * (v2)_y)
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 *  dv2dy
    # ((v1)_y + (v2)_x)
    # stress tensor is symmetric
    tau_12 = dv1dy + dv2dx # = tau_21
    tau_21 = tau_12 # Symmetric, and rewritten for readability
    # (4/3 * (v2)_y - 2/3 * (v1)_x)
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    n = normal_direction / norm(normal_direction)
    force = tau_11*n[1]*Ψl[1] + tau_12*n[2]*Ψl[1] + tau_21*n[1]*Ψl[2] + tau_22*n[2]*Ψl[2]
    force *= mu # The tau had a factor of mu in Ray 2017, but it is not present in the
    # above expressions taken from Trixi.jl and thus it is included here
    factor = 0.5 * rhoinf * uinf^2 * linf
    return force / factor
end

function drag_force(u, normal_direction, equations)
    p = pressure(u, equations)
    return p * normal_direction[1]  / norm(normal_direction)
end

function (drag_force::DragForcePressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψl, rhoinf, uinf, linf = drag_force.force_state
    n = dot(normal_direction, Ψl) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function analyze(quantity::SurfaceQuantitiyViscous,
                 du, u, t, semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic
    cache_parabolic = semi.cache_parabolic
    analyze(quantity, du, u, t, mesh, equations, equations_parabolic, solver, cache,
            cache_parabolic)
end

function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations, dg::DGSEM, cache)
    @unpack boundaries, boundary_cache = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
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
          # 0.5 factor is NOT needed, the norm(normal_direction) is all the factor needed
          dS = weights[node_index] * norm(normal_direction)
          surface_integral += variable(u_node, normal_direction, equations) * dS

          i_node += i_node_step
          j_node += j_node_step
       end
    end
    return surface_integral
end

function analyze(surface_variable::AnalysisSurfaceFrictionCoefficient,
    du, u, t, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
    equations_parabolic, dg::DGSEM, cache, cache_parabolic)
    @unpack boundaries, boundary_cache = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container
    gradients_x, gradients_y = gradients

    dim = 2 # Generalize!
    nvar = nvariables(equations)
    n_nodes = nnodes(dg)
    n_elements = length(indices_)
    avg_array = zeros(n_elements, dim + nvar)
    soln_array = zeros(n_elements*n_nodes, dim + nvar)

    local it = 1
    local element_it = 1

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
          x = get_node_coords(node_coordinates, equations, dg, i_node, j_node, element)
          u_node = Trixi.get_node_vars(boundary_cache.u, equations, dg, node_index, boundary)
          normal_direction = get_normal_direction(direction, contravariant_vectors, i_node, j_node,
                                                  element)
          ux = Trixi.get_node_vars(gradients_x, equations, dg, i_node, j_node, element)
          uy = Trixi.get_node_vars(gradients_y, equations, dg, i_node, j_node, element)

          Cf = surface_skin_friction(u_node, (ux, uy), normal_direction, equations_parabolic)

          soln_array[it, 1:2  ] .= x
          soln_array[it, 3] = Cf
          avg_array[element_it, 1:2  ] .+= x * weights[node_index] / 2.0
          avg_array[element_it, 3] += Cf * weights[node_index] / 2.0

          i_node += i_node_step
          j_node += j_node_step

          it += 1
       end
       element_it += 1
    end
    mkpath("out")
    writedlm(joinpath("out", "Cf_t$t.txt"), soln_array)
    writedlm(joinpath("out", "Cf_avg_t$t.txt"), avg_array)
    return 0.0
end

function analyze(surface_variable::AnalysisSurfaceIntegralViscousCorrectedGrad,
    du, u, t, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
    equations_parabolic, dg::DGSEM, cache, cache_parabolic)
    @unpack boundaries, boundary_cache = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container
    gradients_x, gradients_y = gradients

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
          ux = Trixi.get_node_vars(gradients_x, equations, dg, i_node, j_node, element)
          uy = Trixi.get_node_vars(gradients_y, equations, dg, i_node, j_node, element)

          # L2 norm of normal direction is the surface
          # 0.5 factor is NOT needed, the norm(normal_direction) is all the factor needed
          dS = weights[node_index] * norm(normal_direction)
          surface_integral += variable(u_node, (ux, uy), normal_direction, equations_parabolic) * dS

          i_node += i_node_step
          j_node += j_node_step
       end
    end
    return surface_integral
end

function analyze(surface_variable::AnalysisSurfaceIntegralViscous, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
    equations_parabolic, dg::DGSEM, cache, cache_parabolic)
    @unpack boundaries, boundary_cache = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container
    gradients_x, gradients_y = gradients

    reset_du!(gradients_x, dg, cache)
    reset_du!(gradients_y, dg, cache)

    @unpack derivative_matrix = dg.basis
    @threaded for element in eachelement(dg, cache)

        # Calculate volume terms in one element
        for j in eachnode(dg), i in eachnode(dg)
            # In Trixi, this is u_transformed instead of u. Does that have side-effects?
            # Of course, we compute gradients in conservative variables
            u_node = get_node_vars(u, equations_parabolic, dg, i, j, element)

            for ii in eachnode(dg)
                multiply_add_to_node_vars!(gradients_x, derivative_matrix[ii, i],
                                           u_node, equations_parabolic, dg, ii, j,
                                           element)
            end

            for jj in eachnode(dg)
                multiply_add_to_node_vars!(gradients_y, derivative_matrix[jj, j],
                                           u_node, equations_parabolic, dg, i, jj,
                                           element)
            end
        end
    end

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
          ux = Trixi.get_node_vars(gradients_x, equations, dg, i_node, j_node, element)
          uy = Trixi.get_node_vars(gradients_y, equations, dg, i_node, j_node, element)

          # L2 norm of normal direction is the surface
          # 0.5 factor is NOT needed, the norm(normal_direction) is all the factor needed
          dS = weights[node_index] * norm(normal_direction)
          surface_integral += variable(u_node, (ux, uy), normal_direction, equations_parabolic) * dS

          i_node += i_node_step
          j_node += j_node_step
       end
    end
    return surface_integral
end

function analyze(surface_variable::SaveSurfacePrimitives, du, u, t,
    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
    equations, dg::DGSEM, cache)
    @unpack boundaries, boundary_cache = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    dim = 2 # Generalize!
    nvar = nvariables(equations)
    n_nodes = nnodes(dg)
    n_elements = length(indices_)
    avg_array = zeros(n_elements, dim + nvar)
    soln_array = zeros(n_elements*n_nodes, dim + nvar)

    local it = 1
    local element_it = 1
    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
       # Use the local index to get the global boundary index from the pre-sorted list
       boundary = indices_[local_index]

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
    mkpath("out")
    writedlm(joinpath("out", "soln_t$t.txt"), soln_array)
    writedlm(joinpath("out", "avg_t$t.txt"), avg_array)
    return 0.0
end

pretty_form_ascii(::SaveSurfacePrimitives{<:Any}) = "Dummy value"
pretty_form_utf(::SaveSurfacePrimitives{<:Any}) = "Dummy value"

pretty_form_ascii(::AnalysisSurfaceFrictionCoefficient{<:Any}) = "Dummy value"
pretty_form_utf(::AnalysisSurfaceFrictionCoefficient{<:Any}) = "Dummy value"


pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, typeof(lift_force)}) = "Lift"
pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, typeof(lift_force)}) = "Lift"
pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, typeof(drag_force)}) = "Drag"
pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, typeof(drag_force)}) = "Drag"

pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:LiftForcePressure{<:Any}}) = "Pressure_lift"
pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:LiftForcePressure{<:Any}}) = "Pressure_lift"
pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:DragForcePressure{<:Any}}) = "Pressure_drag"
pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:DragForcePressure{<:Any}}) = "Pressure_drag"

pretty_form_ascii(::AnalysisSurfaceIntegralViscous{<:Any, <:LiftForceViscous{<:Any}}) = "Viscous_lift"
pretty_form_utf(::AnalysisSurfaceIntegralViscous{<:Any, <:LiftForceViscous{<:Any}}) = "Viscous_lift"
pretty_form_ascii(::AnalysisSurfaceIntegralViscous{<:Any, <:DragForceViscous{<:Any}}) = "Viscous_drag"
pretty_form_utf(::AnalysisSurfaceIntegralViscous{<:Any, <:DragForceViscous{<:Any}}) = "Viscous_drag"

pretty_form_ascii(::CFLComputation) = "CFLMin"
pretty_form_utf(::CFLComputation) = "CFLMin"
pretty_form_ascii(::CFLComputationMax) = "CFLMax"
pretty_form_utf(::CFLComputationMax) = "CFLMax"

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

pretty_form_ascii(::RhoRes) = "Rho_res"
pretty_form_utf(::RhoRes) = "Rho_res"
pretty_form_ascii(::RhoV1Res) = "RhoV1_res"
pretty_form_utf(::RhoV1Res) = "RhoV1_res"
pretty_form_ascii(::RhoV2Res) = "RhoV2_res"
pretty_form_utf(::RhoV2Res) = "RhoV2_res"
pretty_form_ascii(::ERes) = "E_res"
pretty_form_utf(::ERes) = "E_res"
