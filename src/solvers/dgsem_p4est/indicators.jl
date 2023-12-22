using Trixi
using Trixi: DGSEM, get_node_coords, norm, @threaded
using StaticArrays

struct RadialIndicator{RealT <: Real, IndicatorCache}
    center::SVector{2, RealT}
    radius::RealT
    cache::IndicatorCache
end

function RadialIndicator(center, radius)
    @assert length(center) == 2
    center = SVector(center...)
    cache = create_radial_cache()
    return RadialIndicator(center, radius, cache)
end

function create_radial_cache()
    alpha = Vector{Float64}()
    cache = (; alpha)
    return cache
end

function (radial_indicator::RadialIndicator)(u::AbstractArray, mesh, equations, dg::DGSEM, cache;
                                             kwargs...)
    @unpack alpha = radial_indicator.cache
    @unpack radius, center = radial_indicator
    @unpack node_coordinates = cache.elements
    resize!(alpha, nelements(dg, cache))

    @threaded for element in eachelement(dg, cache)
        alpha[element] = 0.0
        for j in eachnode(dg), i in eachnode(dg)
            x = get_node_coords(node_coordinates, equations, dg, i, j, element)
            distance = norm(x - center)
            if distance < radius
                alpha[element] = 1.0
            end
        end
    end

    return alpha
end