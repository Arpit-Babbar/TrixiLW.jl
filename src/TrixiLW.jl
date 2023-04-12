module TrixiLW

src_dir()     = @__DIR__ # Directory of file
eq_dir()      = joinpath(src_dir(),"equations")
semi_dir()    = joinpath(src_dir(),"semidiscretization")
solvers_dir() = joinpath(src_dir(),"solvers")
aux_dir()     = joinpath(src_dir(),"auxiliary")

# Basic types
include(joinpath(src_dir(), "basic_types.jl"))
include(aux_dir() * "/auxiliary.jl")

using Trixi: nvariables, eachvariable, AbstractEquations

# Equations
include(eq_dir() * "/equations.jl")
include(eq_dir() * "/numerical_fluxes.jl")

# solver
include(solvers_dir() * "/fr.jl")
include(solvers_dir() * "/lwfr.jl")

# Semi-discretizations
# TODO - Is the ordering correct? rhs!, create_cache functions haven't been defined yet
include(semi_dir() * "/semidiscretization.jl")
include(semi_dir() * "/semidiscretization_hyperbolic.jl")

# Specific solvers

include(solvers_dir() * "/dgsem_tree/dg_2d.jl")
include(solvers_dir() * "/dgsem_tree/containers.jl")

export time_discretization

end # module
