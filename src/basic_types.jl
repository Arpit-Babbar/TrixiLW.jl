abstract type AbstractTimeDiscretization end

struct LW <: AbstractTimeDiscretization end

struct RK <: AbstractTimeDiscretization end

struct Adaptive end
struct CFLBased
   cfl_number::Float64
end

isadaptive(::Adaptive) = true
isadaptive(::CFLBased) = false
