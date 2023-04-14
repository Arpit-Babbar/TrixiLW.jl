abstract type AbstractTimeDiscretization end
abstract type AbstractTimeStagedMethod end

struct LW <: AbstractTimeDiscretization end

struct SingleStaged <: AbstractTimeStagedMethod end
struct TwoStaged <: AbstractTimeStagedMethod end

struct RK <: AbstractTimeDiscretization end

struct Adaptive end
struct CFLBased
   cfl_number::Float64
end

isadaptive(::Adaptive) = true
isadaptive(::CFLBased) = false
