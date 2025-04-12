abstract type AbstractTimeDiscretization end
abstract type AbstractTimeStagedMethod end

abstract type AbstractLWTimeDiscretization <: AbstractTimeDiscretization end
abstract type AbstractLWAD <: AbstractLWTimeDiscretization end
struct LW <: AbstractLWTimeDiscretization end
struct LWADTaylorDiff <: AbstractLWAD end
struct LWADEnzyme <: AbstractLWAD end
struct MDRK <: AbstractLWTimeDiscretization end

struct RK <: AbstractTimeDiscretization end

struct Adaptive end
struct CFLBased
    cfl_number::Float64
end

isadaptive(::Adaptive) = true
isadaptive(::CFLBased) = false
