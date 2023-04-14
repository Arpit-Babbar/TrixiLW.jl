function alloc_for_threads(constructor, cache_size)
   nt = Threads.nthreads()
   SVector{nt}([alloc(constructor, cache_size) for _ in Base.OneTo(nt)])
end

# Construct `cache_size` number of objects with `constructor`
# and store them in an SVector
function alloc(constructor, cache_size)
   SVector{cache_size}(constructor(undef) for _ in Base.OneTo(cache_size))
end

using Trixi

lw_fourier_cfl(N) = (1.000, 0.333, 0.170, 0.103)[N]

# Use this as safety factor to get LW CFL with same safety factor
function trixi2lw(cfl_number, dg)
   N           = Trixi.polydeg(dg)
   nnodes      = N + 1
   fourier_cfl = lw_fourier_cfl(N)
   cfl_number  = cfl_number * nnodes * fourier_cfl
   return cfl_number
end