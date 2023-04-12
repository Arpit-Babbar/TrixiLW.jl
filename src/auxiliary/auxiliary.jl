function alloc_for_threads(constructor, cache_size)
   nt = Threads.nthreads()
   SVector{nt}([alloc(constructor, cache_size) for _ in Base.OneTo(nt)])
end

# Construct `cache_size` number of objects with `constructor`
# and store them in an SVector
function alloc(constructor, cache_size)
   SVector{cache_size}(constructor(undef) for _ in Base.OneTo(cache_size))
end