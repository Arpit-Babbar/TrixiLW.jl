@inline function is_admissible(u, equations::CompressibleEulerEquations2D)
   prim = cons2prim(u, equations)
   return prim[1] > 0.0 && prim[4] > 0.0
end