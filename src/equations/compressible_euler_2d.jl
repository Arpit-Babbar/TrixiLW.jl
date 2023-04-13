@inline function is_admissible(u, equations::CompressibleEulerEquations2D)
   prim = cons2prim(u, equations)
   bool = prim[1] > 0.0 && prim[4] > 0.0
   return bool
end