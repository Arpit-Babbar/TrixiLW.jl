julia  -e 'using Pkg; Pkg.add(PackageSpec(name = "JuliaFormatter", version="1.0.60"))'
julia  -e 'using JuliaFormatter; format(["examples", "src", "utils", "test"])'
