# TrixiLW.jl

An implementation of Lax-Wendroff Flux Reconstruction scheme for curvilinear meshes with adaptive mesh refinement and error based time stepping using [`Trixi.jl`](https://github.com/trixi-framework/Trixi.jl) as a library. To run the code, enter the following the `julia` REPL.

## Users

Using the multiple dispatch of `julia`, most things that you wish to do with the code (including developing your own algorithms) can be done by working with `TrixiLW.jl` as a user. Execute the following in the `julia` REPL.

```julia
julia> import Pkg; Pkg.add(url="https://github.com/arpit-babbar/TrixiLW.jl.git")
```
Then, `TrixiLW.jl` can be loaded as a `julia` package using the following command
```julia
julia> using TrixiLW
```
You can also run any of the [available examples](https://github.com/Arpit-Babbar/TrixiLW.jl/tree/master/examples). The first time you use `using` and the first time you use an example will be slower than the subsequent times.

## Developers

You should do this if you find something in `TrixiLW.jl` to be incompatible with your use case. In this case, I will also be happy to make changes in `TrixiLW.jl` to adapt it to your needs. This is likely to greatly help `TrixiLW.jl`. Thus, feel free to raise an issue, make a pull request or to [email me](mailto:arpit@babbar.dev). For development, clone (ideally, after forking) the repository and then run the following in the `julia` REPL when you are in the `TrixiLW.jl` directory to install the dependencies

```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```
You can skip the `Pkg.activate(".")` command by starting `julia` as `julia --project=.` in the `TrixiLW.jl` directory. You can now run any example, e.g., as
```julia
julia> include("examples/p4est_2d_dgsem/elixir_advection_basic.jl")
```

## Cite us!

If you use this code in your work, please cite us as

```bibtex
@article{BabbarChandrashekar2025,
title = {Lax-Wendroff flux reconstruction on adaptive curvilinear meshes with error based time stepping for hyperbolic conservation laws},
journal = {Journal of Computational Physics},
volume = {522},
pages = {113622},
year = {2025},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2024.113622},
url = {https://www.sciencedirect.com/science/article/pii/S0021999124008702},
author = {Arpit Babbar and Praveen Chandrashekar},
keywords = {Hyperbolic conservation laws, Lax-Wendroff flux reconstruction, Curvilinear grids, Admissibility preservation and shock capturing, Adaptive mesh refinement, Error based time stepping},
abstract = {Lax-Wendroff Flux Reconstruction (LWFR) is a single-stage, high order, quadrature free method for solving hyperbolic conservation laws. This work extends the LWFR scheme to solve conservation laws on curvilinear meshes with adaptive mesh refinement (AMR). The scheme uses a subcell based blending limiter to perform shock capturing and exploits the same subcell structure to obtain admissibility preservation on curvilinear meshes. It is proven that the proposed extension of LWFR scheme to curvilinear grids preserves constant solution (free stream preservation) under the standard metric identities. For curvilinear meshes, linear Fourier stability analysis cannot be used to obtain an optimal CFL number. Thus, an embedded-error based time step computation method is proposed for LWFR method which reduces fine-tuning process required to select a stable CFL number using the wave speed based time step computation. The developments are tested on compressible Euler's equations, validating the blending limiter, admissibility preservation, AMR algorithm, curvilinear meshes and error based time stepping.}
}
```
