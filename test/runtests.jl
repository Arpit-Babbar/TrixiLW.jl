using TrixiBase
using Test
using TrixiLW
using TrixiLW: examples_dir_trixilw
using DelimitedFiles

# Replace solve_lwfr with a "do nothing" function

struct MyNothing end

# Change this to regenerate the testing data
to_overwrite_errors() = false

function TrixiLW.solve_lwfr(::MyNothing, callbacks, dt_initial, tolerances;
                            time_step_computation = CFLBased(), limiters = (;))
    return nothing
end

test_data_dir() = joinpath(@__DIR__, "data")

function get_errors(sol, analysis_callback)
    (; l2, linf) = analysis_callback(sol)
    return l2, linf
end

function compare_errors(sol, analysis_callback, l2_ref, linf_ref; tol = 1e-13)
    (; l2, linf) = analysis_callback(sol)
    nvar = nvariables(sol.prob.p)
    for i in 1:nvar
        @test isapprox(l2[i], l2_ref[i], atol = tol)
        @test isapprox(linf[i], linf_ref[i], atol = tol)
    end
end

function compare_errors_txt(sol, analysis_callback, testname; tol = 1e-13,
                            overwrite_errors = false)
    (; l2, linf) = analysis_callback(sol)
    datafile_l2 = joinpath(test_data_dir(), "$(testname)_l2.txt")
    datafile_linf = joinpath(test_data_dir(), "$(testname)_linf.txt")
    if overwrite_errors
        println("Overwriting $datafile_l2, this should not be triggered in actual testing.")
        writedlm(datafile_l2, l2)
        println("Overwriting $datafile_linf, this should not be triggered in actual testing.")
        writedlm(datafile_linf, linf)
    else
        l2_ref = readdlm(datafile_l2)
        linf_ref = readdlm(datafile_linf)
        compare_errors(sol, analysis_callback, l2_ref, linf_ref; tol = tol)
    end
end

function get_kwarg(args, keyword, default_value)
    val = default_value
    for arg in args
        if arg.head == :(=) && arg.args[1] == keyword
            val = arg.args[2]
            break
        end
    end
    return val
end

macro test_trixilw_include(mesh_name, elixir_name, args...)
    full_test_name = "$(mesh_name)_$(elixir_name)"
    full_elixir_name = joinpath(examples_dir_trixilw(), mesh_name,
                                "elixir_$(elixir_name).jl")

    final_time = get_kwarg(args, :final_time, 0.01)
    trixi_include(@__MODULE__, full_elixir_name,
                  tspan = (0.0, final_time), initial_refinement_level = 2)

    return full_test_name, sol, analysis_callback
end

# TODO - Test if everything below can be replaced with this function and macro
macro test_trixilw_elixir_run(mesh_name, elixir_name, args...)
    full_test_name = "$(mesh_name)_$(elixir_name)"
    full_elixir_name = joinpath(examples_dir_trixilw(), mesh_name,
                                "elixir_$(elixir_name).jl")
    trixi_include(@__MODULE__, full_elixir_name,
                  tspan = (0.0, 0.01), initial_refinement_level = 2)

    return full_test_name, sol, analysis_callback
end

macro test_trixilw_elixir(mesh_name, elixir_name, args...)
    full_test_name, sol, analysis_callback = test_trixilw_elixir_run(mesh_name, elixir_name,
                                                                     args...)
    tol = get_kwarg(args, :tol, 1e-14)
    overwrite_errors = get_kwarg(args, :overwrite_errors, to_overwrite_errors())
    @testset "$full_test_name" begin
        compare_errors_txt(sol, analysis_callback, full_test_name, tol = tol,
                           overwrite_errors = overwrite_errors)
    end
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "advection_basic")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "advection_amr")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "advection_nonperiodic")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "euler_density_wave")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "euler_density_wave_enzyme")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "euler_double_mach_reflection_square")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name, tol = 1e-12)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "euler_source_terms")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "euler_source_terms_nonperiodic")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "advection_diffusion_diff02")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "advection_diffusion_nonperiodic")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "advection_diffusion")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "navierstokes_convergence")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "navierstokes_lid_driven_cavity_ghia")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name, tol = 1e-10)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("tree_2d_dgsem",
                                                               "navierstokes_lid_driven_cavity")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name, tol = 1e-11)
end

# P4estMesh tests

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "advection_basic")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "advection_amr")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "advection_diffusion_diff02_periodic")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "advection_diffusion_nonperiodic_curved",
                                                               final_time=1e-4)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "advection_diffusion_periodic_curved")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "advection_extended")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_density_wave_enzyme")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_astrophysical_jet_amr",
                                                               final_time=1e-6)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name; tol = 1e-6)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_double_mach_reflection_amr",
                                                               final_time=1e-4)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name; tol = 1e-7)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_double_mach_reflection_square")
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name; tol = 1e-5)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_forward_step_amr",
                                                               final_time=1e-4)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name; tol = 1e-7)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_isentropic_hennemann",
                                                               final_time=1e-7)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_naca0012_deep_amr",
                                                               final_time=1e-4)
@testset "$full_test_name" begin
    # TODO - Can this tolerance be lowered?
    compare_errors_txt(sol, analysis_callback, full_test_name; tol = 1.0)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "euler_supersonic_cylinder_amr",
                                                               final_time=1e-4)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name; tol = 1e-7)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "navierstokes_convergence",
                                                               final_time=1e-4)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name)
end

full_test_name, sol, analysis_callback = @test_trixilw_include("p4est_2d_dgsem",
                                                               "navierstokes_naca0012_swanson",
                                                               final_time=1e-4)
@testset "$full_test_name" begin
    compare_errors_txt(sol, analysis_callback, full_test_name; tol = 1e-7)
end
