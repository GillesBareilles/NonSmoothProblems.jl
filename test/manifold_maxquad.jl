using Test
using NonSmoothProblems
using LinearAlgebra
using Zygote
using Random
const NSP = NonSmoothProblems

@testset "Generic manifold" begin
    pb = MaxQuadBGLS()
    M = NSP.MaxQuadManifold(pb, [2, 3, 4, 5])

    @testset "projection on tangent space" begin
        x = rand(pb.n)
        proj_JuMP = similar(x)
        proj_expl = similar(x)

        Jₕx = NSP.Jac_h(M, x) # this function is tested in maxquad.jl
        tangentspacebasis = nullspace(Jₕx)
        for i in 1:10
            d = rand(pb.n)

            NSP.project_tangent!(proj_JuMP, M, x, d)
            @test norm(proj_JuMP - tangentspacebasis * tangentspacebasis' * proj_JuMP) < 1e-14

            λ = NSP.get_normalspace_coordinates(Jₕx, d)
            proj_expl = NSP.normalcoords_to_tangentrep(Jₕx, λ, d)
            @test norm(proj_expl - tangentspacebasis * tangentspacebasis' * proj_expl) < 1e-14
            @test d ≈ proj_expl + NSP.normalcoords_to_normalrep(Jₕx, λ, d)
        end

    end


    @testset "Riemannian gradient and smooth extension" begin
        @testset "Manifold dimension $mandim" for mandim in 1:pb.k
            Random.seed!(16534 + mandim)
            x = randn(pb.n)
            d = rand(pb.n)

            # generate a random manifold inclduing the active function at x.
            ind_actstrata = argmax(NSP.g(pb, x))
            inds_nonactstrata = setdiff(1:pb.k, ind_actstrata)
            manifold_activefuncitons = union(ind_actstrata, inds_nonactstrata[randperm(pb.k-1)[1:mandim-1]])
            M = NSP.MaxQuadManifold(pb, union(ind_actstrata, manifold_activefuncitons))

            # Riemannian objects
            gradFx, HessFx! = NSP.grad_Hess(pb, M, x)

            # lagrangian based objects
            Jₕx = NSP.Jac_h(M, x)
            ∇F̃x = NSP.∇F̃(pb, M, x)
            λ = NSP.get_normalspace_coordinates(Jₕx, ∇F̃x)

            @test gradFx ≈ project_tangent(M, x, ∇F̃x)

            # Order 2
            lagrangian(x, λ) = F(pb, x) - dot(λ, NSP.h(M, x))

            @test gradFx ≈ project_tangent(M, x, Zygote.gradient(x -> lagrangian(x, λ), x)[1])

            dtan = project_tangent(M, x, d)
            HessFxd = Zygote.hessian(x -> lagrangian(x, λ), x) * dtan
            project_tangent!(HessFxd, M, x, HessFxd)

            HessFxd = similar(d)
            HessFx!(HessFxd, dtan)
            @test HessFxd ≈ HessFxd
        end
    end
end
