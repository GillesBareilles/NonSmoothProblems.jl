using Random
using Test
using Zygote


@testset "MaxQuad oracles: g and h" begin
    pb = MaxQuadBGLS()
    M = NSP.MaxQuadManifold(pb, [2, 3, 4 ,5])

    Random.seed!(1465)
    @testset "g derivatives" begin
        @testset "point $v" for v in 1:10
            x = rand(pb.n)
            d = rand(pb.n)
            i = rand(1:pb.k)

            @test NSP.g(pb, x) ≈ [NSP.gᵢ(pb, x, i) for i in 1:pb.k]
            @test NSP.Dg(pb, x) ≈ Zygote.jacobian(x -> NSP.g(pb, x), x)[1]
            @test NSP.Dg(pb, x) * d ≈ NSP.Dg(pb, x, d)
            @test NSP.∇²gᵢ(pb, x, i, d) ≈ Zygote.hessian(x -> NSP.gᵢ(pb, x, i), x) * d
        end
    end

    @testset "h derivatives" begin
        @testset "h derivatives" for v in 1:10
	          x = rand(pb.n)
            d = rand(pb.n)
            i = rand(1:length(M.active_fᵢ_indices)-1)

            @test NSP.manifold_codim(M) == length(M.active_fᵢ_indices) - 1

            @test NSP.h(M, x) ≈ [NSP.hᵢ(M, x, i) for i in 1:NSP.manifold_codim(M)]
            @test NSP.Jac_h(M, x) ≈ Zygote.jacobian(x -> NSP.h(M, x), x)[1]
            @test NSP.Jac_h(M, x) * d ≈ NSP.Dh(M, x, d)
            @test NSP.∇²hᵢ(M, x, i, d) ≈ Zygote.hessian(x -> NSP.hᵢ(M, x, i), x) * d
        end
    end
end
