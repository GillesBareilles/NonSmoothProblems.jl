module NonSmoothProblems

import Base: show, ==

using DocStringExtensions

using LinearAlgebra
using GenericLinearAlgebra
using GenericSchur
using SparseArrays
using Random
using Distributions
using DataStructures

using EigenDerivatives
import EigenDerivatives: g, Dgconj

const NSP = NonSmoothProblems

"""
    NonSmoothPb

Abstract type for generic nonsmooth problem.
"""
abstract type NonSmoothPb end

F(pb::NonSmoothPb, x) = throw(error("F(): Not implemented for problem type $(typeof(pb))."))
∂F_elt(pb::NonSmoothPb, x) = throw(error("∂F_elt(): Not implemented for problem type $(typeof(pb))."))
is_differentiable(pb::NonSmoothPb, x) = throw(error("is_differentiable(): Not implemented for problem type $(typeof(pb))."))

"""
    $TYPEDSIGNATURES

Compute function value, a subgradient and tells wether the function is
differentiable at point `x`.
"""
function firstorderoracle(pb, x)
    return F(pb, x), ∂F_elt(pb, x), is_differentiable(pb, x)
end

include("simpleNSPb.jl")


"""
    CompositePb

Abstract type for nonsmooth problem having some structure, see...
"""
abstract type CompositePb <: NonSmoothPb end

"""
    AdditiveCompoPb

Nonsmooth problems that write as min_x F(x) = f(x) + g(x), where f is a smooth function and g a nonsmooth proxsimple function.
"""
abstract type AdditiveCompoPb <: CompositePb end


export f, ∇f!, proxg!

"""
    CompositionCompoPb

Nonsmooth problems which writes as min_x F(x) = f o Φ (x).
"""
abstract type CompositionCompoPb <: CompositePb end


"""
    AbstractManifold

Manifolds embedded in ℝⁿ.
"""
abstract type AbstractManifold end



include("compositionpb/maxquad.jl")
include("compositionpb/maxquad_instances.jl")

# include("compositionpb/eigmax_linear.jl")
# include("compositionpb/eigenderivatives.jl")
# include("compositionpb/eigmax_linear_instances.jl")

include("compositionpb/eigmax.jl")
include("compositionpb/eigmax_instances.jl")

include("additivepb/logitl1_manifold.jl")
include("additivepb/logitl1.jl")
include("additivepb/logitl1_instances.jl")

include("additivepb/lasso.jl")
include("additivepb/utils_MCP.jl")
include("additivepb/leastsquaresMCP.jl")
include("additivepb/leastsquaresMCP_instances.jl")

include("halfhalf.jl")

"""
    $(TYPEDSIGNATURES)

Compute the hessian of the lagrangian of the problem of minimizing a smooth
extension of the objective function of `pb` on manifold `M` constrained on
that manifold.
"""
function ∇²Lagrangian!(res, pb, M, x, λ, d)
    res .= ∇²F̃(pb, M, x, d)

    for i in axes(λ, 1)
        res .-= λ[i] .* ∇²hᵢ(M, x, i, d)
    end
    return res
end

"""
    $TYPEDSIGNATURES

Compute the lagrangian matrix corresponding assciated with the inplace method
`∇²Lagrangian!`.

Note: meant as a helper for developping methods, rather inefficient.
"""
function ∇²L(pb, M, x::Vector{Tf}, λ::Vector{Tf}) where Tf
    n = pb.n
    res = zeros(Tf, pb.n, pb.n)
    d = similar(x)
    for i in 1:n
        d .= 0
        d[i] = 1
        resᵢ = @view res[:, i]
        ∇²Lagrangian!(resᵢ, pb, M, x, λ, d)
    end
    return res
end

export NSP

export g, Dg, Dgconj
export manifold_dim, manifold_codim

export NonSmoothPb
export F, ∂F_elt, is_differentiable, firstorderoracle

export SimpleQuad, SmoothQuad, Simplel1
export SmoothQuad1d, SmoothQuad2d_1, SmoothQuad2d_2

export MaxQuadPb, MaxQuadManifold
export MaxQuad2d, MaxQuadAL, MaxQuadMaratos, MaxQuadBGLS

export Eigmax, EigmaxManifold
export get_eigmax_affine, get_eigmax_powercoord, get_eigmax_nlmap
export get_eigmax_AL33
export AffineMap, PowerCoordMap, NonLinearMap


export FixedSparsityManifold
export LogitL1
export get_logit_MLE
export Lasso
export MCPLeastSquares

export Halfhalf, HalfhalfManifold

end # module
