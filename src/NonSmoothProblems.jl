module NonSmoothProblems

using TimerOutputs

import Base.show

using DocStringExtensions

using LinearAlgebra
using GenericLinearAlgebra
using GenericSchur
using SparseArrays
using Random
using Distributions
using DataStructures
using Manifolds

using EigenDerivatives
import EigenDerivatives: g, Dg, Dgconj, h, Dh, Jac_h, L, ∇L, ∇²L

using FiniteDifferences
using Infiltrator


const NSP = NonSmoothProblems

"""
    NonSmoothPb

Abstract type for generic nonsmooth problem.
"""
abstract type NonSmoothPb end

F(pb::NonSmoothPb, x) = throw(error("F(): Not implemented for problem type $(typeof(pb))."))
∂F_elt(pb::NonSmoothPb, x) = throw(error("∂F_elt(): Not implemented for problem type $(typeof(pb))."))
∂F_minnormelt(pb::NonSmoothPb, x) = throw(error("∂F_minnormelt(): Not implemented for problem type $(typeof(pb))."))
is_differentiable(pb::NonSmoothPb, x) = throw(error("is_differentiable(): Not implemented for problem type $(typeof(pb))."))

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

include("additivepb/logitl1.jl")
include("additivepb/logitl1_instances.jl")

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


export NSP

export g, Dg, Dgconj

export NonSmoothPb
export F, ∂F_elt, ∂F_minnormelt, is_differentiable

export SimpleQuad, SmoothQuad, Simplel1
export SmoothQuad1d, SmoothQuad2d_1, SmoothQuad2d_2

export MaxQuadPb, MaxQuadManifold
export MaxQuad2d, MaxQuadAL, MaxQuadMaratos, MaxQuadBGLS

# export EigmaxLinear, EigmaxLinearManifold
# export get_eigmaxlinear_pb

export Eigmax, EigmaxManifold
export get_eigmax_linear_pb

export LogitL1, L1Manifold
export get_logit_MLE

export Halfhalf, HalfhalfManifold

end # module
