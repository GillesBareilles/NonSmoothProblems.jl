module NonSmoothProblems

import Base.show

using LinearAlgebra
using Random
using Distributions
using DataStructures

using FiniteDifferences

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

include("compositionpb/eigmax_linear.jl")
include("compositionpb/eigmax_linear_instances.jl")


export NonSmoothPb
export F, ∂F_elt, ∂F_minnormelt, is_differentiable

export SimpleQuad, SmoothQuad, Simplel1
export SmoothQuad1d, SmoothQuad2d_1, SmoothQuad2d_2

export MaxQuadPb
export MaxQuad2d, MaxQuadAL, MaxQuadMaratos, MaxQuadBGLS

export EigmaxLinear
export EigmaxLinearManifold
export get_eigmaxlinear_pb

end # module
