"""
    LogitL1

Problem defined as:
´min_x  1/m * ∑_{i=1}^m log(1 + exp(-y_i dot(A_i, x))) + λ₁|x|₁ + λ₂|x|₂²´

## Parameters
- x : R^n
- y : R^m observations, *values -1 and 1*
- A : mxn matrix of samples
"""
mutable struct LogitL1{Tf} <: AdditiveCompoPb
    A::Matrix{Tf}
    y::Vector{Tf}
    λ₁::Tf
    λ₂::Tf
    n::Int64
    x0::Vector{Tf}
    function LogitL1(
            A::Matrix{Tf},
            y::Vector{Tf},
            λ₁::Tf,
            λ₂::Tf,
            n::Int64,
            x0::Vector{Tf},
        ) where {Tf}
        @assert Set(y) ⊆ Set(Tf[-1.0, 1.0]) "Logistic rhs vector shoudl take values -1.0, 1.0, here: $(Set(y))."
        return new{Tf}(A, y, λ₁, λ₂, n, x0)
    end
end


#
### Shared methods
#
function F(pb::LogitL1, x)
    return f(pb, x) + pb.λ₁ * norm(x, 1)
end

function ∂F_elt(pb::LogitL1, x)
    throw(error("Not implemented."))
end

function is_differentiable(pb::LogitL1, x)
    throw(error("Not implemented."))
end

################################################################################
### Corresponding manifold
################################################################################
"""
    L1Manifold

A manifold associated with the problem `LogitL1`. All points `x` such that the
coordinates indexed `nz_coords` are non null.
"""
struct L1Manifold{Tf} <: AbstractManifold
    pb::LogitL1{Tf}
    nz_coords::BitArray{1}
end
L1Manifold(pb::LogitL1, nz_coords) = L1Manifold(pb, convert(BitArray, nz_coords))
Base.show(io::IO, M::L1Manifold{Tf}) where {Tf} = print(io, "L1(", findall(!, M.nz_coords), ")")

manifold_codim(M::L1Manifold) = sum(.!M.nz_coords)

function select_activestrata(M::L1Manifold, x)
    throw(error("Not implemented."))
end

function h(M::L1Manifold{Tf}, x) where {Tf}
    manifold_codim(M) == 0 && return Tf[]
    return x[.!M.nz_coords]
end

function Jac_h(M::L1Manifold{Tf}, x) where Tf
    m = manifold_codim(M)
    n = M.pb.n
    Is = 1:m
    Js = findall(.!M.nz_coords)
    Vs = ones(Tf, m)
    return sparse(Is, Js, Vs, m, n)
end

function ∇²hᵢ(::L1Manifold{Tf}, x, i, η) where {Tf}
    return zeros(Tf, size(η))
end

function point_manifold(pb::LogitL1, x)
    return L1Manifold(pb, map(t -> abs(t) > 1e-3, x))
end


################################################################################
# Smooth extension on manifold
################################################################################
F̃(pb::LogitL1, ::L1Manifold, x) = F(pb, x)

function ∇F̃(pb::LogitL1, ::L1Manifold, x)
    res = ∇f(pb, x)
    res .+= pb.λ₁ .* sign.(x)

    return res
end

function ∇²F̃(pb::LogitL1, ::L1Manifold, x, h)
    return ∇²f(pb, x, h)
end

################################################################################
# Problem specific methods
################################################################################
"""
    logsig(t)
Compute the logarithm of sigmoid `-log(1+exp(-t))` with higher precision than plain
implementation.
Reference:
- F. Pedragosa's blog post http://fa.bianp.net/blog/2019/evaluate_logistic/
"""
@inline function logsig(t)
    if t < -33.3
        return t
    elseif t <= -18
        return t - exp(t)
    elseif t <= 37
        return -log1p(exp(-t))
    else
        return -exp(-t)
    end
end

σ(x) = 1/(1+exp(-x))

∇σ(x) = σ(x) * σ(-x)

function f(pb::LogitL1, x)
    m = size(pb.A, 1)

    Ax = pb.A * x
    fval = 0.0
    @inbounds @simd for i in 1:m
        fval -= logsig(pb.y[i] * Ax[i])
    end

    return fval / m + 0.5 * pb.λ₂ * norm(x, 2)^2
end

function ∇f(pb::LogitL1, x)
    m = size(pb.A, 1)

    σyAx = pb.A * x
    σyAx .*= -pb.y
    σyAx .= σ.(σyAx)
    res = transpose(pb.A) * (σyAx .* pb.y)
    res ./= -m
    res .+= pb.λ₂ .* x

    return res
end

function ∇²f(pb::LogitL1, x, h)
    m = size(pb.A, 1)

    yAx = -pb.y .* (pb.A * x)
    Ah = pb.A * h

    res = transpose(pb.A) * (Ah .* ∇σ.(yAx))
    res ./= m
    res .+= pb.λ₂ .* h

    return res
end
