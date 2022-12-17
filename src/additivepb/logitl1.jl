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
    Axtemp::Vector{Tf}
    Ahtemp::Vector{Tf}
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
        return new{Tf}(A, y, λ₁, λ₂, n, similar(y), similar(y), x0)
    end
end


#
### Shared methods
#
function F(pb::LogitL1, x)
    return f(pb, x) + g(pb, x)
end

function ∂F_elt(pb::LogitL1, x)
    throw(error("Not implemented."))
end

function is_differentiable(pb::LogitL1, x)
    throw(error("Not implemented."))
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
    end
    return -exp(-t)
end

σ(x) = 1/(1+exp(-x))

∇σ(x) = σ(x) * σ(-x)

function f(pb::LogitL1{Tf}, x::Vector{Tf}) where Tf
    m = size(pb.A, 1)
    Ax = pb.Axtemp

    mul!(Ax, pb.A, x)
    fval = 0.0
    for i in axes(pb.A, 1)
        fval -= logsig(pb.y[i] * Ax[i])
    end

    res = fval / m + 0.5 * pb.λ₂ * norm(x, 2)^2
    return res
end

function ∇f!(res, pb::LogitL1, x)
    m = size(pb.A, 1)
    σyAx = pb.Axtemp

    mul!(σyAx, pb.A, x)
    σyAx .= σ.(-1 .* pb.y .* σyAx)

    σyAx .*= pb.y
    mul!(res, pb.A', σyAx)
    res .= res ./ (-m) .+ pb.λ₂ .* x
    return nothing
end

function ∇f(pb::LogitL1, x)
    res = similar(x)
    ∇f!(res, pb, x)
    return res
end

function ∇²f!(res, pb::LogitL1, x, h)
    m = size(pb.A, 1)
    yAx = pb.Axtemp

    # yAx = -1 .* pb.y .* (pb.A * x)
    mul!(yAx, pb.A, x)
    yAx .= -1 .* pb.y .* (yAx)

    # mul!(res, pb.A', Ah .* ∇σ.(yAx))
    mul!(pb.Ahtemp, pb.A, h)
    pb.Ahtemp .*= ∇σ.(yAx)
    mul!(res, pb.A', pb.Ahtemp)

    res .= res ./ m .+ pb.λ₂ .* h
    return nothing
end

function ∇²f(pb::LogitL1, x, h)
    res = similar(x)
    ∇²f!(res, pb, x, h)
    return res
end


function g(pb::LogitL1, x)
    return pb.λ₁ * norm(x, 1)
end

softthresh(x, γ) = sign(x) * max(0, abs(x) - γ)

function proxg!(res, pb::LogitL1, x, γ)
    res .= softthresh.(x, pb.λ₁ * γ)
    M = L1Manifold(pb, map(t->t == 0, res))
    return M
end


################################################################################
# Smooth extension on manifold
################################################################################
FixedSparsityManifold(pb::LogitL1, nz_coords::Vector{Bool}) = FixedSparsityManifold(pb, convert(BitArray, nz_coords))

function point_manifold(pb::LogitL1, x)
    return FixedSparsityManifold(pb, map(t -> abs(t) > 1e-3, x))
end


### Smooth extensions
F̃(pb::LogitL1, ::FixedSparsityManifold, x) = F(pb, x)

function ∇F̃(pb::LogitL1, ::FixedSparsityManifold, x)
    res = ∇f(pb, x)
    res .+= pb.λ₁ .* sign.(x)

    return res
end

function ∇²F̃(pb::LogitL1, ::FixedSparsityManifold, x, h)
    return ∇²f(pb, x, h)
end
