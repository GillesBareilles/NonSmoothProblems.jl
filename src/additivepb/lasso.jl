raw"""
    $TYPEDSIGNATURES

Problem defined as:
´min_x  1/2 * \|Ax - y\|_{2}^{2} + \lambda|x|₁´

## Parameters
- x : R^n
- y : R^m observations
- A : mxn matrix of samples
"""
struct Lasso{Tf} <: AdditiveCompoPb
    A::Matrix{Tf}
    y::Vector{Tf}
    λ::Tf
    n::Int64
    # function Lasso(
    #         A::Matrix{Tf},
    #         y::Vector{Tf},
    #         λ₁::Tf,
    #         λ₂::Tf,
    #         n::Int64,
    #         x0::Vector{Tf},
    #     ) where {Tf}
    #     @assert Set(y) ⊆ Set(Tf[-1.0, 1.0]) "Logistic rhs vector shoudl take values -1.0, 1.0, here: $(Set(y))."
    #     return new{Tf}(A, y, λ₁, λ₂, n, similar(y), similar(y), x0)
    # end
end


#
### Shared methods
#
function F(pb::Lasso, x)
    return f(pb, x) + g(pb, x)
end

function ∂F_elt(pb::Lasso, x)
    res = similar(x)
    ∇f!(pb, res, x)
    @. res += pb.λ * sign(x)
    return res
end

function is_differentiable(::Lasso, x)
    return isnothing(findfirst(t->t==0, x))
end




## f
# 0th order
f(pb::Lasso, x) = 0.5 * norm(pb.A * x - pb.y)^2

# 1st order
∇f!(pb::Lasso, res, x) = (res .= transpose(pb.A) * (pb.A * x - pb.y))
∇f(pb::Lasso, x) = transpose(pb.A) * (pb.A * x - pb.y)

g(pb::Lasso, x) = pb.λ * norm(x, 1)
