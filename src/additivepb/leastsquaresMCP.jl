"""
    $TYPEDSIGNATURES

Description of the least square MCP problem.

## Reference:
[1] C.H. Zhang. Nearly umbiased variable selection under minimax concave penalty.
    The Annals of Statistics 38(2):894-942, 2010
"""
struct MCPLeastSquares{Tf}
    A::Matrix{Tf}
    y::Vector{Tf}
    λ::Tf
    β::Tf
    n::Int64
end

#
### Shared methods
#
F(pb::MCPLeastSquares, x) = f(pb, x) + r(pb, x)

function ∂F_elt(pb::MCPLeastSquares, x)
    v = similar(x)
    ∇f!(v, pb, x)
    add_∇r!(v, pb, x)
    return v
end

is_differentiable(::MCPLeastSquares, x) = isnothing(findfirst(t->t==0, x))


#### Writing this as a composition
g(::MCPLeastSquares, x) = x # NOTE: the smooth term f(x) is irrelevant for structure detection


## f
# 0th order
f(pb::MCPLeastSquares, x) = 0.5 * norm(pb.A * x - pb.y)^2

# 1st order
∇f!(res, pb::MCPLeastSquares, x) = (res .= transpose(pb.A) * (pb.A * x - pb.y))
∇f(pb::MCPLeastSquares, x) = transpose(pb.A) * (pb.A * x - pb.y)

# 2nd order
∇²f_h!(res, pb::MCPLeastSquares, x, h) = (res .= transpose(pb.A) * (pb.A * h))
∇²f_h(pb::MCPLeastSquares, x, h) = transpose(pb.A) * (pb.A * h)

## r
function r(pb::MCPLeastSquares{Tf}, x::Vector{Tf}) where Tf
    res = Tf(0)
    for xᵢ in x
        res += MCP(xᵢ, pb.β, pb.λ)
    end
    return res
end

function proxr(pb::MCPLeastSquares, x, γ)
    res = similar(x)
    proxr!(res, pb, x, γ)
    return res
end
function proxr!(res, pb::MCPLeastSquares, x, γ)
    res .= prox_MCP.(x, γ, pb.β, pb.λ)
    return nothing
end

function add_∇r!(res, pb::MCPLeastSquares, x)
    res .+= ∇MCP.(x, pb.β, pb.λ)
    return nothing
end

"""
    $TYPEDSIGNATURES

"""
function add_∇²r_ξ!(res, pb::MCPLeastSquares, x, ξ)
    for i in axes(x, 1)
        res[i] += ∇²MCP(x[i], ξ[i], pb.β, pb.λ)
    end
    return nothing
end


################################################################################
# Smooth extension on manifold
################################################################################
FixedSparsityManifold(pb::MCPLeastSquares, nz_coords::Vector{Bool}) = FixedSparsityManifold(pb, convert(BitArray, nz_coords))

function point_manifold(pb::MCPLeastSquares, x)
    return FixedSparsityManifold(pb, map(t -> abs(t) > 1e-3, x))
end


### Smooth extensions
F̃(pb::MCPLeastSquares, ::FixedSparsityManifold, x) = F(pb, x)

function ∇F̃(pb::MCPLeastSquares, M::FixedSparsityManifold, x)
    res = ∇f(pb, x)
    add_∇r!(res, pb, x)

    # for i in findall(M.nz_coords)
    #     res .+= ∇MCP(x[i], pb.β, pb.λ)
    # end
    return res
end

function ∇²F̃(pb::MCPLeastSquares, M::FixedSparsityManifold, x, h)
    res = ∇²f_h(pb, x, h)
    add_∇²r_ξ!(res, pb, x, h)

    return res
end

function ∇²Lagrangian!(res, pb::MCPLeastSquares, M, x, λ, d)
    ∇²f_h!(res, pb, x, d)
    add_∇²r_ξ!(res, pb, x, d)

    return nothing
end
