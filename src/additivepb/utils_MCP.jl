"""
    $TYPEDSIGNATURES

Compute the value of the MCP penalty at a scalar `x`, with parameters `β` and `λ`.

Reference:
- http://proximity-operator.net/nonconvexfunctions.html
"""

function MCP(x::Tf, β::Tf, λ::Tf) where Tf
    if abs(x) ≥ β * λ
        return β * λ^2 / 2
    else
        return λ * abs(x) - x^2 / (2β)
    end
end

"""
    $TYPEDSIGNATURES

Compute the proximal operator of the MCP penalty, with stepsize `γ`.
"""
function prox_MCP(x::Tf, γ::Tf, β::Tf, λ::Tf) where Tf
    if β <= γ
        return abs(x) > sqrt(γ*β)*λ ? x : Tf(0)
    else
        return sign(x) * min( β/(β-γ) * max(abs(x)-λ*γ, 0), abs(x) )
    end
end

"""
    $TYPEDSIGNATURES

Gradient of the (one dimensional) MCP function (at nonzero points).
"""
function ∇MCP(x::Tf, β::Tf, λ::Tf) where Tf
    if abs(x) ≥ β * λ
        return Tf(0)
    else
        return λ * sign(x) - x / β
    end
end

"""
    $TYPEDSIGNATURES

Second-derivative of the (one dimensional) MCP function at `x`, in direction `h`.
"""
function ∇²MCP(x::Tf, h::Tf, β::Tf, λ::Tf) where Tf
    if abs(x) ≥ β * λ
        return Tf(0)
    else
        return - h / β
    end
end
