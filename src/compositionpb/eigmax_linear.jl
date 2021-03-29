
raw"""
EigmaxLinear

Data structure for problem
```math
\min λ₁(A₀ + \Sum_1^n x[i] * A_i) for x ∈ ℝⁿ
```
"""
struct EigmaxLinear <: CompositionCompoPb
    m::Int64
    n::Int64
    As::Vector{Matrix{Float64}}
end


#
### Shared methods
#
function F(pb::EigmaxLinear, x)
    return eigmax(g(pb, x))
end
F(pb::EigmaxLinear, x, i) = eigvals(g(pb, x))[pb.m-i]

function ∂F_elt(pb::EigmaxLinear, x)
    decomp = eigen(g(pb, x))

    # Collect indices of eigvals close to max eigval
    active_inds = filter(i->maximum(decomp.values) - decomp.values[i] < 1e-13, 1:pb.m)
    length(active_inds) > 1 && @show active_inds

    subgradient_max = zeros(size(decomp.values))
    for i in active_inds
        subgradient_max[i] = 1/length(active_inds)
    end

    subgradient_λmax = decomp.vectors * Diagonal(subgradient_max) * decomp.vectors'
    subgradient_g = Dgconj(pb, x, subgradient_λmax)

    return subgradient_g
end

function is_differentiable(pb::EigmaxLinear, x)
    gx_eigvals = eigvals(g(pb, x))
    gx_eigvalsmax = maximum(gx_eigvals)
    return length(filter(λᵢ-> gx_eigvalsmax - λᵢ < 1e-13, gx_eigvals)) == 1
end


#
### Problem specific methods
#
function g(pb::EigmaxLinear, x)
    res = zeros(pb.m, pb.m)
    res .= pb.As[1]
    for i in 1:pb.n
        res .+= x[i] .* pb.As[i+1]
    end
    return res
end

function Dgconj(pb::EigmaxLinear, x, M::AbstractMatrix)
    return [dot(pb.As[i], M) for i in 2:length(pb.As)]
end
