
raw"""
EigmaxLinear

Data structure for problem
```math
\min λ₁(A₀ + \Sigma_1^n x[i] * A_i) for x ∈ ℝⁿ
```
"""
struct EigmaxLinear <: CompositionCompoPb
    m::Int64
    n::Int64
    As::Vector{Symmetric}
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
    res = pb.As[1] + sum(x[i] .* pb.As[i+1] for i in 1:pb.n)

    # res = zeros(pb.m, pb.m)
    # for i in 1:pb.n
    #     res .+= x[i] .* pb.As[i+1]
    # end
    return res
end

function Dgconj(pb::EigmaxLinear, x, M::AbstractMatrix)
    return [dot(pb.As[i], M) for i in 2:length(pb.As)]
end



#
### Corresponding manifold
#
"""
    EigmaxLinearManifold

A manifold associated with the problem `eigmaxlinearpb::EigmaxLinear`. Gathers all
points `x` such that the largest eigenvalue of the symmetric real matrix `g(x)` has
multiplicity `r`.
"""
struct EigmaxLinearManifold <: AbstractManifold
    eigmaxlinearpb::EigmaxLinear
    r::Int64
end

"""
    h(M::EigmaxLinearManifold, x)

Mapping that defines `M::EigmaxLinearManifold` as `M = h^{-1}({0})`.
"""
function h(M::EigmaxLinearManifold, x)
    eigvals_gx = eigvals(g(M.eigmaxlinearpb, x))
    return eigvals_gx[end-M.r+1:end] .- mean(eigvals_gx[end-M.r+1:end])
end

# function Dh(M::EigmaxLinearManifold, x)

# end


"""
    point_manifold(pb::EigmaxLinear, x)

Find a manifold of type `EigmaxLinearManifold` near `x`.

Implementing the heuristic mentioned in
Noll & Apkarian, 2005, second order methods, eq. (3).
"""
function point_manifold(pb::EigmaxLinear, x)
    Λ = eigvals(g(pb, x))
    sort!(Λ, rev=true)
    r = 1

    # each iteration tests if r+1 is valid.
    τ = 1e-6
    while true
        if ((Λ[1] - Λ[r]) / max(1, abs(Λ[1])) <= τ) && ((Λ[1] - Λ[r+1]) / max(1, abs(Λ[1])) > τ)
            break
        end
        r += 1
    end

    return EigmaxLinearManifold(pb, r)
end