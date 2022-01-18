raw"""
Eigmax

Data structure for problem
```math
\min λ₁(A(x)) for x ∈ ℝⁿ
```
"""
struct Eigmax{Tf, Ta} <: CompositionCompoPb
    n::Int64
    A::Ta
end

#
### Shared methods
#
f(::Eigmax{BigFloat}, y) = maximum(eigvals(y))
f(::Eigmax, y) = eigmax(y)

F(pb::Eigmax, x) = f(pb, g(pb.A, x))

g(pb::Eigmax, x) = g(pb.A, x)
Dg(pb::Eigmax, x, d) = Dg(pb.A, x, d)
Dgconj(pb::Eigmax, x, D) = Dgconj(pb.A, x, D)

function ∂F_elt(pb::Eigmax{Tf}, x) where Tf
    A = pb.A

    U = eigvecs(g(A, x))
    subgradient_λmax = Symmetric(U[:, end] * U[:, end]')
    subgradient_g = Dgconj(A, x, subgradient_λmax)

    return subgradient_g
end

function is_differentiable(pb::Eigmax{Tf}, x) where Tf<:Real
    gx_eigvals = eigvals(g(pb.A, x))
    gx_eigvalsmax = maximum(gx_eigvals)
    return length(filter(λᵢ-> gx_eigvalsmax - λᵢ < 1e2 * eps(Tf), gx_eigvals)) == 1
end



################################################################################
# Corresponding manifold
################################################################################
"""
    EigmaxManifold

A manifold associated with the problem `pb::Eigmax`. Gathers all
points `x` such that the largest eigenvalue of the symmetric real matrix `g(x)` has
multiplicity `r`.
"""
struct EigmaxManifold{Tf} <: AbstractManifold
    pb::Eigmax{Tf}
    eigmult::EigMult{Tf, Vector{Tf}}
end
EigmaxManifold(pb::Eigmax{Tf}, r::Int64) where Tf = EigmaxManifold(pb, EigMult(1, r, zeros(Tf, pb.n), pb.A))
Base.show(io::IO, M::EigmaxManifold) = print(io, "Eigmax(", M.eigmult.r, ")")

function manifold_codim(M::EigmaxManifold)
    r = M.eigmult.r
    return r==1 ? 0 : r^2
end

"""
    h(M::EigmaxManifold, x)

Mapping that defines `M::EigmaxManifold` as `M = h^{-1}({0})`.
"""
function h(M::EigmaxManifold{Tf}, x::Vector{Tf}) where Tf
    return h(M.eigmult, M.pb.A, x)
end

function Dh(M::EigmaxManifold{Tf}, x::Vector{Tf}, d::Vector{Tf}) where Tf
    return Dh(M.eigmult, M.pb.A, x, d)
end

function Jac_h(M::EigmaxManifold{Tf}, x::Vector{Tf}) where Tf
    return Jac_h(M.eigmult, M.pb.A, x)
end



"""
    point_manifold(pb::Eigmax, x)

Find a manifold of type `EigmaxManifold` near `x`.

Implementing the heuristic mentioned in
Noll & Apkarian, 2005, second order methods, eq. (3).
"""
function point_manifold(pb::Eigmax{Tf}, x::Vector{Tf}) where Tf
    Λ = eigvals(g(pb.A, x))
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

    return EigmaxManifold(pb, r)
end



################################################################################
# Smooth extension on manifold
################################################################################
function F̃(::Eigmax{Tf}, M::EigmaxManifold{Tf}, x::Vector{Tf}) where Tf
    res = Tf(0)
    for i in 1:M.eigmult.r
        res += ϕᵢⱼ(M.eigmult, M.pb.A, x, i, i)
    end
    res /= M.eigmult.r
    return res
end

"""
    $TYPEDSIGNATURES

The smooth extension of the max eigenvalue on `M` defined as the
mean of the `M.eigmult.r` largest eigenvalues.
"""
function ∇F̃(::Eigmax{Tf}, M::EigmaxManifold{Tf}, x::Vector{Tf}) where Tf
    res = zeros(Tf, size(x))
    for i in 1:M.eigmult.r
        res .+= ∇ϕᵢⱼ(M.eigmult, M.pb.A, x, i, i)
    end
    res ./= M.eigmult.r
    return res
end

function ∇²Lagrangian!(res, pb::Eigmax{Tf}, M::EigmaxManifold{Tf}, x::Vector{Tf}, λ::Vector{Tf}, d::Vector{Tf}) where {Tf}
    res .= ∇²L(M.eigmult, M.pb.A, x, λ, d)
    return
end
