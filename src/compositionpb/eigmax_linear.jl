raw"""
EigmaxLinear

Data structure for problem
```math
\min λ₁(A₀ + \Sigma_1^n x[i] * A_i) for x ∈ ℝⁿ
```
"""
struct EigmaxLinear{Tf, Tm} <: CompositionCompoPb
    m::Int64
    n::Int64
    As::Vector{Symmetric{Tf, Tm}}
end

#
### Shared methods
#
f(::EigmaxLinear{BigFloat}, y) = maximum(eigvals(y))
f(::EigmaxLinear, y) = eigmax(y)

F(pb::EigmaxLinear, x) = f(pb, g(pb, x))

function ∂F_elt(pb::EigmaxLinear{Tf}, x) where Tf
    decomp = eigen(g(pb, x))

    # Collect indices of eigvals close to max eigval
    active_inds = filter(i->maximum(decomp.values) - decomp.values[i] < 1e-13, 1:pb.m)
    length(active_inds) > 1 && @show active_inds

    subgradient_max = zeros(Tf, size(decomp.values))
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



################################################################################
# Corresponding manifold
################################################################################
"""
    EigmaxLinearManifold

A manifold associated with the problem `pb::EigmaxLinear`. Gathers all
points `x` such that the largest eigenvalue of the symmetric real matrix `g(x)` has
multiplicity `r`.
"""
struct EigmaxLinearManifold{Tf} <: AbstractManifold
    pb::EigmaxLinear{Tf}
    r::Int64
    xref::Vector{Tf}
end
EigmaxLinearManifold(pb::EigmaxLinear{Tf}, r::Int64) where Tf = EigmaxLinearManifold(pb, r, zeros(Tf, pb.n))
Base.show(io::IO, M::EigmaxLinearManifold) = print(io, "EigmaxLinear(", M.r, ")")

"""
    h(M::EigmaxLinearManifold, x)

Mapping that defines `M::EigmaxLinearManifold` as `M = h^{-1}({0})`.
"""
function h(M::EigmaxLinearManifold{Tf}, x::Vector{Tf}) where Tf
    gx = g(M.pb, x)
    Uᵣ = U(M, x)

    res = Uᵣ' * gx * Uᵣ
    res .-= tr(res) ./ M.r .* Diagonal(ones(M.r))
    return vec(res)
end

function Dh(M::EigmaxLinearManifold{Tf}, x::Vector{Tf}, d::Vector{Tf}) where Tf
    Dgx = Dg(M.pb, x, d)
    Uᵣ = U(M, x)

    res = Uᵣ' * Dgx * Uᵣ
    res .-= tr(res) ./ M.r .* Diagonal(ones(M.r))
    return vec(res)
end

function Jac_h(M::EigmaxLinearManifold{Tf}, x::Vector{Tf}) where Tf
    res = zeros(Tf, M.r^2, length(x))

    for i in axes(x, 1)
        eᵢ = zeros(Tf, size(x))
        eᵢ[i] = 1.0
        res[:, i] = Dh(M, x, eᵢ)
    end
    return res
end

# function ∇²hᵢ(M::EigmaxLinearManifold, x, i, j, d)
#     res = zeros(similar(x))
#     gx = g(M.pb, x)
#     η = Dg(M.pb, x, d)

#     λs, E = eigen(gx)
#     τ(i, k) = dot(E[:, k], η * E[:, i])
#     ν(i, k, l) = dot(E[:, k], M.pb.As[l+1] * E[:, i])
#     ♈(i) = size(gx, 1) - i + 1

#     for l in axes(res, 1), k in M.r+1:M.pb.m
#         scalar = 0.5 * (inv(λs[♈(i)] - λs[♈(k)]) + inv(λs[♈(j)] - λs[♈(k)]))
#         res[l] += scalar * (τ(♈(i), ♈(k)) * ν(♈(j), ♈(k), l) + ν(♈(i), ♈(k), l) * τ(♈(j), ♈(k)))
#     end

#     res .-= tr(res) ./ M.r .* Diagonal(ones(M.r))

#     return res
# end

"""
    point_manifold(pb::EigmaxLinear, x)

Find a manifold of type `EigmaxLinearManifold` near `x`.

Implementing the heuristic mentioned in
Noll & Apkarian, 2005, second order methods, eq. (3).
"""
function point_manifold(pb::EigmaxLinear{Tf}, x::Vector{Tf}) where Tf
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



################################################################################
# Smooth extension on manifold
################################################################################

function ∇F̃(::EigmaxLinear{Tf}, M::EigmaxLinearManifold{Tf}, x::Vector{Tf}) where Tf
    return @timeit_debug "∇ϕᵢⱼ" ∇ϕᵢⱼ(M, x, 1, 1)
end

# function ∇²Lagrangian!(res, pb, M::EigmaxLinearManifold, x, λ, η)
#     @timeit_debug "∇²ϕᵢⱼ" res .= ∇²ϕᵢⱼ(M, x, η, 1, 1)

#     λmat = reshape(λ, (M.r, M.r))
#     for i in 1:M.r, j in 1:M.r
#         @timeit_debug "∇²ϕᵢⱼ" res .-= λmat[i, j] .* ∇²ϕᵢⱼ(M, x, η, i, j)
#     end
#     return res
# end

function build_σζ(M::EigmaxLinearManifold{Tf}, E, m, n, r, η) where Tf
    σ = zeros(Tf, n, r, m)
    for l in 1:n, i in 1:r, k in 1:m
        σ[l, i, k] = dot(E[:, m-k+1], M.pb.As[l+1] * E[:, m-i+1])
    end
    ζ = zeros(Tf, r, m)
    for i in 1:r, k in 1:m
        ζ[i, k] = dot(E[:, m-k+1], η* E[:, m-i+1])
    end
    return σ, ζ
end

function fill_res!(res, M, λs, λmat, σ, ζ)
    i = j = 1
    for l in axes(res, 1), k in M.r+1:M.pb.m
        scalar = 0.5 * (inv(λs[i] - λs[k]) + inv(λs[j] - λs[k]))
        res[l] += scalar * (ζ[i, k] * σ[l, j, k] + σ[l, i, k] * ζ[j, k])
    end

    for i in 1:M.r, j in 1:M.r
        for l in axes(res, 1), k in M.r+1:M.pb.m
            scalar = λmat[i, j] * 0.5 * (inv(λs[i] - λs[k]) + inv(λs[j] - λs[k]))
            res[l] -= scalar * (ζ[i, k] * σ[l, j, k] + σ[l, i, k] * ζ[j, k])
        end
    end
    return
end

function ∇²Lagrangian!(res::Vector{Tf}, pb::EigmaxLinear{Tf}, M::EigmaxLinearManifold{Tf}, x::Vector{Tf}, λ::Vector{Tf}, d::Vector{Tf}) where {Tf}
    M.xref .= x
    gx = g(M.pb, x)::Symmetric{Tf, Matrix{Tf}}
    η = Dg(M.pb, x, d)::Symmetric{Tf, Matrix{Tf}}
    λs, E = eigen(gx)

    reverse!(λs)

    n, m = pb.n, pb.m
    r = M.r

    σ, ζ = build_σζ(M, E, m, n, r, η)

    λmat = reshape(λ, (M.r, M.r))
    res .= 0

    ## obj hessian
    fill_res!(res, M, λs, λmat, σ, ζ)

    return res
end

################################################################################
# Problem specific methods
################################################################################
function g(pb::EigmaxLinear, x)
    res = copy(pb.As[1].data)
    for i in 1:pb.n
        res .+= x[i] .* pb.As[i+1].data
    end
    return Symmetric(res)
end

function Dgconj(pb::EigmaxLinear, x, D::AbstractMatrix)
    return [dot(pb.As[i+1], D) for i in 1:pb.n]
end

function Dg(pb::EigmaxLinear{Tf}, x, η) where Tf
    res = zeros(Tf, size(first(pb.As)))
    for i in 1:pb.n
        res .+= η[i] .* pb.As[i+1]
    end
    return Symmetric(res)
end

