
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
f(pb::EigmaxLinear, y) = eigmax(y)

F(pb::EigmaxLinear, x) = f(pb, g(pb, x))

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
    return Symmetric(pb.As[1].data .+ sum(x[i] .* pb.As[i+1].data for i in 1:pb.n))
end

function Dgconj(pb::EigmaxLinear, x, D::AbstractMatrix)
    return [dot(pb.As[i+1], D) for i in 1:pb.n]
end

function Dg(pb::EigmaxLinear, x, η)
    return sum(η[i] * pb.As[i+1] for i in 1:pb.n)
end


#
### Corresponding manifold
#
"""
    EigmaxLinearManifold

A manifold associated with the problem `pb::EigmaxLinear`. Gathers all
points `x` such that the largest eigenvalue of the symmetric real matrix `g(x)` has
multiplicity `r`.
"""
struct EigmaxLinearManifold <: AbstractManifold
    pb::EigmaxLinear
    r::Int64
    xref::Vector{Float64}
end
EigmaxLinearManifold(pb::EigmaxLinear, r::Int64) = EigmaxLinearManifold(pb, r, zeros(Float64, pb.n))
Base.show(io::IO, M::EigmaxLinearManifold) = print(io, "EigmaxLinear(", M.r, ")")


function F̃(pb::EigmaxLinear, M::EigmaxLinearManifold, x)
    return eigmax(g(pb, x))
end

function ∇F̃(pb::EigmaxLinear, M::EigmaxLinearManifold, x)
    return Dgconj(pb, x, ∇λᵢ(g(pb, x), 1))
end

function ∇²F̃(pb::EigmaxLinear, M::EigmaxLinearManifold, x, η)
    gx = g(pb, x)
    H = ∇²λᵢ(gx, 1, Dg(pb, x, η))
    return Dgconj(pb, x, H)
end



function U(M::EigmaxLinearManifold, gx)
		E = eigvecs(gx)[:, end-M.r+1:end]
		Ē = eigvecs(g(M.pb, M.xref))[:, end-M.r+1:end]
		return E * project(Stiefel(M.r, M.r), E' * Ē)
end
function DU(M::EigmaxLinearManifold, gx, δgx)
		res = zeros(size(gx, 1), M.r)
		λs, E = eigen(gx)

		for j in 1:M.r
			  for k in 1:(size(gx, 1)-M.r)
				    res[:, M.r - j + 1] .+= inv(λs[end] - λs[k]) * E[:, k] * dot(E[:, k], δgx*E[:, size(gx, 1) - j + 1])
			  end
		end
    return res
end

"""
    h(M::EigmaxLinearManifold, x)

Mapping that defines `M::EigmaxLinearManifold` as `M = h^{-1}({0})`.
"""
function h(M::EigmaxLinearManifold, x)
		gx = g(M.pb, x)

		Uᵣ = U(M, gx)
		res = Uᵣ' * gx * Uᵣ
    res .-= tr(res) ./ M.r .* Diagonal(ones(M.r))

		return vec(res)
end

struct FiniteDiffMethod end
struct ExplicitFormulaMethod end

METHOD = central_fdm(10, 1)

function Dh(M::EigmaxLinearManifold, x::AbstractVector, η)
    return Dh(M::EigmaxLinearManifold, x::AbstractVector, η, ExplicitFormulaMethod())
end
function Jac_h(M::EigmaxLinearManifold, x)
    return Jac_h(M, x, ExplicitFormulaMethod())
end

#
### FiniteDiff methods
#
function Dh(M::EigmaxLinearManifold, x::AbstractVector, η, ::FiniteDiffMethod)
    return jvp(METHOD, x -> h(M, x), (x, η))
end

function Jac_h(M::EigmaxLinearManifold, x::AbstractVector, ::FiniteDiffMethod)
    return FiniteDifferences.jacobian(METHOD, x->h(M, x), x)[1]
end

#
### Explicit methods
#
function Dh(M::EigmaxLinearManifold, x, η, ::ExplicitFormulaMethod)
		gx = g(M.pb, x)
		Dgx = Dg(M.pb, x, η)

		Uᵣ = U(M, gx)
		U̇ᵣ = DU(M, gx, Dgx)

		res = U̇ᵣ' * gx * Uᵣ + Uᵣ' * gx * U̇ᵣ + Uᵣ' * Dgx * Uᵣ
		res -= tr(res) ./ M.r .* Matrix(1.0I, M.r, M.r)

		return vec(res)
end

function Jac_h(M::EigmaxLinearManifold, x, ::ExplicitFormulaMethod)
    res = zeros(M.r^2, length(x))

    for i in axes(x, 1)
        eᵢ = zeros(size(x))
        eᵢ[i] = 1.0
        res[:, i] = Dh(M, x, eᵢ)
    end
    return res
end



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

