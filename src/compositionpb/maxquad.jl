"""
    MaxQuadPb

min_x∈ℝⁿ  max_{i=1,...,k}( ⟨Aᵢx,x⟩ + ⟨bᵢ, x⟩ + cᵢ) = f o g(x),
where:
- g(x) = [⟨Aᵢx,x⟩ + ⟨bᵢ, x⟩ + cᵢ for i=1,...,k]
- f(y) = max_i(yᵢ)
"""
struct MaxQuadPb{Tf} <: CompositionCompoPb
    n::Int64
    k::Int64
    As::Vector{Matrix{Tf}}
    bs::Vector{Vector{Tf}}
    cs::Vector{Tf}
end


#
### Shared methods
#
f(pb::MaxQuadPb, y) = maximum(y)

F(pb::MaxQuadPb, x) = f(pb, g(pb, x))


function ∂F_elt(pb::MaxQuadPb, x)
    gx = g(pb, x)

    active_indices = Set{Int64}()
    for (i, gxᵢ) in enumerate(gx)
        (gxᵢ == maximum(gx)) && push!(active_indices, i)
    end

    subgradient = zeros(size(x))
    for i in active_indices
        subgradient .+= (1/length(active_indices)) .* ∇gᵢ(pb, x, i)
    end

    return subgradient
end

function is_differentiable(pb::MaxQuadPb, x)
    gx = g(pb, x)
    gx_max = maximum(gx)
    return length(filter(gᵢ -> gᵢ == gx_max, gx)) == 1
end

#
### Problem specific methods
#
g(pb::MaxQuadPb, x) = [gᵢ(pb, x, i) for i in 1:pb.k]
gᵢ(pb::MaxQuadPb, x, i) = dot(pb.As[i]*x, x) + dot(pb.bs[i], x) + pb.cs[i]
∇gᵢ(pb::MaxQuadPb, x, i) = 2*pb.As[i]*x + pb.bs[i]
function Dg(pb::MaxQuadPb{Tf}, x, η) where Tf
    res = zeros(Tf, pb.k)
    for i in 1:pb.k
        res[i] = dot(∇gᵢ(pb, x, i), η)
    end
    return res
end
function Dg(pb::MaxQuadPb{Tf}, x) where Tf
    res = zeros(Tf, pb.k, pb.n)
    for i in 1:pb.k
        res[i, :] = ∇gᵢ(pb, x, i)'
    end
    return res
end

∇²gᵢ(pb::MaxQuadPb, x, i, η) = 2*pb.As[i]*η




#
### Corresponding manifold
#
"""
    MaxQuadManifold

A manifold associated with the problem `maxquadpb::MaxQuadPb`. Gathers all
points `x` such that the functions indexed `active_fᵢ_indices` by are all equals.
"""
struct MaxQuadManifold{Tf} <: AbstractManifold
    pb::MaxQuadPb{Tf}
    active_fᵢ_indices::Vector{Int64}
    MaxQuadManifold(pb::MaxQuadPb{Tf}, activeinds::AbstractArray) where Tf = new{Tf}(pb, sort(activeinds))
end
Base.show(io::IO, M::MaxQuadManifold{Tf}) where {Tf} = print(io, "MaxQuad(", M.active_fᵢ_indices, ")")


function select_activestrata(M::MaxQuadManifold, x)
    gx = g(M.pb, x)
    iact = argmax(gx)
    # @debug "F̃: smooth extension" gx iact
    iact ∉ M.active_fᵢ_indices && @warn "F̃: active function not in manifold" iact M gx
    return iact
end


"""
    F̃(M, x)

Computes the value of a smooth extension of `F` on manifold `M` at point `x`.
"""
function F̃(pb::MaxQuadPb, M::MaxQuadManifold, x)
	  return gᵢ(pb, x, select_activestrata(M, x))
end

function ∇F̃(pb::MaxQuadPb, M::MaxQuadManifold, x)
    return ∇gᵢ(pb, x, select_activestrata(M, x))
end

function ∇²F̃(pb::MaxQuadPb, M::MaxQuadManifold, x, η)
	  return ∇²gᵢ(pb, x, select_activestrata(M, x), η)
end


"""
    manifold_codim

Compute the codimension of manifold `M`, that is the dimension of the normal
space of `M`.
"""
manifold_codim(M::MaxQuadManifold) = length(M.active_fᵢ_indices)-1

"""
    h(M::MaxQuadManifold, x)

Mapping that defines `M::MaxQuadManifold` as `M = h^{-1}({0})`.
"""
function h(M::MaxQuadManifold{Tf}, x) where Tf
    manifold_codim(M) == 0 && return Tf[]
    res = [gᵢ(M.pb, x, i) for i in M.active_fᵢ_indices[1:end-1]] .- gᵢ(M.pb, x, last(M.active_fᵢ_indices))
    return res
end

function Dh(M::MaxQuadManifold{Tf}, x, η) where Tf
    res = zeros(Tf, length(M.active_fᵢ_indices)-1)

    dot_glast_η = dot(∇gᵢ(M.pb, x, last(M.active_fᵢ_indices)), η)
    for i in 1:length(res)
        res[i] = dot(∇gᵢ(M.pb, x, M.active_fᵢ_indices[i]), η) - dot_glast_η
    end
    return res
end

function Jac_h(M::MaxQuadManifold{Tf}, x) where Tf
    hx = h(M, x)
    hdim = length(hx)

    Jₕx = zeros(Tf, hdim, M.pb.n)
    for i in 1:hdim
        Jₕx[i, :] .= ∇hᵢ(M, x, i)
    end
    return Jₕx
end

hᵢ(M::MaxQuadManifold, x, i) = gᵢ(M.pb, x, M.active_fᵢ_indices[i]) - gᵢ(M.pb, x, last(M.active_fᵢ_indices))
∇hᵢ(M::MaxQuadManifold, x, i) = ∇gᵢ(M.pb, x, M.active_fᵢ_indices[i]) - ∇gᵢ(M.pb, x, last(M.active_fᵢ_indices))
function ∇²hᵢ(M::MaxQuadManifold, x, i, η)
    return ∇²gᵢ(M.pb, x, M.active_fᵢ_indices[i], η) - ∇²gᵢ(M.pb, x, last(M.active_fᵢ_indices), η)
end


"""
    point_manifold(pb::MaxQuadPb, x)

Find a manifold of type `MaxQuadManifold` near `x`.

Implementing the heuristic mentioned in
Noll & Apkarian, 2005, second order methods, eq. (3).
"""
function point_manifold(pb::MaxQuadPb, x)
    gx = g(pb, x)
    gx_max = maximum(gx)

    active_indices = Int64[]
    τ = 1e-6
    for (i, gxᵢ) in enumerate(gx)
        if (gx_max - gxᵢ) / max(1, abs(gx_max)) <= τ
            push!(active_indices, i)
        end
    end
    return MaxQuadManifold(pb, active_indices)
end

