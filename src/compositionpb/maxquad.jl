"""
    MaxQuadPb

min_x∈ℝⁿ  max_{i=1,...,k}( ⟨Aᵢx,x⟩ + ⟨bᵢ, x⟩ + cᵢ) = f o g(x),
where:
- g(x) = [⟨Aᵢx,x⟩ + ⟨bᵢ, x⟩ + cᵢ for i=1,...,k]
- f(y) = max_i(yᵢ)
"""
struct MaxQuadPb <: CompositionCompoPb
    n::Int64
    k::Int64
    As::Vector{Matrix{Float64}}
    bs::Vector{Vector{Float64}}
    cs::Vector{Float64}
end


#
### Shared methods
#
F(pb::MaxQuadPb, x) = maximum(g(pb, x))

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
function Dg(pb::MaxQuadPb, x, η)
    res = zeros(pb.k)
    for i in 1:pb.k
        res[i] = dot(∇gᵢ(pb, x, i), η)
    end
    return res
end
function Dg(pb::MaxQuadPb, x)
    res = zeros(pb.k, pb.n)
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
struct MaxQuadManifold <: AbstractManifold
    pb::MaxQuadPb
    active_fᵢ_indices::Vector{Int64}
    MaxQuadManifold(pb::MaxQuadPb, activeinds::AbstractArray) = new(pb, sort(activeinds))
end

"""
    h(M::MaxQuadManifold, x)

Mapping that defines `M::MaxQuadManifold` as `M = h^{-1}({0})`.
"""
function h(M::MaxQuadManifold, x)
    gᵢx = zeros(length(M.active_fᵢ_indices)-1)
    gx_last = gᵢ(M.pb, x, last(M.active_fᵢ_indices))

    for (i, ind) in enumerate(M.active_fᵢ_indices[1:end-1])
        gᵢx[i] = gᵢ(M.pb, x, ind) - gx_last
    end
    return gᵢx
end

function Dh(M::MaxQuadManifold, x, η)
    res = zeros(length(M.active_fᵢ_indices)-1)

    dot_glast_η = dot(∇gᵢ(M.pb, x, last(M.active_fᵢ_indices)), η)
    for i in 1:length(res)
        res[i] = dot(∇gᵢ(M.pb, x, M.active_fᵢ_indices[i]), η) - dot_glast_η
    end
    return res
end

function Jac_h(pb::MaxQuadPb, M::MaxQuadManifold, x)
    hx = h(M, x)
    hdim = length(hx)
    ∇F = ∇gᵢ(pb, x, first(M.active_fᵢ_indices))

    Jₕx = zeros(hdim, pb.n)
    for i in 1:hdim
        Jₕx[i, :] .= ∇hᵢ(M, x, i)
    end
    return Jₕx
end

hᵢ(M::MaxQuadManifold, x, i) = gᵢ(M.pb, x, M.active_fᵢ_indices[i]) - gᵢ(M.pb, x, last(M.active_fᵢ_indices))
∇hᵢ(M::MaxQuadManifold, x, i) = ∇gᵢ(M.pb, x, M.active_fᵢ_indices[i]) - ∇gᵢ(M.pb, x, last(M.active_fᵢ_indices))
∇²hᵢ(M::MaxQuadManifold, x, i, η) = ∇²gᵢ(M.pb, x, M.active_fᵢ_indices[i], η) - ∇²gᵢ(M.pb, x, last(M.active_fᵢ_indices), η)


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
    # some comment
    return MaxQuadManifold(pb, active_indices)
end
