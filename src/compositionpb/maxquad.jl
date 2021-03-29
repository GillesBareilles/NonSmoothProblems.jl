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