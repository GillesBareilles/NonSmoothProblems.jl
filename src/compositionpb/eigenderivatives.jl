function U(M::EigmaxLinearManifold{Tf}, x::Vector{Tf}) where Tf
    gx = g(M.pb, x)
    E = eigvecs(gx)[:, end-M.r+1:end]
    Ē = eigvecs(g(M.pb, M.xref))[:, end-M.r+1:end]
    res = E * project(Stiefel(M.r, M.r), E' * Ē)
    reverse!(res, dims=2)
    return res
end

function ϕᵢⱼ(M::EigmaxLinearManifold{Tf}, x::Vector{Tf}, i, j) where Tf
    gx = g(M.pb, x)
    Uᵣ = U(M, x)
    return Uᵣ[:, i]' * gx * Uᵣ[:, j]
end

function ∇ϕᵢⱼ(M::EigmaxLinearManifold{Tf}, x::Vector{Tf}, i, j) where Tf
    M.xref .= x
    res = zeros(Tf, size(x))
    Uᵣ = U(M, x)

    for l in axes(res, 1)
        res[l] = Uᵣ[:, i]' * M.pb.As[l+1] * Uᵣ[:, j]
    end
    return res
end

function ∇²ϕᵢⱼ(M::EigmaxLinearManifold{Tf}, x::Vector{Tf}, d::Vector{Tf}, i, j) where Tf
    M.xref .= x
    gx = @timeit_debug "g oracle" g(M.pb, x)
    η = @timeit_debug "Dg oracles" Dg(M.pb, x, d)

    λs, E = @timeit_debug "eigen" eigen(gx)
    τ(i, k) = dot(E[:, k], η * E[:, i])
    ν(i, k, l) = dot(E[:, k], M.pb.As[l+1] * E[:, i])
    ♈(i) = size(gx, 1) - i + 1

    res = zeros(Tf, size(x))
    @timeit_debug "computation" for l in axes(res, 1), k in M.r+1:M.pb.m
        scalar = 0.5 * (inv(λs[♈(i)] - λs[♈(k)]) + inv(λs[♈(j)] - λs[♈(k)]))
        res[l] += scalar * (τ(♈(i), ♈(k)) * ν(♈(j), ♈(k), l) + ν(♈(i), ♈(k), l) * τ(♈(j), ♈(k)))
    end

    return res
end
