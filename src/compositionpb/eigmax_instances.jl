function Eigmax(A::AffineMap{Tf}) where Tf
    return Eigmax{Tf, AffineMap{Tf}}(A.n, A)
end
function Eigmax(A::PowerCoordMap{Tf}) where {Tf}
    return Eigmax{Tf, PowerCoordMap{Tf}}(A.n, A)
end
function Eigmax(A::NonLinearMap{Tf}) where {Tf}
    return Eigmax{Tf, NonLinearMap{Tf}}(A.n, A)
end

"""
    $(TYPEDSIGNATURES)

Return an `Eigmax` problem instance with affine mapping.
"""
function get_eigmax_affine(; m=15, n=2, seed = 1864, Tf=Float64)
    Random.seed!(seed)
    As = rand(Tf, n+1, m, m)
    A₀ = Symmetric(As[n+1, :, :] + As[n+1, :, :]')
    A₀ += Diagonal(1:m)
    As = [Symmetric(As[i, :, :] + As[i, :, :]') for i in 1:n]
    A = AffineMap{Tf}(n, m, A₀, As)
    return Eigmax{Tf, AffineMap{Tf}}(n, A)
end

function get_eigmax_AL33(; Tf=Float64)
    return Eigmax(get_AL33_affinemap(;Tf))
end

function get_eigmax_powercoord(; m=5, n=5, k=2, Tf=Float64)
    A = get_powercoordmap(;n, m, k, Tf)
    return Eigmax(A)
end

function get_eigmax_nlmap(; m=5, n=5, Tf=Float64)
    return Eigmax(get_nlmap(n, m; Tf))
end
