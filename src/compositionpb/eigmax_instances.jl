"""
    $(TYPEDSIGNATURES)

Return an `Eigmax` problem instance with affine mapping.
"""
function get_eigmax_linear_pb(; m=15, n=2, seed = 1864, Tf=Float64)
    Random.seed!(seed)
    As = rand(Tf, n+1, m, m)
    A₀ = Symmetric(As[n+1, :, :] + As[n+1, :, :]')
    A₀ += Diagonal(1:m)
    As = [Symmetric(As[i, :, :] + As[i, :, :]') for i in 1:n]
    A = AffineMap{Tf}(n, m, A₀, As)
    return Eigmax{Tf, AffineMap{Tf}}(n, A)
end
