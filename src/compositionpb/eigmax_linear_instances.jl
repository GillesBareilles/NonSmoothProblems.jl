"""
get_eigmaxlinear_pb(; m)

Return an `EigmaxLinear` problem instance.
"""
function get_eigmaxlinear_pb(; m=15, n=2, seed = 1864)
    Random.seed!(seed)
    As = rand(n+1, m, m)
    As = [As[i, :, :] + As[i, :, :]' for i in 1:n+1]
    As[1] += Diagonal(1:m)
    return EigmaxLinear(m, n, As)
end