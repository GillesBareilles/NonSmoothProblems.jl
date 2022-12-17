function get_2dMCP()
    A = Matrix{Float64}([1. 1.; 0. 1.])
    b = Vector([1., 2.])
    return MCPLeastSquares(A, b, 1.0, 1.0, 2)
end

function get_MCP_instance(;n=20, m=15, sparsity=0.5, δ=0.1, seed=1234)
    Random.seed!(seed)
    A = rand(Normal(), m, n)

    x0 = rand(Normal(), n)
    x0 .*= rand(Bernoulli(1-sparsity), n)
    b = A*x0 + rand(Normal(0, δ^2), m)

    return MCPLeastSquares(A, b, 1.0, 1.0, n)
end

# function get_skglminstance()
#     Tf = Float64

#     skglm = pyimport("skglm")

#     n_features = 1000
#     n_samples = 500
#     density = 0.1

#     nnz = Int64(n_features * density)
#     Random.seed!(1643)
#     supp = randperm(n_features)[1:nnz]
#     w_true = zeros(Tf, n_features)
#     w_true[supp] .= 1

#     X, y, w_true = skglm.utils.make_correlated_data(n_samples=n_samples, n_features=n_features, snr=5, random_state=2, rho=0.5, w_true=w_true)

#     # standardize for MCP
#     for i in axes(X, 1)
#         X[i, :] ./= norm(X[i, :]) / sqrt(n_features)
#     end

#     pb = MCPLeastSquares(X, y, Tf(1), Tf(3), n_features)

#     return pb
# end
