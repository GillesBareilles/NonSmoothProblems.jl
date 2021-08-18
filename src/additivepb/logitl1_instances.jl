function get_logit_MLE(;Tf = Float64, n=20, m=15, sparsity=0.5, seed=1234, λ₁=0.01, λ₂=0.02, diff=0.5)
    @assert 0 ≤ sparsity ≤ 1
    normaldist = Normal()
    Random.seed!(seed)
    A = Tf.(rand(normaldist, m, n))

    Random.seed!(seed+1)
    x0 = Tf.(rand(normaldist, n))
    Random.seed!(seed+2)
    nz_coords = rand(Bernoulli(1-sparsity), n)
    x0 .*= nz_coords

    y = zeros(Tf, m)
    for i in 1:m
        Random.seed!(seed+i)
        if rand(Bernoulli(diff + (1-diff)*σ(dot(A[i, :], x0))))
            y[i] = 1.0
        else
            y[i] = -1.0
        end
    end

    return LogitL1(A, y, Tf(λ₁), Tf(λ₂), n, x0)
end
