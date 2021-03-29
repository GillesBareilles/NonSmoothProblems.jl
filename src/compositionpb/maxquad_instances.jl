function MaxQuad2d(;ε=0.0)
    return MaxQuadPb(2, 2,
    Vector{Matrix{Float64}}([Diagonal([1, 0]), Diagonal([2, 1])]),
    Vector{Vector{Float64}}([[1-ε, 0], [-ε, 0]]),
    Vector{Float64}([0, 0])
    )
end


"""
    MaxQuadBGLS()

Maximum quadratic function studied by Sagastiz{\'a}bal & Mifflin in their VU superlinear algorithm, and
referenced in *Numerical Optimisation*, BGLS, p. 153 (2nd edition).
Minimizer is approx x̄ = (−.1263, −.0344, −.0069, .0264, .0673, −.2784, .0742, .1385, .0840, .0386), first
four quadratic are active there, f̄ = −0.8414080.
"""
function MaxQuadBGLS()
    n = 10
    k = 5
    As = [zeros(Float64, n, n) for i in 1:k]
    bs = [zeros(Float64, n) for i in 1:k]

    for j=1:k
        for i=1:n
            bs[j][i] = exp(i/j)*sin(i*j)
        end

        for i in 1:n, k in i+1:n
            As[j][i, k] = exp(i/k)*cos(i*k)*sin(j)
            As[j][k, i] = As[j][i, k]
        end
        for i in 1:n
            As[j][i, i] = abs(sin(j))*i/n + sum(abs(As[j][i, l]) for l=1:n) - abs(As[j][i, i])
            #last term dispensable since As are intialized as null
        end
    end

    return MaxQuadPb(n, k, As, bs, zeros(k))
end
