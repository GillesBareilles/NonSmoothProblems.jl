function MaxQuad2d(Tf = Float64; ε=0.0)
    return MaxQuadPb{Tf}(2, 2,
    Vector{Matrix{Tf}}([Diagonal([1, 0]), Diagonal([2, 1])]),
    Vector{Vector{Tf}}([[1 - ε, 0], [-ε, 0]]),
    Vector{Tf}([0, 0])
    )
end

function MaxQuadAL(Tf = Float64)
    return MaxQuadPb{Tf}(2, 2,
        Vector{Matrix{Tf}}([Diagonal([3, 1]), Diagonal([1, 1])]),
        Vector{Vector{Tf}}([[0, -1], [1, 0]]),
        Vector{Tf}([0, 0])
    )
end


function MaxQuadMaratos(Tf = Float64; r=1.0)
    return MaxQuadPb{Tf}(2, 2,
        Vector{Matrix{Tf}}([Diagonal([r, r]), Diagonal([-r, -r])]),
        Vector{Vector{Tf}}([[-1, 0], [-1, 0]]),
        Vector{Tf}([-r, r])
    )
end


"""
    MaxQuadBGLS()

Maximum quadratic function studied by Sagastiz{\'a}bal & Mifflin in their VU superlinear algorithm, and
referenced in *Numerical Optimisation*, BGLS, p. 153 (2nd edition).
Minimizer is approx x̄ = (−.1263, −.0344, −.0069, .0264, .0673, −.2784, .0742, .1385, .0840, .0386), first
four quadratic are active there, f̄ = −0.8414080.
"""
function MaxQuadBGLS(Tf = Float64)
    n = 10
    k = 5
    As = [zeros(Tf, n, n) for i in 1:k]
    bs = [zeros(Tf, n) for i in 1:k]

    for j = 1:k
        for i = 1:n
            bs[j][i] = exp(i / j) * sin(i * j)
        end

        for i in 1:n, k in i + 1:n
            As[j][i, k] = exp(i / k) * cos(i * k) * sin(j)
            As[j][k, i] = As[j][i, k]
        end
        for i in 1:n
            As[j][i, i] = abs(sin(j)) * i / n + sum(abs(As[j][i, l]) for l = 1:n) - abs(As[j][i, i])
            # last term dispensable since As are intialized as null
        end
    end

    return MaxQuadPb{Tf}(n, k, As, bs, zeros(k))
end

"""
    $TYPEDSIGNATURES


"""
function F3d_U(ν; Tf = Float64)
    @assert ν in 0:3
    β = Tf[
        0.5 -2 0 0
        0 10 0 0
        -5 10 0 10
        -5.5 10 11 20
    ]

    n = 3
    k = 4

    As = [zeros(Tf, n, n) for i in 1:k]
    bs = [zeros(Tf, n) for i in 1:k]
    cs = zeros(Tf, k)

    As[1][1, 1] = 0.5
    As[1][2, 2] = 0.5
    As[1][3, 3] = 0.05
    bs[1][2:3] .= -1

    As[2][1, 1] = 1
    bs[2][1] = -3

    bs[3][2] = 1
    bs[4][2] = 1

    cs[:] .= -β[ν+1, :]

    return MaxQuadPb{Tf}(n, k, As, bs, cs)
end
