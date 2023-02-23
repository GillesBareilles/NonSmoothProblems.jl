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

@doc raw"""
    $TYPEDSIGNATURES

Return `pb`, `xopt`, `Fopt`, and `Mopt`:
- the nonsmooth (convex) function F3d-Uν depending of the parameter `ν` in 0..3,
- its minimizer,
- its optimal value, and
- its minimizer structure manifold.

The functions originate from p. 609 of
> Mifflin, Sagastizábal (2005) A VU-algorithm for Convex Minimization, Mathematical Programming.

*Note*: there are two typos in the paper which are corrected in this implem:
- for $ν = 2$, $\beta_4^{\nu=2}$ is set to $20$, not $10$
- the fourth function is $x_3 - \beta_4$, rather than $x_2 - \beta_4$.
With these fixes, the optimal value and point given in table 1 of the paper are correct.
"""
function F3d_U(ν; Tf = Float64)
    @assert ν in 0:3
    β = Tf[
        0.5 0  -5 -5.5
        -2  10 10 10
        0   0  0  11
        0   0  20 20
    ]

    n = 3
    k = 4

    As = [zeros(Tf, n, n) for i in 1:k]
    bs = [zeros(Tf, n) for i in 1:k]
    cs = zeros(Tf, k)

    #  0.5*(x(1)**2 + x(2)**2 + 0.1*x(3)**2)-x(2)-x(3)
    As[1][1, 1] = 0.5
    As[1][2, 2] = 0.5
    As[1][3, 3] = 0.05
    bs[1][2:3] .= -1

    # x(1)**2 - 3x(1)
    As[2][1, 1] = 1
    bs[2][1] = -3

    # x(2)
    bs[3][2] = 1

    # x(3)
    bs[4][3] = 1

    cs[:] .= -β[:, ν+1]

    xopt = zeros(Tf, 4)
    Fopt = Tf(0)
    pb = MaxQuadPb{Tf}(n, k, As, bs, cs)
    Mopt = MaxQuadManifold(pb, [1])
    if ν == 0
        xopt = Tf[1, 0, 0]
        Mopt = MaxQuadManifold(pb, [1, 2, 3, 4])
    elseif ν == 1
        xopt = Tf[0, 0, 0]
        Mopt = MaxQuadManifold(pb, [1, 3, 4])
    elseif ν == 2
        xopt = Tf[0, 0, 10]
        Mopt = MaxQuadManifold(pb, [1, 3])
    else # ν == 3
        xopt = Tf[0, 1, 10]
    end

    return pb, xopt, Fopt, Mopt
end
