
raw"""
    ConvexHull{Tf}

Models the convex hull of vectors gs.
"""
struct ConvexHull{Tf}
    gs::Matrix{Tf}
end


f(ch::ConvexHull, α) = 0.5 * norm(ch.gs * α)^2
∇f!(res, ch::ConvexHull, α) = (res .= ch.gs' * ch.gs * α)
function ∇²f!(res, ch::ConvexHull{T}, v, α, β::T) where T
    if β == zero(T)
        res .= α .* (ch.gs' * ch.gs * v)
    else
        res .= α .* (ch.gs' * ch.gs * v) .+ β .* res
    end
end


g(::ConvexHull, α) = sum(α) == 1 && sum( α .>= 0) == length(α)

"""
    prox_γg!(res, ch, α)

Implement the prox of the indicator of the simplex, which amounts to projecting onto the simplex.

### Reference:
Fast Projection onto the Simplex and the ℓ1 Ball, L. Condat, alg. 1
"""
function prox_γg!(res, ch::ConvexHull{Tf}, α) where Tf
    N = length(α)

    # 1. Sorting
    u = sort(α, rev=true)

    # 2.
    k = 1
    sum_u_1k = u[1]
    while (k < N) && (sum_u_1k + u[k+1] - 1) / (k+1) < u[k+1]
        k += 1
        sum_u_1k += u[k]
    end

    # 3.
    τ = (sum_u_1k - 1) / k

    @. res = max(α - τ, Tf(0))

    return nothing
end

function form_projection(set::ConvexHull, x)
    return set.gs * x
end

function get_activities(::ConvexHull, x)
    return filter(i -> x[i] == 0, 1:length(x))
end


"""
    identification_Newtonaccel!(x, set::ConvexHull)

Identifies the structure (a manifold) of `x` and performs a linesearch along the Riemannian Newton step on that structure.

*Notes:*
- that exact identification is possible only since `x` is the output of an *exact* proximity operator, here the projection on the simplex.
- this function is valid for arbitrary real types, including `BigFloat`.
"""
function identification_Newtonaccel!(x, set::ConvexHull{Tf}, minusgradfx) where Tf
    # Identify active manifold
    k = size(set.gs, 2)
    nnzentries = x .!= zero(Tf)
    nnnzentries = sum(nnzentries)

    function projtan!(res, v, α, β::T) where T
        sumvalsv = sum(v[nnzentries])
        if β != zero(T)
            res[nnzentries] .= α .* (v[nnzentries] .- sumvalsv / nnnzentries) .+ β .* v[nnzentries]
            res[.!(nnzentries)] .= β .* v
        else
            res[nnzentries] .= α .* (v[nnzentries] .- sumvalsv / nnnzentries)
            res[.!(nnzentries)] .= zero(T)
        end
    end

    tangentproj = LinearOperator(Tf, k, k, true, true, projtan!)
    ∇²f = LinearOperator(Tf, k, k, true, true,
                         (res, v, α, β) -> ∇²f!(res, set, v, α, β))

    # Compute Riemannian gradient and hessian of f
    ∇f!(minusgradfx, set, x)
    minusgradfx .*= -1
    projtan!(minusgradfx, minusgradfx, one(Tf), zero(Tf))
    Hessfx = tangentproj * ∇²f * tangentproj

    # Solve Newton's equation
    dᴺ, stats = lsmr(Hessfx, minusgradfx)

    # Linesearch step
    ls = BackTracking()
    γ(t) = x .+ t .* dᴺ
    φ(t) = f(set, γ(t)) + g(set, γ(t))

    φ_0 = φ(0.0)
    dφ_0 = -dot(minusgradfx, dᴺ)
    α, fx = ls(φ, 1.0, φ_0, dφ_0)

    x .+= α .* dᴺ

    @debug "ConvexHull projection - Newton acceleration step" nnnzentries norm(Hessfx * dᴺ - minusgradfx) norm(dᴺ) dot(dᴺ, minusgradfx) / (norm(dᴺ)*norm(minusgradfx)) α
    return nothing
end



"""
    projection_zero_JuMP(set::ConvexHull{Float64})

Compute the projection of zero on the given convex hull by formulating a linear SOCP
problem in JuMP and solving with Mosek.
"""
function projection_zero_JuMP(set::ConvexHull{Float64})
    n, k = size(set.gs)
    QUIET = true
	  model = Model(with_optimizer(Mosek.Optimizer;
                                 QUIET,
		                             MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_INFEAS = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-12))

    α = @variable(model, α[1:k])
    η = @variable(model, η)
    gconvhull = sum(α[i] .* set.gs[:, i] for i in 1:k)
    @objective(model, Min, η)
	  socpctr = @constraint(model, vcat(η, gconvhull) in SecondOrderCone())

    @constraint(model, sum(α) == 1)
    cstr_pos = @constraint(model, α .>= 0)

    optimize!(model)

    if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
      @debug "projection_∂ᴹF: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    end

    return value.(α)
end
