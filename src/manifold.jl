function h end
function Dh end
function point_manifold end


"""
    project_tangent!(res, M, x, d)

Compute the orthogonal projection of vector `d` onto the
tangent space of manifold `M` at point `x`.
"""
function project_tangent!(res, M, x, d)
    QUIET = true
	  model = Model(with_optimizer(Mosek.Optimizer;
                                 QUIET,
		                             MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_INFEAS = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-12,
		                             MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-12))

    Jₕx = Jac_h(M, x)
    if size(Jₕx, 1) == 0
        res .= d
        return nothing
    end

    η = @variable(model, η[1:length(x)])
    @objective(model, Min, dot(η - d, η - d))
    @constraint(model, Jₕx * η .== 0)

    optimize!(model)

    res .= value.(η)

    if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.SLOW_PROGRESS, MOI.LOCALLY_SOLVED])
        @error "Project_tangent: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
        throw(error())
    end
    @debug "project_tangent!" norm(Dh(M, x, res))

    return nothing
end

function project_tangent(M, x, d)
    res = similar(d)
    project_tangent!(res, M, x, d)
    return res
end


"""
    get_normalspace_coordiantes!(λ, Jₕx, d)

Assigns to λ the coefficients of vector `d` in the basis defined by the column
vectors of matrix `Jₕx`.
"""
function get_normalspace_coordinates!(λ, Jₕx, d)
    λ .= pinv(Jₕx)' * d
    return nothing
end

function get_normalspace_coordinates(Jₕx, d)
    λ = zeros(size(Jₕx, 1))
    get_normalspace_coordinates!(λ, Jₕx, d)
    return λ
end

"""
    normalcoords_to_normalrep(Jₕx, λ)

Returns the linear combination of the column vectors of matrix `Jₕx` with coordinates `λ`.
"""
normalcoords_to_normalrep(Jₕx, λ, d) = Jₕx' * λ
normalcoords_to_tangentrep(Jₕx, λ, d) = d .- Jₕx' * λ


"""raw
    grad_Hess(pb::CompositionCompoPb, M::AbstractManifold, x)

Compute the Riemannian gradient and hessian of `F(pb, ⋅)` at point `x` relative
to manifold `M`.
"""
function grad_Hess( M::AbstractManifold, x)
    hx = h(M, x)
    hdim = length(hx)
    ∇F = ∇F̃(M, x)

    if hdim == 0
        return ∇F, (res, η) -> (res.=∇²F̃(M, x, η))
    end

    Jₕx = Jac_h(M, x)
    Jₕxtpinv = pinv(Jₕx')

    λ = Jₕxtpinv * ∇F
    gradFx = ∇F - Jₕx' * λ


    # defining the Hessian operator at x
    function Hess_Fx!(res, ηtangent)
        res .= ∇²F̃(M, x, ηtangent)

        # mancurvatureFD = FiniteDifferences.jvp(central_fdm(10, 1), vec -> Jac_h(M, vec)' * λ, (x, ηtangent))
        for (i, λᵢ) in enumerate(λ)
            res .-= λᵢ .* ∇²hᵢ(M, x, i, ηtangent)
        end

        if sum(isnan.(res)) != 0
            @error "Hess_Fx: Riemannian hessian contains nan."
        end

        # Projection on tangent space
        res .-= Jₕx' * Jₕxtpinv * res
        return nothing
    end

    return gradFx, Hess_Fx!
end


raw"""
    projection_∂ᴹF(pb, M, x, y)

Compute the projection of vector `y` onto the smooth extension of $\partial F$ at point $x$, relative to manifold $M$.
"""
function projection_∂ᴹF(pb, M, x, y) end

"""
	    ∂Fᴹ_minnormelt(pb, M, x)

Computes the projection of 0 on the (smooth extension of the) subdifferential
of `F` at point `x` (relative to manifold `M`).
"""
function ∂Fᴹ_minnormelt(pb, M, x)

    minnormelt = projection_∂ᴹF(pb, M, x, zeros(size(x)))
    gradF, _ = grad_Hess(pb, M, x)

    minnormelt_tangent = similar(gradF)
    project_tangent!(minnormelt_tangent, M, x, minnormelt)

    if norm(minnormelt_tangent - gradF) > 1e-10
        @warn "∂F_minnormelt: mismatch between tangent component of minnormelt and Riemannian gradient." norm(minnormelt_tangent - gradF)
    end

    if norm(minnormelt_tangent - gradF) > 1e-5
        @show eigvals(g(pb, x))
        @show M.r
        @assert false
    end

    minnormelt_normal = minnormelt - minnormelt_tangent

    return minnormelt_tangent, minnormelt_normal
end


"""
    normal_step(M, x)

Perform one Newton-Raphson iteration from point `x` for solving `h(x)=0`.
The problem is written such that the returned direction is orthogonal to
the nullspace of Dh(x) (i.e. is a normal direction).
"""
function normal_step(M::AbstractManifold, x)
    # verbose = false
    # model = Model(with_optimizer(OSQP.Optimizer; polish=true, verbose, max_iter=1e8, time_limit=2, eps_abs=1e-14, eps_rel=1e-14))

    Jₕx = Jac_h(M, x)
    hx = h(M, x)

    # @time begin
    # d = @variable(model, [1:length(x)])
    # @objective(model, Min, dot(d, d)) #dot(hx .+ Jₕx * d, hx .+ Jₕx * d))
    # @constraint(model, hx .+ Jₕx * d .== 0)

    # optimize!(model)

    # res = value.(d)
    #     end

    # if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.SLOW_PROGRESS, MOI.LOCALLY_SOLVED])
    #     @warn "Project_tangent: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    # end
    # projtan = similar(res)
    # project_tangent!(projtan, M, x, res)

    res = similar(x)
    res .= - pinv(Jₕx, rtol = sqrt(eps(real(float(one(eltype(Jₕx))))))) * hx

    if norm(hx + Jₕx * res) > 1e-12
        @warn "normal_step: Linear Newton system solve imprecise" norm(hx + Jₕx * res)
    end

    return res
end
