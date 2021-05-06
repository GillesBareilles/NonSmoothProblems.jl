function h end
function Dh end
function point_manifold end


"""raw
    grad_Hess(pb::CompositionCompoPb, M::AbstractManifold, x)

Compute the Riemannian gradient and hessian of `F(pb, ⋅)` at point `x` relative
to manifold `M`.

# TODO:
- make this implement efficient, one way might be
JₕtJₕ = Symmetric(Jₕx * Jₕx')
λ = JₕtJₕ / (Jₕx * ∇F)
grad_Fx = ∇F - Jₕx' * λ
- abstract away tangent projection with struct and dispatch, so that user can plug efficient projection if available.
"""
function grad_Hess(pb::CompositionCompoPb, M::AbstractManifold, x)
    hx = h(M, x)
    hdim = length(hx)
    ∇F = ∇gᵢ(pb, x, first(M.active_fᵢ_indices))

    Jₕx = Jac_h(pb, M, x)


    ## using pinv method
    Jₕpinv = pinv(Jₕx)

    project_tangent!(η) = (η .-= Jₕx' * Jₕpinv' * η)
    function project_tangent(η)
        return η - Jₕx' * Jₕpinv' * η
    end

    λ = Jₕpinv' * ∇F
    grad_Fx = ∇F - Jₕx' * λ
    # defining the Hessian operator at x
    function Hess_Fx(η)
        res = ∇²gᵢ(pb, x, first(M.active_fᵢ_indices), η)
        for (i, λᵢ) in enumerate(λ)
            res .-= λᵢ .* ∇²hᵢ(M, x, i, η)
        end

        # Projection on tangent space
        project_tangent!(res)
        return res
    end

    # tangentbasis = nullspace(Jₕx)
    # proj∇F = sum( tangentbasis[:, i] * dot(tangentbasis[:, i], ∇F) for i in 1:size(tangentbasis, 2))

    if norm(Dh(M, x, grad_Fx)) > 1e-10
        @warn "projection on tangent space inaccurate" norm(Dh(M, x, grad_Fx))
    end
    return grad_Fx, Hess_Fx, project_tangent!
end


"""
    normal_step(M, x)

Perform one Newton-Raphson iteration from point `x` for solving `h(x)=0`.
"""
function normal_step(M::AbstractManifold, pb, x)
    hx = h(M, x)

    Jₕx = Jac_h(pb, M, x)

    ## using pinv method
    Jₕpinv = pinv(Jₕx)

    res = Jₕpinv * (-hx)

    if norm(hx + Jₕx * res) > 1e-12
        @warn "normal_step: Linear Newton system solve imprecise" norm(hx + Jₕx * res)
    end
    if norm(res - Jₕx' * Jₕpinv' * res) > 1e-12
        @warn "normal_step: direction has a significative normal part." norm(res - Jₕx' * Jₕpinv' * res)
    end

    return res
end
