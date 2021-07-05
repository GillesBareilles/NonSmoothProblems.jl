
"""
    projection_zero(set, x0)

Compute the projection of zero on the given `set` starting from point `x0` in arbitrary type. Solved using a projected gradient accelerated by Riemannian Newton steps on the identified manifolds.

### Reference:
- Newton acceleration on manifolds identified by proximal-gradient methods
"""
function projection_zero(set, x0)
    x = similar(x0)
    x .= x0
    x_old = similar(x0)
    x_old .= x
    ∇fx = similar(x0)
    u = similar(x0)

    converged = false
    stopped = false
    γ = 1e5
    it = 0
    while !converged && !stopped
        ∇f!(∇fx, set, x)
        ∇f_norm2 = norm(∇fx)^2

        ## Update estimate of Lipschitz constant of ∇f
        it_ls = 0
        fx = f(set, x)
        while it_ls <= 50
            u .= x .- γ .* ∇fx
            f(set, u) ≤ fx - γ / 2 * ∇f_norm2 && break

            γ /= 1.2
            it_ls += 1
        end

        ## Perform forward backward update
        prox_γg!(x, set, u)

        ## Add Newton acceleration step
        identification_Newtonaccel!(x, set, ∇fx)

        stopped = it > 1000
        converged = norm(x-x_old) < 10 * eps(typeof(first(x)))

        x_old .= x
        it += 1
    end
    @debug "Projection_zero" it norm(x-x_old) converged stopped

    return x
end
