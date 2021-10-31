"""
    Halfhalf

A nonsmooth problem used in:
- Mifflin, R., & Sagastizabal, C. (2012). A Science Fiction Story in Nonsmooth
  Optimization Originating at IIASA. Documenta Mathematica, (), 10.
"""
struct Halfhalf{Tf} <: NonSmoothPb
    n::Int64
    A::Diagonal{Tf}
    B::Diagonal{Tf}
    Halfhalf(;Tf = Float64) = new{Tf}(
        8,
        Diagonal(Tf[isodd(i) ? 1 : 0 for i in 1:8]),
        Diagonal([Tf(1)/Tf(i)^2 for i in 1:8])
    )
end
projectodd(x) = @view x[1:2:7]

F(pb::Halfhalf, x) = f(pb, g(pb, x)) + dot(x, pb.B*x)
f(::Halfhalf, y) = norm(y)
g(::Halfhalf, x) = projectodd(x)


function ∂F_elt(pb::Halfhalf, x)
    Ax = pb.A * x
    if norm(Ax) == 0
        return 2*pb.B*x
    else
        return Ax ./ norm(Ax) + 2*pb.B*x
    end
end

# ∂F_minnormelt(pb::Halfhalf, x) =
is_differentiable(::Halfhalf, x) = (norm(projectodd(x)) != 0)


################################################################################
# Corresponding manifold
################################################################################
struct HalfhalfManifold{Tf} <: AbstractManifold
    pb::Halfhalf{Tf}
    isnormactive::Bool
end
Base.show(io::IO, M::HalfhalfManifold{Tf}) where {Tf} = print(io, "half(", Int(M.isnormactive), ")")

point_manifold(pb::Halfhalf, x) = HalfhalfManifold(pb, norm(projectodd(x)) == 0)

function h(M::HalfhalfManifold{Tf}, x) where Tf
    return M.isnormactive ? projectodd(x) : Tf[]
end

∇²hᵢ(::HalfhalfManifold{Tf}, x, i, η) where Tf = Tf(0.0)


function Jac_h(M::HalfhalfManifold{Tf}, x) where Tf
    if !M.isnormactive
        return zeros(Tf, 0, 8)
    end

    Jₕx = zeros(Tf, 4, 8)
    for i in 1:4
        Jₕx[i, 2*i-1] = 1.0
    end
    return Jₕx
end

function ∇F̃(pb::Halfhalf{Tf}, M::HalfhalfManifold{Tf}, x) where Tf
    res = similar(x)
    if M.isnormactive
        res .= 0
    else
        res .= pb.A*x / norm(pb.A * x)
    end
    res .+= 2 * pb.B * x

    return res
end


function ∇²F̃(pb::Halfhalf{Tf}, M::HalfhalfManifold{Tf}, x, η) where Tf
    res = similar(η)
    res .= 0

    if !M.isnormactive
        Pres = projectodd(res)
        Px = projectodd(x)
        Pη = projectodd(η)

        Pres .= Pη / norm(Px) - Px/norm(Px) * dot(Px/norm(Px)^2, Pη)
    end
    res .+= 2*pb.B*η
    return res
end
