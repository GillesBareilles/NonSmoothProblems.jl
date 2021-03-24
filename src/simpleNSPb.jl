
"""
SimpleQuad

A simple quadratic function centered at [2, 2, ..., 2]: `f(x) = 0.5*||x-[2, ..., 2]ᵀ||²`
"""
struct SimpleQuad <: NonSmoothPb end

F(::SimpleQuad, x) = 0.5*norm(x - 2*ones(size(x)))^2
∂F_elt(::SimpleQuad, x) = x - 2*ones(size(x))
∂F_minnormelt(::SimpleQuad, x) = x - 2*ones(size(x))



"""
SmoothQuad

A quadratic function defined as `f(x) = ⟨Ax, x⟩ + ⟨b, x⟩ + c`.
"""
struct SmoothQuad <: NonSmoothPb
A::Matrix{Float64}
b::Vector{Float64}
c::Float64
end

F(pb::SmoothQuad, x) = dot(pb.A*x, x) + dot(pb.b, x) + pb.c
∂F_elt(pb::SmoothQuad, x) = 2*pb.A*x + pb.b
∂F_minnormelt(pb::SmoothQuad, x) = 2*pb.A*x + pb.b


SmoothQuad1d() = SmoothQuad(
Diagonal([2]),
Vector{Float64}([1]),
0.0
)

SmoothQuad2d_1(; ε=0.0) = SmoothQuad(
Diagonal([1, 0]),
Vector{Float64}([1-ε, 0]),
0.0
)
SmoothQuad2d_2(; ε=0.0) = SmoothQuad(
Diagonal([2, 1]),
Vector{Float64}([-ε, 0]),
0.0
)


"""
Simplel1

l1 norm function.
"""
struct Simplel1 <: NonSmoothPb end

F(::Simplel1, x) = norm(x, 1)
∂F_elt(::Simplel1, x) = sign.(x)
∂F_minnormelt(::Simplel1, x) = sign.(x)
