################################################################################
### Corresponding manifold
################################################################################
"""
    $TYPEDSIGNATURES

A manifold associated with the problem `LogitL1`. All points `x` such that the
coordinates indexed `nz_coords` are non null.
"""
struct FixedSparsityManifold{Tpb} <: AbstractManifold
    pb::Tpb
    nz_coords::BitArray{1}
end
Base.show(io::IO, M::FixedSparsityManifold{Tpb}) where {Tpb} = print(io, "Sparsity(", findall(!, M.nz_coords), ")")

manifold_codim(M::FixedSparsityManifold) = sum(.!M.nz_coords)

manifold_dim(M::FixedSparsityManifold) = M.pb.n - manifold_codim(M)

==(M::FixedSparsityManifold, N::FixedSparsityManifold) = (M.pb == N.pb) && (M.nz_coords == N.nz_coords)

function select_activestrata(::FixedSparsityManifold, x)
    throw(error("Not implemented."))
end

function h(M::FixedSparsityManifold{Tpb}, x::Tv) where {Tpb, Tf, Tv <: AbstractVector{Tf}}
    manifold_codim(M) == 0 && return Tf[]
    return x[.!M.nz_coords]
end

function Jac_h(M::FixedSparsityManifold{Tpb}, x::Tv) where {Tpb, Tf, Tv <: AbstractVector{Tf}}
    m = manifold_codim(M)
    n = M.pb.n
    Is = 1:m
    Js = findall(.!M.nz_coords)
    Vs = ones(Tf, m)
    return sparse(Is, Js, Vs, m, n)
end

function ∇²hᵢ(::FixedSparsityManifold{Tpb}, x::Tv, i, η) where {Tpb, Tf, Tv <: AbstractVector{Tf}}
    return zeros(Tf, size(η))
end
