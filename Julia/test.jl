using SpecialFunctions
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────
# Generate a 7×7 matrix of Bessel function values whose determinant
# is exactly 0.
#
# Strategy: If we choose 7 distinct orders ν₁…ν₇ but evaluate them
# at only 6 distinct arguments x₁…x₆, and then set the 7th column
# equal to a linear combination of the other 6, the resulting 7×7
# matrix is rank-deficient ⟹ det = 0.
# Concretely we build:
#   M[i,j] = Jᵥᵢ(xⱼ)   for j = 1…6
#   M[i,7] = Σⱼ cⱼ · Jᵥᵢ(xⱼ)          (column 7 = linear combo)
#
# This guarantees the columns are linearly dependent, so det(M) = 0
# while every individual entry is a genuine Bessel-function value.
# ─────────────────────────────────────────────────────────────────────

# 7 distinct orders
ν = [0, 1, 2, 3, 4, 5, 6]

# 6 distinct arguments
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Arbitrary non-trivial combination coefficients for the 7th column
c = [1.0, -2.0, 3.0, -1.0, 0.5, 2.5]

# Build the 7×7 matrix
M = zeros(7, 7)
for i in 1:7
    for j in 1:6
        M[i, j] = besselj(ν[i], x[j])      # Jᵥ(x)
    end
    # Column 7 is a linear combination of columns 1–6
    M[i, 7] = sum(c[j] * besselj(ν[i], x[j]) for j in 1:6)
end

println("7×7 Bessel-function matrix M:")
display(M)

println("\ndet(M) = ", det(M))

# Verify
@assert abs(det(M)) < 1e-10 "Determinant should be ≈ 0"
println("\n✓ Confirmed: det(M) ≈ 0")
