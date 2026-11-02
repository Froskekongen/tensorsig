# %% [markdown]
# # Mathematical Verification Guide for Log-Signature Implementation
#
# This document outlines the mathematical properties that should hold for each function in the log-signature implementation. Treat it as a checklist for analytical reasoning and targeted tests (e.g., pytest/property checks), not as an LLM prompt.
#
# ## 1. Hall Basis Generation (`src/tensorsig/algebra/hall_projection.py`)
#
# ### Functions to Verify:
# - `hall_basis(width, depth)`: Generates Hall basis elements
# - `logsigdim(width, depth)`: Computes dimension of log-signature
# - `logsigkeys(width, depth)`: Generates string representations
#
# ### Mathematical Properties to Verify:
#
# 1. **Hall Basis Definition**:
#    - Verify that generated elements satisfy Hall ordering conditions:
#      - Letters are ordered 1..width and precede brackets; brackets are ordered recursively by their left, then right, entries.
#      - For Hall elements a and b with total degree at most depth, include [a, b] exactly when a < b and, if b = [c, d], c <= a.
#
# 2. **Dimension Formula**:
#    - The homogeneous degree-d dimension is l_d = (1/d) * Σ_{k divides d} μ(d/k) * width^k, where μ is the Möbius function.
#    - Verify that `logsigdim(width, depth)` equals the cumulative dimension Σ_{d=1}^{depth} l_d.
#    - For width=2: depth 1→2, depth 2→3, depth 3→5, depth 4→8, etc.
#
# 3. **Basis Completeness**:
#    - Verify that Hall basis elements span the free Lie algebra through the retained depth
#    - Check that all elements are linearly independent
#    - Verify uniqueness of each basis element
#
# 4. **Ordering Properties**:
#    - Elements should be ordered by depth (lower depth first)
#    - Within the same depth, use the recursive Hall order above
#    - Verify that ordering matches esig library output
#
# ## 2. Lie Bracket Operations (`src/tensorsig/tensor_ops/products.py`)
#
# ### Functions to Verify:
# - `lie_bracket(x, y)`: Computes [x, y] = x ⊗ y - y ⊗ x
# - `lie_bracket(x, y, shared_dims=1)`: Batched version
#
# ### Mathematical Properties to Verify:
#
# 1. **Anti-commutativity**:
#    - Verify: [a, b] = -[b, a] for all tensors a, b
#    - This is a fundamental property of Lie brackets
#
# 2. **Jacobi Identity**:
#    - Verify: [a, [b, c]] + [b, [c, a]] + [c, [a, b]] = 0
#    - This is the defining property of Lie algebras
#
# 3. **Bilinearity**:
#    - Verify: [αa + βb, c] = α[a, c] + β[b, c] for scalars α, β
#    - Verify: [a, αb + βc] = α[a, b] + β[a, c]
#
# 4. **Distributivity**:
#    - Verify: [a, b + c] = [a, b] + [a, c]
#    - Verify: [a + b, c] = [a, c] + [b, c]
#
# 5. **Zero Properties**:
#    - Verify: [a, 0] = 0 and [0, a] = 0
#
# ## 3. Baker-Campbell-Hausdorff Formula (`src/tensorsig/algebra/hall_bch.py`)
#
# ### Functions to Verify:
# - `HallBCH.bch(x, y)`: Hall-basis BCH with closed forms through depth 4 and an exact universal Dynkin expansion above it
#
# ### Mathematical Properties to Verify:
#
# 1. **BCH Series Expansion**:
#    - Through depth 4, verify `HallBCH.bch` against the compact closed form:
#      - z = a + b + 1/2 [a, b]
#      - + 1/12 [a, [a, b]] + 1/12 [b, [b, a]]
#      - - 1/24 [b, [a, [a, b]]] (depth ≥ 4)
#
# 2. **Associativity Property**:
#    - BCH(BCH(a, b), c) = BCH(a, BCH(b, c)) when every operation uses the same homogeneous depth truncation.
#    - This identity is exact in the truncated algebra; numerical checks require floating-point tolerances.
#
# 3. **Special Cases**:
#    - BCH(a, 0) = a
#    - BCH(0, b) = b
#    - BCH(a, -a) = 0 in exact arithmetic, without a smallness assumption
#
# 4. **Truncation Convention**:
#    - Verify that all homogeneous terms through depth d are retained correctly.
#    - Discard terms above depth d consistently in every intermediate operation.
#
# ## 4. Log-Signature Computation (`src/tensorsig/classical/log_signature.py`)
#
# ### Functions to Verify:
# - `log_signature(path, depth)`: Main API function
# - `_batch_log_signature(path, depth)`: Signature→log pipeline implementation
# - `_signature_to_logsignature_tensor(sig_tensors, width, depth)`: Conversion function
# - `_project_to_hall_basis(log_sig_tensors, width, depth)`: Projection function
#
# ### Mathematical Properties to Verify:
#
# 1. **Exponential Relationship**:
#    - Reconstruct the Lie element from log-signature coordinates before taking its tensor-algebra exponential; through the retained depth, this equals signature(path).
#    - This is the fundamental relationship between signature and log-signature
#    - Check: exp(log_sig) should match signature for simple paths
#
# 2. **Chen's Identity**:
#    - Verify that log-signature satisfies Chen's identity for path concatenation
#    - For paths X and Y: log_sig(X * Y) = BCH(log_sig(X), log_sig(Y))
#
# 3. **Path Increment Property**:
#    - For a single straight increment dx, the degree-one log-signature is dx and every higher retained degree is zero.
#    - This holds for any increment size in exact arithmetic; numerical checks require floating-point tolerances.
#
# 4. **Projection Correctness**:
#    - Verify that projection onto Hall basis extracts correct coefficients
#    - Check that tensor components map correctly to Hall basis elements
#    - Verify that projection preserves the free Lie algebra structure
#
# 5. **Dimension Consistency**:
#    - Verify that output dimension matches logsigdim(width, depth)
#    - For positive width, log-signature and signature dimensions are equal at depth 1; log-signature dimension is smaller at every depth >= 2.
#
# 6. **Special Cases**:
#    - Zero path: log_sig(0) = 0
#    - Straight line: log_sig should match analytical result
#    - Path reversal: relationship between log_sig(X) and log_sig(X_rev)
#
# 7. **Differentiability**:
#    - Verify that log-signature is differentiable with respect to path
#    - Check that gradients are well-behaved (finite, no NaN)
#    - Verify chain rule applies correctly
#
# ## 5. Structured-Pruning Quotients
#
# ### Functions to Verify:
# - `compile_lie_quotient(...)`: exact quotient-native Hall or trace normal form
# - `compile_quotient_signature_plan(...)`: exact weighted-PBW/right-action compiler
# - `quotient_signature_from_increment_windows(...)`: quotient Chen scan
# - `quotient_log_signature_from_increment_windows(...)`: direct BCH, quotient-PBW, and tensor-bridge evaluation
#
# ### Mathematical Properties to Verify:
#
# 1. **Ideal Closure**:
#    - Upper-set rules remain discarded after bracketing with any generator.
#    - Graph nonedges generate commuting relations closed under the Lie bracket.
#
# 2. **Independent Dimension Formulas**:
#    - Multihomogeneous Witt counts match additive-rule conjunctions.
#    - Clique/independence-polynomial Graph--Witt counts match graph quotients.
#    - Exact-span finite differences match graph-and-span quotient dimensions.
#
# 3. **Exact Normal Forms**:
#    - Retained Hall brackets reduce by antisymmetry and Jacobi.
#    - Trace structure constants match independent associative trace-word brackets.
#    - Compilers do not construct ambient Hall bases or use numerical rank tests.
#
# 4. **Homomorphism and Runtime Parity**:
#    - Projection commutes with brackets, BCH, and Chen concatenation.
#    - `method="direct"`, `method="pbw"`, and `method="tensor"` agree through the tested depths.
#    - Full-offset rules are imposed before any projection can alias phenomena.
#
# 5. **Universal Enveloping Algebra**:
#    - Weighted PBW level dimensions match independent Hilbert-series coefficients.
#    - Compiled associative commutators recover the quotient Lie bracket.
#    - The Eulerian readout is primitive and maps group-like quotient signatures to quotient logs.
#    - Collinear segment subdivision leaves the quotient signature unchanged.
#    - Exact one-generator signatures equal the truncated scalar exponential through depth six.
#
# 6. **Degenerate and Differential Cases**:
#    - Zero-dimensional and abelian quotients have the expected output shapes.
#    - All three runtime routes preserve gradients.
#    - Increment-native tensor evaluation preserves small segments beside large ones.
#
# The principal independent checks live in
# `test_lie_quotient_dimensions.py`,
# `test_quotient_enveloping.py`,
# `test_trace_quotient_compiler.py`, and `test_upper_set_normal_form.py`.
#
# ## 6. Numerical Properties
#
# ### Properties to Verify:
#
# 1. **Stability**:
#    - Verify numerical stability for small/large increments
#    - Check behavior with repeated points
#    - Verify no NaN or Inf in outputs
#
# 2. **Precision**:
#    - Verify that exp(log_sig) matches sig within numerical precision
#    - Check error bounds for truncation
#    - Verify consistency between CPU and GPU implementations
#
# 3. **Edge Cases**:
#    - Zero increments
#    - Very small increments
#    - Very large increments
#    - Degenerate paths (collinear points)
#
# ## Verification Methodology
#
# For each function, the verification should:
#
# 1. **Check Mathematical Definitions**: Verify that implementation matches mathematical definition
# 2. **Verify Properties**: Check that all known mathematical properties hold
# 3. **Compare with References**: Compare with esig/signatory libraries where possible
# 4. **Test Special Cases**: Verify behavior on known analytical cases
# 5. **Check Relationships**: Verify relationships with other functions (e.g., exp(log_sig) = sig)
#
# ## Known Limitations / Checks
#
# 1. **BCH Depth Scaling**: `HallBCH.bch` is exact at every positive depth, but the universal schedule above depth 4 grows quickly; ordinary ambient paths should normally retain the faster default signature→log method.
#
# ## References
#
# - Hall basis: "On the bases of free Lie algebras" by M. Hall (1950)
# - BCH formula: Baker-Campbell-Hausdorff formula in Lie theory
# - Log-signatures: "A Primer on the Signature Method in Machine Learning"
# - esig library: https://github.com/datasig-ac-uk/esig
# - signatory library: https://github.com/patrick-kidger/signatory
