# Project Memory — DDEC / Whitney-Forms Digital Twin for Incompressible Linear Elasticity

This is a standing project summary, written into the project folder so it's visible alongside the code (as opposed to Claude's internal session memory). Update it as the project evolves.

## 1. Goal

Build a data-driven, coarse-grained digital twin surrogate on top of a validated high-fidelity mixed FEM solution for linear incompressible elasticity (`locking_demo.ipynb`). The surrogate should reproduce the fine-mesh physics on a much smaller, learned set of "control volumes," generalize across load cases and the Lamé compressibility parameter λ, and preserve conservation/well-posedness structure exactly (not just approximately, as in penalty-based physics-informed ML).

**Standing constraint: `locking_demo.ipynb` is read-only.** All code is delivered as chat snippets for manual copy-paste — never edited directly.

## 2. Notebook state (`locking_demo.ipynb`, validated, cells 0-35)

- Cells 0-18: P1 displacement-only FEM, cantilever BCs (clamped left, free elsewhere), demonstrates volumetric locking as λ→∞.
- Cells 19-31: P2(displacement)-P1(pressure) Taylor-Hood mixed element curing the locking (`assemble_block_system_vectorized` → saddle-point block system), validated against an MMS exact solution, rigid-body-mode checks, and a Schur-complement solver (`factor_mesh_for_schur`/`solve_via_schur`) that factors the elasticity block once per mesh and reuses it across λ sweeps.
- Cells 32-35: Convergence study confirming no locking across λ = 1..10000.
- Cells 36-38 (original): METIS-based DDEC data extraction — now being replaced (see §5).

## 3. Source papers and how they relate

Three Trask-group papers, in `device_physics_papers/` (folder name is misleading — not "papers/"):

1. **DDEC** — Trask, Huang, Hu, *"Enforcing exact physics in scientific machine learning: a data-driven exterior calculus on graphs,"* JCP 2022 (`DDEC.pdf`). You're *given* a graph a priori (e.g. METIS-coarsened mesh). Learns diagonal "Hodge star" metric weights perturbing the graph's combinatorial div/grad/curl, plus an optional NN nonlinear flux correction. Equality-constrained (Lagrange multiplier / KKT) training loop, not penalty-based.
2. **Data-driven Whitney forms** — Actor, Hu, Huang, Roberts, Trask, JCP 2024 (`first_whitney_form.pdf` — note the *other* PDF, `data_drivenwhitney_forms.pdf`, is actually paper 3 despite its filename). Removes the need to hand-specify a graph: parameterizes a continuous, trainable partition of unity (POU) over the domain, builds Whitney forms from it exactly as classical FEEC does, and shows the resulting operators are mathematically equivalent to a DDEC graph + its metric, except the graph (control volumes, connectivity, metric) is now *learned* end-to-end instead of chosen by a clustering algorithm.
3. **Conditional Neural Whitney Forms / digital twins** — Kinch et al., arXiv:2508.06981, 2025 (`data_drivenwhitney_forms.pdf`). Combines papers 1+2 and additionally conditions both the learned POU and the nonlinear flux law on a latent/sensor variable `Z` via a cross-attention transformer, enabling real-time recalibration — this is what makes it a "digital twin." Demonstrated on battery thermal runaway, shock hydrodynamics, electrostatics.

**Decision made for this project: combine papers 1+2, explicitly skip paper 3's transformer/cross-attention machinery.** We still borrow paper 3's *idea* of conditioning on an external variable, but lightly: our conditioning variable is a single scalar (the Lamé parameter λ), so a small MLP suffices — attention's value only shows up for high-dimensional/structured conditioning inputs.

## 4. Notation conventions (deliberately disambiguated — there were originally 3+ unrelated meanings sharing "λ")

- **λ** — the Lamé/compressibility parameter. This is the conditioning input (analogous to paper 3's `Z`), but it's a *known simulation parameter* across the training ensemble, not a literal real-time sensor reading — "conditioning input," not "sensor."
- **λ_dual** — the Lagrange multiplier / adjoint variable from the KKT training loss (kept distinct from Lamé λ; revisit naming in actual code, e.g. `lam` vs `mult`).
- **ω_a(x)** — the *fine-scale* partition of unity (renamed from the papers' own `λ_a` specifically to avoid the collision above) — ordinary fixed P1 hat functions on the fine mesh, not trained.
- **φ_i(x) / ψ_i^0(x)** — the *coarse*, learned 0-forms: `ψ_i^0 = Σ_a W_ia ω_a`.
- **û** — coefficients (not basis functions) — `u(x) = û_i φ_i(x)`.

## 5. Architecture (papers 1+2 combined, scalar conditioning borrowed from paper 3)

Two neural networks total, plus one trainable scalar:

1. **POU-generator**: `W(λ; θ_W) : λ → R^(N_fine × N_coarse)`, small MLP + softmax over the coarse axis (enforces `Σ_i W_ia = 1`). Replaces METIS hard clustering entirely.
2. **Flux-correction network**: `N(û_i, û_j, geometry, λ; θ_N) → interface flux`, weight-shared across every coarse interface pair (paper 3's `PairwiseFluxModel` role). Learns what the deleted, hand-derived `interface_force` computation used to compute analytically.
3. **ε** (not a network — a trainable scalar): amplitude of the linear/diffusive part of the flux. Paper 3 itself drops ε between its eq. (12) and its discretized eqs. — a bug we caught; here it's kept explicit: `F = ε∇u + N[u]`, parameterized as `ε = softplus(raw_ε)` for positivity (required by the well-posedness/Poincaré argument), trained via the same adjoint gradient as everything else: `∂L/∂ε = λ_dual · (δ0ᵀM1(λ)δ0û)`.

**Boundary/interior POU split:** decided 8 interior + 2 boundary (out of `N0=10` total). Only the Dirichlet boundary (clamped left edge) needs dedicated boundary partitions (Kronecker-delta property there, coefficients pinned to the known zero displacement). Neumann boundaries (free edges) don't need this — they enter naturally via a boundary term on the right-hand side. **Scope decision: this digital twin has fixed boundary conditions (geometry/support config baked into the trained `W`'s block structure) — the load function (body force/traction magnitude) is freely changeable at inference, but which edge is Dirichlet vs. Neumann is not, by design choice (user is fine with this).**

**Non-invasive mass-matrix trick** (avoids re-deriving quadrature every training step):
```
M0(λ) = W(λ)ᵀ M_P1 W(λ)
M1(λ) = (W(λ)⊗W(λ))ᵀ M_Ned (W(λ)⊗W(λ))
```
`M_P1`, `M_Ned` are fixed fine-scale matrices, computed once; only the cheap `W`-sandwich is recomputed per λ during training.

**Discretized forward operator** (solved via Newton each training step):
```
F(û; θ, λ) = ε·δ0ᵀM1(λ)δ0û + δ0ᵀM1(λ)N[û; λ, θ_N] = f_θ
```

**Training loop** (KKT/Lagrangian, Algorithm 1 style — not penalty-based): per training sample, Newton-solve the forward problem for `û` → linear adjoint solve for `λ_dual` → gradient step on `θ = {θ_W, θ_N, ε}`. Loss: `L = ½Σ_d||û·ψ⁰(x_d) − u_target(x_d)||² + λ_dualᵀ[F(û;θ,λ) − f_θ]`.

## 6. Two confirmed errors found in paper 3 while reading it closely

- **ε disappears.** Eq. (12) defines `F = −ε∇u + N[u;θ]` with ε explicitly purposeful, but it's absent from every subsequent equation (13b, 14b, 16, 17) — likely a dropped term. Don't repeat this in our own implementation (see §5, item 3).
- **Boundary term pairing/sign.** Eq. (13a) prints `+⟨F_N,∇q⟩`, but deriving it properly from the divergence theorem (and matching paper 2's own analogous identity) gives `−⟨F_N,q⟩` — paired with the test function itself, minus sign, not paired with its gradient with a plus sign. Use the corrected form.

## 7. Implementation status (updated 2026-06-23)

**Done and validated:**
- Removed all METIS-dependent code (cell 36's `pymetis`/`ddec_adjacency`/`ddec_membership`, cell 38's `stress_at`/`outward_normal`/`local_L`/`ddec_dataset`/conservation-check). Kept: fine mesh construction, `ddec_edge_to_tris`, the full load-case data-generation loop (cell 37, unaffected by any of this), and `ddec_areas`/`ddec_grads` (now both captured from `get_batched_grads`).
- `M_P1` (fine P1 mass matrix) — built via a small dedicated `assemble_M_P1` (same local-mass-matrix pattern as the existing `assemble_global_stiffness`). Validated: shape `(16384,16384)`, nnz=113666, symmetric.
- `M_Ned` (fine Nédélec/edge mass matrix) — built from scratch: vectorized global edge indexing via `np.unique`, gradient Gram matrix `G`, the closed-form local edge-mass contraction `∫ψ_pq·ψ_rs = G[q,s]Mloc[p,r] − G[q,r]Mloc[p,s] − G[p,s]Mloc[q,r] + G[p,r]Mloc[q,s]`, sign-corrected sparse scatter assembly. Validated: 48641 edges (matches the exact combinatorial count for this grid: `n(n-1) + n(n-1) + (n-1)²` for horizontal+vertical+diagonal edges), nnz=242189, symmetric.

**Not yet built:**
- The POU-generator network `W(λ;θ_W)`.
- The flux-correction network `N(û_i,û_j,geometry,λ;θ_N)`.
- The KKT/adjoint training loop.
- Likely also: extending cell 37's load-case loop to sweep multiple λ values (currently fixed at `ddec_lam=1000`), since conditioning on λ requires training data that actually varies λ.

## 8. Partition-count guidance (checked against both papers' actual numerical examples)

Both papers use far fewer partitions than one might guess, and explicitly state the design philosophy of using the *smallest* `N0` that works:
- Paper 2 manufactured problem: 4+4 = 8 total POUs.
- Paper 2 five-strip problem: 5+5 = 10 total (matched to the 5 physically distinct strips).
- Paper 2 battery problem: 8+8 = 16 total, reducing 5.89M DOF → 136 at <1% error.
- Paper 3 charge-in-shell example: only 3 learned partitions.

This project settled on **`N0 = 10` (8 interior + 2 boundary)**, in line with the papers' demonstrated range.
