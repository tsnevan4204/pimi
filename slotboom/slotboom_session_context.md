# Slotboom NPN BJT FEM — Session Context / Implementation Log

Companion to `slotboom_math.md` (the math source of truth — strong form, weak form, mixed
formulation, inf-sup proof). This file tracks **implementation status**: what code exists, what's
verified, what's broken, and what's still undecided. Workflow throughout: all code is delivered in
chat as copy-paste snippets; nothing is auto-written into notebooks. The old `slotboom.ipynb` (plain
finite differences, uniform grid) was diagnosed but **not** fixed — this is a from-scratch rebuild as
a proper mixed FEM.

## Open question — not yet answered
**Which notebook file is the implementation going into?** Asked twice, no answer yet: a brand new
notebook (need a filename), or clearing out/rebuilding inside the existing `slotboom.ipynb`. Resolve
this before there's enough code that it matters where it lands.

## Status at a glance
- [x] Math: strong form → weak form → mixed (dual) formulation → inf-sup proof, all in `slotboom_math.md`.
- [x] BC diagram: `slotboom_bc_diagram.png` / `.py`.
- [~] Mesh generation: code written (below), **currently buggy** — see "Known issue" below. Not resolved.
- [x] Basis functions (P0, P1, RT0): written and spot-verified, not yet blocked by the mesh bug since
      they're per-element/mesh-agnostic formulas.
- [ ] Boundary-segment classification (Γ_E/Γ_B/Γ_C/Γ_N) on the generated mesh.
- [ ] Assembly (primal Poisson Newton solve, RT0-P0 mixed continuity solves).
- [ ] Gummel outer loop.
- [ ] Known normalization fixes to carry over from the old notebook (see `slotboom_math.md` §3):
      `gamma_p=gamma_n=1.0` should be γ_n=1, γ_p=μn/μp≈3.111; SRH lifetime normalization should use
      one shared reference diffusivity for both carriers, not each carrier's own.

## Mesh generation — DistMesh-style, graded to the junction contour

Approach: unstructured triangular mesh (Persson & Strang DistMesh algorithm), sized by distance to
the *numerically extracted* doping N=0 contour (not hand-picked junction locations — this is what
catches a lateral junction at the surface near y≈5.2μm, inside the oxide gap between the emitter and
base contacts, that isn't obvious from inspection). Chosen over a graded structured/tensor mesh
(which is what Slotboom's own E-net/J-net actually are) because the junction set isn't purely
x-aligned. Concrete tunable constants currently in use: `h_min=15e-9` (junction resolution, ~2.6x the
local Debye length at N_ref=5e23 doping), `h_max=300e-9` (far-field), `grade_length=0.3e-6`.

```python
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import matplotlib.pyplot as plt

Lx, Ly = 5e-6, 10e-6   # device size (m)

def doping_profile(x, y):
    C_E, sigma_Ex, sigma_Ey, y_E = 5e26, 0.3e-6, 1.0e-6, 1.5e-6
    N_D_emitter = C_E * np.exp(-x**2/(2*sigma_Ex**2)) * np.exp(-(y-y_E)**2/(2*sigma_Ey**2))
    C_B, sigma_Bx = 5e23, 1.0e-6
    N_A_base = C_B * np.exp(-x**2/(2*sigma_Bx**2))
    C_C = 1e21
    N_D_collector = C_C * np.ones_like(x)
    return (N_D_emitter + N_D_collector) - N_A_base

nfine = 600
xg, yg = np.linspace(0, Lx, nfine), np.linspace(0, Ly, nfine)
Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
Ng = doping_profile(Xg, Yg)

fig, ax = plt.subplots()
cs = ax.contour(Xg, Yg, Ng, levels=[0.0])
plt.close(fig)
junction_pts = np.vstack(cs.allsegs[0])
junction_tree = cKDTree(junction_pts)

def domain_sdf(p):
    """Negative inside the rectangle [0,Lx]x[0,Ly], 0 on boundary, positive outside."""
    x, y = p[:, 0], p[:, 1]
    return -np.minimum(np.minimum(x, Lx - x), np.minimum(y, Ly - y))

h_min       = 15e-9
h_max       = 300e-9
grade_length = 0.3e-6

def size_function(p):
    d, _ = junction_tree.query(p)
    h = h_min + (h_max - h_min) * (1.0 - np.exp(-d / grade_length))
    return np.clip(h, h_min, h_max)

fixed_pts = np.array([
    [0.0, 0.0], [0.0, 3e-6], [0.0, 8e-6], [0.0, Ly],
    [Lx, 0.0], [Lx, Ly],
])

def distmesh2d(fd, fh, h0, bbox, pfix, max_iter=600, dptol=1e-3, ttol=0.1, Fscale=1.2, deltat=0.2):
    geps = 1e-3 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0

    x = np.arange(bbox[0][0], bbox[1][0] + h0, h0)
    y = np.arange(bbox[0][1], bbox[1][1] + h0 * np.sqrt(3)/2, h0 * np.sqrt(3)/2)
    X, Y = np.meshgrid(x, y)
    X[1::2] += h0 / 2.0
    p = np.vstack([X.ravel(), Y.ravel()]).T
    p = p[fd(p) < geps]

    r0 = 1.0 / fh(p)**2
    p = p[np.random.rand(len(p)) < r0 / r0.max()]

    if len(pfix):
        p = p[cKDTree(pfix).query(p)[0] >= geps]
        p = np.vstack([pfix, p])
    nfix = len(pfix)

    pold = np.full_like(p, np.inf)
    for it in range(max_iter):
        if np.max(np.linalg.norm(p - pold, axis=1)) > ttol * h0:
            pold = p.copy()
            t = Delaunay(p).simplices
            pmid = p[t].mean(axis=1)
            t = t[fd(pmid) < -geps]
            bars = np.vstack([t[:, [0,1]], t[:, [1,2]], t[:, [2,0]]])
            bars = np.unique(np.sort(bars, axis=1), axis=0)

        barvec = p[bars[:,0]] - p[bars[:,1]]
        L = np.linalg.norm(barvec, axis=1)
        hbars = fh(p[bars].mean(axis=1))
        L0 = hbars * Fscale * np.sqrt((L**2).sum() / (hbars**2).sum())

        F = np.maximum(L0 - L, 0.0)
        Fvec = (F / L)[:, None] * barvec
        Ftot = np.zeros_like(p)
        np.add.at(Ftot, bars[:,0],  Fvec)
        np.add.at(Ftot, bars[:,1], -Fvec)
        Ftot[:nfix] = 0.0

        p = p + deltat * Ftot

        d = fd(p)
        out = d > 0
        if out.any():
            gx = (fd(p[out] + [deps,0]) - d[out]) / deps
            gy = (fd(p[out] + [0,deps]) - d[out]) / deps
            p[out] -= np.vstack([d[out]*gx, d[out]*gy]).T

        interior = fd(p) < -geps
        move = np.linalg.norm((p - pold)[interior], axis=1)
        if move.size and np.max(move) / h0 < dptol:
            break

    t = Delaunay(p).simplices
    pmid = p[t].mean(axis=1)
    t = t[fd(pmid) < -geps]
    return p, t

pts, tris = distmesh2d(domain_sdf, size_function, h_min,
                        bbox=[[0, 0], [Lx, Ly]], pfix=fixed_pts)

x1,y1 = pts[tris[:,0],0], pts[tris[:,0],1]
x2,y2 = pts[tris[:,1],0], pts[tris[:,1],1]
x3,y3 = pts[tris[:,2],0], pts[tris[:,2],1]
signed_area = 0.5*((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
tris[signed_area < 0] = tris[signed_area < 0][:, [0, 2, 1]]
```

### Known issue — NOT YET RESOLVED

User's runs (~3100-3200 points, ~6200-6300 triangles, consistent across runs) show:
- **Minimum angle reported as 0.0°** — true degenerate/sliver triangles present.
- Quality histogram is **spiky** at specific round values (~18-19°, a large spike ~29-30°, ~41-42°,
  and a large spike at 60° — the equilateral angle) rather than the smooth distribution a properly
  relaxed DistMesh output should show.
- Visually, the mesh did not look clearly denser near the red junction contour in the rendered plot.

My own sandboxed repro of the *identical* code (different random seed, not fixed in the snippet above)
gave a healthier-looking but still imperfect result: min angle 7.19°, median 43.4°, "did not converge"
within `max_iter=600` (the `dptol=1e-3` relative-to-`h0=15e-9` convergence threshold is almost
certainly mis-scaled — works out to an absolute movement threshold of ~1.5e-11 m, unrealistically
tight, so the loop likely always runs the full 600 iterations rather than truly converging early).
That alone shouldn't *cause* degeneracy, just means "did not detect convergence" rather than "didn't
relax" — but the discrepancy between my run (no 0° angles) and the user's runs (0° angles, spiky
histogram) suggests either run-to-run randomness is exposing a real fragility (e.g. the boundary
projection step misbehaving near a corner, or the probabilistic initial-density thinning occasionally
leaving two points pathologically close together with relaxation never fully separating them), or
something else not yet isolated.

**Investigation was in progress via a forked agent looking at the user's mesh-quality screenshot when
it was stopped (interrupted by the user) — not concluded.** Next session: resume that diagnosis (likely
candidates to check first: boundary-projection behavior near the four corners and the two
contact-window fixed points; whether `Ftot` forces are actually nonzero/effective after the first few
iterations; whether degenerate triangles cluster near `fixed_pts` specifically).

## Basis functions — P0, P1, RT0 (written, spot-verified, not blocked by the mesh bug)

These are per-element formulas, independent of mesh quality, so they were built and verified
(on a clean toy 2-triangle mesh, not the buggy graded mesh) while the mesh bug was set aside.

**P0** (for $\Phi_p,\Phi_n$): trivial — one constant per triangle, a length-`n_tris` array. No code needed.

**P1** (for $\delta$, the Poisson correction):
```python
def get_batched_grads(pts, tris):
    """grad(lambda_i) per triangle -- constant since lambda_i is linear. Mesh-agnostic; if this
    already exists in locking_demo.ipynb, it's identical -- reuse that one."""
    coords = pts[tris]
    x1, y1 = coords[:,0,0], coords[:,0,1]
    x2, y2 = coords[:,1,0], coords[:,1,1]
    x3, y3 = coords[:,2,0], coords[:,2,1]
    areas = 0.5*((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    b = np.stack([y2-y3, y3-y1, y1-y2], axis=1)
    c = np.stack([x3-x2, x1-x3, x2-x1], axis=1)
    grads = np.stack([b, c], axis=2) / (2*areas)[:,None,None]
    return grads, areas
```
Values $\lambda_i$ at a quadrature point in barycentric coords $(\xi,\eta)$: just $(1-\xi-\eta,\xi,\eta)$
directly — same `L_arr` pattern as locking_demo.ipynb cell 37, no new function needed.

**RT0** (for $J_p,J_n$, the $H(\mathrm{div})$ space) — edge indexing is a direct port of the
`ddec_edge_to_tris`/`edge_gidx` pattern (locking_demo.ipynb cell 40):
```python
local_edge_verts = [(1,2), (2,0), (0,1)]   # edge k opposite local vertex k

def build_edges(tris):
    N = tris.max() + 1
    p = np.stack([tris[:, lp] for lp, lq in local_edge_verts], axis=1)
    q = np.stack([tris[:, lq] for lp, lq in local_edge_verts], axis=1)
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    edge_sign = np.where(p < q, 1.0, -1.0)
    edge_key = lo.astype(np.int64)*N + hi.astype(np.int64)
    _, edge_gidx = np.unique(edge_key, return_inverse=True)
    edge_gidx = edge_gidx.reshape(tris.shape)
    return edge_gidx, edge_sign, edge_gidx.max() + 1

def rt0_eval(x, tri_idx, local_k, pts, tris, areas, edge_sign):
    """w_k(x) = unit flux through its own edge, zero through the other two, single-valued
    (H(div)-conforming) across shared edges. Verified on a toy 2-triangle mesh."""
    Pk = pts[tris[tri_idx, local_k]]
    return edge_sign[tri_idx, local_k] * (x - Pk) / (2*areas[tri_idx])

def rt0_div(areas, edge_sign):
    """Constant per triangle -- this exactness is what gives exact per-triangle current
    conservation when tested against P0 (the whole point of going mixed, §6.6)."""
    return edge_sign / areas[:, None]
```
Connection worth remembering: $RT_0$ here is built from the same vertex-coordinate machinery as the
Nédélec `M_Ned` edges in the elasticity project — $\mathbf w_k \propto (\mathbf x-P_k)$ instead of
$\lambda_j\nabla\lambda_k-\lambda_k\nabla\lambda_j$ — same infrastructure, $H(\mathrm{div})$ instead of
$H(\mathrm{curl})$.

## Next steps (in order)
1. Resolve the mesh bug (degenerate triangles / weak-looking grading) — currently the blocker.
2. Decide/confirm the target notebook file.
3. Classify mesh boundary edges into Γ_E/Γ_B/Γ_C/Γ_N.
4. Assembly: primal Poisson Newton solve (P1), RT0-P0 mixed continuity solves (holes, electrons).
5. Gummel outer loop tying the three solves together.
6. Apply the γ/lifetime normalization fixes noted in `slotboom_math.md` §3.
