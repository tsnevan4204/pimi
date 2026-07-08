"""
Boundary-condition diagram for the Slotboom-variable NPN BJT BVP.
Standalone, self-contained: reproduces the device geometry/doping/contact-BC
formulas already validated in slotboom.ipynb (read-only, not modified here)
purely to render an accurate, labeled figure -- does not import the notebook.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# Physical constants / material parameters (SI) -- identical to slotboom.ipynb
# ----------------------------------------------------------------------
q    = 1.602e-19
eps0 = 8.854e-12
k_B  = 1.381e-23
T    = 300.0
eps_r = 11.7
eps   = eps_r * eps0
ni    = 1.5e16          # m^-3

Vt = k_B * T / q        # thermal voltage

# Device geometry (SI -> we plot in micrometers)
Lx = 5e-6                # depth (m)
Ly = 10e-6                # lateral (m)

# Contact windows on the top surface (x=0)
yE_lo, yE_hi = 0.0e-6, 3.0e-6     # emitter window
yB_lo, yB_hi = 8.0e-6, 10.0e-6    # base window

# Applied biases
V_E, V_B, V_C = 0.0, 0.65, 0.5

# ----------------------------------------------------------------------
# Doping profile (reproduced from slotboom.ipynb cell 1, for plotting only)
# ----------------------------------------------------------------------
def doping_profile(x, y):
    C_E, sigma_Ex, sigma_Ey, y_E = 5e26, 0.3e-6, 1.0e-6, 1.5e-6
    N_D_emitter = C_E * np.exp(-x**2/(2*sigma_Ex**2)) * np.exp(-(y-y_E)**2/(2*sigma_Ey**2))
    C_B, sigma_Bx = 5e23, 1.0e-6
    N_A_base = C_B * np.exp(-x**2/(2*sigma_Bx**2))
    C_C = 1e21
    N_D_collector = C_C * np.ones_like(x)
    return (N_D_emitter + N_D_collector) - N_A_base

def contact_bc(N_local, V_applied):
    """psi, phi, n, p at an ohmic contact given local doping and applied bias."""
    phi = V_applied / Vt
    psi = phi + np.arcsinh(N_local / (2*ni))
    n = ni*np.exp(psi-phi)
    p = ni*np.exp(phi-psi)
    return psi, phi, n, p

# Representative contact-center evaluation points
N_E = doping_profile(np.array(0.0), np.array(1.5e-6))
N_B = doping_profile(np.array(0.0), np.array(9.0e-6))
N_C = doping_profile(np.array(Lx),  np.array(5.0e-6))

psi_E, phi_E, n_E, p_E = contact_bc(N_E, V_E)
psi_B, phi_B, n_B, p_B = contact_bc(N_B, V_B)
psi_C, phi_C, n_C, p_C = contact_bc(N_C, V_C)

Phi_p_E, Phi_n_E = np.exp(phi_E), np.exp(-phi_E)
Phi_p_B, Phi_n_B = np.exp(phi_B), np.exp(-phi_B)
Phi_p_C, Phi_n_C = np.exp(phi_C), np.exp(-phi_C)

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 13))
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.08)
ax = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])
axt.axis("off")

# --- background: doping sign/magnitude (soft shading) + junction contour ---
nx, ny = 400, 400
xg = np.linspace(0, Lx, nx)
yg = np.linspace(0, Ly, ny)
Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
Ng = doping_profile(Xg, Yg)
signed_log = np.sign(Ng) * np.log10(1.0 + np.abs(Ng))

cf = ax.contourf(Yg*1e6, Xg*1e6, signed_log, levels=40, cmap="RdBu", alpha=0.45)
ax.contour(Yg*1e6, Xg*1e6, Ng, levels=[0], colors="k", linewidths=2.0, linestyles="-")
cbar = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.035)
cbar.set_label(r"sign$(N)\cdot\log_{10}(1+|N|)$   (blue = n-type, red = p-type)", fontsize=9)

ax.set_xlim(-2.0, Ly*1e6 + 2.0)
ax.set_ylim(Lx*1e6 + 1.3, -3.7)   # inverted: depth increases downward; extra top margin for callouts
ax.set_xlabel(r"$y$ — lateral position ($\mu$m)", fontsize=11)
ax.set_ylabel(r"$x$ — depth ($\mu$m)", fontsize=11)
ax.set_title("NPN BJT — boundary-condition specification (Slotboom variables $\\psi,\\ \\Phi_n,\\ \\Phi_p$)",
             fontsize=13, pad=12)

# --- domain outline ---
ax.add_patch(Rectangle((0, 0), Ly*1e6, Lx*1e6, fill=False, edgecolor="black", linewidth=1.2))

# --- colored boundary segments ---
LW = 7
def seg(p0, p1, color, ls="-", z=5):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=LW, linestyle=ls,
             solid_capstyle="butt", zorder=z)

col_E, col_B, col_C, col_N = "#1f5fd1", "#d11f1f", "#1fa05a", "#b21fd1"

seg((yE_lo*1e6, 0), (yE_hi*1e6, 0), col_E)                       # Gamma_E
seg((yE_hi*1e6, 0), (yB_lo*1e6, 0), col_N, ls=(0, (4, 3)))        # Gamma_N (gap, top)
seg((yB_lo*1e6, 0), (yB_hi*1e6, 0), col_B)                       # Gamma_B
seg((0, Lx*1e6), (Ly*1e6, Lx*1e6), col_C)                        # Gamma_C (whole back face)
seg((0, 0), (0, Lx*1e6), col_N, ls=(0, (4, 3)))                  # Gamma_N (left edge)
seg((Ly*1e6, 0), (Ly*1e6, Lx*1e6), col_N, ls=(0, (4, 3)))        # Gamma_N (right edge)

# --- dimension arrows (outside the device, in the margins) ---
def dim_arrow_h(p0, p1, ypos, label):
    ax.annotate("", xy=(p0, ypos), xytext=(p1, ypos),
                arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.0))
    ax.text((p0+p1)/2, ypos, label, ha="center", va="bottom" if ypos < 0 else "top",
             fontsize=8.5, color="0.25")

def dim_arrow_v(p0, p1, xpos, label):
    ax.annotate("", xy=(xpos, p0), xytext=(xpos, p1),
                arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.0))
    ax.text(xpos, (p0+p1)/2, label, ha="right", va="center", fontsize=8.5,
             color="0.25", rotation=90)

dim_arrow_h(0, Ly*1e6, -1.15, r"$L_y = 10\ \mu m$")
dim_arrow_h(yE_lo*1e6, yE_hi*1e6, -0.55, r"$3\ \mu m$")
dim_arrow_h(yE_hi*1e6, yB_lo*1e6, -0.55, r"$5\ \mu m$")
dim_arrow_h(yB_lo*1e6, yB_hi*1e6, -0.55, r"$2\ \mu m$")
dim_arrow_v(0, Lx*1e6, -2.0, r"$L_x = 5\ \mu m$")

# --- callouts with formulas, leader arrows ---
def callout(xy, text, xytext, color):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=8.3, color="black",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.3),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2,
                                 shrinkA=0, shrinkB=4))

callout(((yE_lo+yE_hi)/2*1e6, 0),
        ("$\\Gamma_E$ — Dirichlet\n"
         f"$V_E$={V_E:.2f} V $\\to\\ \\varphi_E$={phi_E:.3f}\n"
         f"$\\psi_E\\approx${float(psi_E):.3f}\n"
         f"$\\Phi_{{p,E}}\\approx${float(Phi_p_E):.2e}, $\\Phi_{{n,E}}\\approx${float(Phi_n_E):.2e}"),
        (1.0, 2.0), col_E)

callout(((yB_lo+yB_hi)/2*1e6, 0),
        ("$\\Gamma_B$ — Dirichlet\n"
         f"$V_B$={V_B:.2f} V $\\to\\ \\varphi_B$={phi_B:.3f}\n"
         f"$\\psi_B\\approx${float(psi_B):.3f}\n"
         f"$\\Phi_{{p,B}}\\approx${float(Phi_p_B):.2e}, $\\Phi_{{n,B}}\\approx${float(Phi_n_B):.2e}"),
        (9.2, 3.55), col_B)

callout((Ly*1e6/2, Lx*1e6),
        ("$\\Gamma_C$ — Dirichlet (entire back face)\n"
         f"$V_C$={V_C:.2f} V $\\to\\ \\varphi_C$={phi_C:.3f}\n"
         f"$\\psi_C\\approx${float(psi_C):.3f}\n"
         f"$\\Phi_{{p,C}}\\approx${float(Phi_p_C):.2e}, $\\Phi_{{n,C}}\\approx${float(Phi_n_C):.2e}"),
        (2.0, 5.95), col_C)

callout((Ly*1e6/2, 0),
        ("$\\Gamma_N$ — Neumann (insulating top surface\n"
         "between contacts, and both side edges)\n"
         r"$\partial\psi/\partial\nu=\partial\Phi_p/\partial\nu=\partial\Phi_n/\partial\nu=0$"),
        (3.3, -3.3), col_N)

ax.text(yE_hi*1e6+0.05, -0.05, "metallurgical\njunction $N=0$", fontsize=7.5,
        color="0.15", ha="left", va="bottom")

legend_handles = [
    Line2D([0], [0], color=col_E, lw=5, label=r"$\Gamma_E$ emitter contact (Dirichlet)"),
    Line2D([0], [0], color=col_B, lw=5, label=r"$\Gamma_B$ base contact (Dirichlet)"),
    Line2D([0], [0], color=col_C, lw=5, label=r"$\Gamma_C$ collector contact (Dirichlet)"),
    Line2D([0], [0], color=col_N, lw=5, ls=(0, (4, 3)), label=r"$\Gamma_N$ insulating/symmetry (Neumann)"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5, frameon=True,
          framealpha=0.9, ncol=1)

# ----------------------------------------------------------------------
# Bottom panel: precise BC table + fixed parameters
# ----------------------------------------------------------------------
table_text = f"""
BOUNDARY CONDITION TABLE  (psi, Phi_n, Phi_p; phi_k := V_k / V_t;  psi_k := phi_k + asinh(N_hat_k/2))
--------------------------------------------------------------------------------------------------------------------
 Segment   Location                          Type        psi                    Phi_p = exp(phi)     Phi_n = exp(-phi)
--------------------------------------------------------------------------------------------------------------------
 Gamma_E   x=0,  y in [0, 3] um  (emitter)    Dirichlet   psi_E  = {float(psi_E):8.4f}     {float(Phi_p_E):10.4e}        {float(Phi_n_E):10.4e}
 Gamma_B   x=0,  y in [8, 10] um (base)       Dirichlet   psi_B  = {float(psi_B):8.4f}     {float(Phi_p_B):10.4e}        {float(Phi_n_B):10.4e}
 Gamma_C   x=Lx, y in [0, 10] um (collector)  Dirichlet   psi_C  = {float(psi_C):8.4f}     {float(Phi_p_C):10.4e}        {float(Phi_n_C):10.4e}
 Gamma_N   rest of boundary d(Omega)          Neumann     d(psi)/d(nu) = d(Phi_p)/d(nu) = d(Phi_n)/d(nu) = 0
           (oxide top y in (3,8) um; side edges y=0 and y=Ly; all x in [0,Lx])
--------------------------------------------------------------------------------------------------------------------
 Newton correction delta (Poisson):  delta = 0 on Gamma_E u Gamma_B u Gamma_C,   d(delta)/d(nu) = 0 on Gamma_N
 Values above are evaluated at each contact's center point; psi/Phi vary pointwise across a window wherever
 the local doping N_hat(x,y) varies (sharpest under the emitter, where sigma_Ey = 1 um vs. 3 um window width).

FIXED PARAMETERS USED THIS PASS  (all changeable later; pinned here for concreteness)
--------------------------------------------------------------------------------------------------------------------
 Geometry        Lx = 5 um (depth),  Ly = 10 um (lateral)
 Emitter window  y in [0, 3] um            Base window     y in [8, 10] um            Gap (oxide) = 5 um
 Bias            V_E = {V_E:.2f} V   V_B = {V_B:.2f} V   V_C = {V_C:.2f} V
 Material (Si)   eps_r = {eps_r},   n_i = {ni:.2e} m^-3,   T = {T:.0f} K,   V_t = k_B T/q = {Vt*1000:.4f} mV
 Doping model    N(x,y) = N_D,emitter(x,y) + N_D,collector  -  N_A,base(x)   (Gaussian bumps, see script)
                 C_E = 5e26 m^-3 (sigma_Ex=0.3um, sigma_Ey=1.0um, y_E=1.5um);  C_B = 5e23 m^-3 (sigma_Bx=1.0um);  C_C = 1e21 m^-3
"""

axt.text(0.0, 1.0, table_text, transform=axt.transAxes, fontsize=9.0,
         family="monospace", va="top", ha="left")

import sys
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "slotboom_bc_diagram.png",
            dpi=180, bbox_inches="tight")
print("done")
