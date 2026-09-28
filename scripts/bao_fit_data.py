"""Generate figures/bao_fit_data.png (BBN Hyperconical PLB paper).

3+1 panel layout:
  Top row (3 panels): DM/rs, DH/rs, DV/rs vs DESI DR1 BAO data (all 12 points).
    Running-alpha model at alpha_high=0.422 (BBN-constrained); LCDM at Omega_m=0.339.
  Bottom row (1 panel, full width): Pantheon+ 50-bin SN distance proxy.
    Same model parameters: alpha_high=0.422, Omega_m=0.339.

Running-alpha (sqrt interpolation):
  alpha(z) = alpha_high - (alpha_high - alpha_low) / sqrt(1+z)
  Implemented as: (1+z)^{-n} with n=0.5.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from hyperconical_model import ExtendedProjectedHyperconical  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
ALPHA_LOW  = 0.283
ALPHA_HIGH = 0.4224   # BBN-constrained alpha_high (sqrt interpolation, gm=1.000)
N_SQRT     = 0.5      # fixed exponent: (1+z)^{-0.5} = 1/sqrt(1+z)
OMEGA_M_LCDM_JNT = 0.339   # LCDM joint BAO+SN optimal (used for all panels)

# ── load BAO data ────────────────────────────────────────────────────────────
DATA_BAO = ROOT / "data" / "Ardra" / "desi_dr1_bao_galqso_lya_mean.csv"
with open(DATA_BAO) as f:
    rows = list(csv.DictReader(f))
z_all   = np.array([float(r["z"])     for r in rows])
d_all   = np.array([float(r["value"]) for r in rows])
obs_all = [r["observable"] for r in rows]
lbl_all = [r["label"]      for r in rows]
cov_raw = np.array([[float(x) for x in l.split(",")]
                    for l in open(ROOT / "data" / "Ardra" / "desi_dr1_bao_galqso_lya_cov.csv")
                    if l.strip()])

mask = np.array([lbl != "Lya" for lbl in lbl_all])
idx  = np.where(mask)[0]
z_bao, d_bao, obs_bao = z_all[mask], d_all[mask], [obs_all[i] for i in idx]
C_inv = np.linalg.inv(cov_raw[np.ix_(idx, idx)])

# ── load SN data ─────────────────────────────────────────────────────────────
DATA_SN = ROOT / "data" / "gapp" / "pantheon_plus_binned_50.csv"
raw_sn  = np.genfromtxt(DATA_SN, names=True, dtype=None, encoding="utf-8", delimiter=",")
z_sn    = np.asarray(raw_sn["z"], float)
d_sn    = np.asarray(raw_sn["d_proxy"], float)
s_sn    = np.asarray(raw_sn["sigma_d"], float)

sigma_all = np.sqrt(np.diag(cov_raw))   # 1-sigma from covariance diagonal

Z_MAX = max(z_bao.max(), z_sn.max()) * 1.01

# ── running-alpha E(z) ───────────────────────────────────────────────────────
def E_running(n: float, z_arr: np.ndarray, ah: float = ALPHA_HIGH) -> np.ndarray:
    """E(z) for sqrt running-alpha model: alpha(z)=ah-(ah-al)/sqrt(1+z)."""
    model = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)
    az = ah - (ah - ALPHA_LOW) * (1 + z_arr) ** (-n)
    x  = model.x_from_lz(np.log1p(z_arr))
    u  = np.sqrt(np.maximum(1.0 / model.k - x ** 2, 1e-14))
    y  = np.arctan2(x, u)
    g  = np.maximum(1.0 - y / model.y0, 1e-12)
    t  = (y / 2.0) / (g ** az)
    rhat = 2.0 * np.arctan(t)
    dr = np.gradient(rhat, z_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = 1.0 / dr
    E /= float(np.interp(0.0, z_arr, E))
    return E

def E_lcdm(omega_m: float, z_arr: np.ndarray) -> np.ndarray:
    return np.sqrt(omega_m * (1 + z_arr) ** 3 + (1 - omega_m))

# ── BAO model predictions (analytically marginalized beta) ───────────────────
def bao_model_vec(zf: np.ndarray, Ef: np.ndarray) -> np.ndarray:
    """Unit-normalized model vector (r_s = 1) for all 10 BAO points."""
    def dc(zv):
        m = zf <= zv
        return float(_trapz((1 / Ef)[m], zf[m])) if m.sum() >= 2 else 0.0
    mv = np.zeros(len(z_bao))
    for i, (zi, oi) in enumerate(zip(z_bao, obs_bao)):
        if oi == "DM_over_rs":  mv[i] = dc(zi)
        elif oi == "DH_over_rs": mv[i] = 1.0 / float(np.interp(zi, zf, Ef))
        elif oi == "DV_over_rs": mv[i] = (zi * dc(zi)**2 / float(np.interp(zi, zf, Ef)))**(1/3)
    return mv

def beta_opt(mv: np.ndarray) -> float:
    mCd = mv @ C_inv @ d_bao
    mCm = mv @ C_inv @ mv
    return float(mCd / mCm)

# ── SN model predictions (analytically marginalized amplitude) ───────────────
def sn_model_vec(zf: np.ndarray, Ef: np.ndarray) -> np.ndarray:
    """Comoving distance (r_s=1) at each SN redshift."""
    return np.array([
        float(_trapz((1 / Ef)[zf <= zv], zf[zf <= zv])) if (zf <= zv).sum() >= 2 else 0.0
        for zv in z_sn
    ])

def A_opt(mv: np.ndarray) -> float:
    w = 1.0 / s_sn ** 2
    return float((w * mv * d_sn).sum() / (w * mv * mv).sum())

# ── dense z grids ────────────────────────────────────────────────────────────
zf_bao  = np.linspace(0, Z_MAX, 4000)
Ef_hyp_bao   = E_running(N_SQRT, zf_bao)
Ef_hyp_joint = E_running(N_SQRT, zf_bao)
Ef_lcdm_bao  = E_lcdm(OMEGA_M_LCDM_JNT, zf_bao)
Ef_lcdm_jnt  = E_lcdm(OMEGA_M_LCDM_JNT, zf_bao)

mv_hyp_bao   = bao_model_vec(zf_bao, Ef_hyp_bao)
mv_lcdm_bao  = bao_model_vec(zf_bao, Ef_lcdm_bao)
beta_hyp     = beta_opt(mv_hyp_bao)
beta_lcdm_b  = beta_opt(mv_lcdm_bao)

mv_sn_hyp    = sn_model_vec(zf_bao, Ef_hyp_joint)
mv_sn_lcdm   = sn_model_vec(zf_bao, Ef_lcdm_jnt)
A_hyp        = A_opt(mv_sn_hyp)
A_lcdm       = A_opt(mv_sn_lcdm)

# ── continuous curves for plotting ───────────────────────────────────────────
z_curve = np.linspace(0.05, 2.6, 400)
Ec_hyp  = E_running(N_SQRT, z_curve)
Ec_lcdm = E_lcdm(OMEGA_M_LCDM_JNT, z_curve)

zf4 = np.linspace(0, z_curve.max() * 1.01, 4000)
Ef4_hyp  = E_running(N_SQRT, zf4)
Ef4_lcdm = E_lcdm(OMEGA_M_LCDM_JNT, zf4)

def dc_curve(zv, zf, Ef):
    m = zf <= zv
    return float(_trapz((1 / Ef)[m], zf[m])) if m.sum() >= 2 else 0.0

dm_hyp  = np.array([beta_hyp    * dc_curve(z, zf4, Ef4_hyp)  for z in z_curve])
dh_hyp  = beta_hyp    / Ec_hyp
dv_hyp  = np.array([beta_hyp    * (z * dc_curve(z, zf4, Ef4_hyp)  ** 2 / Ec_hyp[i])  ** (1/3) for i, z in enumerate(z_curve)])

dm_lcdm = np.array([beta_lcdm_b * dc_curve(z, zf4, Ef4_lcdm) for z in z_curve])
dh_lcdm = beta_lcdm_b / Ec_lcdm
dv_lcdm = np.array([beta_lcdm_b * (z * dc_curve(z, zf4, Ef4_lcdm) ** 2 / Ec_lcdm[i]) ** (1/3) for i, z in enumerate(z_curve)])

# SN curves
z_sn_curve = np.linspace(0.01, z_sn.max() * 1.05, 300)
zf_sn_fine = np.linspace(0, z_sn_curve.max() * 1.01, 3000)
Ef_sn_hyp  = E_running(N_SQRT, zf_sn_fine)
Ef_sn_lcdm = E_lcdm(OMEGA_M_LCDM_JNT, zf_sn_fine)
# Normalise by A_hyp: data → d_proxy/A, model curves → D_C (no free amplitude)
dc_sn_hyp  = np.array([dc_curve(z, zf_sn_fine, Ef_sn_hyp)  for z in z_sn_curve])
dc_sn_lcdm = (A_lcdm / A_hyp) * np.array([dc_curve(z, zf_sn_fine, Ef_sn_lcdm) for z in z_sn_curve])

# Lyα points (excluded from fit, shown grayed in figure)
def _lya_pts(obs):
    return [(z_all[i], d_all[i], sigma_all[i])
            for i in range(len(z_all)) if obs_all[i] == obs and lbl_all[i] == "Lya"]
dm_lya = _lya_pts("DM_over_rs")
dh_lya = _lya_pts("DH_over_rs")
dv_lya = []   # no DV Lyα point

# ── residuals ────────────────────────────────────────────────────────────────
sig_bao      = np.sqrt(np.diag(cov_raw[np.ix_(idx, idx)]))
res_bao_hyp  = d_bao - beta_hyp    * mv_hyp_bao
res_bao_lcdm = d_bao - beta_lcdm_b * mv_lcdm_bao
dm_sel = np.array([o == "DM_over_rs" for o in obs_bao])
dh_sel = np.array([o == "DH_over_rs" for o in obs_bao])
dv_sel = np.array([o == "DV_over_rs" for o in obs_bao])

res_sn_hyp  = d_sn / A_hyp - mv_sn_hyp
res_sn_lcdm = d_sn / A_hyp - (A_lcdm / A_hyp) * mv_sn_lcdm
sig_sn      = s_sn / A_hyp

# chi2 with full covariance (for annotations)
chi2_bao_hyp  = float(res_bao_hyp  @ C_inv @ res_bao_hyp)
chi2_bao_lcdm = float(res_bao_lcdm @ C_inv @ res_bao_lcdm)
chi2_sn_hyp   = float(np.sum((res_sn_hyp  / sig_sn) ** 2))
chi2_sn_lcdm  = float(np.sum((res_sn_lcdm / sig_sn) ** 2))
nu_bao, nu_sn = len(d_bao) - 1, len(z_sn) - 1

# ── layout: (3+1) × (main + residual) ────────────────────────────────────────
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(9.5, 6.5))
outer  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.22)
gs_bao = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0],
                                           height_ratios=[3, 1], hspace=0.07, wspace=0.30)
gs_sn  = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1],
                                           height_ratios=[3, 1], hspace=0.07, wspace=0.30)

ax_dm     = fig.add_subplot(gs_bao[0, 0])
ax_dh     = fig.add_subplot(gs_bao[0, 1])
ax_dv     = fig.add_subplot(gs_bao[0, 2])
ax_dm_res = fig.add_subplot(gs_bao[1, 0], sharex=ax_dm)
ax_dh_res = fig.add_subplot(gs_bao[1, 1], sharex=ax_dh)
ax_dv_res = fig.add_subplot(gs_bao[1, 2], sharex=ax_dv)
ax_sn     = fig.add_subplot(gs_sn[0, 0:2])
ax_sn_res = fig.add_subplot(gs_sn[1, 0:2], sharex=ax_sn)
ax_legend = fig.add_subplot(gs_sn[:, 2])

COL_HYP  = "#4477AA"
COL_LCDM = "#EE6677"
COL_BAO  = "#222222"
COL_SN   = "#228833"

def _resid_panel(ax_res, z_pts, res_h, res_l, sig, xlabel):
    for lv, ls in [(0, "-"), (1, "--"), (-1, "--"), (2, ":"), (-2, ":")]:
        ax_res.axhline(lv, color="gray", lw=0.6, ls=ls)
    ax_res.scatter(z_pts, res_h / sig, color=COL_HYP,  s=22, zorder=3)
    ax_res.scatter(z_pts, res_l / sig, color=COL_LCDM, s=22, marker="s", zorder=3)
    ax_res.set_ylabel(r"$\Delta/\sigma$", fontsize=7)
    ax_res.set_ylim(-3.2, 3.2)
    ax_res.set_xlabel(xlabel)
    ax_res.grid(True, alpha=0.20, lw=0.4)
    ax_res.tick_params(axis="y", labelsize=7)

# ── BAO panels ────────────────────────────────────────────────────────────────
COL_LYA = "0.62"   # gray for excluded Lyα points
panel_cfg = [
    (ax_dm, ax_dm_res, dm_hyp, dm_lcdm, dm_sel, dm_lya, r"$D_M/r_s$"),
    (ax_dh, ax_dh_res, dh_hyp, dh_lcdm, dh_sel, dh_lya, r"$D_H/r_s$"),
    (ax_dv, ax_dv_res, dv_hyp, dv_lcdm, dv_sel, dv_lya, r"$D_V/r_s$"),
]
for ax, ax_res, hyp_c, lc_c, sel, lya_pts, ylabel in panel_cfg:
    ax.plot(z_curve, hyp_c,  color=COL_HYP,  lw=1.8)
    ax.plot(z_curve, lc_c,   color=COL_LCDM, lw=1.4, ls="--")
    ax.errorbar(z_bao[sel], d_bao[sel], yerr=sig_bao[sel], fmt="o", color=COL_BAO,
                ms=5, elinewidth=1.2, capsize=3)
    if lya_pts:
        zp, dp, ep = zip(*lya_pts)
        ax.errorbar(zp, dp, yerr=ep, fmt="o", color=COL_LYA, ms=5,
                    elinewidth=1.2, capsize=3, mfc="white", mec=COL_LYA, zorder=2)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, lw=0.5)
    plt.setp(ax.get_xticklabels(), visible=False)
    _resid_panel(ax_res, z_bao[sel],
                 res_bao_hyp[sel], res_bao_lcdm[sel], sig_bao[sel],
                 r"$z_\mathrm{eff}$")


# ── SN panel ──────────────────────────────────────────────────────────────────
ax_sn.plot(z_sn_curve, dc_sn_hyp,  color=COL_HYP,  lw=1.8)
ax_sn.plot(z_sn_curve, dc_sn_lcdm, color=COL_LCDM, lw=1.4, ls="--")
ax_sn.errorbar(z_sn, d_sn / A_hyp, yerr=sig_sn, fmt="o", color=COL_SN,
               ms=3.5, elinewidth=0.8, alpha=0.7)
ax_sn.set_ylabel(r"$d_{\rm proxy}/A$")
ax_sn.grid(True, alpha=0.25, lw=0.5)
plt.setp(ax_sn.get_xticklabels(), visible=False)
_resid_panel(ax_sn_res, z_sn, res_sn_hyp, res_sn_lcdm, sig_sn, r"$z$")
ax_sn_res.set_ylim(-5, 5)

# ── global legend ─────────────────────────────────────────────────────────────
ax_legend.axis("off")
legend_handles = [
    Line2D([0], [0], color=COL_HYP,  lw=1.8,
           label=r"Hyp. $\alpha$-run ($\alpha_{\rm h}=0.422$)"),
    Line2D([0], [0], color=COL_LCDM, lw=1.4, ls="--",
           label=rf"$\Lambda$CDM ($\Omega_m={OMEGA_M_LCDM_JNT}$)"),
    Line2D([0], [0], color=COL_BAO, marker="o", ms=5, lw=1.2,
           label="DESI DR1 (10 pts, fit)"),
    Line2D([0], [0], color=COL_LYA, marker="o", ms=5, lw=1.2,
           mfc="white", mec=COL_LYA, label=r"DESI DR1 Ly$\alpha$ (excl.)"),
    Line2D([0], [0], color=COL_SN,  marker="o", ms=3.5, lw=0.8, alpha=0.7,
           label=r"Pantheon$+$ 50-bin"),
    Line2D([0], [0], color=COL_HYP,  marker="o", ms=5, lw=0,
           label=r"Hyp. $\Delta/\sigma$"),
    Line2D([0], [0], color=COL_LCDM, marker="s", ms=5, lw=0,
           label=r"$\Lambda$CDM $\Delta/\sigma$"),
]
ax_legend.legend(handles=legend_handles, loc="center", fontsize=8,
                 frameon=True, borderpad=1.2, labelspacing=0.9)

OUT = FIGURES / "bao_fit_data.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
