"""SAT-BH model: alpha(z) = alpha_high - (alpha_high - alpha_low)/(1+z)^(0.5 + b/(1+z))
Free parameters: alpha_high (ah), b
Fixed: alpha_low = 0.283 (geometry), exponent structure ensures:
  - z=0:   alpha = alpha_low (always)
  - z->inf: alpha -> alpha_high (radiation limit)
  - effective exponent at z: 0.5 + b/(1+z), i.e. n_eff(z=1) = 0.5 + b/2

Key insight: at BBN (z>>1), exponent -> 0.5 -> same as SAT-2 with n=0.5 and alpha_high.
So Y_p is controlled only by alpha_high (same as 1-param SAT-2).
b provides extra freedom in the BAO range without affecting BBN.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar, brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))  # bbn_hyperconical.py lives at project root
from hyperconical_model import ExtendedProjectedHyperconical
from explore_two_param import (chi2_bao, chi2_sn, c2j_lcdm,
                                DOF_BAO, DOF_SN, Z_MAX,
                                _H0_SI, _H_std, _T0_EV)
from bbn_hyperconical import compute_abundances

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
ALPHA_LOW = 0.283

# ── model functions ───────────────────────────────────────────────────────────
def alpha_bh(z, ah, b):
    exp = 0.5 + b / (1. + np.asarray(z, float))
    return ah - (ah - ALPHA_LOW) / (1. + np.asarray(z, float))**exp

def E_bh(ah, b, n_grid=5000):
    """E(z)/E(0) for SAT-BH model."""
    zf = np.linspace(0., Z_MAX * 1.01, n_grid)
    model = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)
    x  = model.x_from_lz(np.log1p(zf))
    u  = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y  = np.arctan2(x, u)
    az = alpha_bh(zf, ah, b)
    g  = np.maximum(1. - y/model.y0, 1e-12)
    t  = (y/2.) / (g**az)
    rhat = 2.*np.arctan(t)
    dr = np.gradient(rhat, zf, edge_order=2)
    dr = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h  = 1./dr
    return zf, h / h[0]

def bbn_ratio_bh(ah, b, T_mev, z_ref=1e6):
    """H_model/H_std at a single BBN temperature via analytical extrapolation."""
    z_eval = np.unique(np.concatenate([np.linspace(0., 1., 400),
                                        np.geomspace(1., z_ref * 1.02, 5000)]))
    model = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)
    x  = model.x_from_lz(np.log1p(z_eval))
    u  = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y  = np.arctan2(x, u)
    az = alpha_bh(z_eval, ah, b)
    g  = np.maximum(1. - y/model.y0, 1e-12)
    t  = (y/2.) / (g**az)
    rhat = 2.*np.arctan(t)
    dr = np.gradient(rhat, z_eval, edge_order=2)
    dr = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h  = 1./dr; h /= h[0]
    E_ref = float(np.interp(z_ref, z_eval, h))
    # Integrate nu(z) numerically from z_ref to z_T
    z_T   = float(T_mev * 1e6 / _T0_EV - 1.)
    z_int = np.geomspace(z_ref, z_T, 500)
    az_int = alpha_bh(z_int, ah, b)
    nu_int = 1. + 2.*az_int
    lnz   = np.log1p(z_int)
    E_BBN = E_ref * np.exp(float(_trapz(nu_int, lnz)))
    return float(_H0_SI * E_BBN / _H_std(T_mev))

def yp_for_bh(ah, b):
    res = compute_abundances(lambda T: bbn_ratio_bh(ah, b, T))
    return float(res['Y_p']), float(res['D_H'])

# ── Step 1: verify b doesn't affect BBN (same as SAT-2 at high z) ─────────────
print("=== Step 1: BBN insensitivity to b (ratio@0.8 MeV) ===")
print(f"{'ah':>6} {'b':>8}  {'ratio@0.5':>10} {'ratio@0.8':>10} {'ratio@1.0':>10}")
for ah in [0.422, 0.436, 0.440]:
    for b in [-1.0, -0.3, 0.0, 0.3, 1.0]:
        r05 = bbn_ratio_bh(ah, b, 0.5)
        r08 = bbn_ratio_bh(ah, b, 0.8)
        r10 = bbn_ratio_bh(ah, b, 1.0)
        print(f"{ah:>6.3f} {b:>8.2f}  {r05:>10.4f} {r08:>10.4f} {r10:>10.4f}")
    print()

# ── Step 2: find ah* for Y_p=0.2453 (with b=0, same as SAT-2 n=0.5, al=0.283) ─
print("=== Step 2: find ah* for Y_p=0.2453 with b=0 ===")
YP_OBS = 0.2453
print("Computing Y_p at ah=0.435 and ah=0.440...")
yp_lo, _ = yp_for_bh(0.435, 0.)
yp_hi, _ = yp_for_bh(0.440, 0.)
print(f"  ah=0.435: Y_p={yp_lo:.4f},  ah=0.440: Y_p={yp_hi:.4f}")
ah_star = brentq(lambda ah: yp_for_bh(ah, 0.)[0] - YP_OBS, 0.435, 0.440, xtol=1e-4)
yp_s, dh_s = yp_for_bh(ah_star, 0.)
print(f"  ah*={ah_star:.4f}: Y_p={yp_s:.4f} ({(yp_s-YP_OBS)/0.0034:+.2f}σ), D/H={dh_s:.3e}")

# ── Step 3: grid over b (ah=ah_star fixed) ────────────────────────────────────
print(f"\n=== Step 3: Grid over b with ah={ah_star:.4f} (Y_p-optimal) ===")
print(f"{'b':>8}  {'α(z=0.5)':>9} {'α(z=1)':>8} {'α(z=2)':>8} {'χ²ν_BAO':>9} {'χ²ν_SN':>8} {'ΔAIC':>7}")
b_vals = [-2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
best_dj, best_b = 999., 0.
for b in b_vals:
    try:
        zf, Ef = E_bh(ah_star, b)
        c2b = chi2_bao(zf, Ef); c2s = chi2_sn(zf, Ef)
        dj  = c2b + c2s - c2j_lcdm
        az05 = float(alpha_bh(0.5, ah_star, b))
        az10 = float(alpha_bh(1.0, ah_star, b))
        az20 = float(alpha_bh(2.0, ah_star, b))
        print(f"{b:>8.2f}  {az05:>9.4f} {az10:>8.4f} {az20:>8.4f} "
              f"{c2b/DOF_BAO:>9.4f} {c2s/DOF_SN:>8.4f} {dj:>+7.2f}")
        if dj < best_dj:
            best_dj, best_b = dj, b
    except Exception as e:
        print(f"{b:>8.2f}  ERROR: {e}")

print(f"\nBest b={best_b:.2f}, Δχ²_joint={best_dj:+.4f}")

# ── Step 4: optimize b continuously ───────────────────────────────────────────
print(f"\n=== Step 4: Continuous optimisation of b (ah={ah_star:.4f}) ===")
res = minimize_scalar(
    lambda b: chi2_bao(*E_bh(ah_star, b)) + chi2_sn(*E_bh(ah_star, b)),
    bounds=(-3., 3.), method="bounded")
b_opt = res.x
zf, Ef = E_bh(ah_star, b_opt)
c2b = chi2_bao(zf, Ef); c2s = chi2_sn(zf, Ef)
dj  = c2b + c2s - c2j_lcdm
# ΔAIC: k_model = 3 (ah, b, eta_BBN) vs k_ΛCDM = 2 (Ω_m, eta_fixed) → Δk=1
daic = dj + 0. + 2. * 1   # chi2_bbn(Y_p)=0 by construction, Δk=1
print(f"b_opt = {b_opt:.4f}")
print(f"χ²ν_BAO = {c2b/DOF_BAO:.4f}")
print(f"χ²ν_SN  = {c2s/DOF_SN:.4f}")
print(f"Δχ²_joint = {dj:+.4f}")
print(f"ΔAIC (3 vs 2 par, eta free) = {daic:+.4f}")
print(f"α(z=0) = {float(alpha_bh(0., ah_star, b_opt)):.4f}")
print(f"α(z=0.5) = {float(alpha_bh(0.5, ah_star, b_opt)):.4f}")
print(f"α(z=1) = {float(alpha_bh(1., ah_star, b_opt)):.4f}")
print(f"α(z=2) = {float(alpha_bh(2., ah_star, b_opt)):.4f}")
print(f"α(z→∞) ≈ {ah_star:.4f}")

# ── Step 5: also verify Y_p at b_opt (should be same as b=0 for |b|<1) ────────
print(f"\n=== Step 5: Y_p at optimal b={b_opt:.3f} ===")
r05 = bbn_ratio_bh(ah_star, b_opt, 0.5)
r08 = bbn_ratio_bh(ah_star, b_opt, 0.8)
print(f"ratio@0.5 MeV = {r05:.4f}  (b=0 gives {bbn_ratio_bh(ah_star, 0., 0.5):.4f})")
print(f"ratio@0.8 MeV = {r08:.4f}  (b=0 gives {bbn_ratio_bh(ah_star, 0., 0.8):.4f})")
yp_opt, dh_opt = yp_for_bh(ah_star, b_opt)
print(f"Y_p = {yp_opt:.4f} ({(yp_opt-YP_OBS)/0.0034:+.2f}σ)")
print(f"D/H = {dh_opt:.3e}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"""
=== FINAL SUMMARY ===
SAT-BH model: alpha(z) = {ah_star:.4f} - ({ah_star:.4f}-0.283)/(1+z)^(0.5+{b_opt:.3f}/(1+z))
  Y_p   = {yp_opt:.4f}  ({(yp_opt-YP_OBS)/0.0034:+.2f}σ)
  D/H   = {dh_opt:.3e}  (need eta={6.11e-10*(dh_opt/2.527e-5)**(1/1.6):.3e} for compatibility)
  χ²ν_BAO = {c2b/DOF_BAO:.4f}  (ΛCDM: 1.7893)
  χ²ν_SN  = {c2s/DOF_SN:.4f}   (ΛCDM: 0.7860)
  ΔAIC  = {daic:+.4f}  (k=3 vs k_ΛCDM=2; eta free)

Compare with 2-par (al=0.250, ah=0.4364, eta free):
  Y_p=0.2452, chi2nu_BAO=1.5523, ΔAIC=-2.40
""")
