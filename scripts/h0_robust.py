"""
Robustness of DAIC=-2.9 under re-optimisation at different H0 values.

For each H0 in a grid, runs the full Nelder-Mead over (al, ah, eta)
with that H0 hardwired into the BBN ratio function, then reports DAIC.
This verifies that DAIC=-2.9 is not contingent on H0=68 km/s/Mpc.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import chi2_table as ct
from bbn_hyperconical import compute_abundances
from h0_sensitivity import bbn_rf_h0

# ── Re-optimisation at a given H0 ────────────────────────────────────────────

def chi2_3par_h0(params, H0_kms):
    al, ah, eta = params
    if al < 0.05 or ah <= al + 0.01 or ah > 0.90 or eta < 3e-10 or eta > 15e-10:
        return 1e10
    try:
        zf_, Ef_ = ct.E_hippopede_al(ct.sat(ct.N_SAT_FIX, ah, al), al=al)
        c2bs = ct.chi2_bao(zf_, Ef_) + ct.chi2_sn(zf_, Ef_)
        rf_ = bbn_rf_h0(ct.N_SAT_FIX, ah, al, H0_kms)
        ab_ = compute_abundances(rf_, eta=eta)
        c2_yp = ((ab_['Y_p'] - ct._YP_OBS) / ct._SIG_YP) ** 2
        c2_dh = ((ab_['D_H'] - ct.DH_OBS) / ct.SIG_DH_OBS) ** 2
        return c2bs + c2_yp + c2_dh
    except Exception:
        return 1e10


def daic_at_h0(H0_kms, x0=None):
    """Full 3-par Nelder-Mead at the given H0; returns (DAIC, al, ah, eta, Yp)."""
    if x0 is None:
        x0 = [ct.al_2par, ct.ah_2par, ct._eta_fit]
    res = minimize(lambda p: chi2_3par_h0(p, H0_kms), x0,
                   method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-5, 'maxiter': 8000})
    al, ah, eta = res.x
    zf, Ef = ct.E_hippopede_al(ct.sat(ct.N_SAT_FIX, ah, al), al=al)
    c2bao = ct.chi2_bao(zf, Ef)
    c2sn  = ct.chi2_sn(zf, Ef)
    rf = bbn_rf_h0(ct.N_SAT_FIX, ah, al, H0_kms)
    ab = compute_abundances(rf, eta=eta)
    yp = float(ab['Y_p'])
    dh = float(ab['D_H'])
    c2_yp = ((yp - ct._YP_OBS) / ct._SIG_YP) ** 2
    c2_dh = ((dh - ct.DH_OBS) / ct.SIG_DH_OBS) ** 2
    dc2bao = c2bao - ct.c2b_jnt
    dc2sn  = c2sn  - ct.c2s_jnt
    dc2bbn = c2_yp + c2_dh - ct._c2_yp_lcdm_bbn - ct._c2_dh_lcdm_bbn if hasattr(ct, '_c2_dh_lcdm_bbn') else \
             c2_yp - ct._c2_yp_lcdm_bbn + c2_dh
    daic   = dc2bao + dc2sn + dc2bbn + 2*(3 - 2)
    return daic, al, ah, eta, yp, dh


print("Robustness of DAIC under re-optimisation at each H0")
print("=" * 68)
print(f"{'H0':>5}  {'DAIC':>6}  {'al':>6}  {'ah':>7}  {'eta*1e10':>9}  {'Yp':>7}  {'D/H*1e5':>8}")
print("-" * 68)

H0_grid = [65, 67, 68, 69, 70, 71, 72, 74]
x_prev = None
for H0 in H0_grid:
    daic, al, ah, eta, yp, dh = daic_at_h0(H0, x0=x_prev)
    x_prev = [al, ah, eta]
    marker = " <-- adopted" if H0 == 68 else ""
    print(f"{H0:>5}  {daic:>+6.2f}  {al:>6.4f}  {ah:>7.5f}  {eta*1e10:>9.4f}  {yp:>7.4f}  {dh*1e5:>8.4f}{marker}")

print()
print("NOTE: DAIC at each H0 is the result of a full re-optimisation over")
print("(al, ah, eta), so it is comparable to the adopted fit, not to the")
print("sensitivity analysis in h0_sensitivity.py (which fixed the parameters).")
