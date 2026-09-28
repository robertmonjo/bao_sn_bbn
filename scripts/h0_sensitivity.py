"""H0 sensitivity for DAIC — n=3 hyperconical model at the joint-fit optimum."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import chi2_table as ct
from bbn_hyperconical import compute_abundances

al_opt  = ct.al_2par
ah_opt  = ct.ah_2par
eta_opt = ct._eta_fit

_MPC_SI = 3.0856775814913673e19
_T0_EV  = 2.7255 * 8.617333262e-5

def bbn_rf_h0(n, ah, al, H0_kms, z_ref=1e6):
    """BBN ratio function T_mev -> H_model/H_std with explicit H0."""
    H0_SI = H0_kms / _MPC_SI
    z_eval = np.unique(np.concatenate([
        np.linspace(0., 1., 400),
        np.geomspace(1., z_ref * 1.02, 5000),
    ]))
    e_run = ct._e_sat_on_grid(n, z_eval, ah, al)
    E_ref = float(np.interp(z_ref, z_eval, e_run))
    def ratio(T_mev):
        T = float(np.asarray(T_mev))
        z_T = T * 1e6 / _T0_EV - 1.
        ln_r = np.log((1. + z_T) / (1. + z_ref))
        int_al = ah * ln_r + (ah - al) / n * (
            (1. + z_T)**(-n) - (1. + z_ref)**(-n))
        E_T = E_ref * np.exp(ln_r + 2. * int_al)
        return float(H0_SI * E_T / ct._H_std(T))
    return ratio

def gm_h0(n, ah, al, H0_kms, n_T=80, z_ref=1e6):
    """Geometric mean of ratio over T=0.07-0.10 MeV at given H0."""
    rf = bbn_rf_h0(n, ah, al, H0_kms, z_ref)
    T_arr = np.geomspace(0.07, 0.10, n_T)
    ratios = np.array([rf(T) for T in T_arr])
    return float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-30))))), rf

c2bao = ct.chi2_bao(ct.zf_2par, ct.Ef_2par)
c2sn  = ct.chi2_sn(ct.zf_2par, ct.Ef_2par)

# ── H0 estimate from BAO β ────────────────────────────────────────────────────
# beta_opt = c/(H0*r_s) is the nuisance absorbed by chi2_bao marginalisation.
# We recover it as beta_opt = (mv @ C_inv @ d_bao) / (mv @ C_inv @ mv).
# With the Planck/DESI sound horizon r_s = 147.18 Mpc (Planck 2018, Aghanim+2020)
# and c = 299792.458 km/s, H0 = c / (beta_opt * r_s).
_c_kms = 299792.458
_rs_planck = 147.18   # Mpc — Planck 2018 value

def _beta_opt(zf, Ef):
    """Optimal β = c/(H0·r_s) from the BAO profile marginalisation."""
    def dc(zv):
        m = zf <= zv
        return float(ct._trapz((1/Ef)[m], zf[m])) if m.sum() >= 2 else 0.
    def Eat(zv): return float(np.interp(zv, zf, Ef))
    mv = np.zeros(ct.N_BAO)
    for i, (zi, oi) in enumerate(zip(ct.z_bao, ct.obs_bao)):
        if   oi == "DM_over_rs": mv[i] = dc(zi)
        elif oi == "DH_over_rs": mv[i] = 1./Eat(zi)
        elif oi == "DV_over_rs": mv[i] = (zi * dc(zi)**2 / Eat(zi))**(1/3)
    mCd = float(mv @ ct.C_inv_bao @ ct.d_bao)
    mCm = float(mv @ ct.C_inv_bao @ mv)
    return mCd / mCm

beta_hyp3 = _beta_opt(ct.zf_2par, ct.Ef_2par)
H0_hyp3   = _c_kms / (beta_hyp3 * _rs_planck)

# ΛCDM reference beta and H0 for comparison
from chi2_table import E_lcdm, res_jnt
zf_lc, Ef_lc = E_lcdm(res_jnt.x)
beta_lcdm  = _beta_opt(zf_lc, Ef_lc)
H0_lcdm    = _c_kms / (beta_lcdm * _rs_planck)

print(f"β  (nuisance c/(H0·r_s))  — hyperconical n=3: {beta_hyp3:.4f},  ΛCDM: {beta_lcdm:.4f}")
print(f"H0 (km/s/Mpc, r_s=147.18 Mpc Planck18)   — hyperconical n=3: {H0_hyp3:.1f},  ΛCDM: {H0_lcdm:.1f}")
print(f"Note: assumes r_s is the same as Planck18; if the hyperconical model predicts")
print(f"a different r_s the estimate shifts proportionally.")
print(f"")
print(f"H0 sensitivity — n=3 hyperconical, fixed (al={al_opt:.4f}, ah={ah_opt:.5f}, eta={eta_opt:.3e})")
print(f"BAO+SN chi2 unchanged: chi2_BAO={c2bao:.3f}, chi2_SN={c2sn:.3f}")
print(f"")
print(f"{'H0':>5}  {'gm':>6}  {'Yp':>7}  {'chi2_BBN':>9}  {'Dchi2_BBN':>10}  {'DAIC':>7}")
print("-" * 58)

for H0 in [65, 67, 68, 70, 72, 74, 75]:
    gm, rf = gm_h0(ct.N_SAT_FIX, ah_opt, al_opt, H0)
    ab = compute_abundances(rf, eta=eta_opt)
    yp = float(ab['Y_p'])
    c2_yp = ((yp - ct._YP_OBS) / ct._SIG_YP)**2
    dc2bbn = c2_yp - ct._c2_yp_lcdm_bbn
    dc2bao = c2bao - ct.c2b_jnt
    dc2sn  = c2sn  - ct.c2s_jnt
    daic   = dc2bao + dc2sn + dc2bbn + 2*(3-2)
    print(f"{H0:>5}  {gm:>6.3f}  {yp:>7.4f}  {c2_yp:>9.3f}  {dc2bbn:>10.3f}  {daic:>7.2f}")

print("")
print(f"LCDM ref: chi2_BBN={ct._c2_yp_lcdm_bbn:.3f} (Yp_LCDM vs obs), DAIC=0 by def.")
print(f"Note: eta fixed at joint-fit optimum ({eta_opt:.3e}); no re-optimisation per H0.")
