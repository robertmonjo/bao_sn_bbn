"""Explore 2-parameter SAT-2 model: alpha_low and alpha_high both free.

Wide BBN window: T=0.07-1.00 MeV (includes n/p freeze-out at T~0.8 MeV).
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar, brentq

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from hyperconical_model import ExtendedProjectedHyperconical

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ── physical constants ────────────────────────────────────────────────────────
_T0_EV      = 2.7255 * 8.617333262e-5
_H0_SI      = 70.0 / 3.0856775814913673e19
_MPL_GEV    = 1.220890e19
_HBAR_GEV_S = 6.582119569e-25

_fd_u = np.linspace(1e-5, 60., 5000)
_fd_x = np.unique(np.concatenate([np.linspace(0,1,60), np.linspace(1,5,100), np.linspace(5,20,60)]))
_fd_eps = np.sqrt(_fd_u[None,:]**2 + _fd_x[:,None]**2)
_fd_h = (120./(7.*np.pi**4)) * _trapz(
    _fd_u**2 * _fd_eps / (np.exp(np.minimum(_fd_eps, 500.)) + 1.), _fd_u, axis=1)

def _g_star(T_mev):
    x_e = 0.511 / np.asarray(T_mev, float)
    h_e = np.interp(x_e, _fd_x, _fd_h, left=1., right=0.)
    nu4 = ((4. + 7.*h_e) / 11.) ** (4./3.)
    return 2. + (7./8.)*4.*h_e + (7./8.)*6.*nu4

def _H_std(T_mev):
    T_gev = np.asarray(T_mev, float) / 1000.
    return 1.66 * np.sqrt(_g_star(T_mev)) * T_gev**2 / (_MPL_GEV * _HBAR_GEV_S)

# ── BAO / SN data ─────────────────────────────────────────────────────────────
DATA = ROOT / "data" / "Ardra"
rows = list(csv.DictReader(open(DATA / "desi_dr1_bao_galqso_lya_mean.csv")))
z_bao_all = np.array([float(r["z"]) for r in rows])
d_bao_all = np.array([float(r["value"]) for r in rows])
obs_all   = [r["observable"] for r in rows]
lbl_all   = [r["label"]      for r in rows]
cov_all   = np.array([[float(x) for x in l.split(",")]
                      for l in open(DATA / "desi_dr1_bao_galqso_lya_cov.csv") if l.strip()])
mask       = np.array([lbl != "Lya" for lbl in lbl_all])
idx        = np.where(mask)[0]
z_bao, d_bao = z_bao_all[mask], d_bao_all[mask]
obs_bao = [obs_all[i] for i in idx]
C_inv_bao = np.linalg.inv(cov_all[np.ix_(idx, idx)])
N_BAO = len(z_bao); DOF_BAO = N_BAO - 1

raw = np.genfromtxt(ROOT/"data"/"gapp"/"pantheon_plus_binned_50.csv",
                    names=True, dtype=None, encoding="utf-8", delimiter=",")
z_sn = np.asarray(raw["z"], float)
d_sn = np.asarray(raw["d_proxy"], float)
s_sn = np.asarray(raw["sigma_d"], float)
N_SN = len(z_sn); DOF_SN = N_SN - 1
Z_MAX = max(z_bao.max(), z_sn.max())

N_SAT = 0.5  # fixed exponent for SAT-2

# ── model and chi2 ───────────────────────────────────────────────────────────
def E_sat2(al, ah, n=N_SAT, n_grid=5000):
    """E(z)/E(0) for SAT-2 with both al and ah free; geometry at al."""
    zf = np.linspace(0., Z_MAX * 1.01, n_grid)
    model = ExtendedProjectedHyperconical(alpha=al)
    x  = model.x_from_lz(np.log1p(zf))
    u  = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y  = np.arctan2(x, u)
    az = ah - (ah - al) * (1 + zf)**(-n)
    g  = np.maximum(1. - y/model.y0, 1e-12)
    t  = (y/2.) / (g**az)
    rhat = 2.*np.arctan(t)
    dr = np.gradient(rhat, zf, edge_order=2)
    dr = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h  = 1./dr
    return zf, h / h[0]

def chi2_bao(zf, Ef):
    def dc(zv):
        m = zf <= zv; return float(_trapz((1/Ef)[m], zf[m])) if m.sum()>=2 else 0.
    def Eat(zv): return float(np.interp(zv, zf, Ef))
    mv = np.zeros(N_BAO)
    for i,(zi,oi) in enumerate(zip(z_bao, obs_bao)):
        if   oi=="DM_over_rs":  mv[i] = dc(zi)
        elif oi=="DH_over_rs":  mv[i] = 1./Eat(zi)
        elif oi=="DV_over_rs":  mv[i] = (zi*dc(zi)**2/Eat(zi))**(1/3)
    mCm=mv@C_inv_bao@mv; mCd=mv@C_inv_bao@d_bao; dCd=d_bao@C_inv_bao@d_bao
    return float(dCd - mCd**2/mCm)

def chi2_sn(zf, Ef):
    def dc(zv):
        m = zf <= zv; return float(_trapz((1/Ef)[m], zf[m])) if m.sum()>=2 else 0.
    mv = np.array([dc(zi) for zi in z_sn])
    w=1/s_sn**2; mwm=float((w*mv**2).sum()); mwd=float((w*mv*d_sn).sum()); dwd=float((w*d_sn**2).sum())
    return float(dwd - mwd**2/mwm)

def gm_window(al, ah, T_lo, T_hi, n=N_SAT, n_T=80, z_ref=1e6):
    """Geometric mean of H_model/H_std over [T_lo, T_hi] MeV."""
    z_eval = np.unique(np.concatenate([
        np.linspace(0., 1., 400),
        np.geomspace(1., z_ref * 1.02, 5000),
    ]))
    model = ExtendedProjectedHyperconical(alpha=al)
    x  = model.x_from_lz(np.log1p(z_eval))
    u  = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y  = np.arctan2(x, u)
    az = ah - (ah - al) * (1 + z_eval)**(-n)
    g  = np.maximum(1. - y/model.y0, 1e-12)
    t  = (y/2.) / (g**az)
    rhat = 2.*np.arctan(t)
    dr = np.gradient(rhat, z_eval, edge_order=2)
    dr = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h  = 1./dr; h /= h[0]
    E_ref = float(np.interp(z_ref, z_eval, h))
    T_arr = np.geomspace(T_lo, T_hi, n_T)
    z_arr = T_arr * 1.e6 / _T0_EV - 1.
    ln_r  = np.log((1. + z_arr) / (1. + z_ref))
    integral_alpha = ah * ln_r + (ah - al) / n * (
        (1. + z_arr)**(-n) - (1. + z_ref)**(-n))
    E_BBN = E_ref * np.exp(ln_r + 2. * integral_alpha)
    ratio = _H0_SI * E_BBN / _H_std(T_arr)
    return float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-30)))))

def ratio_at_T(al, ah, T_mev, n=N_SAT, z_ref=1e6):
    """H_model/H_std at a single temperature T_mev."""
    z_eval = np.unique(np.concatenate([np.linspace(0.,1.,400),
                                        np.geomspace(1., z_ref*1.02, 5000)]))
    model = ExtendedProjectedHyperconical(alpha=al)
    x  = model.x_from_lz(np.log1p(z_eval))
    u  = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y  = np.arctan2(x, u)
    az = ah - (ah - al) * (1 + z_eval)**(-n)
    g  = np.maximum(1. - y/model.y0, 1e-12)
    t  = (y/2.) / (g**az)
    rhat = 2.*np.arctan(t)
    dr = np.gradient(rhat, z_eval, edge_order=2)
    dr = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h  = 1./dr; h /= h[0]
    E_ref = float(np.interp(z_ref, z_eval, h))
    z_T  = float(T_mev * 1e6 / _T0_EV - 1.)
    ln_r  = np.log((1. + z_T) / (1. + z_ref))
    integral_alpha = ah * ln_r + (ah - al) / n * (
        (1. + z_T)**(-n) - (1. + z_ref)**(-n))
    E_BBN = E_ref * np.exp(ln_r + 2. * integral_alpha)
    return float(_H0_SI * E_BBN / _H_std(T_mev))

# ── ΛCDM reference ────────────────────────────────────────────────────────────
def E_lcdm(om, n=5000):
    zf = np.linspace(0, Z_MAX*1.01, n)
    return zf, np.sqrt(om*(1+zf)**3 + (1-om))

res_lcdm = minimize_scalar(
    lambda om: chi2_bao(*E_lcdm(om)) + chi2_sn(*E_lcdm(om)),
    bounds=(0.1, 0.6), method="bounded")
c2j_lcdm = res_lcdm.fun
c2b_jnt  = chi2_bao(*E_lcdm(res_lcdm.x))
c2s_jnt  = chi2_sn(*E_lcdm(res_lcdm.x))
if __name__ == "__main__":
    print(f"ΛCDM joint: Ω_m={res_lcdm.x:.4f}, χ²_joint={c2j_lcdm:.4f} "
          f"(χ²ν_BAO={c2b_jnt/DOF_BAO:.4f}, χ²ν_SN={c2s_jnt/DOF_SN:.4f})\n")

    # ── 1. Narrow gm=1 (0.07-0.10 MeV) as function of al (ah fixed at 0.4224) ─
    print("=== Effect of freeing α_low (narrow window, ah=0.4224 fixed) ===")
    ah_bbn = 0.4224
    print(f"{'al':>6}  {'gm_narrow':>10}  {'gm_wide':>10}  {'ratio@0.8MeV':>13}  {'χ²ν_BAO':>9}  {'χ²ν_SN':>9}  {'ΔAIC':>7}")
    for al in np.arange(0.20, 0.43, 0.02):
        if al >= ah_bbn:
            break
        try:
            gm_n = gm_window(al, ah_bbn, 0.07, 0.10)
            gm_w = gm_window(al, ah_bbn, 0.07, 1.00)
            r08  = ratio_at_T(al, ah_bbn, 0.80)
            zf, Ef = E_sat2(al, ah_bbn)
            c2b  = chi2_bao(zf, Ef); c2s = chi2_sn(zf, Ef)
            dj   = (c2b + c2s - c2j_lcdm)  # k_model=2=k_lcdm -> Δk=0
            print(f"{al:>6.3f}  {gm_n:>10.3f}  {gm_w:>10.3f}  {r08:>13.3f}  "
                  f"{c2b/DOF_BAO:>9.3f}  {c2s/DOF_SN:>9.3f}  {dj:>+7.2f}")
        except Exception as e:
            print(f"{al:>6.3f}  ERROR: {e}")

    # ── 2. Grid over (al, ah): find where gm_wide≈1 ──────────────────────────
    print("\n=== Grid (al, ah): gm_wide(0.07-1.00 MeV) and BAO+SN quality ===")
    print(f"{'al':>6}  {'ah':>6}  {'gm_wide':>9}  {'ratio@0.8':>10}  {'χ²ν_BAO':>9}  {'χ²ν_SN':>9}  {'ΔAIC':>7}")
    al_vals = np.array([0.20, 0.25, 0.283, 0.30, 0.32, 0.35, 0.38, 0.40])
    ah_vals = np.array([0.40, 0.42, 0.44, 0.46, 0.48, 0.50])
    for al in al_vals:
        for ah in ah_vals:
            if ah <= al + 0.04:
                continue
            try:
                gm_w = gm_window(al, ah, 0.07, 1.00)
                r08  = ratio_at_T(al, ah, 0.80)
                zf, Ef = E_sat2(al, ah)
                c2b = chi2_bao(zf, Ef); c2s = chi2_sn(zf, Ef)
                dj  = c2b + c2s - c2j_lcdm  # Δk=0 (k_model=2=k_lcdm)
                print(f"{al:>6.3f}  {ah:>6.3f}  {gm_w:>9.3f}  {r08:>10.3f}  "
                      f"{c2b/DOF_BAO:>9.3f}  {c2s/DOF_SN:>9.3f}  {dj:>+7.2f}")
            except Exception as e:
                print(f"{al:>6.3f}  {ah:>6.3f}  ERROR: {e}")

    # ── 3. For each al, find ah_wide such that gm_wide(0.07-1.00 MeV)=1 ──────
    print("\n=== Optimal ah for gm_wide=1 (wide BBN window) at each al ===")
    print(f"{'al':>6}  {'ah_wide*':>9}  {'gm_n(0.07-0.10)':>16}  {'ratio@0.8':>10}  {'χ²ν_BAO':>9}  {'χ²ν_SN':>9}  {'ΔAIC':>7}")
    for al in np.array([0.20, 0.25, 0.283, 0.30, 0.32, 0.35, 0.38]):
        try:
            lo, hi = al + 0.05, 0.55
            if gm_window(al, lo, 0.07, 1.00) * gm_window(al, hi, 0.07, 1.00) > 0:
                print(f"{al:>6.3f}  no bracket for gm_wide=1")
                continue
            ah_w = brentq(lambda ah: np.log(gm_window(al, ah, 0.07, 1.00)),
                          lo, hi, xtol=1e-3)
            gm_n = gm_window(al, ah_w, 0.07, 0.10)
            r08  = ratio_at_T(al, ah_w, 0.80)
            zf, Ef = E_sat2(al, ah_w)
            c2b = chi2_bao(zf, Ef); c2s = chi2_sn(zf, Ef)
            dj  = c2b + c2s - c2j_lcdm
            print(f"{al:>6.3f}  {ah_w:>9.4f}  {gm_n:>16.3f}  {r08:>10.3f}  "
                  f"{c2b/DOF_BAO:>9.3f}  {c2s/DOF_SN:>9.3f}  {dj:>+7.2f}")
        except Exception as e:
            print(f"{al:>6.3f}  ERROR: {e}")
