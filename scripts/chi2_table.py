"""
Joint chi² table: BAO+SN, BAO+BBN, BAO+BBN+SN, ΛCDM
ΔAIC_joint = chi2_joint_model - chi2_joint_ΛCDM (shared Ω_m)
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar, brentq

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from hyperconical_model import ExtendedProjectedHyperconical

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
ALPHA_LOW, ALPHA_HIGH = 0.283, 0.500

# ── BBN physical constants ─────────────────────────────────────────────────────
_T0_EV      = 2.7255 * 8.617333262e-5         # CMB temperature in eV
_H0_SI      = 70.0 / 3.0856775814913673e19    # H0=70 km/s/Mpc in s^-1
_MPL_GEV    = 1.220890e19                      # Planck mass in GeV
_HBAR_GEV_S = 6.582119569e-25                 # ℏ in GeV·s
_T_BBN_LO, _T_BBN_HI = 0.07, 0.10            # BBN window in MeV

# Fermi-Dirac energy integral table for g*(T): built once at import.
# h(x) = (120/7π⁴) ∫ u² √(u²+x²)/(e^{√(u²+x²)}+1) du, x=m_e/T.
_fd_u = np.linspace(1e-5, 60., 5000)
_fd_x = np.unique(np.concatenate([np.linspace(0,1,60), np.linspace(1,5,100), np.linspace(5,20,60)]))
_fd_eps = np.sqrt(_fd_u[None,:]**2 + _fd_x[:,None]**2)      # (n_x, n_u)
_fd_h = (120./(7.*np.pi**4)) * _trapz(
    _fd_u**2 * _fd_eps / (np.exp(np.minimum(_fd_eps, 500.)) + 1.), _fd_u, axis=1)

def _g_star(T_mev):
    x_e = 0.511 / np.asarray(T_mev, float)
    h_e = np.interp(x_e, _fd_x, _fd_h, left=1., right=0.)
    nu4 = ((4. + 7.*h_e) / 11.) ** (4./3.)
    return 2. + (7./8.)*4.*h_e + (7./8.)*6.*nu4

def _H_std(T_mev):
    """Standard radiation-dominated H(T) [s^-1] from Kolb & Turner."""
    T_gev = np.asarray(T_mev, float) / 1000.
    return 1.66 * np.sqrt(_g_star(T_mev)) * T_gev**2 / (_MPL_GEV * _HBAR_GEV_S)

# ── data ─────────────────────────────────────────────────────────────────────
DATA = ROOT / "data" / "Ardra"
rows = list(csv.DictReader(open(DATA / "desi_dr1_bao_galqso_lya_mean.csv")))
z_bao_all  = np.array([float(r["z"])     for r in rows])
d_bao_all  = np.array([float(r["value"]) for r in rows])
obs_all    = [r["observable"] for r in rows]
lbl_all    = [r["label"]      for r in rows]
cov_all    = np.array([[float(x) for x in l.split(",")]
                       for l in open(DATA / "desi_dr1_bao_galqso_lya_cov.csv") if l.strip()])
mask       = np.array([lbl != "Lya" for lbl in lbl_all])
idx        = np.where(mask)[0]
z_bao, d_bao, obs_bao = z_bao_all[mask], d_bao_all[mask], [obs_all[i] for i in idx]
C_inv_bao  = np.linalg.inv(cov_all[np.ix_(idx, idx)])
N_BAO = len(z_bao); DOF_BAO = N_BAO - 1

raw = np.genfromtxt(ROOT/"data"/"gapp"/"pantheon_plus_binned_50.csv",
                    names=True, dtype=None, encoding="utf-8", delimiter=",")
z_sn = np.asarray(raw["z"], float)
d_sn = np.asarray(raw["d_proxy"], float)
s_sn = np.asarray(raw["sigma_d"], float)
N_SN = len(z_sn); DOF_SN = N_SN - 1

Z_MAX = max(z_bao.max(), z_sn.max())

# ── E(z) builders ────────────────────────────────────────────────────────────

def E_lcdm(om, n=5000):
    zf = np.linspace(0, Z_MAX*1.01, n)
    return zf, np.sqrt(om*(1+zf)**3 + (1-om))

def E_hippopede(alpha_z_func, n=5000):
    model = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)
    zf = np.linspace(0, Z_MAX*1.01, n)
    az = alpha_z_func(zf)
    x = model.x_from_lz(np.log1p(zf))
    u = np.sqrt(np.maximum(1.0/model.k - x**2, 1e-14))
    y = np.arctan2(x, u)
    g = np.maximum(1.0 - y/model.y0, 1e-12)
    t = (y/2.0)/(g**az)
    rhat = 2.0*np.arctan(t)
    dr = np.gradient(rhat, zf)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = 1.0/dr
    E /= float(np.interp(0.0, zf, E))
    return zf, E

def E_hippopede_const(alpha, n=3000):
    model = ExtendedProjectedHyperconical(alpha=alpha)
    zf = np.linspace(0, Z_MAX*1.01, n)
    Ef, _ = model.e_and_q(zf)
    return zf, Ef

# ── chi² functions ───────────────────────────────────────────────────────────

def chi2_bao(zf, Ef):
    def dc(zv):
        m = zf <= zv; return float(_trapz((1/Ef)[m], zf[m])) if m.sum()>=2 else 0.
    def Eat(zv): return float(np.interp(zv, zf, Ef))
    mv = np.zeros(N_BAO)
    for i,(zi,oi) in enumerate(zip(z_bao,obs_bao)):
        if oi=="DM_over_rs": mv[i]=dc(zi)
        elif oi=="DH_over_rs": mv[i]=1./Eat(zi)
        elif oi=="DV_over_rs": mv[i]=(zi*dc(zi)**2/Eat(zi))**(1/3)
    mCm=mv@C_inv_bao@mv; mCd=mv@C_inv_bao@d_bao; dCd=d_bao@C_inv_bao@d_bao
    return float(dCd - mCd**2/mCm)

def chi2_sn(zf, Ef):
    def dc(zv):
        m = zf <= zv; return float(_trapz((1/Ef)[m], zf[m])) if m.sum()>=2 else 0.
    mv = np.array([dc(zi) for zi in z_sn])
    w=1/s_sn**2; mwm=float((w*mv*mv).sum()); mwd=float((w*mv*d_sn).sum()); dwd=float((w*d_sn*d_sn).sum())
    return float(dwd - mwd**2/mwm)

def chi2_joint(zf, Ef):
    return chi2_bao(zf,Ef) + chi2_sn(zf,Ef)

# ── ΛCDM references ───────────────────────────────────────────────────────────

res_bao = minimize_scalar(lambda om: chi2_bao(*E_lcdm(om)), bounds=(0.1,0.6), method="bounded")
res_sn  = minimize_scalar(lambda om: chi2_sn(*E_lcdm(om)),  bounds=(0.1,0.6), method="bounded")
res_jnt = minimize_scalar(lambda om: chi2_joint(*E_lcdm(om)),bounds=(0.1,0.6),method="bounded")

c2b_lcdm = chi2_bao(*E_lcdm(res_bao.x))
c2s_lcdm = chi2_sn(*E_lcdm(res_sn.x))
c2j_lcdm = chi2_joint(*E_lcdm(res_jnt.x))
# consistent reference: BAO and SN chi2 at the joint-optimal Omega_m
c2b_jnt = chi2_bao(*E_lcdm(res_jnt.x))
c2s_jnt = chi2_sn(*E_lcdm(res_jnt.x))

print(f"ΛCDM BAO: Ω_m={res_bao.x:.4f}, χ²_ν={c2b_lcdm/DOF_BAO:.4f}")
print(f"ΛCDM SN:  Ω_m={res_sn.x:.4f},  χ²_ν={c2s_lcdm/DOF_SN:.4f}")
print(f"ΛCDM joint: Ω_m={res_jnt.x:.4f}, χ²_joint={c2j_lcdm:.4f}")

# ── ansatz functions ──────────────────────────────────────────────────────────

def sat(n):   return lambda z: ALPHA_HIGH - (ALPHA_HIGH-ALPHA_LOW)*(1+z)**(-n)
def pade(al,zc): return lambda z: al + (ALPHA_HIGH-al)*z/(z+zc)

def _e_sat_on_grid(n, z_eval):
    """E(z)/E(0) for SAT running-alpha on a provided z_eval grid (must start at 0).

    Numerically stable only for z_eval well below z_BBN~3e8; beyond ~1e7 the
    rhat→pi saturation kills the gradient.  Use bbn_geomean for BBN extrapolation.
    """
    z = np.asarray(z_eval, float)
    model = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)
    x   = model.x_from_lz(np.log1p(z))
    u   = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y   = np.arctan2(x, u)
    az  = sat(n)(z)
    g   = np.maximum(1. - y/model.y0, 1e-12)
    t   = (y/2.) / (g**az)
    rhat = 2.*np.arctan(t)
    dr  = np.gradient(rhat, z, edge_order=2)
    dr  = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h   = 1./dr
    return h / h[0]   # h[0] = 1 by geometry (drhat/dz|_{z=0} = 1 for any alpha)

def bbn_geomean(n, n_T=80, z_ref=1e6):
    """Geometric mean of H_perc(T)/H_std(T) over T=0.07-0.10 MeV.

    Analytical extrapolation: integrates nu(z) = 1 + 2*alpha(z) from z_ref to
    z_BBN~3e8, where g propto (1+z)^{-2} is exact for the hyperconical metric.
    This is more accurate than the power-law fit (which uses a transient nu from
    the fit range z=10^3-10^6, where alpha is still running, overestimating E).

    For SAT running alpha = alpha_high - (alpha_high-alpha_low)*(1+z)^{-n}:
      integral_{z_ref}^{z} alpha dlnz = alpha_h*ln(r) + (alpha_h-alpha_l)/n *
                                         ((1+z)^{-n} - (1+z_ref)^{-n})
      log(E(z)/E(z_ref)) = ln(r) + 2 * integral
    """
    z_eval = np.unique(np.concatenate([
        np.linspace(0., 1., 400),
        np.geomspace(1., z_ref * 1.02, 5000),
    ]))
    e_run = _e_sat_on_grid(n, z_eval)
    E_ref = float(np.interp(z_ref, z_eval, e_run))

    # Extrapolate to BBN temperatures (z~3e8)
    T_arr = np.geomspace(_T_BBN_LO, _T_BBN_HI, n_T)
    z_arr = T_arr * 1.e6 / _T0_EV - 1.
    ln_r = np.log((1. + z_arr) / (1. + z_ref))
    integral_alpha = ALPHA_HIGH * ln_r + (ALPHA_HIGH - ALPHA_LOW) / n * (
        (1. + z_arr)**(-n) - (1. + z_ref)**(-n))
    E_BBN = E_ref * np.exp(ln_r + 2. * integral_alpha)
    ratio  = _H0_SI * E_BBN / _H_std(T_arr)
    return float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-30)))))

# Find optimal n for BAO-only and joint
res_n_bao = minimize_scalar(
    lambda n: chi2_bao(*E_hippopede(sat(n))), bounds=(0.05, 0.8), method="bounded")
n_bao_opt = res_n_bao.x
res_n_joint = minimize_scalar(
    lambda n: chi2_joint(*E_hippopede(sat(n))), bounds=(0.05, 0.8), method="bounded")
n_joint_opt = res_n_joint.x
print(f"\nSat BAO-optimal  n={n_bao_opt:.4f}, chi2_BAO={chi2_bao(*E_hippopede(sat(n_bao_opt))):.4f}")
print(f"Sat joint-optimal n={n_joint_opt:.4f}, chi2_joint={res_n_joint.fun:.4f}")

print("Computing BBN normalisations (z~3e8) ...")
gm_bao   = bbn_geomean(n_bao_opt)
gm_joint = bbn_geomean(n_joint_opt)
print(f"  n_bao={n_bao_opt:.4f} -> gm={gm_bao:.4f};  n_joint={n_joint_opt:.4f} -> gm={gm_joint:.4f}")

# Find n_bbn_opt: the unique n in (0.04, 0.12) where gm_bbn = 1.
# gm(0.04)<1<gm(0.12) from the sweep; use bisection on log(gm).
n_bbn_opt    = brentq(lambda n: np.log(bbn_geomean(n)), 0.04, 0.12, xtol=1e-4)
gm_bbn_opt   = bbn_geomean(n_bbn_opt)
print(f"  n_bbn_opt={n_bbn_opt:.4f} (gm={gm_bbn_opt:.4f})")

MODEL = "Hyperconical a-run"

def row4(constraint, model, param, zf, Ef, bbn_norm, k_model=1, k_lcdm=1):
    """k_model: free params of the model being compared (1 for hyperconical).
       k_lcdm:  free params of the ΛCDM reference (1 for non-BBN rows, 2 for BBN rows).
       ΔAIC_tot = Δχ²_tot + 2*(k_model - k_lcdm); ΔAIC_BAO/SN are pure Δχ² components."""
    c2b = chi2_bao(zf, Ef); c2s = chi2_sn(zf, Ef); c2j = c2b + c2s
    nb = c2b / DOF_BAO; ns = c2s / DOF_SN
    db = c2b - c2b_jnt; ds = c2s - c2s_jnt
    dj = (c2j - c2j_lcdm) + 2*(k_model - k_lcdm)
    print(f"{constraint:<14} {model:<22} {param:<12} {nb:>8.3f} {ns:>8.3f} "
          f"{bbn_norm:>10}  {db:>+8.2f} {ds:>+8.2f} {dj:>+8.2f}")

print(f"\n{'Constraint':<14} {'Model':<22} {'Param':<12} {'χ²ν_BAO':>8} {'χ²ν_SN':>8} "
      f"{'BBN_norm':>10}  {'ΔAIC_BAO':>8} {'ΔAIC_SN':>8} {'ΔAIC_tot':>8}")
print(f"  1-par ΛCDM ref: Ω_m={res_jnt.x:.4f} (joint BAO+SN)  |  2-par ΛCDM ref: same Ω_m, k=2 (BBN trivially satisfied)")
print("-"*110)

zf, Ef = E_hippopede(sat(n_bao_opt))
row4("BAO",        MODEL, f"n={n_bao_opt:.3f}", zf, Ef, f"~{gm_bao:.1f}", k_lcdm=1)

zf, Ef = E_hippopede(sat(n_joint_opt))
row4("BAO+SN",     MODEL, f"n={n_joint_opt:.3f}", zf, Ef, f"~{gm_joint:.1f}", k_lcdm=1)

zf, Ef = E_hippopede(sat(n_bbn_opt))
row4("BAO+BBN",    MODEL, f"n={n_bbn_opt:.3f}", zf, Ef, f"{gm_bbn_opt:.3f}", k_lcdm=2)

zf, Ef = E_hippopede(sat(n_bbn_opt))
row4("BAO+SN+BBN", MODEL, f"n={n_bbn_opt:.3f}", zf, Ef, f"{gm_bbn_opt:.3f}", k_lcdm=2)

zf_lcdm, Ef_lcdm = E_lcdm(res_jnt.x)
print("-"*110)
row4("BAO+SN",     "ΛCDM (1-par ref)", f"Ω_m={res_jnt.x:.3f}", zf_lcdm, Ef_lcdm, "1", k_model=1, k_lcdm=1)
# 2-par LCDM reference: k_model=k_lcdm=2 so DAIC=0 (self-reference for BBN group)
# Omega_r is calibrated by BBN (not fitted to BAO/SN); chi2 values same as 1-par (Omega_r negligible at DESI z)
row4("BAO+SN+BBN", "ΛCDM (2-par ref)", f"Ω_m={res_jnt.x:.3f}", zf_lcdm, Ef_lcdm, "1", k_model=2, k_lcdm=2)
