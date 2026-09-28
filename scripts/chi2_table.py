"""
Joint chi² table: BAO+SN and BAO+SN+BBN — Table 1 reproduction.

Columns (12):
  Constraint | Model | n | Parameters | H_norm | Y_p | D/H(×10⁻⁵) |
  χ²ν_BAO | χ²ν_SN | Δχ²_BAO | Δχ²_SN | ΔAIC_tot

ΔAIC_tot = Δ(χ²_BAO + χ²_SN + χ²_BBN) + 2·(k_model − k_ΛCDM)
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.interpolate import interp1d as _interp1d

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from hyperconical_model import ExtendedProjectedHyperconical
from bbn_hyperconical import compute_abundances, ETA_STD

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
ALPHA_LOW, ALPHA_HIGH = 0.283, 0.500
N_SAT_FIX = 0.5    # fixed SAT exponent for the 1-par running-alpha parametrisation

# ── ΛCDM radiation (fixed externally from FIRAS, Fixsen 2009, ApJ 707, 916) ──
_TCMB_K      = 2.72548                         # T_CMB from FIRAS spectroscopy
ORH2_FIX     = 4.18e-5                         # Ω_r h² = (π²/15) T⁴/ρ_c,0; FIRAS value
H_FIX        = 0.70                            # h used only to evaluate Ω_r = Ω_r h²/h²
OMEGA_R_LCDM = ORH2_FIX / H_FIX**2            # ≈ 8.53e-5; enters E_lcdm only

# ── BBN physical constants ─────────────────────────────────────────────────────
_T0_EV      = 2.7255 * 8.617333262e-5         # CMB temperature in eV
_H0_SI      = 68.0 / 3.0856775814913673e19    # H0=68 km/s/Mpc in s^-1 (hyperconical model prediction, Monjo 2024 ApJ)
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
    zf  = np.linspace(0, Z_MAX*1.01, n)
    om_l = 1.0 - om - OMEGA_R_LCDM
    return zf, np.sqrt(om*(1+zf)**3 + OMEGA_R_LCDM*(1+zf)**4 + om_l)

def E_hippopede(alpha_z_func, n=5000):
    """E(z) grid for running-alpha model using ALPHA_LOW for the geometry."""
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

def E_hippopede_al(alpha_z_func, al=ALPHA_LOW, n=5000):
    """E(z) grid for running-alpha model using al for the geometry (not global ALPHA_LOW)."""
    model = ExtendedProjectedHyperconical(alpha=al)
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
    """E(z) grid for constant-alpha hyperconical model."""
    zf = np.linspace(0., Z_MAX*1.01, n)
    return zf, _e_const_on_grid(alpha, zf)

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

# ── Profile-likelihood uncertainty helpers ────────────────────────────────────

def _profile_sig(f_c2, x_opt, c2_opt, x_lo, x_hi, n=501):
    """1σ uncertainty from Δχ²=1 profile."""
    xg = np.linspace(x_lo, x_hi, n)
    cg = np.array([f_c2(x) for x in xg])
    fi = _interp1d(xg, cg - c2_opt, kind='cubic', fill_value='extrapolate')
    try:
        xl = brentq(lambda x: fi(x)-1., xg[0], x_opt)
        xh = brentq(lambda x: fi(x)-1., x_opt, xg[-1])
        return (xh - xl) / 2.
    except Exception:
        return float('nan')

def _paren(val, sig):
    """Parenthesized notation: e.g. 0.320(10), 1.000(40).

    Uses 2 significant figures in the parenthesized uncertainty, consistent
    with the physics convention for errors < 1.
    """
    import math
    if sig <= 0 or np.isnan(sig) or np.isinf(sig):
        return f"{val:.3g}"
    p = max(0, -int(math.floor(math.log10(abs(sig)))) + 1)
    return f"{val:.{p}f}({round(sig * 10**p):.0f})"

# ── ΛCDM references ───────────────────────────────────────────────────────────

res_bao = minimize_scalar(lambda om: chi2_bao(*E_lcdm(om)), bounds=(0.1,0.6), method="bounded")
res_sn  = minimize_scalar(lambda om: chi2_sn(*E_lcdm(om)),  bounds=(0.1,0.6), method="bounded")
res_jnt = minimize_scalar(lambda om: chi2_joint(*E_lcdm(om)),bounds=(0.1,0.6),method="bounded")

c2b_lcdm = chi2_bao(*E_lcdm(res_bao.x))
c2s_lcdm = chi2_sn(*E_lcdm(res_sn.x))
c2j_lcdm = chi2_joint(*E_lcdm(res_jnt.x))
# Consistent reference: BAO and SN chi2 at the joint-optimal Omega_m
c2b_jnt = chi2_bao(*E_lcdm(res_jnt.x))
c2s_jnt = chi2_sn(*E_lcdm(res_jnt.x))

# ── ansatz functions ──────────────────────────────────────────────────────────

def sat(n, ah=ALPHA_HIGH, al=ALPHA_LOW):
    """Saturating running alpha: alpha(z) = ah - (ah-al)*(1+z)^{-n}."""
    return lambda z: ah - (ah-al)*(1+z)**(-n)

def pade(al,zc): return lambda z: al + (ALPHA_HIGH-al)*z/(z+zc)

def _e_sat_on_grid(n, z_eval, ah=ALPHA_HIGH, al=ALPHA_LOW):
    """E(z)/E(0) for SAT running-alpha on a provided z_eval grid (must start at 0).

    Numerically stable only for z_eval well below z_BBN~3e8; beyond ~1e7 the
    rhat->pi saturation kills the gradient.  Use bbn_geomean for BBN extrapolation.
    """
    z = np.asarray(z_eval, float)
    model = ExtendedProjectedHyperconical(alpha=al)
    x   = model.x_from_lz(np.log1p(z))
    u   = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y   = np.arctan2(x, u)
    az  = sat(n, ah, al)(z)
    g   = np.maximum(1. - y/model.y0, 1e-12)
    t   = (y/2.) / (g**az)
    rhat = 2.*np.arctan(t)
    dr  = np.gradient(rhat, z, edge_order=2)
    dr  = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h   = 1./dr
    return h / h[0]   # h[0] = 1 by geometry (drhat/dz|_{z=0} = 1 for any alpha)

def _e_const_on_grid(alpha, z_eval):
    """E(z)/E(0) for constant alpha, using the full model geometry at that alpha.

    Analogous to _e_sat_on_grid but uses ExtendedProjectedHyperconical(alpha=alpha)
    (not ALPHA_LOW) for the geometry, and applies a constant exponent az=alpha.
    Numerically stable up to z~10^6; use bbn_geomean_const for z >> 10^6.
    """
    z = np.asarray(z_eval, float)
    model = ExtendedProjectedHyperconical(alpha=alpha)
    x   = model.x_from_lz(np.log1p(z))
    u   = np.sqrt(np.maximum(1./model.k - x**2, 1e-14))
    y   = np.arctan2(x, u)
    g   = np.maximum(1. - y/model.y0, 1e-12)
    t   = (y/2.) / (g**alpha)
    rhat = 2.*np.arctan(t)
    dr  = np.gradient(rhat, z, edge_order=2)
    dr  = np.where(np.abs(dr) < 1e-18, np.sign(dr)*1e-18 + (dr == 0.)*1e-18, dr)
    h   = 1./dr
    return h / h[0]

def bbn_geomean_const(alpha, n_T=80, z_ref=1e6):
    """BBN geometric mean for constant alpha using the full model geometry at alpha.

    E_ref at z_ref is obtained from _e_const_on_grid; extrapolated analytically
    via E_BBN = E_ref * ((1+z_BBN)/(1+z_ref))^{1+2*alpha}, valid when g->0 (high-z).
    """
    z_eval = np.unique(np.concatenate([
        np.linspace(0., 1., 400),
        np.geomspace(1., z_ref * 1.02, 5000),
    ]))
    E_ref = float(np.interp(z_ref, z_eval, _e_const_on_grid(alpha, z_eval)))
    T_arr = np.geomspace(_T_BBN_LO, _T_BBN_HI, n_T)
    z_arr = T_arr * 1.e6 / _T0_EV - 1.
    ln_r  = np.log((1. + z_arr) / (1. + z_ref))
    E_BBN = E_ref * np.exp((1. + 2.*alpha) * ln_r)
    ratio = _H0_SI * E_BBN / _H_std(T_arr)
    return float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-30)))))

def bbn_geomean(n, ah=ALPHA_HIGH, n_T=80, z_ref=1e6, al=ALPHA_LOW):
    """Geometric mean of H_perc(T)/H_std(T) over T=0.07-0.10 MeV.

    Analytical extrapolation: integrates nu(z) = 1 + 2*alpha(z) from z_ref to
    z_BBN~3e8, where g propto (1+z)^{-2} is exact for the hyperconical metric.
    This is more accurate than the power-law fit (which uses a transient nu from
    the fit range z=10^3-10^6, where alpha is still running, overestimating E).

    For SAT running alpha = ah - (ah-al)*(1+z)^{-n}:
      integral_{z_ref}^{z} alpha dlnz = ah*ln(r) + (ah-al)/n *
                                         ((1+z)^{-n} - (1+z_ref)^{-n})
      log(E(z)/E(z_ref)) = ln(r) + 2 * integral
    """
    z_eval = np.unique(np.concatenate([
        np.linspace(0., 1., 400),
        np.geomspace(1., z_ref * 1.02, 5000),
    ]))
    e_run = _e_sat_on_grid(n, z_eval, ah, al)
    E_ref = float(np.interp(z_ref, z_eval, e_run))

    # Extrapolate to BBN temperatures (z~3e8)
    T_arr = np.geomspace(_T_BBN_LO, _T_BBN_HI, n_T)
    z_arr = T_arr * 1.e6 / _T0_EV - 1.
    ln_r = np.log((1. + z_arr) / (1. + z_ref))
    integral_alpha = ah * ln_r + (ah - al) / n * (
        (1. + z_arr)**(-n) - (1. + z_ref)**(-n))
    E_BBN = E_ref * np.exp(ln_r + 2. * integral_alpha)
    ratio  = _H0_SI * E_BBN / _H_std(T_arr)
    return float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-30)))))

def bbn_ratio_func(n, ah, al=ALPHA_LOW, z_ref=1e6):
    """Return callable T_mev -> H_model(T)/H_std(T) using the SAT analytical extrapolation.

    Uses the same high-z extrapolation as bbn_geomean, but returns a pointwise
    ratio function suitable for passing to compute_abundances.
    """
    z_eval = np.unique(np.concatenate([
        np.linspace(0., 1., 400),
        np.geomspace(1., z_ref * 1.02, 5000),
    ]))
    e_run = _e_sat_on_grid(n, z_eval, ah, al)
    E_ref = float(np.interp(z_ref, z_eval, e_run))

    def ratio(T_mev):
        T_scalar = float(np.asarray(T_mev))
        z_T = T_scalar * 1.e6 / _T0_EV - 1.
        ln_r = np.log((1. + z_T) / (1. + z_ref))
        integral_alpha = ah * ln_r + (ah - al) / n * (
            (1. + z_T)**(-n) - (1. + z_ref)**(-n))
        E_T = E_ref * np.exp(ln_r + 2. * integral_alpha)
        return float(_H0_SI * E_T / _H_std(T_scalar))

    return ratio

# ── BBN likelihood ─────────────────────────────────────────────────────────────
# σ_H/H from primordial Y_p: Aver, Olive & Skillman 2021 (arXiv:2010.04180),
# Yp=0.2453±0.0034, dYp/dNeff≈0.013 → σ(Neff)≈0.26 → σH/H=σ(Neff)/(2×3.046)≈4%.
# Consistent with Fields, Olive, Yeh & Young 2020 (arXiv:1912.01132, JCAP 03,010):
# Nν=2.86±0.15 from BBN alone → σH/H≈2.5% (tighter, includes CMB priors).
SIGMA_H_BBN = 0.04   # 4% (1σ), conservative: Y_p alone, no CMB N_eff prior

def chi2_bbn(gm_val, sigma_H=SIGMA_H_BBN):
    """χ²_BBN = ((gm−1)/σ_H)²: one effective BBN data point.

    gm_val is the geometric mean of H_model(T)/H_std(T) over T=0.07-0.10 MeV.
    σ_H=0.04 from Aver et al. 2021 (arXiv:2010.04180).
    ΛCDM satisfies BBN exactly (gm=1), so χ²_BBN(ΛCDM)=0.
    """
    return float((gm_val - 1.) ** 2 / sigma_H ** 2)

# ── Optimise BAO+SN running-alpha (1-par, N_SAT_FIX exponent, ah free) ────────

res_ah_bao = minimize_scalar(
    lambda ah: chi2_joint(*E_hippopede(sat(N_SAT_FIX, ah))),
    bounds=(0.30, 0.65), method="bounded")
ah_bao_opt = res_ah_bao.x
gm_ah_bao  = bbn_geomean(N_SAT_FIX, ah_bao_opt)

# BBN-constrained alpha_high for N_SAT_FIX exponent (gm=1)
ah_bbn_lo, ah_bbn_hi = 0.30, 0.55
ah_bbn_opt = brentq(lambda ah: np.log(bbn_geomean(N_SAT_FIX, ah)),
                    ah_bbn_lo, ah_bbn_hi, xtol=1e-4)
gm_ah_bbn  = bbn_geomean(N_SAT_FIX, ah_bbn_opt)

# ── Constant-alpha BAO+SN-optimal ─────────────────────────────────────────────

res_alpha_const = minimize_scalar(
    lambda a: chi2_joint(*E_hippopede_const(a)), bounds=(0.20, 0.50), method="bounded")
alpha_const_opt = res_alpha_const.x
gm_const_opt  = bbn_geomean_const(alpha_const_opt)
gm_const_half = bbn_geomean_const(0.500)

# ── 3-par joint fit: (al, ah, eta) minimises chi2_BAO+SN+Yp+DH ──────────────

DH_OBS = 2.527e-5; SIG_DH_OBS = 0.030e-5
_YP_OBS = 0.2453; _SIG_YP = 0.0034

def _chi2_3par(params):
    al, ah, eta = params
    if al < 0.05 or ah <= al + 0.01 or ah > 0.90 or eta < 3e-10 or eta > 15e-10:
        return 1e10
    try:
        zf_, Ef_ = E_hippopede_al(sat(N_SAT_FIX, ah, al), al=al)
        c2bs = chi2_bao(zf_, Ef_) + chi2_sn(zf_, Ef_)
        rf_ = bbn_ratio_func(N_SAT_FIX, ah, al=al)
        ab_ = compute_abundances(rf_, eta=eta)
        c2_yp = ((ab_['Y_p'] - _YP_OBS) / _SIG_YP) ** 2
        c2_dh = ((ab_['D_H'] - DH_OBS) / SIG_DH_OBS) ** 2
        return c2bs + c2_yp + c2_dh
    except Exception:
        return 1e10

_res_3par = minimize(_chi2_3par, [0.255, 0.4364, 7.658e-10],
                     method='Nelder-Mead',
                     options={'xatol': 1e-6, 'fatol': 1e-5, 'maxiter': 8000})
al_2par, ah_2par, _eta_fit = _res_3par.x
zf_2par, Ef_2par = E_hippopede_al(sat(N_SAT_FIX, ah_2par, al_2par), al=al_2par)
gm_2par = bbn_geomean(N_SAT_FIX, ah_2par, al=al_2par)
rf_2par = bbn_ratio_func(N_SAT_FIX, ah_2par, al=al_2par)
abund_2par = compute_abundances(rf_2par, eta=_eta_fit)

# ── Y_p / D/H for BAO+SN running-alpha rows (0-par and 1-par, at ETA_STD) ────

# 0-par: ah=1/2, N_SAT_FIX exponent
rf_0par = bbn_ratio_func(N_SAT_FIX, 0.500)
abund_0par = compute_abundances(rf_0par, eta=ETA_STD)

# 1-par BAO+SN-optimal
rf_1par_bao = bbn_ratio_func(N_SAT_FIX, ah_bao_opt)
abund_1par_bao = compute_abundances(rf_1par_bao, eta=ETA_STD)

# 1-par BBN-constrained
rf_1par_bbn = bbn_ratio_func(N_SAT_FIX, ah_bbn_opt)
abund_1par_bbn = compute_abundances(rf_1par_bbn, eta=ETA_STD)

# 1-par+η BBN: same ah as 1-par but η fitted to D/H (n=2 vs ΛCDM n=2)
eta_1par_bbn_fit = brentq(lambda e: compute_abundances(rf_1par_bbn, eta=e)['D_H'] - DH_OBS, 4e-10, 11e-10)
abund_1par_bbn_eta = compute_abundances(rf_1par_bbn, eta=eta_1par_bbn_fit)
_dDH_deta_1par = (compute_abundances(rf_1par_bbn, eta=eta_1par_bbn_fit+1e-12)['D_H'] -
                  compute_abundances(rf_1par_bbn, eta=eta_1par_bbn_fit-1e-12)['D_H']) / 2e-12
sig_eta_1par_bbn = SIG_DH_OBS / abs(_dDH_deta_1par) if abs(_dDH_deta_1par) > 1e-20 else float('nan')

# SBBN reference (ratio=1) for ΛCDM BBN row
abund_sbbn = compute_abundances(lambda T: 1.0, eta=ETA_STD)

# ── σ for each fitted parameter ───────────────────────────────────────────────
_dah = 0.0002

# σ(α_const) from profile χ²_joint
sig_alpha_const = _profile_sig(
    lambda a: chi2_joint(*E_hippopede_const(a)),
    alpha_const_opt, res_alpha_const.fun, 0.20, 0.50)

# σ(Ω_m) from profile χ²_joint
sig_om_jnt = _profile_sig(
    lambda om: chi2_joint(*E_lcdm(om)),
    res_jnt.x, res_jnt.fun, 0.1, 0.6)

# σ(ah_bao) from profile χ²_joint
sig_ah_bao = _profile_sig(
    lambda ah: chi2_joint(*E_hippopede(sat(N_SAT_FIX, ah))),
    ah_bao_opt, res_ah_bao.fun, 0.30, 0.65)

# σ(ah_bbn) from d(H_norm)/d(ah) and σ_H=4%
_dhn_dah_bbn = (bbn_geomean(N_SAT_FIX, ah_bbn_opt+_dah) - bbn_geomean(N_SAT_FIX, ah_bbn_opt-_dah)) / (2*_dah)
sig_ah_bbn = SIGMA_H_BBN / abs(_dhn_dah_bbn)

# σ(al_2par) from profile χ²_joint at fixed ah_2par
_c2_al2 = chi2_joint(*E_hippopede_al(sat(N_SAT_FIX, ah_2par, al_2par), al=al_2par))
sig_al_2par = _profile_sig(
    lambda al: chi2_joint(*E_hippopede_al(sat(N_SAT_FIX, ah_2par, al), al=al)),
    al_2par, _c2_al2, 0.15, 0.42)

# σ(ah_2par) from d(Y_p)/d(ah_2par) and σ_Yp=0.0034
_r2_lo = compute_abundances(bbn_ratio_func(N_SAT_FIX, ah_2par-_dah, al=al_2par), eta=_eta_fit)
_r2_hi = compute_abundances(bbn_ratio_func(N_SAT_FIX, ah_2par+_dah, al=al_2par), eta=_eta_fit)
_dYp_dah2 = (_r2_hi['Y_p'] - _r2_lo['Y_p']) / (2*_dah)
sig_ah_2par = _SIG_YP / abs(_dYp_dah2) if abs(_dYp_dah2) > 1e-10 else float('nan')

# ── Propagated BBN predictions ────────────────────────────────────────────────

# const-alpha: Y_p≈0, D/H=0.31(1) (al=ah=alpha_const)
_rf_const_bbn = bbn_ratio_func(N_SAT_FIX, alpha_const_opt, al=alpha_const_opt)
_r_const_bbn = compute_abundances(_rf_const_bbn, eta=ETA_STD)
_yp_const = float(_r_const_bbn['Y_p'])
_dh_const = float(_r_const_bbn['D_H']*1e5)
_dh_c_lo = compute_abundances(bbn_ratio_func(N_SAT_FIX, alpha_const_opt-_dah, al=alpha_const_opt-_dah), eta=ETA_STD)['D_H']*1e5
_dh_c_hi = compute_abundances(bbn_ratio_func(N_SAT_FIX, alpha_const_opt+_dah, al=alpha_const_opt+_dah), eta=ETA_STD)['D_H']*1e5
sig_dh_const = abs(_dh_c_hi - _dh_c_lo) / (2*_dah) * sig_alpha_const

# Const-alpha H_norm sigma
_dgm_da_const = (bbn_geomean_const(alpha_const_opt+_dah) - bbn_geomean_const(alpha_const_opt-_dah)) / (2*_dah)
sig_gm_const = abs(_dgm_da_const) * sig_alpha_const

# n=1 BAO+SN: H_norm, Y_p, D/H from σ(ah_bao)
sig_gm_bao = abs(bbn_geomean(N_SAT_FIX, ah_bao_opt+_dah) - bbn_geomean(N_SAT_FIX, ah_bao_opt-_dah)) / (2*_dah) * sig_ah_bao
_r_lo_bao = compute_abundances(bbn_ratio_func(N_SAT_FIX, ah_bao_opt-_dah), eta=ETA_STD)
_r_hi_bao = compute_abundances(bbn_ratio_func(N_SAT_FIX, ah_bao_opt+_dah), eta=ETA_STD)
sig_yp_bao = abs(_r_hi_bao['Y_p'] - _r_lo_bao['Y_p']) / (2*_dah) * sig_ah_bao
sig_dh_bao = abs(_r_hi_bao['D_H'] - _r_lo_bao['D_H']) / (2*_dah) * sig_ah_bao * 1e5

# n=1 BBN: Y_p, D/H from σ(ah_bbn)
_r_lo_bbn2 = compute_abundances(bbn_ratio_func(N_SAT_FIX, ah_bbn_opt-_dah), eta=ETA_STD)
_r_hi_bbn2 = compute_abundances(bbn_ratio_func(N_SAT_FIX, ah_bbn_opt+_dah), eta=ETA_STD)
sig_yp_bbn = abs(_r_hi_bbn2['Y_p'] - _r_lo_bbn2['Y_p']) / (2*_dah) * sig_ah_bbn
sig_dh_bbn = abs(_r_hi_bbn2['D_H'] - _r_lo_bbn2['D_H']) / (2*_dah) * sig_ah_bbn * 1e5

# n=3 BBN: H_norm from σ(ah_2par)
sig_gm_2par = abs(bbn_geomean(N_SAT_FIX, ah_2par+_dah, al=al_2par) - bbn_geomean(N_SAT_FIX, ah_2par-_dah, al=al_2par)) / (2*_dah) * sig_ah_2par

# η_fit from D/H inversion; σ(η) from σ(D/H_obs) — DH_OBS and _eta_fit defined above
_dDH_deta = (compute_abundances(rf_2par, eta=_eta_fit+1e-12)['D_H'] - compute_abundances(rf_2par, eta=_eta_fit-1e-12)['D_H']) / 2e-12
sig_eta = SIG_DH_OBS / abs(_dDH_deta) if abs(_dDH_deta) > 1e-20 else float('nan')

# η_ΛCDM: invert D/H to match D/H^obs with H=H_std (ΛCDM second free param in BBN fit)
_eta_lcdm = brentq(lambda e: compute_abundances(lambda T: 1.0, eta=e)['D_H'] - DH_OBS, 5e-10, 10e-10)
_dDH_deta_lcdm = (compute_abundances(lambda T: 1.0, eta=_eta_lcdm+1e-12)['D_H'] -
                  compute_abundances(lambda T: 1.0, eta=_eta_lcdm-1e-12)['D_H']) / 2e-12
sig_eta_lcdm = SIG_DH_OBS / abs(_dDH_deta_lcdm) if abs(_dDH_deta_lcdm) > 1e-20 else float('nan')
abund_lcdm_bbn = compute_abundances(lambda T: 1.0, eta=_eta_lcdm)

# ── Y_p chi² for BBN block rows ───────────────────────────────────────────────
# η is fitted to D/H (chi²_DH = 0 by construction); only Y_p residual enters.
_c2_yp_2par     = ((abund_2par['Y_p']    - _YP_OBS) / _SIG_YP)**2
_c2_yp_lcdm_bbn = ((abund_lcdm_bbn['Y_p'] - _YP_OBS) / _SIG_YP)**2

# 1-par BBN row: eta fixed at ETA_STD, so Y_p and D/H both contribute to chi²_BBN
_c2_yp_1par_bbn = ((abund_1par_bbn['Y_p'] - _YP_OBS) / _SIG_YP)**2
_c2_dh_1par_bbn = ((abund_1par_bbn['D_H'] - DH_OBS) / SIG_DH_OBS)**2
_c2_bbn_1par    = _c2_yp_1par_bbn + _c2_dh_1par_bbn

# 1-par+η BBN: η absorbs D/H, so only Y_p residual contributes; dof_bbn=1
_c2_yp_1par_bbn_eta = ((abund_1par_bbn_eta['Y_p'] - _YP_OBS) / _SIG_YP)**2

MODEL  = "Hyp a-run"

# ── Table-1 output ────────────────────────────────────────────────────────────

def row_t1(constraint, model, n_free, param, zf, Ef, bbn_norm,
           yp="---", dh="---", k_model=1, k_lcdm=1, c2_bbn=0.0, c2_bbn_ref=0.0,
           dof_bbn=None):
    """Print one Table 1 row (14 columns).

    bbn_norm   : string shown in H_norm column (geometric mean H_model/H_std at BBN)
    yp         : Y_p value (float) or "---"
    dh         : D/H × 10^5 (float) or "---"
    k_model    : free parameters in the model (for ΔAIC)
    k_lcdm     : free parameters in the ΛCDM reference (for ΔAIC)
    c2_bbn     : χ²_BBN contribution for this model row
    c2_bbn_ref : χ²_BBN contribution for the ΛCDM reference (subtracted from ΔAIC)
    dof_bbn    : BBN degrees of freedom (None → print '---' for BBN chi2 columns)
    """
    c2b = chi2_bao(zf, Ef); c2s = chi2_sn(zf, Ef)
    c2j = c2b + c2s + c2_bbn
    nb = c2b / (N_BAO - k_model); ns = c2s / (N_SN - k_model)
    db = c2b - c2b_jnt; ds = c2s - c2s_jnt
    dj = (c2j - c2j_lcdm - c2_bbn_ref) + 2*(k_model - k_lcdm)
    yp_s = str(yp)
    dh_s = str(dh)
    if dof_bbn is not None:
        _c2b_nu = c2_bbn / dof_bbn
        nb_bbn  = f"{_c2b_nu:.0f}" if _c2b_nu >= 10 else f"{_c2b_nu:.2f}"
        _dc2b   = c2_bbn - c2_bbn_ref
        dc2_bbn = f"{_dc2b:+.0f}" if abs(_dc2b) >= 10 else f"{_dc2b:+.2f}"
    else:
        nb_bbn = "---"; dc2_bbn = "---"
    print(f"{constraint:<14} {model:<18} {n_free:>2}  {param:<24} "
          f"{bbn_norm:>7}  {yp_s:>5}  {dh_s:>5}  "
          f"{nb:>6.2f}  {ns:>6.2f}  {nb_bbn:>7}  "
          f"{db:>+6.1f}  {ds:>+6.1f}  {dc2_bbn:>7}  {dj:>+7.1f}")

hdr = (f"{'Constraint':<14} {'Model':<18}  n  {'Parameters':<24} "
       f"{'H_norm':>7}  {'Y_p':>5}  {'D/H':>5}  "
       f"{'χ²ν_BAO':>7}  {'χ²ν_SN':>7}  {'χ²ν_BBN':>7}  "
       f"{'Δχ²_BAO':>7}  {'Δχ²_SN':>7}  {'Δχ²_BBN':>7}  {'ΔAIC':>7}")
sep = "-" * len(hdr)
print(hdr)
print(f"  (D/H in units of 1e-5; H_norm = geometric mean H_model/H_std at T=0.07–0.10 MeV)")
print(f"  ΛCDM ref (BAO+SN): Ω_m={res_jnt.x:.4f}  |  ΛCDM ref (BBN): same Ω_m, k=2")
print(sep)

# ── BAO+SN block ──────────────────────────────────────────────────────────────
print("BAO+SN block:")

# Hyp. const-α (BAO+SN-optimal, k=1 vs ΛCDM k=1)
zfc_opt, Efc_opt = E_hippopede_const(alpha_const_opt)
row_t1("BAO+SN", "Hyp const-α", 1,
        f"α={_paren(alpha_const_opt, sig_alpha_const)}",
        zfc_opt, Efc_opt, _paren(gm_const_opt, sig_gm_const),
        yp="≈0" if _yp_const < 1e-6 else f"{_yp_const:.3f}",
        dh=_paren(_dh_const, sig_dh_const),
        k_model=1, k_lcdm=1)

# Hyp. a-run, 0-par (ah=1/2, N_SAT_FIX exponent both fixed; k=0 vs ΛCDM k=1)
zf0, Ef0 = E_hippopede(sat(N_SAT_FIX, 0.500))
row_t1("BAO+SN", MODEL, 0,
        "ah=1/2 (fix)",
        zf0, Ef0, f"~{bbn_geomean(N_SAT_FIX, 0.500):.1f}",
        yp=f"{float(abund_0par['Y_p']):.2f}",
        dh=f"{float(abund_0par['D_H']*1e5):.1f}",
        k_model=0, k_lcdm=1)

# Hyp. a-run, 1-par (ah free, joint BAO+SN optimal; k=1 vs ΛCDM k=1)
zf1, Ef1 = E_hippopede(sat(N_SAT_FIX, ah_bao_opt))
row_t1("BAO+SN", MODEL, 1,
        f"ah={_paren(ah_bao_opt, sig_ah_bao)}",
        zf1, Ef1, _paren(gm_ah_bao, sig_gm_bao),
        yp=f"{abund_1par_bao['Y_p']:.2f}±{sig_yp_bao:.2f}",
        dh=_paren(float(abund_1par_bao['D_H']*1e5), sig_dh_bao),
        k_model=1, k_lcdm=1)

# ΛCDM reference (k=1); BBN prediction at ETA_STD with H=H_std
zf_lcdm, Ef_lcdm = E_lcdm(res_jnt.x)
row_t1("BAO+SN", "ΛCDM (ref)", 1,
        f"Ω_m={_paren(res_jnt.x, sig_om_jnt)}",
        zf_lcdm, Ef_lcdm, "1.000",
        yp=f"{float(abund_sbbn['Y_p']):.3f}",
        dh=_paren(float(abund_sbbn['D_H']*1e5), float(SIG_DH_OBS*1e5)),
        k_model=1, k_lcdm=1)

print(sep)
print("BAO+SN+BBN block:")

# Hyp. a-run, 1-par BBN-constrained (ah free, gm=1; k=1 vs ΛCDM k=2; ΔAIC from Y_p+D/H residuals)
zfb, Efb = E_hippopede(sat(N_SAT_FIX, ah_bbn_opt))
row_t1("BAO+SN+BBN", MODEL, 1,
        f"ah={_paren(ah_bbn_opt, sig_ah_bbn)}",
        zfb, Efb, _paren(gm_ah_bbn, SIGMA_H_BBN),
        yp=_paren(float(abund_1par_bbn['Y_p']), sig_yp_bbn),
        dh=_paren(float(abund_1par_bbn['D_H']*1e5), sig_dh_bbn),
        k_model=1, k_lcdm=2,
        c2_bbn=_c2_bbn_1par, c2_bbn_ref=_c2_yp_lcdm_bbn,
        dof_bbn=2)

# Hyp. a-run, 2-par (al and ah fixed; k=3 vs ΛCDM k=2; chi2_BBN from Y_p residual)
row_t1("BAO+SN+BBN", MODEL, 3,
        f"al={_paren(al_2par, sig_al_2par)}, ah={_paren(ah_2par, sig_ah_2par)}, eta={_paren(_eta_fit*1e10, sig_eta*1e10)}e-10",
        zf_2par, Ef_2par, _paren(gm_2par, sig_gm_2par),
        yp=_paren(float(abund_2par['Y_p']), _SIG_YP),
        dh=_paren(float(abund_2par['D_H']*1e5), float(SIG_DH_OBS*1e5)),
        k_model=3, k_lcdm=2,
        c2_bbn=_c2_yp_2par, c2_bbn_ref=_c2_yp_lcdm_bbn,
        dof_bbn=1)

# ΛCDM reference (k=2: Ω_m + η; chi2_BBN from Y_p residual; by construction ΔAIC=0)
row_t1("BAO+SN+BBN", "ΛCDM (ref)", 2,
        f"Ω_m={_paren(res_jnt.x, sig_om_jnt)}, η={_paren(_eta_lcdm*1e10, sig_eta_lcdm*1e10)}e-10",
        zf_lcdm, Ef_lcdm, "1.000",
        yp=f"{float(abund_lcdm_bbn['Y_p']):.3f}",
        dh=_paren(float(abund_lcdm_bbn['D_H']*1e5), float(SIG_DH_OBS*1e5)),
        k_model=2, k_lcdm=2,
        c2_bbn=_c2_yp_lcdm_bbn, c2_bbn_ref=_c2_yp_lcdm_bbn,
        dof_bbn=1)

print(sep)
