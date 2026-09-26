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

# Find optimal n for BAO-only and joint
res_n_bao = minimize_scalar(
    lambda n: chi2_bao(*E_hippopede(sat(n))), bounds=(0.05, 0.8), method="bounded")
n_bao_opt = res_n_bao.x
res_n_joint = minimize_scalar(
    lambda n: chi2_joint(*E_hippopede(sat(n))), bounds=(0.05, 0.8), method="bounded")
n_joint_opt = res_n_joint.x
print(f"\nSat BAO-optimal  n={n_bao_opt:.4f}, chi2_BAO={chi2_bao(*E_hippopede(sat(n_bao_opt))):.4f}")
print(f"Sat joint-optimal n={n_joint_opt:.4f}, chi2_joint={res_n_joint.fun:.4f}")

# BBN geomean from analyze_hippopede_dipole_bbn.py (z~3e8, numerically verified):
#   n≈0.353 → geomean ≈ 0.14 (interpolated from n=0.355 → 0.141)
#   n=0.468 → geomean = 1.002 (calibration zero)
gm_bao = 0.14
gm_bbn = 1.002

MODEL = "Hyperconical α-run"

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
row4("BAO",        MODEL, f"n={n_bao_opt:.3f}", zf, Ef, f"~{gm_bao:.2f}", k_lcdm=1)

zf, Ef = E_hippopede(sat(n_joint_opt))
row4("BAO+SN",     MODEL, f"n={n_joint_opt:.3f}", zf, Ef, "<0.14", k_lcdm=1)

zf, Ef = E_hippopede(sat(0.468))
row4("BAO+BBN",    MODEL, "n=0.468", zf, Ef, f"{gm_bbn:.3f}", k_lcdm=2)

zf, Ef = E_hippopede(sat(0.468))
row4("BAO+SN+BBN", MODEL, "n=0.468", zf, Ef, f"{gm_bbn:.3f}", k_lcdm=2)

zf_lcdm, Ef_lcdm = E_lcdm(res_jnt.x)
print("-"*110)
row4("BAO+SN",     "ΛCDM (1-par ref)", f"Ω_m={res_jnt.x:.3f}", zf_lcdm, Ef_lcdm, "1", k_model=1, k_lcdm=1)
# 2-par LCDM reference: k_model=k_lcdm=2 so DAIC=0 (self-reference for BBN group)
# Omega_r is calibrated by BBN (not fitted to BAO/SN); chi2 values same as 1-par (Omega_r negligible at DESI z)
row4("BAO+SN+BBN", "ΛCDM (2-par ref)", f"Ω_m={res_jnt.x:.3f}", zf_lcdm, Ef_lcdm, "1", k_model=2, k_lcdm=2)
