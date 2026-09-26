"""Compute best-fit BAO parameters for the projected hyperconical model.

Uses DESI DR1 compressed BAO data (galaxy + QSO bins, with and without Lyalpha).
Analytically marginalizes over the scale parameter beta = c/(H0 r_s).
Writes results to data/Ardra/hippopede_bao_fit_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
DATA = ROOT / "data" / "Ardra"

for extra in (str(SCRIPT_ROOT), str(SCRIPT_ROOT / "obsolete_bbn")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from analyze_hippopede_dipole_bbn import ExtendedProjectedHyperconical  # noqa: E402

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

DATA_FILE = DATA / "desi_dr1_bao_galqso_lya_mean.csv"
COV_FILE = DATA / "desi_dr1_bao_galqso_lya_cov.csv"
OUT_JSON = DATA / "hippopede_bao_fit_results.json"


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (z, values, observables, labels) from CSV in covariance-matrix order."""
    import csv
    rows = []
    with open(DATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    z = np.array([float(r["z"]) for r in rows])
    values = np.array([float(r["value"]) for r in rows])
    obs = [r["observable"] for r in rows]
    labels = [r["label"] for r in rows]
    return z, values, obs, labels


def load_cov() -> np.ndarray:
    cov = []
    with open(COV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cov.append([float(x) for x in line.split(",")])
    return np.array(cov)


def build_model_vector(z: np.ndarray, obs: list[str], alpha: float) -> np.ndarray:
    """Dimensionless model predictions (divided by beta)."""
    model = ExtendedProjectedHyperconical(alpha=alpha)

    z_fine = np.linspace(0.0, max(z.max() * 1.01, 0.1), 3000)
    E_fine, _ = model.e_and_q(z_fine)

    def comoving(z_max: float) -> float:
        mask = z_fine <= z_max
        if mask.sum() < 2:
            return 0.0
        return float(_trapz((1.0 / E_fine)[mask], z_fine[mask]))

    def E_at(z_val: float) -> float:
        return float(np.interp(z_val, z_fine, E_fine))

    vec = np.zeros(len(z))
    for i, (zi, oi) in enumerate(zip(z, obs)):
        if oi == "DM_over_rs":
            vec[i] = comoving(zi)
        elif oi == "DH_over_rs":
            vec[i] = 1.0 / E_at(zi)
        elif oi == "DV_over_rs":
            dm = comoving(zi)
            dh = 1.0 / E_at(zi)
            vec[i] = (zi * dm**2 * dh) ** (1.0 / 3.0)
    return vec


def chi2_marginalized(alpha: float, d: np.ndarray, C_inv: np.ndarray,
                      obs: list[str], z: np.ndarray) -> tuple[float, float]:
    """chi2 with beta analytically marginalized. Returns (chi2_min, beta_best)."""
    m = build_model_vector(z, obs, alpha)
    mTCm = m @ C_inv @ m
    mTCd = m @ C_inv @ d
    dTCd = d @ C_inv @ d
    beta = mTCd / mTCm
    chi2 = dTCd - mTCd**2 / mTCm
    return float(chi2), float(beta)


def fit_lcdm(d: np.ndarray, C_inv: np.ndarray, obs: list[str],
             z: np.ndarray) -> tuple[float, float, float]:
    """Best-fit flat LambdaCDM (Omega_m free, beta marginalized). Returns (omega_m, chi2, beta)."""
    def neg_chi2(omega_m):
        E_vec = np.array([
            np.sqrt(omega_m * (1 + zi)**3 + (1 - omega_m))
            if oi == "DH_over_rs" else
            (  # DM or DV computed analytically
                None
            )
            for zi, oi in zip(z, obs)
        ])
        # Compute model vector for LambdaCDM
        z_fine = np.linspace(0, z.max() * 1.01, 3000)
        E_fine = np.sqrt(omega_m * (1 + z_fine)**3 + (1 - omega_m))

        def comoving_lcdm(z_max):
            mask = z_fine <= z_max
            if mask.sum() < 2:
                return 0.0
            return float(_trapz((1.0 / E_fine)[mask], z_fine[mask]))

        def Elcdm(zv):
            return float(np.sqrt(omega_m * (1 + zv)**3 + (1 - omega_m)))

        m = np.zeros(len(z))
        for i, (zi, oi) in enumerate(zip(z, obs)):
            if oi == "DM_over_rs":
                m[i] = comoving_lcdm(zi)
            elif oi == "DH_over_rs":
                m[i] = 1.0 / Elcdm(zi)
            elif oi == "DV_over_rs":
                dm = comoving_lcdm(zi)
                dh = 1.0 / Elcdm(zi)
                m[i] = (zi * dm**2 * dh) ** (1.0 / 3.0)

        mTCm = m @ C_inv @ m
        mTCd = m @ C_inv @ d
        dTCd = d @ C_inv @ d
        beta = mTCd / mTCm
        chi2 = dTCd - mTCd**2 / mTCm
        return chi2

    res = minimize_scalar(neg_chi2, bounds=(0.1, 0.6), method="bounded")
    omega_m_best = res.x
    chi2_best = res.fun
    # recover beta
    E_fine = np.sqrt(omega_m_best * (1 + np.linspace(0, z.max() * 1.01, 3000))**3
                     + (1 - omega_m_best))
    z_fine = np.linspace(0, z.max() * 1.01, 3000)

    def comoving_lcdm(z_max):
        mask = z_fine <= z_max
        if mask.sum() < 2:
            return 0.0
        return float(_trapz((1.0 / E_fine)[mask], z_fine[mask]))

    def Elcdm(zv):
        return float(np.sqrt(omega_m_best * (1 + zv)**3 + (1 - omega_m_best)))

    m = np.zeros(len(z))
    for i, (zi, oi) in enumerate(zip(z, obs)):
        if oi == "DM_over_rs":
            m[i] = comoving_lcdm(zi)
        elif oi == "DH_over_rs":
            m[i] = 1.0 / Elcdm(zi)
        elif oi == "DV_over_rs":
            dm = comoving_lcdm(zi)
            dh = 1.0 / Elcdm(zi)
            m[i] = (zi * dm**2 * dh) ** (1.0 / 3.0)

    beta = (m @ C_inv @ d) / (m @ C_inv @ m)
    return float(omega_m_best), float(chi2_best), float(beta)


def run_fit(d: np.ndarray, C_inv: np.ndarray, obs: list[str],
            z: np.ndarray, label: str) -> dict:
    """Scan alpha in [0.1, 0.9] and find best-fit."""
    alpha_grid = np.linspace(0.10, 0.90, 161)
    chi2_grid = []
    beta_grid = []
    print(f"  Scanning alpha for {label} ({len(d)} points)...")
    for a in alpha_grid:
        c2, b = chi2_marginalized(a, d, C_inv, obs, z)
        chi2_grid.append(c2)
        beta_grid.append(b)
    chi2_arr = np.array(chi2_grid)
    i_min = int(np.argmin(chi2_arr))
    alpha_best = float(alpha_grid[i_min])
    chi2_best = float(chi2_arr[i_min])
    beta_best = float(beta_grid[i_min])

    # Refine with scipy
    res = minimize_scalar(
        lambda a: chi2_marginalized(a, d, C_inv, obs, z)[0],
        bounds=(max(0.10, alpha_best - 0.05), min(0.90, alpha_best + 0.05)),
        method="bounded",
    )
    alpha_best = float(res.x)
    chi2_best_refined, beta_best = chi2_marginalized(alpha_best, d, C_inv, obs, z)
    chi2_best = chi2_best_refined

    # 1σ uncertainty: Deltachi2 = 1 — find exact roots via brentq
    chi2_1sig = chi2_best + 1.0

    def chi2_profile(a):
        return chi2_marginalized(a, d, C_inv, obs, z)[0]

    # Lower bound: scan left from refined minimum; bracket the first crossing
    alpha_lo = 0.10
    for i in range(len(alpha_grid) - 2, -1, -1):
        if alpha_grid[i + 1] > alpha_best:
            continue
        if chi2_arr[i] >= chi2_1sig and chi2_arr[i + 1] < chi2_1sig:
            alpha_lo = brentq(lambda a: chi2_profile(a) - chi2_1sig,
                              alpha_grid[i], alpha_grid[i + 1], xtol=1e-6)
            break

    # Upper bound: scan right from refined minimum; bracket the first crossing
    alpha_hi = 0.90
    for i in range(len(alpha_grid) - 1):
        if alpha_grid[i] < alpha_best:
            continue
        if chi2_arr[i] < chi2_1sig and chi2_arr[i + 1] >= chi2_1sig:
            alpha_hi = brentq(lambda a: chi2_profile(a) - chi2_1sig,
                              alpha_grid[i], alpha_grid[i + 1], xtol=1e-6)
            break

    n_data = len(d)
    n_params = 2  # alpha and beta (beta marginalized analytically)
    dof = n_data - n_params
    aic = chi2_best + 2 * n_params

    print(f"    alpha = {alpha_best:.5f} + {alpha_hi - alpha_best:.5f} - {alpha_best - alpha_lo:.5f}")
    print(f"    beta = {beta_best:.5f}")
    print(f"    chi2 = {chi2_best:.4f}  (dof={dof})  AIC = {aic:.4f}")

    return {
        "label": label,
        "n_data": n_data,
        "alpha_best": alpha_best,
        "alpha_1sigma_low": alpha_lo,
        "alpha_1sigma_high": alpha_hi,
        "beta_best": beta_best,
        "chi2_best": chi2_best,
        "chi2_dof": chi2_best / dof,
        "AIC": aic,
    }


# ── main ───────────────────────────────────────────────────────────────────
z_all, d_all, obs_all, labels_all = load_data()
cov_all = load_cov()
C_inv_all = np.linalg.inv(cov_all)

# 10-point fit: exclude Lyalpha (last 2 rows/cols)
mask_no_lya = np.array([lbl != "Lya" for lbl in labels_all])
idx_no_lya = np.where(mask_no_lya)[0]
z_10 = z_all[mask_no_lya]
d_10 = d_all[mask_no_lya]
obs_10 = [obs_all[i] for i in idx_no_lya]
cov_10 = cov_all[np.ix_(idx_no_lya, idx_no_lya)]
C_inv_10 = np.linalg.inv(cov_10)

# 12-point fit: all data
z_12 = z_all
d_12 = d_all
obs_12 = obs_all
C_inv_12 = C_inv_all

print("=== BAO fit: projected hyperconical model vs DESI DR1 ===\n")
result_10 = run_fit(d_10, C_inv_10, obs_10, z_10, "10-point (no Lya)")
result_12 = run_fit(d_12, C_inv_12, obs_12, z_12, "12-point (with Lya)")

print("\n=== LambdaCDM reference fits ===")
omega_m_10, chi2_lcdm_10, beta_lcdm_10 = fit_lcdm(d_10, C_inv_10, obs_10, z_10)
print(f"  10-pt: Omega_m = {omega_m_10:.4f}, chi2 = {chi2_lcdm_10:.4f}, beta = {beta_lcdm_10:.4f}")

omega_m_12, chi2_lcdm_12, beta_lcdm_12 = fit_lcdm(d_12, C_inv_12, obs_12, z_12)
print(f"  12-pt: Omega_m = {omega_m_12:.4f}, chi2 = {chi2_lcdm_12:.4f}, beta = {beta_lcdm_12:.4f}")

output = {
    "dataset": "DESI DR1 compressed BAO",
    "10point_no_lya": {
        **result_10,
        "lcdm_omega_m": omega_m_10,
        "lcdm_chi2": chi2_lcdm_10,
        "lcdm_beta": beta_lcdm_10,
        "lcdm_AIC": chi2_lcdm_10 + 2 * 2,
        "delta_AIC_hyp_minus_lcdm": result_10["AIC"] - (chi2_lcdm_10 + 2 * 2),
    },
    "12point_with_lya": {
        **result_12,
        "lcdm_omega_m": omega_m_12,
        "lcdm_chi2": chi2_lcdm_12,
        "lcdm_beta": beta_lcdm_12,
        "lcdm_AIC": chi2_lcdm_12 + 2 * 2,
        "delta_AIC_hyp_minus_lcdm": result_12["AIC"] - (chi2_lcdm_12 + 2 * 2),
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {OUT_JSON}")

print("\n=== Comparison with manuscript ===")
print(f"  Manuscript 10-pt: alpha=0.375, beta=28.85, chi2=12.39, DeltaAIC=0.68")
print(f"  This script 10-pt: alpha={result_10['alpha_best']:.5f}, beta={result_10['beta_best']:.5f}, chi2={result_10['chi2_best']:.4f}, DeltaAIC={output['10point_no_lya']['delta_AIC_hyp_minus_lcdm']:.4f}")
print(f"  JSON versionated:  alpha=0.36445, beta=29.107, chi2=14.311, DeltaAIC=1.555 (12-pt)")
print(f"  This script 12-pt: alpha={result_12['alpha_best']:.5f}, beta={result_12['beta_best']:.5f}, chi2={result_12['chi2_best']:.4f}, DeltaAIC={output['12point_with_lya']['delta_AIC_hyp_minus_lcdm']:.4f}")
