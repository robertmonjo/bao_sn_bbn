"""
H0 from Cosmic Chronometers for the hyperconical n=3 model.

CC data: hippopede project, hz_curated_chronometers.csv
        (Moresco+12, Moresco+15, Moresco+16, Zhang+14, Ratsimbazafy+17,
         Borghi+22, Jiao+23, Tomasetti+23 — only cosmic_chronometer rows).

Method: marginalise H0 analytically at fixed (al, ah) from chi2_table joint fit.
  w_i = 1/sigma_i^2
  H0_opt = (E^T w H_obs) / (E^T w E)
  sigma(H0) = 1/sqrt(E^T w E)
  chi2_CC(H0) = sum_i (H_i - H0 * E_i)^2 / sigma_i^2

Then feeds H0_opt into h0_sensitivity logic to get the implied DAIC.
"""
import os
import sys
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import chi2_table as ct
from bbn_hyperconical import compute_abundances
from h0_sensitivity import bbn_rf_h0, gm_h0

# ── CC data ───────────────────────────────────────────────────────────────────
# Data lives in the sibling hippopede project; override with env var CC_DATA_FILE if needed.
_CC_DEFAULT = ROOT.parent / "hippopede" / "data" / "hz_background" / "hz_curated_chronometers.csv"
CC_FILE = Path(os.environ.get("CC_DATA_FILE", _CC_DEFAULT))
if not CC_FILE.exists():
    raise FileNotFoundError(
        f"CC data not found at {CC_FILE}.\n"
        "Set the CC_DATA_FILE environment variable to the path of hz_curated_chronometers.csv."
    )

def _load_cc(path=CC_FILE):
    z_cc, h_cc, s_cc = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["source_kind"].strip() != "cosmic_chronometer":
                continue
            z_cc.append(float(row["z"]))
            h_cc.append(float(row["h_km_s_mpc"]))
            sp = float(row["sigma_plus"])
            sm = float(row["sigma_minus"])
            s_cc.append(max(sp, sm))   # symmetrise: conservative (larger side)
    return np.array(z_cc), np.array(h_cc), np.array(s_cc)

z_cc, h_cc, s_cc = _load_cc()
w_cc = 1.0 / s_cc**2

# ── E(z_CC) interpolated from chi2_table grid ─────────────────────────────────
def _E_at_zcc(zf, Ef, z_obs):
    return np.array([float(np.interp(zi, zf, Ef)) for zi in z_obs])

def h0_from_cc(zf, Ef):
    """Analytically marginalised H0 and its 1-sigma uncertainty."""
    Ev = _E_at_zcc(zf, Ef, z_cc)
    wEE = float((w_cc * Ev**2).sum())
    wEH = float((w_cc * Ev * h_cc).sum())
    H0_opt = wEH / wEE
    sigma_H0 = 1.0 / np.sqrt(wEE)
    chi2_min = float(((h_cc - H0_opt * Ev)**2 * w_cc).sum())
    return H0_opt, sigma_H0, chi2_min

def chi2_cc_at(zf, Ef, H0_kms):
    Ev = _E_at_zcc(zf, Ef, z_cc)
    return float(((h_cc - H0_kms * Ev)**2 * w_cc).sum())

# ── ΛCDM reference CC fit ─────────────────────────────────────────────────────
zf_lc, Ef_lc = ct.E_lcdm(ct.res_jnt.x)
H0_lcdm, sig_lcdm, c2_lcdm = h0_from_cc(zf_lc, Ef_lc)

# ── Hyperconical n=3 CC fit ───────────────────────────────────────────────────
al_opt  = ct.al_2par
ah_opt  = ct.ah_2par
eta_opt = ct._eta_fit

H0_hyp, sig_hyp, c2_hyp = h0_from_cc(ct.zf_2par, ct.Ef_2par)

# ── BBN at H0_CC ──────────────────────────────────────────────────────────────
gm_cc, rf_cc = gm_h0(ct.N_SAT_FIX, ah_opt, al_opt, H0_hyp)
ab_cc = compute_abundances(rf_cc, eta=eta_opt)
yp_cc = float(ab_cc['Y_p'])
c2_yp_cc = ((yp_cc - ct._YP_OBS) / ct._SIG_YP)**2

# DAIC at H0_CC (vs. H0=70 reference in chi2_table):
# DC2_BAO and DC2_SN are unchanged (CC does not enter BAO/SN marginalisation).
c2bao = ct.chi2_bao(ct.zf_2par, ct.Ef_2par)
c2sn  = ct.chi2_sn(ct.zf_2par, ct.Ef_2par)
dc2bbn_cc = c2_yp_cc - ct._c2_yp_lcdm_bbn
dc2bao    = c2bao - ct.c2b_jnt
dc2sn     = c2sn  - ct.c2s_jnt
daic_cc   = dc2bao + dc2sn + dc2bbn_cc + 2*(3 - 2)

N_CC = len(z_cc)

print("=" * 62)
print("H0 from Cosmic Chronometers — hyperconical n=3")
print("=" * 62)
print(f"  CC points used : {N_CC}  (z = {z_cc.min():.3f} – {z_cc.max():.3f})")
print()
print(f"  H0_hyp  = {H0_hyp:.1f} ± {sig_hyp:.1f}  km/s/Mpc   chi2/dof = {c2_hyp:.1f}/{N_CC-1}")
print(f"  H0_ΛCDM = {H0_lcdm:.1f} ± {sig_lcdm:.1f}  km/s/Mpc   chi2/dof = {c2_lcdm:.1f}/{N_CC-1}")
print()
print(f"  BBN at H0_CC = {H0_hyp:.1f} km/s/Mpc:")
print(f"    gm_ratio = {gm_cc:.3f}")
print(f"    Y_p      = {yp_cc:.4f}  (obs 0.2453 ± 0.0034)")
print(f"    chi2_BBN = {c2_yp_cc:.3f}  (ref ΛCDM = {ct._c2_yp_lcdm_bbn:.3f})")
print()
print(f"  ΔAIC at H0_CC = {H0_hyp:.1f}:  {daic_cc:+.2f}")
print()
print("  Context (from h0_sensitivity.py, no re-optimisation):")
print("    H0=70  →  ΔAIC = −2.87   (assumed in BBN calculation)")
print(f"    H0={H0_hyp:.0f}  →  ΔAIC = {daic_cc:+.2f}   (CC-implied)")
print("    H0=67  →  ΔAIC = +1.77   (BAO β-implied, Planck r_s)")
print()
print("NOTE: (al, ah, eta) are fixed at the joint-fit optimum.")
print("      A full 4-par fit including H0 would shift all three.")
