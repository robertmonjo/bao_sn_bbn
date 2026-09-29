"""
Joint parameter uncertainties for the 3-par hyperconical fit.

Computes the numerical Hessian of chi2(al, ah, eta) at the joint optimum,
inverts it to get the covariance matrix, and reports 1-sigma uncertainties
and correlations.

Parameterisation: x = [al, ah, eta10] where eta10 = eta * 1e10.
All three parameters are then O(0.1)–O(10), making finite differences commensurable.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv, eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import chi2_table as ct

# ── Scaled chi2 in [al, ah, eta10] space ─────────────────────────────────────

def chi2_scaled(x):
    al, ah, eta10 = x
    eta = eta10 * 1e-10
    if al < 0.05 or ah <= al + 0.01 or ah > 0.90 or eta10 < 0.3 or eta10 > 15.0:
        return 1e10
    try:
        zf_, Ef_ = ct.E_hippopede_al(ct.sat(ct.N_SAT_FIX, ah, al), al=al)
        c2bs = ct.chi2_bao(zf_, Ef_) + ct.chi2_sn(zf_, Ef_)
        rf_ = ct.bbn_ratio_func(ct.N_SAT_FIX, ah, al=al)
        from bbn_hyperconical import compute_abundances
        ab_ = compute_abundances(rf_, eta=eta)
        c2_yp = ((ab_['Y_p'] - ct._YP_OBS) / ct._SIG_YP) ** 2
        c2_dh = ((ab_['D_H'] - ct.DH_OBS) / ct.SIG_DH_OBS) ** 2
        return c2bs + c2_yp + c2_dh
    except Exception:
        return 1e10

# ── Re-optimise in scaled space ───────────────────────────────────────────────

x0_scaled = [ct.al_2par, ct.ah_2par, ct._eta_fit * 1e10]
res = minimize(chi2_scaled, x0_scaled, method='Nelder-Mead',
               options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 20000})
if not res.success:
    print(f"WARNING: joint Nelder-Mead did not converge ({res.nit} iter): {res.message}", file=sys.stderr)
al_opt, ah_opt, eta10_opt = res.x
chi2_min = res.fun

print("=" * 60)
print("Joint 3-par optimum (scaled Nelder-Mead)")
print("=" * 60)
print(f"  al     = {al_opt:.6f}")
print(f"  ah     = {ah_opt:.6f}")
print(f"  eta10  = {eta10_opt:.5f}  (eta = {eta10_opt*1e-10:.4e})")
print(f"  chi2   = {chi2_min:.5f}")

# ── Numerical Hessian via central differences ─────────────────────────────────
# H_ij = d2(chi2)/dx_i dx_j at x_opt
# eps chosen ~5% of expected 1-sigma for each parameter

x_opt = np.array([al_opt, ah_opt, eta10_opt])
eps   = np.array([5e-4, 5e-5, 5e-3])   # ~5% of expected σ

n = 3
H = np.zeros((n, n))
f0 = chi2_scaled(x_opt)

for i in range(n):
    # diagonal: central 2nd derivative
    xp = x_opt.copy(); xp[i] += eps[i]
    xm = x_opt.copy(); xm[i] -= eps[i]
    H[i, i] = (chi2_scaled(xp) - 2.0 * f0 + chi2_scaled(xm)) / eps[i]**2
    # off-diagonal: mixed central 2nd derivative
    for j in range(i + 1, n):
        xpp = x_opt.copy(); xpp[i] += eps[i]; xpp[j] += eps[j]
        xpm = x_opt.copy(); xpm[i] += eps[i]; xpm[j] -= eps[j]
        xmp = x_opt.copy(); xmp[i] -= eps[i]; xmp[j] += eps[j]
        xmm = x_opt.copy(); xmm[i] -= eps[i]; xmm[j] -= eps[j]
        H[i, j] = H[j, i] = (
            chi2_scaled(xpp) - chi2_scaled(xpm)
            - chi2_scaled(xmp) + chi2_scaled(xmm)
        ) / (4.0 * eps[i] * eps[j])

# Check positive definiteness
eigvals, _ = eigh(H)
print(f"\nHessian eigenvalues: {eigvals}")
if np.any(eigvals <= 0):
    print("WARNING: Hessian is not positive definite — optimum may not be a minimum")

# Covariance matrix: C = 2 * H^{-1}
# (from log L = -chi2/2 => Fisher info F = H/2 => Cov = F^{-1} = 2*H^{-1})
C = 2.0 * inv(H)

# Unscaled variances: sigma(eta) = sigma(eta10) * 1e-10
sigma_al   = np.sqrt(C[0, 0])
sigma_ah   = np.sqrt(C[1, 1])
sigma_eta10 = np.sqrt(C[2, 2])
sigma_eta  = sigma_eta10 * 1e-10

# Correlation matrix
D_inv = np.diag(1.0 / np.sqrt(np.diag(C)))
corr  = D_inv @ C @ D_inv

print()
print("=" * 60)
print("Joint 1-sigma uncertainties (Hessian covariance)")
print("=" * 60)
print(f"  sigma(al)    = {sigma_al:.5f}  -> al    = {al_opt:.4f} +/- {sigma_al:.4f}")
print(f"  sigma(ah)    = {sigma_ah:.5f}  -> ah    = {ah_opt:.5f} +/- {sigma_ah:.5f}")
print(f"  sigma(eta10) = {sigma_eta10:.4f}   -> eta10 = {eta10_opt:.4f} +/- {sigma_eta10:.4f}")
print(f"  sigma(eta)   = {sigma_eta:.2e}  -> eta   = {eta10_opt*1e-10:.4e} +/- {sigma_eta:.2e}")
print()
print("Correlation matrix (al, ah, eta10):")
for row in corr:
    print("  " + "  ".join(f"{v:+.3f}" for v in row))
print()
print("Covariance matrix (al, ah, eta10):")
for row in C:
    print("  " + "  ".join(f"{v:+.2e}" for v in row))
print()

# Summary for paper table
def _paren(v, s, digits=None):
    """Format v(s) in last-digit notation."""
    if digits is None:
        import math
        exp = math.floor(math.log10(abs(s))) if s > 0 else -4
        digits = max(0, -exp + 1)
    return f"{v:.{digits}f}({int(round(s * 10**digits))})"

print("=" * 60)
print("Table values (last-digit notation)")
print("=" * 60)
print(f"  al    = {_paren(al_opt,    sigma_al)}")
print(f"  ah    = {_paren(ah_opt,    sigma_ah)}")
print(f"  eta10 = {_paren(eta10_opt, sigma_eta10)}")

# Compare with current (separate) uncertainties
print()
print("Current (separate) uncertainties in main.tex:")
print(f"  al    = 0.255(16)   sigma_al   = 0.016")
print(f"  ah    = 0.4379(5)   sigma_ah   = 0.0005")
print(f"  eta   = 7.725(57)   sigma_eta10= 0.057")
