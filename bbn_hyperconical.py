#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBN abundance calculator for a modified Hubble rate H_model(T)/H_std(T).

Physics: Kolb & Turner (1990), Mukhanov (2005) analytical approximations.
Accuracy target: ~5% in Y_p; D/H uses empirical power-law scaling.

Usage:
    python bbn_hyperconical.py
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ---- Physical constants -------------------------------------------------------
Q_np    = 1.2933      # MeV  neutron-proton mass difference
lam_n   = 1 / 878.4  # s^-1  neutron decay rate (PDG 2022)
hbar    = 6.582e-22   # MeV*s
M_pl    = 1.2209e22   # MeV  Planck mass (non-reduced)
ETA_STD = 6.11e-10    # baryon-to-photon ratio (Planck 2018)
D_H_STD = 2.527e-5    # SBBN reference D/H at ETA_STD
T_f_std = 0.808       # MeV  standard weak freeze-out calibration point


def g_star(T):
    """Effective relativistic d.o.f. for energy density at T (MeV)."""
    # Above e+e- mass: photons(2) + e+e-(3.5) + 3*nu(5.25) = 10.75
    # Below:           photons(2) + 3*nu reheated (effectively 3.91 as g_*S)
    return 10.75 if T > 0.511 else 3.91


def H_std_MeV(T):
    """Standard Hubble rate H(T) in MeV, radiation domination (Friedmann eq)."""
    return np.sqrt(4 * np.pi**3 * g_star(T) / 45) * T**2 / M_pl


# Weak n<->p rate: Gamma ~ T^5 (Fermi theory, all reaction channels).
# Calibrated so Gamma_weak(T_f_std) = H_std(T_f_std), giving T_f = 0.808 MeV
# for the standard case.
_Gamma_ref = H_std_MeV(T_f_std)


def Gamma_weak(T):
    """Effective weak n<->p rate in MeV (natural units)."""
    return _Gamma_ref * (T / T_f_std)**5


def compute_abundances(ratio_func, eta=ETA_STD):
    """
    Compute Y_p (He-4 mass fraction) and D/H for a modified Hubble rate.

    Parameters
    ----------
    ratio_func : callable   T_MeV -> H_model(T) / H_std(T)
    eta        : float      baryon-to-photon ratio

    Returns
    -------
    dict  with keys: Y_p, D_H, n_p_freeze, n_p_nuc, T_freeze, t_elapsed_s
    """
    # 1. Weak freeze-out temperature: Gamma_weak(T_f) = ratio(T_f) * H_std(T_f)
    def eq(T):
        return Gamma_weak(T) - ratio_func(T) * H_std_MeV(T)

    try:
        T_f = brentq(eq, 0.10, 10.0, xtol=1e-5)
    except ValueError:
        # Fallback from (T/T_f_std)^3 = ratio self-consistency
        T_f = T_f_std * ratio_func(T_f_std)**(1.0/3)

    # 2. n/p ratio at freeze-out (Boltzmann equilibrium)
    n_p_f = np.exp(-Q_np / T_f)

    # 3. Time from freeze-out to nucleosynthesis (deuterium bottleneck)
    #    From dT/dt = -H*T  =>  dt = dT / (H_mod * T)
    T_nuc = 0.066  # MeV  (Wagoner 1973; D bottleneck breaks here)
    split = [0.511] if T_f > 0.511 > T_nuc else []

    def dt_dT(T):
        H_mod = ratio_func(T) * H_std_MeV(T)   # MeV (natural units)
        return hbar / (H_mod * T)               # s / MeV

    t_elapsed, _ = quad(dt_dT, T_nuc, T_f, limit=300, points=split)

    # 4. n/p corrected for neutron decay during that window
    n_p_nuc = n_p_f * np.exp(-lam_n * t_elapsed)

    # 5. Y_p: essentially all neutrons captured into He-4 nuclei
    #    Y_p = 4 * n_He4 / (n_b) = 2*(n/p) / (1 + n/p)  [with n/p < 1]
    Y_p = 2 * n_p_nuc / (1 + n_p_nuc)

    # 6. D/H: power-law scaling from BBN network results (Cyburt 2004)
    #    D/H ~ eta^{-1.6} * (H/H_std)^{0.6}  evaluated at T_nuc
    #    Physical reason: faster expansion leaves more D unburned.
    r_nuc = ratio_func(T_nuc)
    D_H = D_H_STD * (eta / ETA_STD)**(-1.6) * r_nuc**0.6

    return dict(Y_p=Y_p, D_H=D_H,
                n_p_freeze=n_p_f, n_p_nuc=n_p_nuc,
                T_freeze=T_f, t_elapsed_s=t_elapsed)


# ---- Test with hyperconical model --------------------------------------------
if __name__ == "__main__":

    # Given H_model/H_std values at specific temperatures
    T_pts = np.array([0.07, 0.08, 0.09, 0.10, 0.50, 1.00])
    R_pts = np.array([1.073, 1.022, 0.970, 0.920, 0.471, 0.413])

    # Log-log linear interpolation (power law between nodes)
    _log_interp = interp1d(np.log(T_pts), np.log(R_pts),
                           kind='linear', fill_value='extrapolate')
    def ratio_model(T):
        return float(np.exp(_log_interp(np.log(np.clip(T, 1e-4, 100)))))

    results = {
        "Standard (ratio=1)": compute_abundances(lambda T: 1.0),
        "Hyperconical model": compute_abundances(ratio_model),
    }

    SEP  = "-" * 50
    SEP2 = "=" * 50
    for label, r in results.items():
        print(f"\n{SEP}")
        print(f"  {label}")
        print(f"{SEP}")
        print(f"  T_freeze       = {r['T_freeze']:.4f} MeV")
        print(f"  n/p freeze-out = {r['n_p_freeze']:.5f}")
        print(f"  t(T_f->T_nuc)  = {r['t_elapsed_s']:.2f} s")
        print(f"  n/p at nuc.    = {r['n_p_nuc']:.5f}")
        print(f"  Y_p            = {r['Y_p']:.5f}")
        print(f"  D/H            = {r['D_H']:.4e}")

    std = results["Standard (ratio=1)"]
    mod = results["Hyperconical model"]
    print(f"\n{SEP2}")
    print("  Comparison: hyperconical vs standard analytical")
    print(f"{SEP2}")
    print(f"  Y_p : {std['Y_p']:.5f} -> {mod['Y_p']:.5f}  "
          f"({(mod['Y_p']/std['Y_p'] - 1)*100:+.2f}%)")
    print(f"  D/H : {std['D_H']:.4e} -> {mod['D_H']:.4e}  "
          f"({(mod['D_H']/std['D_H'] - 1)*100:+.2f}%)")
    print()
    print("  Primordial observations:")
    print("    Y_p = 0.2470 +/- 0.0040  (Aver et al. 2021)")
    print("    D/H = 2.527e-5 +/- 0.030e-5  (Cooke et al. 2018)")
    print()
    print(f"  Note: analytical Y_p_std = {std['Y_p']:.4f} "
          f"(~{(std['Y_p']/0.247-1)*100:+.1f}% vs observed,")
    print("  expected from ~5% analytical approximation accuracy).")
    print()
    print("  Key physics: H_model < H_std at T~0.5-1 MeV (ratio~0.41-0.47)")
    print("  => later weak freeze-out => lower n/p => lower Y_p.")
    print("  H_model > H_std at T~0.07 MeV (ratio~1.07)")
    print("  => slightly faster nucleosynthesis => slightly higher D/H.")
