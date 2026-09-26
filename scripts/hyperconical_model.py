"""Standalone projected hyperconical model classes (no GaPP dependency).

Extracted from:
  hippopede/scripts/plot_hippopede_qz_double_panel_with_gapp.py (MonjoProjectedHyperconical)
  hippopede/scripts/obsolete_bbn/analyze_hippopede_dipole_bbn.py (ExtendedProjectedHyperconical)
"""

from __future__ import annotations
import numpy as np


class MonjoProjectedHyperconical:
    def __init__(self, alpha=0.283, k=1.0):
        self.alpha = alpha
        self.k = k
        self._prepare_lookup()

    def _ttp(self, r):
        T = 1.0
        t = 1.0
        k = self.k
        return t * np.sqrt(((T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) / (np.sqrt((-k * r**2 + T**2) / T**2) * T**2 * k)))

    def _grrp(self, r):
        T = 1.0
        t = 1.0
        k = self.k
        return -t**2 * (((T**2 * (k - 2) + k * r**2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) / ((T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) * (-k * r**2 + T**2)))

    def _f(self, x):
        return -np.sqrt(-self._grrp(x)) / self._ttp(x)

    def _prepare_lookup(self):
        k = self.k
        self.mx = np.sqrt((1.0 - (1.0 - k / 2.0) ** 2) / k)
        seq0 = np.linspace(0.0, 10.8, 18000)
        x_linear = np.linspace(0.0, self.mx, 4000)
        x_cluster = self.mx * (1.0 - 10.0 ** (-seq0))
        x_pos = np.unique(np.concatenate([x_linear, x_cluster]))
        x_pos = x_pos[(x_pos >= 0.0) & (x_pos < self.mx)]
        fx_pos = self._f(x_pos)
        dx = np.diff(x_pos)
        integ = np.zeros_like(x_pos)
        integ[1:] = np.cumsum(0.5 * (fx_pos[1:] + fx_pos[:-1]) * dx)
        I_pos = -integ
        self.x_grid = np.concatenate([-x_pos[:0:-1], x_pos])
        self.I_grid = np.concatenate([-I_pos[:0:-1], I_pos])
        self.f_grid = np.concatenate([fx_pos[:0:-1], fx_pos])
        ux = np.sqrt(1.0 / k - self.mx**2)
        self.y0 = np.arctan2(self.mx, ux)

    def x_from_lz(self, lz):
        return np.interp(lz, self.I_grid, self.x_grid, left=self.x_grid[0], right=self.x_grid[-1])

    def dxdLZ(self, x):
        f = np.interp(x, self.x_grid, self.f_grid)
        return -1.0 / f

    def dinvll_dx(self, x):
        k = self.k
        u = np.sqrt(np.maximum(1.0 / k - x**2, 1e-14))
        y = np.arctan2(x, u)
        dy_dx = 1.0 / u
        s = y / 2.0
        g = np.maximum(1.0 - y / self.y0, 1e-12)
        t = s / g**self.alpha
        dt_dx = dy_dx * (0.5 / g**self.alpha + s * self.alpha / self.y0 * g ** (-self.alpha - 1.0))
        return 2.0 * dt_dx / (1.0 + t**2)


class ExtendedProjectedHyperconical(MonjoProjectedHyperconical):
    """Same Monjo-type map but with a denser boundary lookup for high-z tests."""

    def _prepare_lookup(self):
        k = self.k
        self.mx = np.sqrt((1.0 - (1.0 - k / 2.0) ** 2) / k)
        seq0 = np.linspace(0.0, 35.0, 250000)
        x_linear = np.linspace(0.0, self.mx, 10000)
        x_cluster = self.mx * (1.0 - 10.0 ** (-seq0))
        x_pos = np.unique(np.concatenate([x_linear, x_cluster]))
        x_pos = x_pos[(x_pos >= 0.0) & (x_pos < self.mx)]
        fx_pos = self._f(x_pos)
        dx = np.diff(x_pos)
        integ = np.zeros_like(x_pos)
        integ[1:] = np.cumsum(0.5 * (fx_pos[1:] + fx_pos[:-1]) * dx)
        I_pos = -integ
        self.x_grid = np.concatenate([-x_pos[:0:-1], x_pos])
        self.I_grid = np.concatenate([-I_pos[:0:-1], I_pos])
        self.f_grid = np.concatenate([fx_pos[:0:-1], fx_pos])
        ux = np.sqrt(1.0 / k - self.mx**2)
        self.y0 = np.arctan2(self.mx, ux)

    def projected_hubble_unnormalized(self, z):
        z = np.asarray(z, dtype=float)
        lz = np.log1p(z)
        x = self.x_from_lz(lz)
        dr_dz = self.dinvll_dx(x) * self.dxdLZ(x) / (1.0 + z)
        return 1.0 / dr_dz
