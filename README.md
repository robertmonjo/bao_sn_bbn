# bao_sn_bbn

Reproducibility repository for the paper:

**"BAO, SNe Ia, and BBN constraints on the hyperconical universe with a running cosmological index"**
R. Monjo (2026), submitted to Physics Letters B.

## Contents

| Path | Description |
|------|-------------|
| `data/Ardra/` | DESI DR1 BAO data (mean and covariance); from Adame et al. 2024 (arXiv:2404.03001, 2404.03004) |
| `data/gapp/` | Pantheon+ SNe Ia binned data (50 bins); from Brout et al. 2022 |
| `scripts/hyperconical_model.py` | Hyperconical model: `MonjoProjected` and `MonjoExtended` classes |
| `scripts/bao_fit_compute.py` | Computes BAO chi2 and writes fit results |
| `scripts/bao_fit_data.py` | Produces Figure 1 (BAO Hubble diagram with model fits) |
| `scripts/chi2_table.py` | Produces Table 1 (chi2, AIC comparisons across models and datasets) |
| `figures/bao_fit_data.png` | Figure 1 as published |

## Requirements

```
numpy scipy matplotlib
```

## Usage

```bash
python scripts/chi2_table.py          # print Table 1 to stdout
python scripts/bao_fit_data.py        # write figures/bao_fit_data.png
```

## Model

The running-alpha extension uses:

    alpha(z) = alpha_high - (alpha_high - alpha_low) / sqrt(1 + z)

with `alpha_low = 0.283` (late-time value, fixed) and `alpha_high = 0.422` (radiation-era
asymptote, constrained by BBN). The BBN normalisation is computed analytically by integrating
the effective expansion rate over the nucleosynthesis window T = 0.07–0.10 MeV.

## License

MIT License. Data files retain the license of their original sources (DESI DR1, Pantheon+).
