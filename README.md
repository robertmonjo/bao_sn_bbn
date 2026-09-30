# Coasting cosmology compatible with BAO and nucleosynthesis

Reproducibility repository for the paper submitted to *Physics Letters B*.

The hyperconical universe (coasting expansion $a(t)\propto t$) is fitted to
DESI DR1 BAO, Pantheon+ SNe Ia, and primordial nucleosynthesis observations.
A two-parameter running projection index $\alpha(z)$ reproduces the observed
helium fraction $Y_p$ and deuterium-to-hydrogen ratio D/H.

## Requirements

Python 3.9 or later with:

```
numpy >= 1.24
scipy >= 1.11
matplotlib >= 3.7
```

Install with:

```bash
pip install -r requirements.txt
```

## Reproducing Table 1

```bash
python scripts/chi2_table.py
```

This script reads the data files in `data/`, calls the BBN calculator
`bbn_hyperconical.py` and the hyperconical model `scripts/hyperconical_model.py`,
and prints Table 1 of the paper to stdout.

## Repository structure

```
bbn_hyperconical.py               BBN abundance calculator (Kolb & Turner 1990)
scripts/
  hyperconical_model.py           Hyperconical E(z) model
  chi2_table.py                   Reproduces Table 1 (BAO + SN + BBN fits)  [canonical]
  joint_uncertainties.py          Joint 1σ uncertainties via Hessian         [canonical]
  bao_fit_compute.py              BAO-only and joint ΛCDM fits; writes JSON  [canonical]
  h0_robust.py                    Robustness of ΔAIC across H0 values        [canonical]
  bao_fit_data.py                 BAO data preparation and figures
  h0_sensitivity.py               H0 sensitivity at fixed (al, ah)
  chi2_cc.py                      Cosmic chronometer H0 constraint           [exploratory]
data/
  Ardra/                          DESI DR1 BAO compressed statistics
  gapp/                           Pantheon+ binned distance moduli
main.tex                          Manuscript
main-ref.bib                      Bibliography
```

Scripts marked **[canonical]** reproduce numbers in the paper. Scripts marked
**[exploratory]** depend on data not included in this repository (see inline
comments) and are provided for transparency.

## Data sources

- **BAO:** DESI Collaboration, JCAP 2025, 02, 021
  ([DOI 10.1088/1475-7516/2025/02/021](https://doi.org/10.1088/1475-7516/2025/02/021))
- **SN Ia:** Pantheon+ (Scolnic et al. 2022, ApJ 938, 113;
  Brout et al. 2022, ApJ 938, 110)
- **Helium:** Aver, Olive & Skillman 2021, JCAP 2021, 03, 027
- **Deuterium:** Schöneberg 2024, JCAP 2024, 06, 006
- **CMB temperature:** Fixsen 2009, ApJ 707, 916

## Citation

If you use this code, please cite the accompanying paper (reference to be
added upon acceptance).

<!-- MEMORIA-ESTRUCTURADA:INICI
## memory/ — memoria estructurada (LLEGIR PRIMER)

Aquest projecte manté el seu estat de treball a **memory/**, no a la
conversa. Qualsevol agent (Claude Code, Codex, ...) ha de:

1. Obrir **memory/INDEX.md** abans de respondre qualsevol prompt i
   anar al .md que corresponga (l'entrada 0 es el protocol complet:
   memory/README.md).
2. Actualitzar memory/ESTAT.md, memory/PENDENTS.md i el .md tematic despres de
   cada pas rellevant. Si no consta a memory/, no s'ha fet.
3. Abans de qualsevol compactacio de context, bolcar-hi tot el que encara nomes viu a la
   conversa.

La conversa es volatil i es compacta; memory/ no. No preguntes a l'usuari res que ja
estiga escrit aci.
MEMORIA-ESTRUCTURADA:FI -->
