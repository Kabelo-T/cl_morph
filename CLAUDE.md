# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase analyzing 305 galaxy clusters simulated with GadgetX-3k (full-physics, standard
dark matter), each viewed along 3 lines-of-sight (xy, yz, zx projections). The science goal is
relating a cluster's mass accretion history (MAH) to its dynamical state and stellar-mass
morphology (measured with `statmorph` on projected stellar mass maps). There is a `paper/`
directory (LaTeX, `aastex7`) that this analysis supports.

Environment is a conda env (`environment.yml`, env name `cluster_morph`) — no build step; this is
a library + scripts + notebooks project, not a package that gets installed or shipped.

## Commands

There is no test runner, linter, or build configured (no pytest/tox/Makefile). "Running" the
project means executing a script or notebook end to end.

Run scripts from the repo root (they use `utils` as a package via relative imports and read
paths relative to cwd):

```bash
python predict.py --features sm    # or --features ds
python run_corrs.py                # thin wrapper around utils.corrs.main()
python dynamical_state.py
python measure_morphs.py --map_dir <dir> [--annulus] [--r1 1.0] [--r2 50.0] [--out_dir NAME] [--vir]
```

`measure_morphs.py` is the statmorph pipeline: it multiprocesses over FITS mass maps in
`--map_dir`, computes non-parametric morphology + Sersic fits per map, and writes a CSV to
`results/<out_dir>/`. This is the slow, expensive step — everything downstream reads its CSV
output rather than recomputing morphology.

Most exploratory/plotting work happens in Jupyter notebooks at the repo root (`mah.ipynb`,
`multicam.ipynb`, `score.ipynb`, `dynamical_state.ipynb`, `cluster_morphs.ipynb`, etc.) that import
the same `utils` modules the scripts use — treat the `.py` scripts as the canonical, de-duplicated
logic and the notebooks as callers/exploration on top of it.

## Architecture

**`utils/file.py`** — all disk I/O. Loads AHF halo catalogs and MAHs from
`data/AHF_HaloHistory/NewMDCLUSTER_XXXX_halo_*.dat`, dynamical-state parameters from
`data/G3X_progenitors/DS_G3X_snap_NNN_center-cluster_progenitors.txt`, splashback/truncation radii
from `data/cluster_rsp_rtrunc.npz`, magnitude-gap data from `data/mag_diff_GadgetX_3k.csv`, and
statmorph morphology CSVs from `results/{xy,yz,zx}/*.csv`. Halo/cluster IDs are parsed out of
filenames via `find_id()` and used as the DataFrame index throughout — this ID is the join key
across every other module. Note: several function defaults here (e.g. `get_mah_all`'s
`mah_dir='data/gadgetx3k/GadgetX'`, `get_ahf_all`'s default, `get_ds`/`get_m14`'s hardcoded
`data/gadgetx3k/...` paths) reference a `data/gadgetx3k/` prefix that doesn't exist in the current
`data/` layout — pass explicit paths matching the real layout above rather than trusting the
defaults.

**`utils/data.py`** — MAH math, no I/O. Interpolates each cluster's M(a)/M(a=1) history onto a
common set of scale factors (`interp_ma`, PCHIP interpolation), and inverts m(a) into a(m) (scale
factor at which a cluster first reaches a given mass fraction — `build_am`, following the
Rockstar/MultiCAM convention documented in its docstring). `prepare_ma_corrs`/`prepare_am_corrs`
reshape per-cluster MAHs into per-timestep DataFrames merged against a dynamical-state/morphology
table, keyed by cluster ID, for correlation analysis.

**`utils/corrs.py`** — wraps the external `multicam` package (a sibling project at
`~/multicam`, not pip-installed into this env's site-packages — imports will fail unless that
package is importable, e.g. via `pip install -e ~/multicam` or `PYTHONPATH`) to fit
`MultiCAM` (ridge regression) predicting MAH from dynamical-state/morphology features, via
repeated random train/test splits (Monte Carlo CV), reporting Spearman correlation, RMSE, R² per
target time/mass bin.

**`utils/score.py`** — turns fitted weights (or plain Spearman correlations, `build_rho`) into a
per-cluster scalar "score" (`wsum`) using quantile-Gaussianization (`multicam.qt`) so that
different features are combined in the same units. `calc_score` picks the score at a given target
time; `split_mah`/`split_mah_idx` bucket clusters into MAH quartiles by that score, which is what
the "split" plots (e.g. `plots/mah_split.pdf`) visualize.

**`utils/plots.py`** — shared matplotlib styling (colorblind-safe palette) and plot builders
(`plot_corrs`, `plot_split`, `plot_2dcorr`, lookback-time twin axes) consumed by the top-level
scripts/notebooks.

**`utils/image.py`** — segmentation-map helpers (circular/annular masks) used by
`measure_morphs.py` before handing a mass map + segmap to `statmorph`.

**Recurring domain pattern**: dynamical-state (`DS`) parameters (`eta_200[3]`, `delta_200[4]`,
`fm_200[5]`, `fm2_200[6]`, plus `r_sp`/`r_trunc`) are projection-independent (one value per
cluster), while stellar-morphology (`SM`) features (concentration, asymmetry, Sersic amplitude,
etc.) are projection-dependent (one value per cluster per line-of-sight, so DataFrames get
stacked/repeated 3x — look for `np.vstack([...]*3)` and `proj=True` when a function needs to know
this). Code paths, variable names (`ds`/`sm`), and CLI flags (`--features sm|ds`) consistently
branch on this distinction — check which regime a function is in before changing it.
