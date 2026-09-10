import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13

import utils.file as futils
import utils.data as datutils
import utils.score as score
import utils.corrs as corrs
import utils.plots as plots

warnings.filterwarnings('ignore')

# Load data
mah_dict = futils.get_mah_all()
ma, _, aexp_bins = datutils.interp_ma(mah_dict)
am, mass_bins = datutils.build_am(ma=ma, scales=aexp_bins,
                                  min_mass_bin=0.01, log_spacing=True)

halo_ids = np.array([int(k) for k in mah_dict.keys()])
m14 = futils.get_m14(halo_ids)

dsdf = futils.get_ds(halo_ids)
dsdf.drop(columns=['eta_500[8]', 'delta_500[9]',
          'fm_500[10]', 'fm2_500[11]'], inplace=True)

# SM (statmorph) data is only valid for a subset of halos: measurement
# quality/map availability means core concentration ('iconc') isn't
# measurable for every halo in every projection, so we restrict to the
# halos with a valid measurement in all three projections.
ixy = futils.get_morph(sm_dir=futils.find_result_csv('results/xy', 'r_0.03Mpc'))
iyz = futils.get_morph(sm_dir=futils.find_result_csv('results/yz', 'r_0.03Mpc'))
izx = futils.get_morph(sm_dir=futils.find_result_csv('results/zx', 'r_0.03Mpc'))
sm_halo_ids = futils.valid_ids(ixy, iyz, izx)

inner_df = pd.concat(
    [ixy.loc[sm_halo_ids], iyz.loc[sm_halo_ids], izx.loc[sm_halo_ids]])
iconc = inner_df['C']
iconc.name = 'core_C'

m14_sm = futils.get_m14(sm_halo_ids)
m14_xyz = pd.Series(
    np.hstack([m14_sm['proj_x'], m14_sm['proj_y'], m14_sm['proj_z']]))
m14_xyz.name = 'm14'

smdf = futils.get_morphologies(sm_halo_ids)

n_sm_halos = len(sm_halo_ids)
assert smdf.shape[0] == 3 * n_sm_halos
assert iconc.shape[0] == 3 * n_sm_halos
assert m14_xyz.shape[0] == 3 * n_sm_halos

mah_dict_sm = futils.get_mah_all(halo_ids=sm_halo_ids)
ma_sm, _, aexp_bins_sm = datutils.interp_ma(mah_dict_sm)
stacked_ma_sm = np.vstack([ma_sm]*3)  # because we have 3 projections

t_d = (3.0857e19)/(10*Planck13.H0.value*60*60*24*365e9)
print(f'Dynamical time = {t_d:.2f} Gyr')

# Prepare all data
df = pd.concat([dsdf, m14['3d']], axis=1, join='inner')
x = df.to_numpy()
y = ma
_, _, _, _, ds_weights, ds_rmse, ds_r2, model = corrs.multicorr(
    x, y, proj=False, scoring=True)
wsum_ds = score.wsum(xng=df, weights=ds_weights)  # scores as function of mah
# scores at target time, target time index
scores, indx = score.calc_score(time=aexp_bins, target=0.75, wsum=wsum_ds)
mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50 = score.split_mah(
    mah_dict, scores)  # split mah into lower and upper 25%
mcam_ds = (mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50)
print(f'[MultiCAM DS]  n_lower25={len(mah_25)} n_upper25={len(mah_75)} '
      f'diff@a={aexp_bins[indx]:.3g}: {u25[1][indx] - l25[1][indx]:.4f}')

df = pd.concat([smdf.reset_index(drop=True), m14_xyz.reset_index(drop=True),
                iconc.reset_index(drop=True)], axis=1)
x = df.to_numpy()
y = stacked_ma_sm
_, _, _, _, sm_weights, sm_rmse, sm_r2, model = corrs.multicorr(
    x, y, proj=True, scoring=True)
wsum_sm = score.wsum(xng=df, model=model, weights=sm_weights)
scores, indx_sm = score.calc_score(time=aexp_bins_sm, target=0.75, wsum=wsum_sm)
mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50 = score.split_mah(
    mah_dict_sm, scores)
mcam_sm = (mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50)
print(f'[MultiCAM SM]  n_lower25={len(mah_25)} n_upper25={len(mah_75)} '
      f'diff@a={aexp_bins_sm[indx_sm]:.3g}: {u25[1][indx_sm] - l25[1][indx_sm]:.4f}')

df = pd.concat([dsdf, m14['3d']], axis=1)
rho = score.build_rho(df=df, mah=ma)
wsum_ds = score.wsum(xng=df, weights=rho)
scores, indx = score.calc_score(time=aexp_bins, target=0.75, wsum=wsum_ds)
mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50 = score.split_mah(
    mah_dict, scores)
rho_ds = (mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50)
print(f'[rho DS]       n_lower25={len(mah_25)} n_upper25={len(mah_75)} '
      f'diff@a={aexp_bins[indx]:.3g}: {u25[1][indx] - l25[1][indx]:.4f}')

df = pd.concat([smdf.reset_index(drop=True), m14_xyz.reset_index(drop=True),
                iconc.reset_index(drop=True)], axis=1)
rho = score.build_rho(df=df, mah=stacked_ma_sm)
wsum_sm = score.wsum(xng=df, weights=rho)
scores, indx_sm = score.calc_score(time=aexp_bins_sm, target=0.75, wsum=wsum_sm)
mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50 = score.split_mah(
    mah_dict_sm, scores)
rho_sm = (mah_25, mah_50, mah_75, l25, u25, mah_u50, mah_l50)
print(f'[rho SM]       n_lower25={len(mah_25)} n_upper25={len(mah_75)} '
      f'diff@a={aexp_bins_sm[indx_sm]:.3g}: {u25[1][indx_sm] - l25[1][indx_sm]:.4f}')

# Big plot
plot_rc = {
    'font.size': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
}
with plt.rc_context(plot_rc):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    xticks = np.arange(0, 1.1, 0.1)
    xlim = (0, 1.01)
    ylim = (0., 1.01)

    plots.plot_split(axs[0, 0], mcam_ds[0], mcam_ds[1], mcam_ds[2],
                      mcam_ds[3], mcam_ds[4], aexp_bins, indx)
    axs[0, 0].annotate(text=r'MultiCAM weighted DS', xy=(0.06, 0.8))
    axs[0, 0].set_xticks(xticks)

    plots.plot_split(axs[0, 1], mcam_sm[0], mcam_sm[1], mcam_sm[2],
                      mcam_sm[3], mcam_sm[4], aexp_bins_sm, indx_sm)
    axs[0, 1].yaxis.set_visible(False)
    axs[0, 1].annotate(text=r'MultiCAM weighted SM', xy=(0.06, 0.8))
    axs[0, 1].set_xticks(xticks)

    plots.plot_split(axs[1, 0], rho_ds[0], rho_ds[1], rho_ds[2],
                      rho_ds[3], rho_ds[4], aexp_bins, indx)
    axs[1, 0].annotate(text=r'$\rho_{\mathrm{sp}}$ weighted DS', xy=(0.06, 0.8))
    axs[1, 0].set_xticks(xticks)

    plots.plot_split(axs[1, 1], rho_sm[0], rho_sm[1], rho_sm[2],
                      rho_sm[3], rho_sm[4], aexp_bins_sm, indx_sm)
    axs[1, 1].annotate(text=r'$\rho_{\mathrm{sp}}$ weighted SM', xy=(0.06, 0.8))
    axs[1, 1].yaxis.set_visible(False)
    axs[1, 1].set_xticks(xticks)

    plots.plot_lookback_time(axs=axs, xlim=xlim, xticks=xticks, fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/mah_split.pdf')
    plt.show()
