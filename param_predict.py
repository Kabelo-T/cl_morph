import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.file as futils
import utils.data as datutils
import utils.plots as plots
import utils.corrs as corrs

warnings.filterwarnings('ignore')

DS_PARAMS = ['eta_200[3]', 'delta_200[4]', 'fm_200[5]', 'fm2_200[6]', 'DS_200[2]',
             'eta_500[8]', 'delta_500[9]', 'fm_500[10]', 'fm2_500[11]', 'DS_500[7]',
             'r_sp', 'r_trunc']
DS_LABELS = [r'$\eta_{200c}$', r'$\Delta_{200c}$', r'$f_{m, 200c}$', r'$f_{m2, 200c}$',
             r'$DS_{200c}$ (relaxed/disturbed)',
             r'$\eta_{500c}$', r'$\Delta_{500c}$', r'$f_{m, 500c}$', r'$f_{m2, 500c}$',
             r'$DS_{500c}$ (relaxed/disturbed)',
             'Splashback Radius', 'Truncation Radius']

AHF_PARAMS = futils.AHF_Z0_PARAMS
AHF_LABELS = ['Number of Substructures', 'Virial Mass', 'Particle Count', 'Virial Radius',
              'Rmax (radius of Vmax)', 'r2 (radius where slope = -2)',
              'Most-Bound-Particle Offset', 'Center-of-Mass Offset', 'Vmax',
              'Escape Velocity', 'Velocity Dispersion', 'Spin Parameter (Bullock)',
              'Spin Parameter (Peebles)', 'Minor/Major Axis Ratio (b/a)',
              'Minor/Major Axis Ratio (c/a)', 'Overdensity', 'Fraction High-res Particles',
              'Kinetic Energy', 'Potential Energy', 'NFW Concentration', 'Gas Mass',
              'Stellar Mass']

SM_CORE_XY_FILE = futils.find_result_csv('results/xy', 'r_0.03Mpc')
SM_CORE_YZ_FILE = futils.find_result_csv('results/yz', 'r_0.03Mpc')
SM_CORE_ZX_FILE = futils.find_result_csv('results/zx', 'r_0.03Mpc')

SM_LABELS_MAP = {
    'xc_centroid': 'X Centroid',
    'yc_centroid': 'Y Centroid',
    'ellipticity_asymmetry': 'Ellipticity (asymmetry-based)',
    'ellipticity_centroid': 'Ellipticity (centroid-based)',
    'elongation_asymmetry': 'Elongation (asymmetry-based)',
    'elongation_centroid': 'Elongation (centroid-based)',
    'orientation_centroid': 'Orientation (centroid-based)',
    'xc_asymmetry': 'X Asymmetry Center',
    'yc_asymmetry': 'Y Asymmetry Center',
    'orientation_asymmetry': 'Orientation (asymmetry-based)',
    'rpetro_circ': 'Petrosian Radius (circular)',
    'rpetro_ellip': 'Petrosian Radius (elliptical)',
    'rhalf_circ': 'Half-light Radius (circular)',
    'rhalf_ellip': 'Half-light Radius (elliptical)',
    'r20': 'Radius Enclosing 20% of Light',
    'r50': 'Radius Enclosing 50% of Light',
    'r80': 'Radius Enclosing 80% of Light',
    'Gini': 'Gini Coefficient',
    'M20': 'M20 Statistic',
    'F(G, M20)': 'Gini-M20 Bulge Statistic',
    'S(G, M20)': 'Gini-M20 Merger Statistic',
    'C': 'Concentration',
    'A': 'Asymmetry',
    'S': 'Smoothness',
    'sersic_amplitude': 'Sersic Amplitude',
    'sersic_rhalf': 'Sersic Half-light Radius',
    'sersic_n': 'Sersic Index',
    'sersic_xc': 'Sersic X Center',
    'sersic_yc': 'Sersic Y Center',
    'sersic_ellip': 'Sersic Ellipticity',
    'sersic_theta': 'Sersic Orientation Angle',
    'sersic_chi2_dof': r'Sersic Fit Reduced $\chi^2$',
    'core_C': 'Core Concentration (inner 0.03 Mpc aperture)',
}


def load_ds():
    dsdf = futils.get_ds(clean=False)
    dsdf.drop(columns=['Hid[1]'], inplace=True)
    dsdf.sort_index(inplace=True)
    halo_ids = list(dsdf.index)
    return dsdf, halo_ids


def load_ahf():
    ahfdf = futils.get_ahf_z0()
    ahfdf.sort_index(inplace=True)
    halo_ids = list(ahfdf.index)
    return ahfdf, halo_ids


def load_sm():
    ixy = futils.get_morph(sm_dir=SM_CORE_XY_FILE)
    iyz = futils.get_morph(sm_dir=SM_CORE_YZ_FILE)
    izx = futils.get_morph(sm_dir=SM_CORE_ZX_FILE)
    halo_ids = futils.valid_ids(ixy, iyz, izx)

    inner_df = pd.concat([ixy.loc[halo_ids], iyz.loc[halo_ids], izx.loc[halo_ids]])
    core_c = inner_df['C']
    core_c.name = 'core_C'

    smdf = futils.get_morphologies(halo_ids, all=True).drop(columns=['ID'])
    df = pd.concat([smdf.reset_index(drop=True), core_c.reset_index(drop=True)], axis=1)

    params = list(df.columns)
    labels = [SM_LABELS_MAP.get(p, p) for p in params]
    return df, halo_ids, params, labels


def load(group):
    if group == 'ds':
        df, halo_ids = load_ds()
        params, labels = DS_PARAMS, DS_LABELS
    elif group == 'ahf':
        df, halo_ids = load_ahf()
        params, labels = AHF_PARAMS, AHF_LABELS
    else:
        df, halo_ids, params, labels = load_sm()

    mah_dict = futils.get_mah_all(halo_ids=halo_ids)
    ma, _, aexp_bins = datutils.interp_ma(mah_dict)
    return mah_dict, ma, aexp_bins, df, params, labels


def print_menu(params, labels):
    print('\nAvailable features:')
    for i, (param, label) in enumerate(zip(params, labels)):
        print(f'  [{i}] {param}  ({label})')


def choose_param(params, labels):
    print_menu(params, labels)
    while True:
        choice = input(f'\nEnter a number [0-{len(params) - 1}]: ').strip()
        try:
            idx = int(choice)
        except ValueError:
            print('Please enter an integer.')
            continue
        if idx < 0 or idx >= len(params):
            print(f'Please enter a number between 0 and {len(params) - 1}.')
            continue
        return idx


def single_predict(ma, df, param, proj=False):
    x = df[[param]].to_numpy()
    y = np.vstack([ma] * 3) if proj else ma
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=proj)
    curve = np.array([[p25, p50, p75]])
    return curve


def predict(group):
    _, ma, tbins, df, params, labels = load(group)
    idx = choose_param(params, labels)
    param, label = params[idx], labels[idx]

    curves = single_predict(ma, df, param, proj=(group == 'sm'))

    fig, axs = plt.subplots(figsize=(8, 6), constrained_layout=True)
    plots.plot_preds(axs, curves, tbins, [label], colors=['k'], fill=[True])
    fig.suptitle(label)
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature', type=str, default='ds', choices=['ds', 'ahf', 'sm'],
        help="feature group to pick a single parameter from, ds, ahf, or sm")
    args = parser.parse_args()
    predict(args.feature)
