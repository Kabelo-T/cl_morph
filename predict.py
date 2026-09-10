import numpy as np
import matplotlib.pyplot as plt

import utils.file as futils
import utils.data as datutils
import utils.plots as plots
import utils.corrs as corrs
import argparse
import warnings
warnings.filterwarnings('ignore')

MAH_DIR = 'data/AHF_HaloHistory'
DS_PARAMS = ['eta_200[3]', 'delta_200[4]', 'fm_200[5]', 'fm2_200[6]', '3d',
             'r_sp', 'r_trunc']
DS_LABELS = [r'$\eta_{200c}$', r'$\Delta_{200c}$',
             r'$f_{m, 200c}$', r'$f_{m2, 200c}$', r'$m_{14, 3D}$',
             'Splashback', 'Truncation']
SM_PARAMS = ['rhalf_circ', 'C', 'sersic_amplitude', 'A', 'm14', 'core_C']
SM_PARAM_LABELS = ['Half-light Radius',
                    'Concentration',
                    'Sersic Amplitude',
                    'Asymmetry',
                    r'$m_{14}$',
                    'Core Concentration']
SM_LABELS = [r'$\mathrm{SM} + m_{14} + \mathrm{Core\,Conc.}$',
             r'$\mathrm{SM} + m_{14}$', r'$\mathrm{SM}$', 'Asymmetry']

DS_CURVE_LABELS = ['All', ' + '.join([DS_LABELS[0], DS_LABELS[1], DS_LABELS[4]]),
                    DS_LABELS[2], DS_LABELS[4]]
DS_CURVE_COLORS = ['k', '#009E73', '#F0E442', '#CC79A7']
DS_CURVE_FILL = [True, False, False, True]

def sm_predict(features, mah='ma'):
    mah_dict, ma, am, aexp_bins, mass_bins, df = futils.load(features)
    if mah == 'am':
        corr_dict = datutils.prepare_am_corrs(am, mass_bins, df)
        tbins = mass_bins
        y = np.vstack([am]*3)  # because we have 3 projections
    else:
        corr_dict, _ = datutils.prepare_ma_corrs(mah_dict, df)
        tbins = aexp_bins
        y = np.vstack([ma]*3)  # because we have 3 projections
    curves = []

    x = np.vstack([df['rhalf_circ'], df['C'], df['A'],
           df['sersic_amplitude'], df['m14'], df['core_C']]).T
    _, p25, p50, p75, sm_weights, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))
    np.save('results/multicam_sm_weights.npy', sm_weights)

    x = np.vstack([df['rhalf_circ'], df['C'], df['A'],
           df['sersic_amplitude'], df['m14']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['rhalf_circ'], df['C'], df['A'],
           df['sersic_amplitude']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['A']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    curves = np.array(curves)
    return mah_dict, curves, tbins, corr_dict


def ds_predict(features, mah='ma'):
    mah_dict, ma, am, aexp_bins, mass_bins, df = futils.load(features)
    if mah == 'am':
        corr_dict = datutils.prepare_am_corrs(am, mass_bins, df)
        tbins = mass_bins
        y = am
    else:
        corr_dict, _ = datutils.prepare_ma_corrs(mah_dict, df)
        tbins = aexp_bins
        y = ma
    curves = []

    x = np.vstack([df['eta_200[3]'], df['delta_200[4]'], df['fm_200[5]'],
                   df['fm2_200[6]'], df['3d']]).T
    _, p25, p50, p75, ds_weights, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))
    np.save('results/multicam_ds_weights.npy', ds_weights)

    x = np.vstack([df['eta_200[3]'], df['delta_200[4]'], df['3d']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['fm_200[5]']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['3d']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    curves = np.array(curves)
    return mah_dict, curves, tbins, corr_dict


def predict(features, save=False, radius=False, mah='ma'):
    am_flag = mah == 'am'
    if features == 'sm':
        mah_dict, curves, tbins, corr_dict = sm_predict(features, mah)
        labels, params, param_labels, sm_flag = SM_LABELS, SM_PARAMS, SM_PARAM_LABELS, True
        curve_colors, curve_fill = plots.CURVE_COLORS, plots.CURVE_FILL
    else:
        mah_dict, curves, tbins, corr_dict = ds_predict(features, mah)
        labels = DS_CURVE_LABELS
        if radius:
            params, param_labels = DS_PARAMS, DS_LABELS
        else:
            params, param_labels = DS_PARAMS[:5], DS_LABELS[:5]
        sm_flag = False
        curve_colors, curve_fill = DS_CURVE_COLORS, DS_CURVE_FILL

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), constrained_layout=True)
    plots.plot_feature_corrs(axs[0], corr_dict, params, param_labels, tbins, sm=sm_flag, am=am_flag)
    plots.plot_preds(axs[1], curves, tbins, labels, colors=curve_colors, fill=curve_fill, am=am_flag)
    if save:
        if features =='sm':
            plt.savefig(f'plots/sm_{mah}_preds.pdf')
        else:
            plt.savefig(f'plots/ds_{mah}_preds.pdf')
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features', type=str,
        default='sm', help="features to make predictions, sm or ds")
    parser.add_argument('-s', '--save', action="store_true",
                        help="Save plots")
    parser.add_argument('--radius', action="store_true",
                        help="Include splashback/truncation radii in DS predictions")
    parser.add_argument('--mah', type=str, default='ma', choices=['ma', 'am'],
                        help="mass accretion history representation: m(a) (default) or a(m)")
    args = parser.parse_args()
    predict(args.features, args.save, args.radius, args.mah)
