import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.file as futils
import utils.data as datutils
import utils.plots as plots
import warnings
warnings.filterwarnings('ignore')

MAH_DIR = 'data/AHF_HaloHistory'

# label -> (csv filename prefix, extends to cluster outskirts)
# "wide" apertures (out to 1 Mpc) are drawn solid, small core-only apertures dashed
# (the cluster-count suffix on the actual filename varies by projection/run, so
# it's resolved dynamically in compute() via futils.find_result_csv)
APERTURES = {
    '50 kpc - 1 Mpc': ('rin50.0kpc_rout1.0Mpc', True),
    '30 kpc - 1 Mpc': ('rin30.0kpc_rout1.0Mpc', True),
    '10 kpc - 1 Mpc': ('rin10.0kpc_rout1.0Mpc', True),
    '1 Mpc': ('r_1.0Mpc', True),
    '50 kpc': ('r_0.05Mpc', False),
    '30 kpc': ('r_0.03Mpc', False),
    '10 kpc': ('r_0.01Mpc', False),
}


def aperture_corrs(smdf, mah_dir=MAH_DIR):
    """Correlate concentration/asymmetry against m(a) and a(m) for one aperture.

    Parameters
    ----------
    smdf : pd.DataFrame
        statmorph measurements indexed by cluster ID, for a single aperture.
    mah_dir : str, optional

    Returns
    -------
    dict
        aexp_bins, mass_bins and the (p10, p25, p50, p75, p90) percentile curves
        of the Spearman correlation for 'C' and 'A' against m(a) and a(m).
    """
    halo_ids = sorted(smdf.index.tolist())
    mah_dict = futils.get_mah_all(mah_dir=mah_dir, halo_ids=halo_ids)
    smdf = smdf.reindex(list(mah_dict.keys()))

    ma, _, aexp_bins = datutils.interp_ma(mah_dict)
    am, mass_bins = datutils.build_am(ma=ma, scales=aexp_bins,
                                      min_mass_bin=0.01, log_spacing=True)

    ma_dict, _ = datutils.prepare_ma_corrs(mah_dict, smdf)
    am_dict = datutils.prepare_am_corrs(am, mass_bins, smdf)

    # prepare_ma_corrs keys its dict by ascending redshift (i.e. descending aexp),
    # and get_percs iterates the dict in that insertion order, so the x-axis
    # paired with C_ma/A_ma must be reversed to match (cf. predict.py's tbins[::-1]).
    return {
        'aexp_bins': aexp_bins[::-1],
        'mass_bins': mass_bins,
        'C_ma': datutils.get_percs(ma_dict, param='C', history='M/M0'),
        'A_ma': datutils.get_percs(ma_dict, param='A', history='M/M0'),
        'C_am': datutils.get_percs(am_dict, param='C', history='am'),
        'A_am': datutils.get_percs(am_dict, param='A', history='am'),
    }


def compute(proj_dir='results/zx', mah_dir=MAH_DIR):
    curves = {}
    for label, (prefix, wide) in APERTURES.items():
        smdf = futils.get_morph(sm_dir=futils.find_result_csv(proj_dir, prefix))
        curves[label] = {'wide': wide, **aperture_corrs(smdf, mah_dir=mah_dir)}
    return curves


def to_frame(curves):
    """Flatten the per-aperture percentile curves into a tidy long-format DataFrame."""
    rows = []
    for label, data in curves.items():
        for param in ('C', 'A'):
            for history, x in (('ma', data['aexp_bins']), ('am', data['mass_bins'])):
                p10, p25, p50, p75, p90 = data[f'{param}_{history}']
                for xi, v10, v25, v50, v75, v90 in zip(x, p10, p25, p50, p75, p90):
                    rows.append({'aperture': label, 'wide': data['wide'], 'param': param,
                                'history': history, 'x': xi, 'p10': v10, 'p25': v25,
                                'p50': v50, 'p75': v75, 'p90': v90})
    return pd.DataFrame(rows)


FILL_APERTURES = {'50 kpc - 1 Mpc', '30 kpc'}


def plot_param(curves, param, ylabel_ma, ylabel_am, title):
    xticks = np.arange(0, 1.1, 0.1)
    yticks = np.arange(-0.6, 0.6, 0.1)
    ylim = (-0.65, 0.55)
    xlim = (0.09, 1.05)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for i, (label, data) in enumerate(curves.items()):
        color = plots.colors[i % len(plots.colors)]
        linestyle = '-' if data['wide'] else '--'
        fill = label in FILL_APERTURES

        _, p25, p50, p75, _ = data[f'{param}_ma']
        axs[0].plot(data['aexp_bins'], p50, color=color, linestyle=linestyle, label=label)
        if fill:
            axs[0].fill_between(data['aexp_bins'], p25, p75, color=color, alpha=0.2)

        _, p25, p50, p75, _ = data[f'{param}_am']
        axs[1].plot(data['mass_bins'], p50, color=color, linestyle=linestyle, label=label)
        if fill:
            axs[1].fill_between(data['mass_bins'], p25, p75, color=color, alpha=0.2)

    axs[0].set_xlabel(r'Scale Factor $a = 1/(1+z)$')
    axs[0].set_ylabel(ylabel_ma)
    axs[0].set_xticks(xticks)
    axs[0].set_yticks(yticks)
    axs[0].set_ylim(ylim)
    axs[0].set_xlim(xlim)
    axs[0].grid()
    axs[0].legend(fontsize='small')
    plots.add_lbt_twiny(axs[0], xticks, xlim)

    axs[1].set_xlabel(r'$m = M/M(a=1)$')
    axs[1].set_ylabel(ylabel_am)
    axs[1].set_yticks(yticks)
    axs[1].set_ylim(ylim)
    axs[1].set_xlim(xlim)
    axs[1].grid()
    axs[1].legend(fontsize='small')

    fig.suptitle(title, fontsize=18)
    return fig


def regional_morphs(proj_dir='results/zx', mah_dir=MAH_DIR, save=False):
    curves = compute(proj_dir=proj_dir, mah_dir=mah_dir)

    fig_conc = plot_param(curves, 'C', r'$\rho_s (SM_{a=0.94}, m(a))$',
                          r'$\rho_s (SM_{a=0.94}, a(m))$', 'Median Concentration Correlation')
    fig_asym = plot_param(curves, 'A', r'$\rho_s (SM_{a=0.94}, m(a))$',
                          r'$\rho_s (SM_{a=0.94}, a(m))$', 'Median Asymmetry Correlation')

    if save:
        fig_conc.savefig('plots/radial_conc.pdf')
        fig_asym.savefig('plots/radial_asym.pdf')
        to_frame(curves).to_csv('results/regional_morphs.csv', index=False)

    plt.show()
    return curves


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_dir', type=str, default='results/zx',
                        help="directory of per-aperture statmorph CSVs")
    parser.add_argument('--mah_dir', type=str, default=MAH_DIR,
                        help="directory of AHF halo mass accretion histories")
    parser.add_argument('-s', '--save', action="store_true",
                        help="Save plots and underlying correlation curves")
    args = parser.parse_args()
    regional_morphs(args.proj_dir, args.mah_dir, args.save)
