"""Rank clusters at a target scale factor and compare the median splashback/
truncation radius across rank quartiles.

By default (no --features) clusters are ranked by their *true* mass accretion
history, m(a) = M(a)/M(a=1); pass --features ds/sm to instead rank by their
MultiCAM DS/SM score (a proxy for m(a) built from dynamical-state or
stellar-morphology features) at that same scale factor.
"""
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

import utils.file as futils
import utils.score as score
from utils.plots import colors

warnings.filterwarnings('ignore')

# feature order used when the corresponding results/multicam_{features}_weights.npy
# was fit (see predict.py:ds_predict / predict.py:sm_predict)
FEATURE_SETS = {
    'ds': ['eta_200[3]', 'delta_200[4]', 'fm_200[5]', 'fm2_200[6]', '3d'],
    'sm': ['rhalf_circ', 'C', 'A', 'sersic_amplitude', 'm14', 'core_C'],
}
DEFAULT_WEIGHTS = {
    'ds': 'results/multicam_ds_weights.npy',
    'sm': 'results/multicam_sm_weights.npy',
}


def quartile_split(scores: np.ndarray) -> list[np.ndarray]:
    """Rank clusters by score (ascending) and split into 4 equal-size groups.

    Returns a list of index arrays into `scores`, Q1 = lowest score .. Q4 = highest score.
    """
    order = np.argsort(scores)
    return np.array_split(order, 4)


def quartile_stats(values: np.ndarray, groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Median of `values` within each quartile group, with error bars of +/-25% of the median."""
    medians = np.array([np.median(values[g]) for g in groups])
    errs = 0.25 * medians
    return medians, errs


def get_radii(df):
    """Splashback/truncation radius aligned to each row of `df`.

    DS features are already one row per cluster with r_sp/r_trunc attached
    (utils.file.get_ds). SM features are projection-stacked (3 rows per
    cluster, one per line-of-sight, no r_sp/r_trunc columns), so those
    projection-independent radii are looked up per cluster ID and repeated.
    """
    ids = df['ID'].to_numpy() if 'ID' in df.columns else df.index.to_numpy()
    dsdf = futils.get_ds(halo_ids=list(np.unique(ids)))
    r_sp = dsdf['r_sp'].reindex(ids).to_numpy()
    r_trunc = dsdf['r_trunc'].reindex(ids).to_numpy()
    return r_sp, r_trunc


def main(features=None, weights_path=None, target=0.75, save=False):
    if features is None:
        # true rank ordering: m(a) = M(a)/M(a=1) itself, no MultiCAM proxy.
        # Higher m(a) means more of the final mass was already assembled by
        # this scale factor, i.e. a slower/earlier-forming accretion history.
        # Rank by -m(a) so Q1 = highest m(a) = slowest accreters, matching
        # the sense of the MultiCAM DS/SM score quartiles below.
        _, ma, _, aexp_bins, _, df = futils.load('ds')
        indx = int(np.abs(aexp_bins - target).argmin())
        scores = -ma[:, indx]
        rank_label = 'True MAH'
    else:
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS[features]

        _, _, _, aexp_bins, _, df = futils.load(features)
        weights = np.load(weights_path)

        x = df[FEATURE_SETS[features]].to_numpy()
        wsum_x = score.wsum(xng=x, weights=weights)
        scores, indx = score.calc_score(time=aexp_bins, target=target, wsum=wsum_x)
        rank_label = f'MultiCAM {features.upper()} Score'

    print(f'Ranking by {rank_label} at a = {aexp_bins[indx]:.3f} '
          f'(target a = {target}), n_rows = {len(scores)}')

    groups = quartile_split(scores)
    quartiles = np.arange(1, 5)

    r_sp, r_trunc = get_radii(df)

    rsp_med, rsp_std = quartile_stats(r_sp, groups)
    rtrunc_med, rtrunc_std = quartile_stats(r_trunc, groups)

    for q, g, rm, re, tm, te in zip(quartiles, groups, rsp_med, rsp_std,
                                     rtrunc_med, rtrunc_std):
        print(f'  Q{q} (n={len(g)}): '
              f'r_sp = {rm:.3f} +/- {re:.3f}, r_trunc = {tm:.3f} +/- {te:.3f}')

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.errorbar(quartiles, rsp_med, yerr=rsp_std, marker='o', capsize=4,
                color=colors[1], label='Splashback radius')
    ax.errorbar(quartiles, rtrunc_med, yerr=rtrunc_std, marker='s', capsize=4,
                color=colors[5], label='Truncation radius')
    ax.set_xticks(quartiles)
    ax.set_xticklabels([f'Q{q}' for q in quartiles])
    ax.set_xlabel(f'{rank_label} Quartile (a = {aexp_bins[indx]:.2f})')
    ax.set_ylabel(r'Radius [$h^{-1}\,\mathrm{Mpc}$]')
    ax.legend()
    ax.grid(alpha=0.3)

    if save:
        out_tag = features if features is not None else 'true'
        plt.savefig(f'plots/rsp_weights_{out_tag}.pdf')
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features', type=str, choices=['ds', 'sm'], default=None,
                        help='rank clusters by a MultiCAM DS/SM score instead of the true '
                             'mass accretion history (default: true m(a), no MultiCAM proxy)')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to saved MultiCAM weights (n_times x n_features); '
                             'defaults to results/multicam_{features}_weights.npy; '
                             'ignored unless --features is set')
    parser.add_argument('--target', type=float, default=0.75,
                        help='target scale factor at which to rank clusters')
    parser.add_argument('-s', '--save', action='store_true',
                        help='save plot to plots/rsp_weights_{features|true}.pdf')
    args = parser.parse_args()
    main(args.features, args.weights, args.target, args.save)
