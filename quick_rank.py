"""Rank clusters by true m(a), MultiCAM DS score, and MultiCAM SM score at a
target scale factor, and print or save each ranking to its own CSV.

Rank 1 = slowest accreter (highest m(a) / lowest MultiCAM score), consistent
with the Q1 = slowest-accreter convention used in rsp_weights.py.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

import utils.file as futils
import utils.score as score
from rsp_weights import FEATURE_SETS, DEFAULT_WEIGHTS

warnings.filterwarnings('ignore')

TARGET = 0.75
OUT_DIR = 'ranks'


def aexp_tag(target: float) -> str:
    """Format a scale factor for use in a filename, e.g. 0.75 -> '0_75', 0.9 -> '0_9'."""
    return str(target).replace('.', '_')


def rank_ascending(values: np.ndarray) -> np.ndarray:
    """1-based rank, 1 = smallest value."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def true_rank(target: float) -> pd.Series:
    """True m(a) rank for all clusters with a DS/MAH entry (rank 1 = slowest)."""
    _, ma, _, aexp_bins, _, df = futils.load('ds')
    indx = int(np.abs(aexp_bins - target).argmin())
    scores = -ma[:, indx]
    ranks = pd.Series(rank_ascending(scores), index=df.index, name='true_rank')
    print(f'true m(a) rank computed at a = {aexp_bins[indx]:.3f} (target a = {target}), '
          f'n = {len(ranks)}')
    return ranks


def multicam_rank(features: str, target: float) -> pd.Series:
    """MultiCAM {features} score rank, averaged over projections for 'sm' (rank 1 = slowest)."""
    _, _, _, aexp_bins, _, df = futils.load(features)
    weights = np.load(DEFAULT_WEIGHTS[features])
    x = df[FEATURE_SETS[features]].to_numpy()
    wsum_x = score.wsum(xng=x, weights=weights)
    scores, indx = score.calc_score(time=aexp_bins, target=target, wsum=wsum_x)
    print(f'MultiCAM {features.upper()} rank computed at a = {aexp_bins[indx]:.3f} '
          f'(target a = {target}), n_rows = {len(scores)}')

    name = f'{features}_rank'
    if 'ID' in df.columns:
        # projection-stacked (sm): average the score per cluster before ranking
        mean_scores = pd.Series(scores, index=df['ID'].to_numpy()).groupby(level=0).mean()
        return pd.Series(rank_ascending(mean_scores.to_numpy()),
                         index=mean_scores.index, name=name)
    return pd.Series(rank_ascending(scores), index=df.index, name=name)


def main(target: float = TARGET, save: bool = False, out_dir: str = OUT_DIR):
    ranks = {
        'true': true_rank(target),
        'ds': multicam_rank('ds', target),
        'sm': multicam_rank('sm', target),
    }

    tag = aexp_tag(target)
    if save:
        os.makedirs(out_dir, exist_ok=True)

    for kind, rank in ranks.items():
        out = rank.sort_values().reset_index()
        out.columns = ['ID', rank.name]
        out = out[[rank.name, 'ID']]
        if save:
            fname = os.path.join(out_dir, f'{kind}_rank{tag}.csv')
            out.to_csv(fname, index=False)
            print(f'wrote {out.shape[0]} rows to {fname}')
        else:
            print(f'\n{kind}_rank{tag}:')
            print(out.to_string(index=False))
    return ranks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--aexp', type=float, default=TARGET,
                        help='target scale factor to rank clusters at')
    parser.add_argument('-s', '--save', action='store_true',
                        help='save rankings to CSVs ({out_dir}/{kind}_rank{aexp}.csv) instead of printing')
    parser.add_argument('-o', '--out-dir', type=str, default=OUT_DIR,
                        help='directory to save ranking CSVs into')
    args = parser.parse_args()
    main(args.aexp, args.save, args.out_dir)
