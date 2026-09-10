import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm, BoundaryNorm
from cycler import cycler
from astropy.cosmology import Planck13
import seaborn as sns

import utils.file as futils
import utils.data as datutils


plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

# Colorblind-friendly color palette https://www.nature.com/articles/nmeth.1618.pdf
colors = [
    '#000000',      # Black
    '#0072B2',      # Blue
    '#F0E442',      # Yellow
    '#009E73',      # Bluish Green
    '#CC79A7',      # Reddish Purple
    '#D55E00',      # Vermilion
    '#56B4E9',      # Sky Blue
    '#E69F00'       # Orange
]

plt.rc('axes', prop_cycle=cycler('color', colors))


def display_img(image: np.ndarray, axs, segmap=None,
                mask=None, vmin=None, vmax=None, **kwargs):
    """Display an image, masked or unmasked. May plot alongside segmentation map

    Parameters
    ----------
    image : np.array
        the original image
    segmap : np.array, optional
        segmentation map that labels distinct sources in image, by default None
    mask : np.array[bool], optional
        a mask for cropping the image, by default None
    """
    if mask is not None:
        image = np.where(mask, image, 0)

    if vmin is not None and vmax is not None:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = LogNorm()

    if isinstance(axs, np.ndarray):
        axs[0].imshow(image, norm=norm)
        axs[1].imshow(segmap, norm=norm, cmap='gray')
    else:
        axs.imshow(image, norm=norm)
    return


def plot_light(object, zoomed_region, zoom_size):
    """
    For a given patch of the image plot the region and overplot half-light radius r50, r80 and r20
    """
    fig, ax = plt.subplots()
    ax.imshow(zoomed_region, cmap='gray', origin='lower', vmin=0, vmax=0.05)

    circle_half = Circle((zoom_size, zoom_size), object.rhalf_circ,
                         color='red', fill=False, linewidth=1.5, label='r_half')
    circle_80 = Circle((zoom_size, zoom_size), object.r80,
                       color='white', fill=False, linewidth=1.5, label='r_80')
    circle_20 = Circle((zoom_size, zoom_size), object.r20,
                       color='yellow', fill=False, linewidth=1.5, label='r_20')

    ax.add_patch(circle_half)
    ax.add_patch(circle_80)
    ax.add_patch(circle_20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()

    return


def plot_corr_matrix(corr_matrix: pd.DataFrame, title: str, fname: str,
                     save: bool = False, show_cbar: bool = False,
                     annot_size: int = 20, **kwargs):
    """Plot the Correlation Matrix as a heatmap

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        correlations of all the parameters
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(14, 14))
    fhm = sns.heatmap(data=corr_matrix, mask=mask, cmap='coolwarm',
                      square=True, cbar=show_cbar, cbar_kws={"shrink": 0.3},
                      annot_kws={"size": annot_size, "weight": "bold"}, fmt='.2f',  **kwargs)
    plt.title(title)
    plt.tight_layout()
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    if save:
        plt.savefig(f'plots/{fname}')
    plt.show()
    return


def high_corr_cols(corr_matrix: pd.DataFrame, tolerance=0.7) -> list[bool]:
    """Select columns that are highly correlated and anti-correlated

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        correlations of all the parameters

    Returns
    -------
    list[bool]
        _description_
    """
    selection = []
    good_cols = []
    for i, x in enumerate(corr_matrix):
        lst = corr_matrix[x][np.abs(corr_matrix[x]) > 0.8]
        good_cols.append(x)
        for c in lst.index:
            if c in good_cols:
                continue
            else:
                selection.append((x, c))
    return selection


def plot_joint_dist(corr_matrix: pd.DataFrame, morph_df: pd.DataFrame,
                    low: int = 30, high: int = 40) -> None:
    """Plot joint distributions of the highly correlated and anti-correlated
    parameters

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        correlations of all the parameters
    morph_df : pd.DataFrame
        statmorph morphological measurements
    low : int
        lower bound index
    high : int
        upper bound index
    """

    selection = high_corr_cols(corr_matrix=corr_matrix)
    for i, row in enumerate(selection):
        if i in range(30, 40):
            rho = corr_matrix[row[0]][row[1]]
            df = morph_df[[row[0], row[1]]]
            joint = sns.jointplot(data=df, x=row[0], y=row[1], height=4)
            joint.figure.suptitle("r = " + str(rho))
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()


def plot_mah(indx: int, axs, am=False, highlight=False,
             mah_dir: str = 'data/AHF_HaloHistory/',
             show_relaxed: bool = False, show_disturbed: bool = False,
             show_all: bool = False) -> None:

    state = {0: "Disturbed",
             1: "Relaxed"}

    dsdf = futils.get_ds(snap=125, clean=False)
    mah_df_dict = futils.get_mah_all(mah_dir=mah_dir)
    ds = int(dsdf.loc[[indx]]['DS_200[2]'].values[0])
    axs.set_title(f'ID {indx}')

    if am:
        axs.set_xlabel('m = M/M(z=0)')
        axs.set_ylabel('a(m)')
        for k, df in mah_df_dict.items():
            if k == indx:
                axs.plot(df['M/M0'], df['aexp'], lw=5, color='red')
            elif show_all:
                axs.plot(df['M/M0'], df['aexp'],
                         color='black', alpha=0.2, lw=1)
            else:
                ds = int(dsdf.loc[[k]]['DS_200[2]'].values[0])
                if (ds == 1 and show_relaxed) or (ds == 0 and show_disturbed):
                    axs.plot(df['M/M0'], df['aexp'],
                             color='black', alpha=0.3, lw=1)

    else:
        axs.set_xlabel('Scale Factor a = 1/(1+z)')
        axs.set_ylabel('m(a)')
        for k, df in mah_df_dict.items():
            if k == indx:
                axs.plot(df['aexp'], df['M/M0'], lw=5, color='red')
            elif show_all:
                axs.plot(df['aexp'], df['M/M0'],
                         color='black', alpha=0.2, lw=1)
            else:
                ds = int(dsdf.loc[[k]]['DS_200[2]'].values[0])
                if (ds == 1 and show_relaxed) or (ds == 0 and show_disturbed):
                    axs.plot(df['aexp'], df['M/M0'],
                             color='black', alpha=0.3, lw=1)

    return


def check_neg(percentiles: list):
    percs = np.array(percentiles)
    r = np.sum(percs < 0) / len(percs)
    if r > 0.5:
        return True
    else:
        return False


LATEX_FONTSIZE = 18
LEGEND_LATEX_FONTSIZE = 16


def bump_latex_legend_fontsize(legend, size=LEGEND_LATEX_FONTSIZE):
    for text in legend.get_texts():
        if '$' in text.get_text():
            text.set_fontsize(size)


def plot_corrs(data_dict: dict, params: list[str], labels: list[str], time: np.ndarray,
               axs, sm=True, am=False, history='M/M0') -> None:

    yticks = np.arange(0, 0.9, 0.1)
    if am:
        t = 'a(m)'
        time_label = r'$m = M/M(a=1)$'
    else:
        t = 'm(a)'
        time_label = r'Scale Factor $a = 1/(1+z)$'

    axs.set_xlabel(time_label)

    if sm:
        axs.set_ylabel(
            r'$|\rho_{\mathrm{sp}} (SM_{a=0.94}, ' + t + r')|$', fontsize=LATEX_FONTSIZE)
    else:
        axs.set_ylabel(
            r'$|\rho_{\mathrm{sp}} (DS_{a=1}, ' + t + r')|$', fontsize=LATEX_FONTSIZE)
    ylim = (0., 0.8)

    for i, param in enumerate(params):
        p50 = datutils.get_perc(
            data_dict, param=param, q=50, history=history)
        flag = check_neg(p50)
        p50 = np.abs(p50)
        if flag:
            axs.plot(time, p50, color=colors[i],
                     label=labels[i], linestyle='dashed')
        else:
            axs.plot(time, p50, color=colors[i],
                     label=labels[i], linestyle='solid')

        if param == 'A' or param == 'core_C' or param == 'fm_200[5]' or param == '3d':
            p25 = datutils.get_perc(
                data_dict, param=param, q=25, history=history)
            p75 = datutils.get_perc(
                data_dict, param=param, q=75, history=history)

            if param == 'A':
                p25[3:] = np.abs(p25[3:])       # hack to fix tail of A corrs
                p75 = np.abs(p75)
            else:
                p25, p75 = np.abs(p25), np.abs(p75)

            axs.fill_between(time, p25, p75, color=colors[i], alpha=0.2)
            axs.fill_between(time, p25, p75, color=colors[i], alpha=0.2)

    axs.set_yticks(yticks)
    axs.set_ylim(ylim)
    axs.grid()
    bump_latex_legend_fontsize(axs.legend())


CURVE_COLORS = ['k', '#56B4E9', '#F0E442', '#009E73', '#E69F00', '#CC79A7']
CURVE_FILL = [True, False, False, True, True, True]


def calc_lbt(aexp):
    redshifts = 1 / aexp - 1
    lookback_time = Planck13.lookback_time(redshifts).value  # in Gyr

    return lookback_time


def add_lbt_twiny(axs, xticks, xlim):
    ax2 = axs.twiny()
    ax2.set_xlim(xlim)
    a_ticks = xticks[(xticks > 0) & (xticks <= xlim[1])]
    ax2.set_xticks(a_ticks)
    ax2.set_xticklabels([f"{t:.1f}" for t in calc_lbt(a_ticks)])
    ax2.set_xlabel("Lookback Time (Gyr)", size=16)


def plot_preds(axs, curves, tbins, labels, colors=CURVE_COLORS, fill=CURVE_FILL, am=False):
    xticks = np.arange(0, 1.1, 0.1)
    xlim = (0, 1.01)
    yticks = np.arange(0, 0.9, 0.1)
    ylim = (0., 0.8)

    for curve, label, color, fl in zip(curves, labels, colors, fill):
        axs.plot(tbins, np.abs(curve[1]), color=color, label=label)
        if fl:
            axs.fill_between(tbins, np.abs(curve[0]), np.abs(curve[2]),
                              color=color, alpha=0.2)

    if am:
        axs.set_xlabel(r'$m = M/M(a=1)$')
        axs.set_ylabel(r'$\rho_s (a(m)_{test}, a(m)_{pred})$', fontsize=LATEX_FONTSIZE)
    else:
        axs.set_xlabel(r'Scale Factor $a = 1/(1+z)$')
        axs.set_ylabel(r'$\rho_s (m(a)_{test}, m(a)_{pred})$', fontsize=LATEX_FONTSIZE)
    axs.set_xticks(xticks)
    axs.set_xlim(xlim)
    axs.set_yticks(yticks)
    axs.set_ylim(ylim)
    axs.grid()
    bump_latex_legend_fontsize(axs.legend())

    if not am:
        add_lbt_twiny(axs, xticks, xlim)


def plot_feature_corrs(axs, corr_dict, params, labels, tbins, sm=True, am=False):
    xticks = np.arange(0, 1.1, 0.1)
    xlim = (0, 1.01)
    history = 'am' if am else 'M/M0'
    time = tbins if am else tbins[::-1]

    plot_corrs(corr_dict, params=params, labels=labels,
               time=time, axs=axs, sm=sm, am=am, history=history)
    axs.set_xticks(xticks)
    axs.set_xlim(xlim)

    if not am:
        add_lbt_twiny(axs, xticks, xlim)


def plot_dynamical_time(axs, xlim, xticks, scales, lookback_time):
    for ax in axs.ravel():
        ax2 = ax.twiny()
        ax2.set_xlim(xlim)
        idx = np.searchsorted(scales, xticks)
        tlabels = [f'{lookback_time[i]:.1f}' for i in idx]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(tlabels)
        ax2.set_xlabel(
            r'$\Delta t / t_{\mathrm{dyn}}$', fontsize=16, labelpad=10)


def plot_lookback_time(axs, xlim, xticks, fontsize=14):
    a_ticks = xticks[(xticks > 0) & (xticks <= xlim[1])]
    redshifts = 1 / a_ticks - 1
    lookback_time = Planck13.lookback_time(redshifts).value  # in Gyr

    for ax in axs.ravel():
        ax2 = ax.twiny()
        ax2.set_xlim(xlim)
        ax2.set_xticks(a_ticks)
        ax2.set_xticklabels(
            [f'{t:.1f}' for t in lookback_time], fontsize=fontsize)
        ax2.set_xlabel('Lookback Time (Gyr)', fontsize=fontsize, labelpad=10)


def plot_split(axs, mah_25, mah_50, mah_75, l25, u25, time, indx):
    axs.set_xlabel('Scale Factor a = 1/(1+z)')
    axs.set_ylabel('m(a)')

    for i, mah in enumerate(mah_25):
        if i == 0:
            axs.plot(time, mah,
                     color='blue', alpha=0.3, lw=1)
        else:
            axs.plot(time, mah,
                     color='blue', alpha=0.3, lw=1)

    for i, mah in enumerate(mah_75):
        if i == 0:
            axs.plot(time, mah,
                     color='red', alpha=0.3, lw=1)
        else:
            axs.plot(time, mah,
                     color='red', alpha=0.3, lw=1)

    # Lower 25%
    axs.plot(
        time, l25[1],
        color='blue', lw=3, label='lower 25%'
    )
    # fill between 16th and 84th percentiles
    axs.fill_between(
        time, l25[0], l25[2],
        color='blue', alpha=0.25, linewidth=0
    )

    # Upper 25%
    axs.plot(
        time, u25[1],
        color='red', lw=3, label='upper 25%'
    )
    # fill between 16th and 84th percentiles
    axs.fill_between(
        time, u25[0], u25[2],
        color='red', alpha=0.25, linewidth=0
    )

    axs.vlines(time[indx], ymin=-1, ymax=1.5, color='k', ls='--')
    axs.set_ylim(0.0, 1.01)
    axs.legend()

    return


def plot_2dcorr(corr_matrix, scales, xlabel, ylabel, ax, title=''):
    cmap = plt.get_cmap('bwr', 10)                   # 10 discrete color bins
    # 10 bins -> 11 boundaries
    boundaries = np.linspace(-1.0, 1.0, 11)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # compute extent so pixel centers map to the provided scales array
    ny, nx = corr_matrix.shape
    xmin, xmax = scales[0], scales[-1]
    ymin, ymax = scales[0], scales[-1]
    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 0.0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 0.0
    extent = [xmin - dx/2.0, xmax + dx/2.0, ymin - dy/2.0, ymax + dy/2.0]

    im = ax.imshow(corr_matrix, origin='lower',
                   extent=extent,
                   cmap=cmap, norm=norm,
                   interpolation='nearest', aspect='equal')

    ax.plot(scales, scales, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return im, boundaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true',
                        help='display a map')
    parser.add_argument('--id', type=int, default=1,
                        help='region id')

    args = parser.parse_args()
    # if args.display:
