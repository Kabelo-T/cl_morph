from multicam.multicam import MultiCAM  # type: ignore
from multicam.train import get_tt_indices  # type: ignore
from multicam.qt import qt_gauss_base, qt_inverse_gauss_base  # type: ignore
from scipy.stats import spearmanr
import numpy as np
from sklearn.model_selection import train_test_split


def build_mcam_corrs(spearm_corr: np.ndarray):
    """Build the correlation arrays with errors

    Parameters
    ----------
    spearm_corr : np.ndarray
        Spearman correlation array of shape (n_seeds, n_targets)
    """
    p25, p50, p75 = [], [], []

    for x in spearm_corr.T:
        p25.append(np.percentile(x, q=25))
        p50.append(np.percentile(x, q=50))
        p75.append(np.percentile(x, q=75))

    return p25, p50, p75


def multicorr(x: np.ndarray, y: np.ndarray, alpha: float = 6.0, proj=False, scoring=False) -> tuple:
    """Calculate the Spearman correlation, RMSE, and R^2 between predicted and true values

    Parameters
    ----------
    x : np.ndarray
        features
    y : np.ndarray
        targets
    alpha : float, optional
        regularization strength for Ridge regression in MultiCAM, by default 0.0
    proj : bool, optional
        if using projected features/targets need to handle projection leakage in train-test split, default False
    """
    n_targets = y.shape[1]
    n_features = x.shape[1]

    if proj:
        n_halos = x.shape[0] // 3
    else:
        n_halos = x.shape[0]
    halo_idx = np.arange(n_halos)
    sp_corr = []
    weights = np.zeros((n_targets, n_features))

    r = 100
    rmse = np.zeros(n_targets)
    r2 = np.zeros(n_targets)
    # Monte carlo cross validation loop for train-test splits and model fitting
    for seed in range(r):
        train_idx, test_idx = train_test_split(
            halo_idx, test_size=0.25, random_state=seed
        )

        if proj:
            train_idx = np.concatenate(
                [train_idx, train_idx + n_halos, train_idx + 2*n_halos])

            test_idx = np.concatenate(
                [test_idx,  test_idx + n_halos, test_idx + 2*n_halos])

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        mv_model = MultiCAM(n_features, n_targets, alpha=alpha)
        mv_model.fit(x_train, y_train)

        coeff = mv_model.reg.coef_
        weights += coeff / r
        y_pred = mv_model.predict(x_test)

        sp_corr.append([
            spearmanr(yt, yp).correlation
            for yt, yp in zip(y_test.T, y_pred.T)
        ])

        error = y_test - y_pred
        rmse += np.sqrt(np.mean(error**2, axis=0)) / r

        ss_res = np.sum(error**2, axis=0)
        ss_tot = np.sum((y_test - np.mean(y_test, axis=0))**2, axis=0)

        # Guard against divide-by-zero if a target is constant
        r2_split = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

        r2 += r2_split / r

    sp_corr = np.array(sp_corr)
    p25, p50, p75 = build_mcam_corrs(sp_corr)

    if scoring:
        # if trying to rank order mah, need to return the last model to get the predicted values for the test set to calculate the scores
        return sp_corr, p25, p50, p75, weights, rmse, r2, mv_model
    else:
        return sp_corr, p25, p50, p75, weights, rmse, r2


def main():
    from . import data as datutils
    from . import file as futils

    mah_dict = futils.get_mah_all()
    ma, _, aexp_bins = datutils.interp_ma(mah_dict)
    am, mass_bins = datutils.build_am(ma=ma, scales=aexp_bins,
                                      min_mass_bin=0.01, log_spacing=True)

    stacked_ma = np.vstack([ma]*3)  # because we have 3 projections
    stacked_am = np.vstack([am]*3)  # because we have 3 projections

    halo_ids = np.array([int(k) for k in mah_dict.keys()])
    m14 = futils.get_m14(halo_ids)
    m14_stacked = np.hstack([m14['proj_x'], m14['proj_y'], m14['proj_z']])

    dsdf = futils.get_ds(halo_ids)
    dsdf.drop(columns=['eta_500[8]', 'delta_500[9]',
                       'fm_500[10]', 'fm2_500[11]'], inplace=True)
    dsdf = dsdf.join(m14['3d'])

    smdf = futils.get_morphologies(halo_ids)
    inner_df = futils.get_morphologies(halo_ids, 'rin30.0kpc_rout1.0Mpc_305.csv')
    iconc = inner_df['C']
    iconc.name = 'core_C'

    mass = futils.get_virial_masses(halo_ids)

    logmass = np.log10(mass)
    logmass_stacked = np.hstack([logmass]*3)  # because we have 3 projections

    n_halos = len(am)
    assert m14_stacked.shape[0] == 3 * n_halos
    assert stacked_am.shape[0] == 3 * n_halos
    assert stacked_ma.shape[0] == 3 * n_halos
    assert smdf.shape[0] == 3 * n_halos
    assert iconc.shape[0] == 3 * n_halos

    x = np.vstack([dsdf['eta_200[3]'], dsdf['delta_200[4]'],
                  dsdf['fm_200[5]'], dsdf['fm2_200[6]'], dsdf['3d']]).T
    y = ma
    _, ap25, ap50, ap75, _, _, _ = multicorr(x, y, proj=False)


if __name__ == "__main__":
    main()
