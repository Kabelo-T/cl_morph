import glob
import os

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy import units as u

from . import data as datutils


def print_src_morphs(source_morphs, index=0):

    print('BASIC MEASUREMENTS (NON-PARAMETRIC)')
    print('xc_centroid =', source_morphs[index].xc_centroid)
    print('yc_centroid =', source_morphs[index].yc_centroid)
    print('ellipticity_centroid =', source_morphs[index].ellipticity_centroid)
    print('elongation_centroid =', source_morphs[index].elongation_centroid)
    print('orientation_centroid =', source_morphs[index].orientation_centroid)
    print('xc_asymmetry =', source_morphs[index].xc_asymmetry)
    print('yc_asymmetry =', source_morphs[index].yc_asymmetry)
    print('ellipticity_asymmetry =',
          source_morphs[index].ellipticity_asymmetry)
    print('elongation_asymmetry =', source_morphs[index].elongation_asymmetry)
    print('orientation_asymmetry =',
          source_morphs[index].orientation_asymmetry)
    print('rpetro_circ =', source_morphs[index].rpetro_circ)
    print('rpetro_ellip =', source_morphs[index].rpetro_ellip)
    print('rhalf_circ =', source_morphs[index].rhalf_circ)
    print('rhalf_ellip =', source_morphs[index].rhalf_ellip)
    print('r20 =', source_morphs[index].r20)
    print('r80 =', source_morphs[index].r80)
    print('Gini =', source_morphs[index].gini)
    print('M20 =', source_morphs[index].m20)
    print('F(G, M20) =', source_morphs[index].gini_m20_bulge)
    print('S(G, M20) =', source_morphs[index].gini_m20_merger)
    print('sn_per_pixel =', source_morphs[index].sn_per_pixel)
    print('C =', source_morphs[index].concentration)
    print('A =', source_morphs[index].asymmetry)
    print('S =', source_morphs[index].smoothness)
    print()
    print('SERSIC MODEL')
    print('sersic_amplitude =', source_morphs[index].sersic_amplitude)
    print('sersic_rhalf =', source_morphs[index].sersic_rhalf)
    print('sersic_n =', source_morphs[index].sersic_n)
    print('sersic_xc =', source_morphs[index].sersic_xc)
    print('sersic_yc =', source_morphs[index].sersic_yc)
    print('sersic_ellip =', source_morphs[index].sersic_ellip)
    print('sersic_theta =', source_morphs[index].sersic_theta)
    print('sersic_chi2_dof =', source_morphs[index].sersic_chi2_dof)
    print()
    print('OTHER')
    print('sky_mean =', source_morphs[index].sky_mean)
    print('sky_median =', source_morphs[index].sky_median)
    print('sky_sigma =', source_morphs[index].sky_sigma)
    print('flag =', source_morphs[index].flag)
    print('flag_sersic =', source_morphs[index].flag_sersic)
    return


def find_id(file: str) -> int:
    """Find halo ID for mass accretion histories

    Parameters
    ----------
    file : str
        file name

    Returns
    -------
    idx  
    """

    ind = file.find('_0')
    sub = file[ind+1:ind+5]
    idx = int(sub.lstrip('0'))
    return idx


def create_morph_df(source_morphs, name=None, save=False):

    sources = []
    for idx, src in source_morphs:
        sources.append({
            'ID': idx,
            'xc_centroid': src.xc_centroid,
            'yc_centroid': src.yc_centroid,
            'ellipticity_asymmetry': src.ellipticity_asymmetry,
            'ellipticity_centroid': src.ellipticity_centroid,
            'elongation_asymmetry': src.elongation_asymmetry,
            'elongation_centroid': src.elongation_centroid,
            'orientation_centroid': src.orientation_centroid,
            'xc_asymmetry': src.xc_asymmetry,
            'yc_asymmetry': src.yc_asymmetry,
            'ellipticity_asymmetry': src.ellipticity_asymmetry,
            'elongation_asymmetry': src.elongation_asymmetry,
            'orientation_asymmetry': src.orientation_asymmetry,
            'rpetro_circ': src.rpetro_circ,
            'rpetro_ellip': src.rpetro_ellip,
            'rhalf_circ': src.rhalf_circ,
            'rhalf_ellip': src.rhalf_ellip,
            'r20': src.r20,
            'r50': src.r50,
            'r80': src.r80,
            'Gini': src.gini,
            'M20': src.m20,
            'F(G, M20)': src.gini_m20_bulge,
            'S(G, M20)': src.gini_m20_merger,
            'sn_per_pixel': src.sn_per_pixel,
            'C': src.concentration,
            'A': src.asymmetry,
            'S': src.smoothness,
            'sersic_amplitude': src.sersic_amplitude,
            'sersic_rhalf': src.sersic_rhalf,
            'sersic_n': src.sersic_n,
            'sersic_xc': src.sersic_xc,
            'sersic_yc': src.sersic_yc,
            'sersic_ellip': src.sersic_ellip,
            'sersic_theta': src.sersic_theta,
            'sersic_chi2_dof': src.sersic_chi2_dof,
            'flag': src.flag,
            'flag_sersic': src.flag_sersic,
            'flux_circ': src.flux_circ,
            'flux_ellip': src.flux_ellip,
            'runtime (s)': src.runtime
        })

    sources = pd.DataFrame(sources)
    sources.set_index('ID', inplace=True)
    sources.sort_index(inplace=True)
    if save:
        if name is not None:
            sources.to_csv(name)
        else:
            sources.to_csv('source_morphs.csv')
    return sources


def load_map(file, map_dir='/Users/kabelo/clusters/data/maps/zx/', id=None):

    if id is not None:
        map_file = f'/Users/kabelo/clusters/data/maps/zx/bcg_{id:04}_125_2.fits'
    else:
        if '.fits' in file:
            map_file = map_dir + file

    try:
        map = fits.open(map_file)
    except FileNotFoundError:
        print(f'File {map_file} not found.')
        return None

    map = fits.open(map_file)
    map = map[0].data
    return map


def get_mah(file: str) -> pd.DataFrame:
    """Load the mass accretion histories for a cluster.

    Parameters
    ----------
    file : str

    Returns
    -------
    mm0 : pd.DataFrame
        mass accretion history as M(z)/M(z=0)
    """
    if '.dat' not in file:
        return  # type: ignore

    mah_df = pd.read_csv(file, sep=r'\s+', index_col=False)
    masses = mah_df['Mvir(4)'][::-1]
    mpeak_a1 = np.max(mah_df['Mvir(4)'])
    norm_masses = []
    for i, m in enumerate(masses):
        mpeak = np.max(masses[:i+1])
        norm_masses.append(mpeak/mpeak_a1)

    ma = pd.DataFrame(columns=['aexp', 'mass', 'M/M0', 'redshift'])
    ma['mass'] = masses
    ma['M/M0'] = norm_masses
    ma['Redshift'] = mah_df['Redshift(0)']
    ma['aexp'] = 1 / (1+ma['Redshift'])
    ma.sort_values(by='aexp', inplace=True)

    return ma


def get_mah_all(mah_dir: str = 'data/AHF_HaloHistory', return_masses: bool = False,
                halo_ids: list = None) -> dict:
    df_dict = {}
    for f in sorted(os.listdir(mah_dir)):
        file = os.path.join(mah_dir, f)
        idx = find_id(file)
        if halo_ids is not None and idx not in halo_ids:
            continue
        # at this point m(a) is calculated for different snapshots so inhomogenous in aexp
        ma = get_mah(file)
        df_dict[idx] = ma

    mah, masses, common_aexp = datutils.interp_ma(df_dict)

    mah_df_dict = {}
    for i, key in enumerate(df_dict.keys()):
        df = pd.DataFrame(columns=['M/M0', 'mass', 'Redshift', 'aexp'])
        df['M/M0'] = mah[i]
        df['aexp'] = common_aexp
        df['mass'] = masses[i]
        df['Redshift'] = (1/df['aexp']) - 1
        mah_df_dict[key] = df

    return mah_df_dict


def get_ahf(file: str, clean=True) -> pd.DataFrame:
    """Load the Amiga Halo Finder properties for a cluster.

    Parameters
    ----------
    file : str

    Returns
    -------
    ahf : pd.DataFrame
    """
    if '.dat' not in file:
        return  # type: ignore

    ahf_df = pd.read_csv(file, sep=r'\s+', index_col=False)
    ahf_df['aexp'] = 1 / (1+ahf_df['Redshift(0)'])
    ahf_df.reset_index(inplace=True)
    ahf_df.sort_values(by='aexp', inplace=True)
    ahf_df.drop(columns=['ID(1)', 'hostHalo(2)', 'index'], inplace=True)

    return ahf_df


def get_ahf_all(ahf_dir: str = 'data/AHF_HaloHistory/', clean=True) -> dict:
    df_dict = {}
    for f in sorted(os.listdir(ahf_dir)):
        file = ahf_dir + f
        ahf = get_ahf(file, clean)
        idx = find_id(file)
        df_dict[idx] = ahf

    return df_dict


def get_virial_masses(halo_ids=None) -> np.ndarray:
    ahf = get_ahf_all()

    if halo_ids is None:    # then just use them all
        halo_ids = list(ahf.keys())

    mass = np.array([ahf[halo_id]['Mvir(4)'][ahf[halo_id]['Redshift(0)'] == 0].values[0]
                    for halo_id in halo_ids])
    return mass


AHF_Z0_PARAMS = ['numSubStruct(3)', 'Mvir(4)', 'npart(5)', 'Rvir(12)', 'Rmax(13)', 'r2(14)',
                 'mbp_offset(15)', 'com_offset(16)', 'Vmax(17)', 'v_esc(18)', 'sigV(19)',
                 'lambda(20)', 'lambdaE(21)', 'b(25)', 'c(26)', 'ovdens(36)', 'fMhires(38)',
                 'Ekin(39)', 'Epot(40)', 'cNFW(43)', 'M_gas(45)', 'M_star(65)']


def get_ahf_z0(halo_ids=None, params: list = AHF_Z0_PARAMS) -> pd.DataFrame:
    """Curated AHF scalar parameters at redshift 0, one row per halo.

    Parameters
    ----------
    halo_ids : list, optional
        restrict to these halo IDs, by default None (use all halos)
    params : list, optional
        AHF columns to include, by default AHF_Z0_PARAMS

    Returns
    -------
    pd.DataFrame
        indexed by halo ID, one column per param
    """
    ahf_dict = get_ahf_all()
    if halo_ids is None:
        halo_ids = list(ahf_dict.keys())

    rows = []
    for halo_id in halo_ids:
        df = ahf_dict[halo_id]
        row = df.loc[df['Redshift(0)'] == 0, params].iloc[0]
        row.name = halo_id
        rows.append(row)

    z0df = pd.DataFrame(rows)
    z0df.index.name = 'ID'
    return z0df


def get_virial_radius(idx: int = None, file: str = None) -> tuple:
    """Get the virial radius for a given halo at z = 0, in units of h^-1 kpc.

    Virial radius is defined at some overdensity above critical density, defined by the ovdens column. 

    Parameters
    ----------
    idx : int, optional
        cluster id, by default None
    file : str, optional
        can also specify by file, by default None

    Returns
    -------
    float
        virial radius in units of h^-1 kpc
    """
    if idx is None and file is None:
        print('Must specify either ID or file')
        return None

    if file is None:
        file = f'data/AHF_HaloHistory/NewMDCLUSTER_{str(idx).zfill(4)}_halo_128000000000001.dat'

    ahf = get_ahf(file)
    rvir = ahf.loc[ahf['Redshift(0)'] == 0.0, 'Rvir(12)'].values[0]
    ovdens = ahf.loc[ahf['Redshift(0)'] == 0.0, 'ovdens(36)'].values[0]
    rvir = rvir * u.kpc
    return rvir, ovdens


def find_result_csv(proj_dir: str, prefix: str) -> str:
    """Locate a statmorph results CSV by aperture prefix (e.g. 'r_0.03Mpc'),
    tolerating the trailing cluster-count suffix that measure_morphs.py bakes
    into the filename, which varies by run and by projection.
    """
    matches = glob.glob(os.path.join(proj_dir, f'{prefix}_*.csv'))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one match for prefix {prefix!r} in {proj_dir!r}, found {matches}")
    return matches[0]


def get_morph(halo_ids=None, sm_dir: str = 'results/zx/rin50.0kpc_rout1.0Mpc_305.csv', all=False) -> pd.DataFrame:
    sm_df = pd.read_csv(sm_dir)
    sm_df.set_index('ID', inplace=True)
    sm_df.drop(columns=['flag', 'flag_sersic', 'flux_circ', 'flux_ellip',
                        'sn_per_pixel', 'runtime (s)'], inplace=True)
    if not all:
        sm_df.drop(columns=['xc_centroid', 'yc_centroid', 'ellipticity_asymmetry',
                            'ellipticity_centroid', 'elongation_asymmetry', 'elongation_centroid',
                            'orientation_centroid', 'xc_asymmetry', 'yc_asymmetry',
                            'orientation_asymmetry', 'rpetro_circ', 'rpetro_ellip',
                            'r20', 'r50', 'r80', 'F(G, M20)',
                            'S(G, M20)', 'sersic_n', 'sersic_xc', 'sersic_yc',
                            'sersic_ellip', 'sersic_theta', 'sersic_chi2_dof',
                            'rhalf_ellip', 'M20', 'S', 'sersic_rhalf', 'Gini'], inplace=True)

    if halo_ids is not None:
        sm_df = sm_df.reindex(halo_ids)
    return sm_df


def get_morphologies(halo_ids=None, f='rin50.0kpc_rout1.0Mpc_305.csv', all=False):
    sm_df_zx = get_morph(halo_ids, f'results/zx/{f}', all=all)
    sm_df_xy = get_morph(halo_ids, f'results/xy/{f}', all=all)
    sm_df_yz = get_morph(halo_ids, f'results/yz/{f}', all=all)
    smdf = pd.concat(
        [sm_df_xy.reset_index(), sm_df_yz.reset_index(), sm_df_zx.reset_index()])
    return smdf


def get_ds_theory_today(file_path: str = 'data/GadgetX-DS-theory-snap-128.txt'):
    ds_z0 = {}
    with open(file_path) as f:
        for i, x in enumerate(f):
            if i == 0:
                continue
            k, v = list(map(int, x.split()))
            ds_z0[k] = v
    return ds_z0


def get_ds(halo_ids=None, snap: int = 128, clean=True) -> pd.DataFrame:
    """Gets dynamical state dataframe for all 324 regions.

    Parameters
    ----------
    snap : int, optional
        snapshot number [32, 128], by default 128

    Returns
    -------
    pd.DataFrame
    """
    file = f'data/G3X_progenitors/DS_G3X_snap_{str(snap).zfill(3)}_center-cluster_progenitors.txt'
    dsdf = pd.read_csv(file, sep=r'\s+', header=0)
    int_columns = [0, 1, 2, 7]
    column_names = dsdf.columns
    for idx in range(len(column_names)):
        col_name = column_names[idx]
        if idx in int_columns:
            dsdf[col_name] = dsdf[col_name].astype(int)
        else:
            dsdf[col_name] = dsdf[col_name].astype(float)
    if clean:
        dsdf.drop(columns=['Hid[1]', 'DS_200[2]', 'DS_500[7]'], inplace=True)

    dsdf.rename(columns={'rID[0]': 'ID'}, inplace=True)
    dsdf.set_index('ID', inplace=True)
    rads = np.load('data/cluster_rsp_rtrunc.npz')
    rdf = pd.DataFrame()
    rdf['ID'] = rads['cluster_ID']
    rdf['r_trunc'] = rads['r_trunc']
    rdf['r_sp'] = rads['r_sp']
    rdf.set_index(['ID'], inplace=True)
    dsdf = dsdf.join(rdf, on='ID')
    if halo_ids is not None:
        dsdf = dsdf.loc[halo_ids]

    return dsdf


def get_ds_all():
    df_dict = {}
    for i in range(32, 129):
        dsdf = get_ds(snap=i)
        dsdf['snap'] = i
        df_dict[i] = dsdf

    return df_dict


def get_m14(halo_ids=None, file_path: str = 'data/mag_diff_GadgetX_3k.csv') -> pd.DataFrame:
    m14 = pd.read_csv(file_path)
    m14.set_index('ID', inplace=True)
    if halo_ids is not None:
        m14 = m14.loc[halo_ids]
    return m14


def valid_ids(ixy, iyz, izx):
    idxy = {int(k) for k in ixy.index}
    idyz = {int(k) for k in iyz.index}
    idzx = {int(k) for k in izx.index}
    halo_ids = list(idxy & idyz & idzx)

    return halo_ids


def load_sm(stack=True):
    """Load SM (statmorph) features.

    Parameters
    ----------
    stack : bool, optional
        if True, stack all 3 projections (xy, yz, zx) into one dataframe;
        if False, use only the zx projection, by default True
    """
    ixy = get_morph(sm_dir=find_result_csv('results/xy', 'r_0.03Mpc'))
    iyz = get_morph(sm_dir=find_result_csv('results/yz', 'r_0.03Mpc'))
    izx = get_morph(sm_dir=find_result_csv('results/zx', 'r_0.03Mpc'))
    halo_ids = valid_ids(ixy, iyz, izx)
    m14 = get_m14()
    m14 = m14.loc[halo_ids]

    if stack:
        inner_df = pd.concat(
            [ixy.loc[halo_ids], iyz.loc[halo_ids], izx.loc[halo_ids]])
        iconc = inner_df['C']
        iconc.name = 'core_C'
        m14_xyz = pd.Series(
            np.hstack([m14['proj_x'], m14['proj_y'], m14['proj_z']]), name='m14')
        smdf = get_morphologies(halo_ids)
        n_halos = len(halo_ids)
        assert m14_xyz.shape[0] == 3 * n_halos
        assert smdf.shape[0] == 3 * n_halos
        assert iconc.shape[0] == 3 * n_halos
    else:
        iconc = izx.loc[halo_ids]['C']
        iconc.name = 'core_C'
        m14_xyz = m14['proj_z']
        m14_xyz.name = 'm14'
        smdf = get_morph(halo_ids)  # defaults to the zx projection

    df = pd.concat([smdf.reset_index(), m14_xyz.reset_index(drop=True),
                    iconc.reset_index(drop=True)], axis=1)

    return df, halo_ids


def load(features, stack=True):
    if features == 'sm':
        df, halo_ids = load_sm(stack=stack)
    else:
        dsdf = get_ds()
        dsdf.drop(columns=['eta_500[8]', 'delta_500[9]',
                           'fm_500[10]', 'fm2_500[11]'], inplace=True)
        m14 = get_m14()
        df = pd.concat([dsdf, m14['3d']], axis=1, join='inner')
        df.sort_index(inplace=True)
        halo_ids = list(df.index)

    mah_dict = get_mah_all(halo_ids=halo_ids)
    ma, _, aexp_bins = datutils.interp_ma(mah_dict)
    am, mass_bins = datutils.build_am(ma=ma, scales=aexp_bins,
                                      min_mass_bin=0.01, log_spacing=True)
    return mah_dict, ma, am, aexp_bins, mass_bins, df
