"""Calabrese et al. (2019) density validation - reconstructed 2026-07-25, reproducible from source.

Source data: `Papers/Calabrese_Table6.xlsx`, their Table 6 digitised - the
[x CO2 + (1-x) NaCl(aq)] system at m = 0.77 and 2.50 mol/kg, 275-449 K, 1-100 MPa,
303 points of which 96 are CO2-free (x = 0) and 207 are CO2-loaded.
Paper: J. Chem. Eng. Data 64, 3831-3847, DOI 10.1021/acs.jced.9b00248.

WHY THIS EXISTS: the original analysis survived only as `calabrese_frac_validation.csv`;
its generating script was lost, and the stored implied volumes could not be
reconstructed because they used the MEASURED CO2-free densities as the baseline
(per Mark's implied-V_phi directive) rather than Spivey. This script rebuilds the
analysis from the spreadsheet with every step explicit, and self-checks against the
archived CSV.

METHOD. Calabrese model the density with their Eq. 18

    rho = (x M_CO2 + (1-x) M_b) / (x V_CO2 + (1-x) V_b)

where M_b = M_w (1 + m M_s)/(1 + m M_w) is the mean molar mass of the CO2-free
brine, V_b = M_b/rho_b its molar volume, and x is the CO2 mole fraction on the
paired-NaCl basis. Note this is exactly the corrected Garcia solvent basis: the
dissolved-gas volume is added to the volume of the WHOLE CO2-free brine, salt
included. Their own formulation is therefore independent confirmation of the
2026-07-25 basis correction, and the identity against our per-kg-of-water form is
asserted below.

Inverting Eq. 18 for the volume each measured point implies:

    V_CO2 = [ (x M_CO2 + (1-x) M_b)/rho - (1-x) V_b ] / x

Baseline: the CO2-free rows sit on the same (T, P) grid as the loaded rows but at
temperatures offset by up to 0.2 K, so rho_b is taken from a least-squares surface
in (T, P) fitted per molality to the measured x = 0 rows, and the fit residual is
reported. A nearest-neighbour baseline is computed alongside as a cross-check.

Comparators: the Plyasunov V2inf used throughout this project, and Calabrese's own
Eq. 19 fit to the (CO2 + H2O) system of McBride-Wright et al., whose Table 10
coefficients are transcribed below. Those coefficients were recovered from a
rendered page image, not pdftotext, because the exponent minus signs are dropped in
the text layer; they are verified against the archived V_cmw column to 1e-14.
"""

# --- brine_props repository layout (inserted by build_public_repo.py) ---
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_HERE = _os.path.dirname(_os.path.abspath(__file__))
# when run as a script, drop the script's own directory from sys.path so the
# subpackage name (e.g. `validation`) resolves to the package, not to a module
if _sys.path and _os.path.abspath(_sys.path[0]) == _HERE:
    _sys.path.pop(0)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_DATA = _os.path.join(_ROOT, 'data')
_RESULTS = _os.path.join(_ROOT, 'validation', 'results')
_FIGOUT = _os.path.join(_ROOT, 'figures', 'out')
_TABLES = _os.path.join(_ROOT, 'examples', 'tables')
for _d in (_RESULTS, _FIGOUT, _TABLES):
    _os.makedirs(_d, exist_ok=True)
# ------------------------------------------------------------------------

import sys, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Analysis scripts must use the SHIPPED default route, not a hard-wired one.
# Before 2026-07-25 this imported V_phi from plyasunov_model directly, so after
# the default moved to PR+VSHIFT this script kept reporting the old route.
from brine_gas.plyasunov_model import gas_mw
from brine_gas.vphi_route import V_phi
from brine_gas.brine_properties import M_NACL

ROOT = os.path.dirname(HERE)
XLSX = os.path.join(_DATA, 'calabrese2019_table6.csv')   # extracted from the authors' xlsx
ARCHIVE = os.path.join(_DATA, 'calabrese_frac_validation.csv')
OUT = os.path.join(_RESULTS, 'calabrese_density_results.csv')

MW_WATER = 18.015268          # g/mol
M_CO2 = gas_mw('CO2')         # g/mol

# Calabrese Eq. 19 / Table 10: V_CO2/(cm3 mol-1) = sum_i sum_j a_ij (T/K)^i (p/MPa)^j
_A = {(0, 0): 51.19, (1, 0): -0.15575, (2, 0): 3.2955e-4,
      (0, 1): -6.0708e-2, (1, 1): 5.5026e-4, (2, 1): -1.2114e-6}



def x_saltfree_from_pseudo(x, m):
    """Calabrese's pseudo-component mole fraction x = n_g/(n_g + n_w + n_s) to
    the salt-free basis x_sf = n_g/(n_g + n_w) at NaCl molality m.

    Per mole of mixture n_g = x and n_w + n_s = 1 - x with n_w : n_s = n_w0 : m,
    n_w0 = 1000/M_w, so x_sf = x (n_w0 + m) / (n_w0 + x m). Both expressions
    this replaced (2026-09-05, independent review item 2.3) were wrong; the
    round trip against a mole inventory is pinned in validation.py.
    """
    nw0 = 1000.0 / MW_WATER
    return x * (nw0 + m) / (nw0 + x * m)

def V_co2_calabrese(T, P):
    """Apparent molar volume of CO2(aq) from Calabrese Eq. 19 (cm3/mol)."""
    return sum(a * T ** i * P ** j for (i, j), a in _A.items())


def brine_molar_mass(m):
    """Mean molar mass of CO2-free NaCl brine, g/mol (paired-NaCl basis)."""
    return (1000.0 + m * M_NACL) / (1000.0 / MW_WATER + m)


def implied_V(x, rho_gcc, m, rho_b_gcc):
    """Invert Calabrese Eq. 18 for the apparent molar volume of CO2 (cm3/mol)."""
    Mb = brine_molar_mass(m)
    Vb = Mb / rho_b_gcc
    return ((x * M_CO2 + (1.0 - x) * Mb) / rho_gcc - (1.0 - x) * Vb) / x


# Baseline surface order, chosen by leave-one-out cross-validation over the
# measured CO2-free rows (see _fit_baseline). Quartic in T, quadratic in P.
_NT, _NP = 4, 2
_TC, _TS, _PC, _PS = 360.0, 90.0, 50.0, 50.0     # centring/scaling for conditioning


def _design(T, P):
    t = (np.asarray(T, float) - _TC) / _TS
    p = (np.asarray(P, float) - _PC) / _PS
    return np.column_stack([t ** i * p ** j
                            for i in range(_NT + 1) for j in range(_NP + 1)])


def _fit_baseline(sub):
    """
    Least-squares rho_b(T, P) surface through the measured CO2-free rows.

    The 48 x = 0 rows per molality lie near an 8 isotherm x 6 isobar grid, but
    the actual pressures scatter by up to 0.2 MPa about each nominal level, so
    they do not form a rectangular grid and cannot be spline-interpolated
    directly. A polynomial surface is fitted instead, quartic in T and quadratic
    in P, on centred and scaled variables. Order was chosen by leave-one-out
    cross-validation; raw (unscaled) powers are badly conditioned above T^3 and
    were what made a first attempt look hopeless at ~1 kg/m3.

    Returns (evaluator, LOO RMS, LOO max) in kg/m3. The LOO RMS is the honest
    baseline uncertainty and propagates directly into the implied volumes.
    """
    T, P, y = sub['T'].values, sub['P'].values, sub['rho'].values
    A = _design(T, P)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    loo = []
    for k in range(len(y)):
        msk = np.ones(len(y), bool)
        msk[k] = False
        c, *_ = np.linalg.lstsq(A[msk], y[msk], rcond=None)
        loo.append(A[k] @ c - y[k])
    loo = np.asarray(loo)
    return (lambda Tq, Pq: float((_design([Tq], [Pq]) @ coef)[0]),
            float(np.sqrt(np.mean(loo ** 2))), float(np.max(np.abs(loo))))


def main():
    raw = pd.read_csv(XLSX)
    raw.columns = ['m', 'x', 'T', 'P', 'rho']
    base = raw[raw.x == 0.0].copy()
    load = raw[raw.x > 0.0].copy().reset_index(drop=True)
    print(f'Source: {os.path.relpath(XLSX, ROOT)}  '
          f'({len(raw)} rows, {len(base)} CO2-free, {len(load)} loaded)')

    rows = []
    for m, sub in base.groupby('m'):
        surf, rms, mx = _fit_baseline(sub)
        # propagate the baseline uncertainty into V_phi: dV/drho_b = -Vb/rho_b
        # per unit (1-x)/x, i.e. roughly (Mb/rho_b^2)*(1-x)/x
        Mb = brine_molar_mass(m)
        rho_typ = sub.rho.mean() / 1000.0
        x_typ = load[load.m == m].x.mean()
        dV = (Mb / rho_typ ** 2) * (1 - x_typ) / x_typ * (rms / 1000.0)
        print(f'  CO2-free baseline, m = {m:.2f} mol/kg: {len(sub)} points, '
              f'leave-one-out RMS {rms:.3f} kg/m3 (max {mx:.3f}) '
              f'-> +/-{dV:.2f} cm3/mol on implied V_phi')
        for _, r in load[load.m == m].iterrows():
            rho_b = surf(r['T'], r['P'])
            # nearest-measured cross-check baseline: closest node in a scaled
            # (T, P) metric, so a small T offset never outranks a 70 MPa gap
            dist = (((sub['T'].values - r['T']) / 25.0) ** 2
                    + ((sub['P'].values - r['P']) / 15.0) ** 2)
            rho_b_near = float(sub['rho'].values[int(np.argmin(dist))])

            V_imp = implied_V(r.x, r.rho / 1000.0, m, rho_b / 1000.0)
            V_imp_near = implied_V(r.x, r.rho / 1000.0, m, rho_b_near / 1000.0)
            # A few measured pressures sit marginally above the IAPWS-IF97
            # Region 1 limit (up to 100.12 MPa). V_phi varies by under
            # 0.01 cm3/mol over that excess, so the model is evaluated at the
            # limit and the row is flagged rather than silently extrapolated.
            P_model = min(r['P'], 100.0)
            V_ply = float(V_phi('CO2', r['T'], P_model, 'auto', r['m']))
            V_cal = V_co2_calabrese(r['T'], r['P'])

            # superseded water-mass basis, for the size of the basis correction
            x_sf = x_saltfree_from_pseudo(r.x, m)
            W = (1.0 - x_sf) * MW_WATER
            V_old = ((W + x_sf * M_CO2) / (r.rho / 1000.0)
                     - W / (rho_b / 1000.0)) / x_sf

            rows.append(dict(m=m, x=r.x, T=r['T'], P=r['P'],
                             rho_meas=r.rho, rho_base_fit=rho_b,
                             rho_base_nearest=rho_b_near,
                             f_exp_pct=100.0 * (r.rho / rho_b - 1.0),
                             V_imp=V_imp, V_imp_nearest=V_imp_near,
                             V_imp_watermass=V_old,
                             V_ply=V_ply, V_cal=V_cal,
                             P_clamped=bool(r['P'] > 100.0),
                             err_ply_pct=100.0 * (V_ply / V_imp - 1.0),
                             err_cal_pct=100.0 * (V_cal / V_imp - 1.0)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    # ---------------- self-checks ----------------
    print('\nSELF-CHECKS')
    a = pd.read_csv(ARCHIVE)
    chk = np.max(np.abs([V_co2_calabrese(r['T'], r['P']) - r.V_cmw
                         for _, r in a.iterrows()]))
    print(f'  Eq. 19 vs archived V_cmw column          max |diff| {chk:.2e} cm3/mol')

    # Eq. 18 must be identical to the per-kg-of-water Garcia form used in the code
    worst = 0.0
    for _, r in d.iterrows():
        nb = 1000.0 / MW_WATER + r.m                 # brine pseudo-moles per kg water
        n2 = r.x * nb / (1.0 - r.x)                  # CO2 moles per kg water
        Wb = 1000.0 + r.m * M_NACL
        rho_pkw = ((Wb + n2 * M_CO2)
                   / (Wb / (r.rho_base_fit / 1000.0) + n2 * r.V_imp))
        worst = max(worst, abs(rho_pkw * 1000.0 - r.rho_meas))
    print(f'  Eq. 18 vs per-kg-water Garcia form       max |diff| {worst:.2e} kg/m3')

    mrg = d.merge(a[['m', 'x', 'T', 'P', 'V_imp']].rename(columns={'V_imp': 'V_imp_archive'}),
                  on=['m', 'x', 'T', 'P'], how='inner')
    print(f'  matched {len(mrg)} of {len(d)} rows against the archived CSV')
    if len(mrg):
        dv = mrg.V_imp - mrg.V_imp_archive
        dvo = mrg.V_imp_watermass - mrg.V_imp_archive
        print(f'  new (brine-mass) vs archive: mean {dv.mean():+.3f}, '
              f'max |{dv.abs().max():.3f}| cm3/mol')
        print(f'  water-mass form vs archive:  mean {dvo.mean():+.3f}, '
              f'max |{dvo.abs().max():.3f}| cm3/mol')
    bd = d.V_imp - d.V_imp_nearest
    print(f'  fitted vs nearest-measured baseline:     mean {bd.mean():+.3f}, '
          f'max |{bd.abs().max():.3f}| cm3/mol')

    # ---------------- results ----------------
    print('\nIMPLIED V_phi (cm3/mol) AND MODEL ERROR, by molality')
    g = d.groupby('m')
    print(pd.DataFrame({
        'n': g.size(),
        'V_imp_mean': g.V_imp.mean(),
        'V_ply_mean': g.V_ply.mean(),
        'V_cal_mean': g.V_cal.mean(),
        'ply_err_mean_%': g.err_ply_pct.mean(),
        'ply_err_meanabs_%': g.err_ply_pct.apply(lambda s: s.abs().mean()),
        'cal_err_mean_%': g.err_cal_pct.mean(),
        'cal_err_meanabs_%': g.err_cal_pct.apply(lambda s: s.abs().mean()),
    }).round(2).to_string())

    hot = d[d['T'] > 300]
    print('\nAbove 300 K (the manuscript quote), model vs implied:')
    print(pd.DataFrame({
        'n': hot.groupby('m').size(),
        'dV_mean_cm3': hot.groupby('m').apply(
            lambda s: (s.V_ply - s.V_imp).mean(), include_groups=False),
        'err_mean_%': hot.groupby('m').err_ply_pct.mean(),
        'err_meanabs_%': hot.groupby('m').err_ply_pct.apply(lambda s: s.abs().mean()),
    }).round(2).to_string())

    # density-side framing: fractional increase predicted with the model V_phi
    Mb_of = {m: brine_molar_mass(m) for m in d.m.unique()}
    f_mod = []
    for _, r in d.iterrows():
        Mb = Mb_of[r.m]
        rb = r.rho_base_fit / 1000.0
        rho_mod = ((r.x * M_CO2 + (1 - r.x) * Mb)
                   / (r.x * r.V_ply + (1 - r.x) * Mb / rb))
        f_mod.append(100.0 * (rho_mod / rb - 1.0))
    d['f_model_pct'] = f_mod
    d['f_err_pp'] = d.f_model_pct - d.f_exp_pct
    print('\nDENSITY SIDE, error in the fractional increase (percentage points):')
    print(pd.DataFrame({
        'n': d.groupby('m').size(),
        'f_exp_min_pp': d.groupby('m').f_exp_pct.min(),
        'f_exp_max_pp': d.groupby('m').f_exp_pct.max(),
        'err_mean_pp': d.groupby('m').f_err_pp.mean(),
        'err_meanabs_pp': d.groupby('m').f_err_pp.apply(lambda s: s.abs().mean()),
        'err_maxabs_pp': d.groupby('m').f_err_pp.apply(lambda s: s.abs().max()),
    }).round(3).to_string())
    hot2 = d[d['T'] > 300]
    print('  above 300 K only, mean error (pp): '
          + ', '.join(f'm={k}: {v:+.3f}'
                      for k, v in hot2.groupby('m').f_err_pp.mean().items()))

    print('\nBy isotherm (all pressures pooled), model error % (via vphi_route):')
    piv = d.assign(Tr=d['T'].round(0)).pivot_table(
        index='Tr', columns='m', values='err_ply_pct', aggfunc='mean')
    print(piv.round(1).to_string())

    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
