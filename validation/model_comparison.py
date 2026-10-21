"""Model comparison on the same data: the delivered V_phi route against the
published correlations of the same quantity (Reviewer 2, FPE submission:
"there is no clear comparison with existing models on the same data").

PART A - freshwater infinite-dilution volume, five gases, every point at its
own (T, P). Scored on the calibration set (fit_pr_vshift.calibration_set) and
on the held-out determinations of manuscript Section 4.2 (fit_pr_vshift.HELD_OUT
less Lauder's two finite-loading values, plus Zhou & Battino's excluded 303.15 K
N2 value, which Section 4.2 lists). Models:
  route      vphi_route.V_phi, default PR + VSHIFT
  plyasunov  plyasunov_model.V2_inf (Part II CO2, Part III H2S, Part IV H2/N2/CH4)
  ad03       Akinfiev & Diamond 2003, formula of akinfiev_diamond_h2s.py,
             Table 1 parameters. All five gases checked against the primary
             (Papers/52, Table 1) and against CHNOSZ OBIGT AD.csv (AD03.1):
             identical. H2S is the Table 1 'c' (below-200 degC) set. With IF97
             water it returns Table 1's 298 K V2 to 0.16 cm3/mol (H2 0.33).
  garcia     Garcia 2001 Eq. 3, CO2 only, T only
Because the route's shifts are FITTED to the calibration set, a leave-one-
SOURCE-out score is added: drop one source, refit s by the closed form
(point-weighted mean of (raw - measured)/b, which is what minimising the RMS
in fit_pr_vshift gives), score the dropped source.

What the comparators were themselves fitted to (their papers):
  Plyasunov  Hnedkovsky 1996 V2inf with higher weight (CH4 30 pts Part IV,
             CO2 45 with Crovetto Part II, H2S 34 Part III), and the 298.15 K
             selected values compiled from Moore, Tiepel, Zhou, Bignell,
             Barbero, Murphy & Gaines (Part I Table 2, Part II/III tables).
  AD03       Henry-constant T dependence plus ONE 298 K V2inf per gas: Moore
             (N2 35.7), Tiepel (H2 25.2, CH4 37.3), Barbero 1983 (CO2 32.8,
             H2S 34.9). Hnedkovsky is not used.
  Garcia     fitted to a literature compilation of CO2 V_phi (Paper 1).

PART B - CO2-loaded NaCl brine at MEASURED loading. The same per-kg-water
balance for every model, rho = (Wb + n2 M2)/(Wb/rho1 + n2 V), Wb = 1000 +
m M_NaCl; only V changes. Baselines: for Yan 2011 the Rowe-Chou brine they
calibrated against (IF97 at m = 0; Spivey before 2026-10-21), for Calabrese
their own measured gas-free surface.
Metric: error in the increment rho/rho1 - 1, percentage points.

Spycher & Pruess's constant CO2 partial molar volume is OMITTED: no Spycher-
Pruess paper is in Papers/, so the value cannot be confirmed from the primary.

Usage:  python model_comparison.py [--check]
Writes code/model_comparison_results.csv.
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

import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

import brine_gas.pr_vphi_model as PR
import validation.akinfiev_diamond_h2s as AD
import validation.calabrese_density_validation as CAL
from fits.fit_pr_vshift import calibration_set, HELD_OUT
from brine_gas.vphi_route import V_phi, salt_fraction
from brine_gas.plyasunov_model import V2_inf as V_ply, gas_mw
from validation.rowe_chou import yan_baseline
from brine_gas.brine_properties import rho_brine, salinity_from_molality, M_NACL
from brine_gas.water_properties import rho_w

OUT = os.path.join(_RESULTS, 'model_comparison_results.csv')
GASES = ('CH4', 'CO2', 'H2S', 'N2', 'H2')
MW_WATER = 18.015268

# Akinfiev & Diamond 2003 Table 1 (Papers/52), xi, a / cm3 g-1, b / cm3 K0.5 g-1.
# H2S, CO2, CH4 are already in AD.AD03; N2 and H2 added here so the shared
# formula evaluates them. Identical to CHNOSZ AD.csv (AD03.1) for all five.
AD.AD03.update({'N2': dict(a=-11.5380, b=14.6278, xi=-0.0320),
                'H2': dict(a=-8.4596, b=10.8301, xi=0.3090)})

# Zhou & Battino 2001 N2 at 303.15 K, 32.5 +/- 0.8; excluded with that paper's
# 303 K column, listed as held out in manuscript Section 4.2.
ZHOU_N2_303 = (303.15, 0.1, 32.5, 'Zhou 2001 303.15 K (excluded column)')


def garcia_CO2_vphi(T_C):
    """Garcia (2001) Eq. 3, copied from validation.py (which runs on import)."""
    return 37.51 - 9.585e-2 * T_C + 8.740e-4 * T_C**2 - 5.044e-7 * T_C**3


FRESH = {
    'route': lambda g, T, P: V_phi(g, T, P),
    'plyasunov': lambda g, T, P: float(V_ply(g, T, P)),
    # AD's central-difference kappa_T steps 0.1 MPa past the IF97 limit at 100
    'ad03': lambda g, T, P: AD.v_inf_ad03(g, T, min(P, 99.9)),
    'garcia': lambda g, T, P: garcia_CO2_vphi(T - 273.15) if g == 'CO2' else np.nan,
}


def held_out_set():
    out = {g: [p for p in HELD_OUT.get(g, []) if 'finite loading' not in p[3]]
           for g in GASES}
    out['N2'].append(ZHOU_N2_303)
    return out


def _stats(e):
    e = np.asarray(e, float)
    return dict(n=len(e), bias=float(e.mean()), mae=float(np.abs(e).mean()),
                max=float(np.abs(e).max()))


def score_fresh(pts, fn):
    return _stats([100 * (fn(T, P) - V) / V for T, P, V, _ in pts])


def loso(gas, pts):
    """Leave-one-source-out: refit s without each source, score that source."""
    b = PR.b_covolume(gas)
    raw = np.array([PR.V2_inf_raw(gas, T, P) for T, P, _, _ in pts])
    meas = np.array([V for _, _, V, _ in pts])
    src = np.array([p[3] for p in pts])
    s_all = float(np.mean((raw - meas) / b))
    errs, per = [], {}
    for k in sorted(set(src)):
        out = src == k
        s_k = float(np.mean((raw[~out] - meas[~out]) / b))
        e = 100 * ((raw[out] - s_k * b) - meas[out]) / meas[out]
        errs.extend(e)
        per[k] = (s_k, int(out.sum()), float(np.abs(e).mean()))
    return s_all, _stats(errs), per


def part_a():
    cal, held = calibration_set(), held_out_set()
    rows, lo = [], {}
    for g in GASES:
        for subset, pts in (('calibration', cal[g]), ('held-out', held[g])):
            if not pts:    # H2S has no held-out determination
                continue
            for name, fn in FRESH.items():
                if name == 'garcia' and g != 'CO2':
                    continue
                r = score_fresh(pts, lambda T, P: fn(g, T, P))
                rows.append(dict(part='A', set=g, subset=subset, model=name, **r))
        # Per-source split of the calibration set (2026-10-21, adversarial read):
        # a pooled MAE can hide that a verdict rests on one source.
        for k in sorted({p[3] for p in cal[g]}):
            sub_pts = [p for p in cal[g] if p[3] == k]
            for name, fn in FRESH.items():
                if name == 'garcia' and g != 'CO2':
                    continue
                r = score_fresh(sub_pts, lambda T, P: fn(g, T, P))
                rows.append(dict(part='A', set=g, subset=f'source:{k}', model=name, **r))
        s_all, r, per = loso(g, cal[g])
        assert abs(s_all - PR.VSHIFT[g]) < 5e-6, (g, s_all)  # closed form == shipped
        rows.append(dict(part='A', set=g, subset='LOSO', model='route', **r,
                         note='; '.join(f'{k}: n={n} s={s:+.4f} MAE {m:.2f}%'
                                        for k, (s, n, m) in per.items())))
        lo[g] = per
    return pd.DataFrame(rows), lo


# ------------------------------------------------------------------- PART B
def yan_data():
    """Yan DATA list parsed from the source; importing the module runs it."""
    tree = ast.parse(open(os.path.join(_ROOT, 'validation', 'yan2011_validation.py')).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == 'DATA':
            return ast.literal_eval(node.value)
    raise RuntimeError('DATA not found in yan2011_validation.py')


def brine_models():
    """V(T, P, m) for each Part B variant."""
    ply = lambda T, P: float(V_ply('CO2', T, P))
    return {
        'route+salt': lambda T, P, m: V_phi('CO2', T, P, m_nacl=m),
        'route': lambda T, P, m: V_phi('CO2', T, P),
        'plyasunov+salt': lambda T, P, m: ply(T, P) * (1 + salt_fraction(m)),
        'plyasunov': lambda T, P, m: ply(T, P),
        'ad03': lambda T, P, m: AD.v_inf_ad03('CO2', T, min(P, 99.9)),
        'garcia': lambda T, P, m: garcia_CO2_vphi(T - 273.15),
    }


def brine_points():
    """(dataset, m, T, P, n2 per kg water, rho_meas, rho1), densities in g/cm3."""
    pts = []
    for T, P, m, mco2, x, rho in yan_data():
        rho1 = yan_baseline(T, P, m)  # Rowe-Chou, Yan's own reference
        pts.append(('Yan 2011', float(m), T, P, mco2, rho, rho1))
    raw = pd.read_csv(CAL.XLSX)
    raw.columns = ['m', 'x', 'T', 'P', 'rho']
    for m, sub in raw[raw.x == 0.0].groupby('m'):
        surf = CAL._fit_baseline(sub)[0]
        nb = 1000.0 / MW_WATER + m
        for _, r in raw[(raw.m == m) & (raw.x > 0)].iterrows():
            # Calabrese x is on the paired-NaCl pseudo basis (their Eq. 18)
            pts.append(('Calabrese 2019', float(m), r['T'], r['P'],
                        r.x * nb / (1 - r.x), r.rho / 1000.0,
                        surf(r['T'], r['P']) / 1000.0))
    # 2026-10-21: Song's own same-campaign gas-free baseline
    import validation.song2013_validation as SONG
    for m, T, P, n2, rho, rho1, _tn, _w in SONG.points():
        pts.append(('Song 2013', m, T, P, n2, rho, rho1))
    return pts


def part_b():
    M2 = gas_mw('CO2')
    models = brine_models()
    rec = []
    for ds, m, T, P, n2, rho, rho1 in brine_points():
        Wb = 1000.0 + m * M_NACL
        row = dict(dataset=ds, m=m, T=T, P=P, inc_meas=100 * (rho / rho1 - 1))
        for name, fn in models.items():
            V = fn(T, min(P, 100.0), m)    # IF97 Region 1 limit; see CAL
            rho_mod = (Wb + n2 * M2) / (Wb / rho1 + n2 * V)
            row[f'inc_{name}'] = 100 * (rho_mod / rho1 - 1)
            row[f'rho_{name}'] = 100 * (rho_mod / rho - 1)
        rec.append(row)
    d = pd.DataFrame(rec)
    rows = []
    for (ds, m), s in d.groupby(['dataset', 'm'], sort=False):
        for name in models:
            e = s[f'inc_{name}'] - s.inc_meas
            rows.append(dict(part='B', set=ds, subset=f'm={m:g}', model=name,
                             n=len(s), bias=e.mean(), mae=e.abs().mean(),
                             max=e.abs().max(),
                             mae_pct_of_inc=100 * e.abs().mean() / s.inc_meas.mean(),
                             inc_meas_mean=s.inc_meas.mean(),
                             rho_mae_pct=s[f'rho_{name}'].abs().mean()))
    return pd.DataFrame(rows)


# Headline pins, generated by this script 2026-10-21. Part A: MAE %, Part B:
# increment MAE in percentage points.
PINS = {
    ('A', 'CH4', 'calibration', 'route'): 1.444,
    ('A', 'CH4', 'calibration', 'plyasunov'): 1.537,
    ('A', 'CH4', 'calibration', 'ad03'): 7.269,
    ('A', 'CH4', 'held-out', 'route'): 2.778,
    ('A', 'CH4', 'held-out', 'plyasunov'): 0.543,
    ('A', 'CH4', 'held-out', 'ad03'): 0.580,
    ('A', 'CH4', 'LOSO', 'route'): 2.057,
    ('A', 'CO2', 'calibration', 'route'): 1.057,
    ('A', 'CO2', 'calibration', 'plyasunov'): 5.306,
    ('A', 'CO2', 'calibration', 'ad03'): 12.795,
    ('A', 'CO2', 'calibration', 'garcia'): 4.355,
    ('A', 'CO2', 'held-out', 'route'): 3.300,
    ('A', 'CO2', 'held-out', 'plyasunov'): 3.677,
    ('A', 'CO2', 'held-out', 'ad03'): 9.271,
    ('A', 'CO2', 'held-out', 'garcia'): 2.706,
    ('A', 'CO2', 'LOSO', 'route'): 2.484,
    # Per-source split, added 2026-10-21 (adversarial read): the CO2 verdict
    # reverses between direct V-inf data and finite-loading solution densities.
    ('A', 'CO2', 'source:Hnedkovsky 1996', 'route'): 2.33,
    ('A', 'CO2', 'source:Hnedkovsky 1996', 'plyasunov'): 0.71,
    ('A', 'CO2', 'source:McBride-Wright 2015', 'route'): 0.92,
    ('A', 'CO2', 'source:McBride-Wright 2015', 'plyasunov'): 5.83,
    ('A', 'H2S', 'calibration', 'route'): 0.484,
    ('A', 'H2S', 'calibration', 'plyasunov'): 1.439,
    ('A', 'H2S', 'calibration', 'ad03'): 3.879,
    ('A', 'H2S', 'LOSO', 'route'): 0.569,
    ('A', 'N2', 'calibration', 'route'): 2.290,
    ('A', 'N2', 'calibration', 'plyasunov'): 4.433,
    ('A', 'N2', 'calibration', 'ad03'): 5.147,
    ('A', 'N2', 'held-out', 'route'): 7.769,
    ('A', 'N2', 'held-out', 'plyasunov'): 6.197,
    ('A', 'N2', 'held-out', 'ad03'): 8.494,
    ('A', 'N2', 'LOSO', 'route'): 3.051,
    ('A', 'H2', 'calibration', 'route'): 5.129,
    ('A', 'H2', 'calibration', 'plyasunov'): 6.221,
    ('A', 'H2', 'calibration', 'ad03'): 5.397,
    ('A', 'H2', 'held-out', 'route'): 11.630,
    ('A', 'H2', 'held-out', 'plyasunov'): 19.782,
    ('A', 'H2', 'held-out', 'ad03'): 17.360,
    ('A', 'H2', 'LOSO', 'route'): 7.693,
    ('B', 'Song 2013', 'm=1', 'route+salt'): 0.048,
    ('B', 'Song 2013', 'm=1', 'route'): 0.043,
    ('B', 'Song 2013', 'm=1', 'plyasunov+salt'): 0.073,
    ('B', 'Song 2013', 'm=1', 'plyasunov'): 0.060,
    ('B', 'Song 2013', 'm=1', 'ad03'): 0.212,
    ('B', 'Song 2013', 'm=1', 'garcia'): 0.086,
    ('B', 'Song 2013', 'm=2', 'route+salt'): 0.067,
    ('B', 'Song 2013', 'm=2', 'route'): 0.093,
    ('B', 'Song 2013', 'm=2', 'plyasunov+salt'): 0.052,
    ('B', 'Song 2013', 'm=2', 'plyasunov'): 0.055,
    ('B', 'Song 2013', 'm=2', 'ad03'): 0.160,
    ('B', 'Song 2013', 'm=2', 'garcia'): 0.084,
    ('B', 'Song 2013', 'm=3', 'route+salt'): 0.103,
    ('B', 'Song 2013', 'm=3', 'route'): 0.175,
    ('B', 'Song 2013', 'm=3', 'plyasunov+salt'): 0.062,
    ('B', 'Song 2013', 'm=3', 'plyasunov'): 0.124,
    ('B', 'Song 2013', 'm=3', 'ad03'): 0.127,
    ('B', 'Song 2013', 'm=3', 'garcia'): 0.079,
    ('B', 'Song 2013', 'm=4', 'route+salt'): 0.074,
    ('B', 'Song 2013', 'm=4', 'route'): 0.089,
    ('B', 'Song 2013', 'm=4', 'plyasunov+salt'): 0.085,
    ('B', 'Song 2013', 'm=4', 'plyasunov'): 0.058,
    ('B', 'Song 2013', 'm=4', 'ad03'): 0.188,
    ('B', 'Song 2013', 'm=4', 'garcia'): 0.088,
    ('B', 'Yan 2011', 'm=0', 'route+salt'): 0.039,
    ('B', 'Yan 2011', 'm=0', 'route'): 0.039,
    ('B', 'Yan 2011', 'm=0', 'plyasunov+salt'): 0.138,
    ('B', 'Yan 2011', 'm=0', 'plyasunov'): 0.138,
    ('B', 'Yan 2011', 'm=0', 'ad03'): 0.527,
    ('B', 'Yan 2011', 'm=0', 'garcia'): 0.177,
    ('B', 'Yan 2011', 'm=1', 'route+salt'): 0.043,
    ('B', 'Yan 2011', 'm=1', 'route'): 0.060,
    ('B', 'Yan 2011', 'm=1', 'plyasunov+salt'): 0.132,
    ('B', 'Yan 2011', 'm=1', 'plyasunov'): 0.089,
    ('B', 'Yan 2011', 'm=1', 'ad03'): 0.400,
    ('B', 'Yan 2011', 'm=1', 'garcia'): 0.112,
    ('B', 'Yan 2011', 'm=5', 'route+salt'): 0.147,
    ('B', 'Yan 2011', 'm=5', 'route'): 0.231,
    ('B', 'Yan 2011', 'm=5', 'plyasunov+salt'): 0.099,
    ('B', 'Yan 2011', 'm=5', 'plyasunov'): 0.178,
    ('B', 'Yan 2011', 'm=5', 'ad03'): 0.115,
    ('B', 'Yan 2011', 'm=5', 'garcia'): 0.165,
    ('B', 'Calabrese 2019', 'm=0.77', 'route+salt'): 0.152,
    ('B', 'Calabrese 2019', 'm=0.77', 'route'): 0.125,
    ('B', 'Calabrese 2019', 'm=0.77', 'plyasunov+salt'): 0.224,
    ('B', 'Calabrese 2019', 'm=0.77', 'plyasunov'): 0.206,
    ('B', 'Calabrese 2019', 'm=0.77', 'ad03'): 0.424,
    ('B', 'Calabrese 2019', 'm=0.77', 'garcia'): 0.126,
    ('B', 'Calabrese 2019', 'm=2.5', 'route+salt'): 0.050,
    ('B', 'Calabrese 2019', 'm=2.5', 'route'): 0.077,
    ('B', 'Calabrese 2019', 'm=2.5', 'plyasunov+salt'): 0.088,
    ('B', 'Calabrese 2019', 'm=2.5', 'plyasunov'): 0.091,
    ('B', 'Calabrese 2019', 'm=2.5', 'ad03'): 0.148,
    ('B', 'Calabrese 2019', 'm=2.5', 'garcia'): 0.116,
}


def main(check=False):
    a, per = part_a()
    b = part_b()
    res = pd.concat([a, b], ignore_index=True)
    res.to_csv(OUT, index=False, float_format='%.5g')

    pd.set_option('display.width', 160)
    print('PART A  freshwater V_phi, % error at each point\'s own (T, P)')
    for sub in ('calibration', 'held-out'):
        t = a[a.subset == sub].pivot_table(index='set', columns='model',
                                           values='mae', sort=False)
        n = a[(a.subset == sub) & (a.model == 'route')].set_index('set').n
        print(f'\n  {sub}: mean |error| %  (n per gas: '
              + ', '.join(f'{g} {n.get(g, 0)}' for g in GASES) + ')')
        print(t.reindex(GASES)[['route', 'plyasunov', 'ad03', 'garcia']].round(2).to_string())
        t = a[a.subset == sub].pivot_table(index='set', columns='model',
                                           values='bias', sort=False)
        print(f'  {sub}: mean signed error %')
        print(t.reindex(GASES)[['route', 'plyasunov', 'ad03', 'garcia']].round(2).to_string())
    print('\n  Calibration set by source: mean |error| %')
    srcs = a[a.subset.str.startswith('source:')]
    for g in GASES:
        t = srcs[srcs.set == g].pivot_table(index='subset', columns='model', values='mae', sort=False)
        nn = srcs[(srcs.set == g) & (srcs.model == 'route')].set_index('subset').n
        t['n'] = nn
        print(f'    {g}'); print(t.round(2).to_string())
    print('\n  LOSO, route (refit without each source, score that source):')
    for g in GASES:
        r = a[(a.set == g) & (a.subset == 'LOSO')].iloc[0]
        print(f'    {g:<4} MAE {r.mae:5.2f}%  bias {r.bias:+5.2f}%  max {r["max"]:5.1f}%  n={r.n}')
        for k, (s, n, m) in per[g].items():
            print(f'         without {k:<30} s={s:+.5f}  n={n:3d}  MAE {m:5.2f}%')

    print('\nPART B  CO2-loaded NaCl brine at measured loading; increment rho/rho1-1')
    cols = ['n', 'inc_meas_mean', 'mae', 'bias', 'max', 'mae_pct_of_inc', 'rho_mae_pct']
    for (ds, sub), s in b.groupby(['set', 'subset'], sort=False):
        print(f'\n  {ds} {sub}  (mae/bias/max in pp; mae_pct_of_inc = MAE / mean measured increment; rho_mae_pct = total density MAE %)')
        print(s.set_index('model')[cols].round(3).to_string())

    if check:
        flat = {}
        for _, r in a.iterrows():
            flat[('A', r.set, r.subset, r.model)] = r.mae
        for _, r in b.iterrows():
            flat[('B', r.set, r.subset, r.model)] = r.mae
        bad = [(k, v, flat.get(k)) for k, v in PINS.items()
               if flat.get(k) is None or abs(flat[k] - v) > 0.006]
        for k, v, got in bad:
            print('PIN FAIL', k, 'pinned', v, 'got', got)
        assert PINS and not bad
        print(f'\nCHECK PASSED ({len(PINS)} pins)')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main(check='--check' in sys.argv)
