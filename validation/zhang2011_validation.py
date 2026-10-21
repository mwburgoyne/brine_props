"""Zhang et al. (2011) CO2-loaded Tianjin brine densities: near-freshwater test.

Zhang, Chang, Song, Zhao, Zhan & Jian, J. Chem. Eng. Data 56:565 (2011), DOI
10.1021/je101214e (Paper 68; zhang2011_data.csv, parsed from Tables 2 and 3).
97 CO2-loaded rows at CO2 mass fractions 0.010-0.040, five nominal isotherms
313-353 K, 10-18 MPa, plus 25 gas-free rows, same magnetic-suspension-balance
apparatus as Song et al. 2013. The brine is dilute (about 1.25 g dissolved
solids per kg water, 0.020 mol/kg Na), so it is scored as 0.02 mol/kg NaCl
with the salt factor on; at this salinity the factor is negligible. The paper
gives no analysis method for the CO2 content and no phase check, but omits the
w=0.040 rows at 343 K/10 MPa and 353 K/10-12 MPa, the lowest-solubility states.

Baseline and scoring follow song2013_validation.py: the paper's own gas-free
densities, linear in P per isotherm, quadratic-in-T derivative for the small
T offsets. No parameter is fitted to the loaded rows.

--check asserts the numbers quoted in the manuscript and docs.
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

import os
import sys

import numpy as np
import pandas as pd

from brine_gas.plyasunov_model import gas_mw
from validation.song2013_validation import w_sat
from brine_gas.vphi_route import V_phi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_DATA, 'zhang2011_data.csv')
T_NOM = (313, 323, 333, 343, 353)
M_EQ = 0.02            # NaCl-equivalent molality used for V_phi and w_sat
W_SOLIDS = 1.2514      # g dissolved solids per kg water, from Table 1


def load():
    return pd.read_csv(DATA, comment='#')


def score(salt=True):
    d = load()
    M2 = gas_mw('CO2')
    Wb = 1000.0 + W_SOLIDS
    base = d[d.w_CO2 == 0]
    A = np.column_stack([np.ones(len(base)), base.T_K, base.T_K ** 2, base.P_MPa])
    g, *_ = np.linalg.lstsq(A, base.rho_kg_m3, rcond=None)
    dTd = lambda T: g[1] + 2 * g[2] * T
    rows = []
    for Tn in T_NOM:
        b = base[np.abs(base.T_K - Tn) < 5]
        Tb = b.T_K.mean()
        cb = np.polyfit(b.P_MPa, b.rho_kg_m3 - dTd(Tb) * (b.T_K - Tb), 1)
        for _, r in d[(d.w_CO2 > 0) & (np.abs(d.T_K - Tn) < 5)].iterrows():
            rho1 = (np.polyval(cb, r.P_MPa) + dTd(Tb) * (r.T_K - Tb)) / 1000.0
            rho = r.rho_kg_m3 / 1000.0
            n2 = r.w_CO2 / (1.0 - r.w_CO2) * Wb / M2
            V = float(V_phi('CO2', r.T_K, r.P_MPa, m_nacl=M_EQ, salt_effect=salt))
            rmod = (Wb + n2 * M2) / (Wb / rho1 + n2 * V)
            Vimp = ((Wb + n2 * M2) / rho - Wb / rho1) / n2
            rows.append(dict(w=r.w_CO2, Tn=Tn, T=r.T_K, P=r.P_MPa,
                             inc_meas=100 * (rho / rho1 - 1), inc_mod=100 * (rmod / rho1 - 1),
                             rho_err=100 * (rmod / rho - 1), V=V, Vimp=Vimp,
                             over=r.w_CO2 > w_sat(r.T_K, r.P_MPa, M_EQ)))
    s = pd.DataFrame(rows)
    s['d'] = s.inc_mod - s.inc_meas
    return s


def summary(s, by):
    return s.groupby(by).agg(n=('d', 'size'), inc_meas=('inc_meas', 'mean'),
                             mae=('d', lambda v: np.abs(v).mean()), bias=('d', 'mean'),
                             rel=('d', lambda v: np.abs(v).sum()),
                             V_med=('V', 'median'), Vimp_med=('Vimp', 'median'),
                             rho_mae=('rho_err', lambda v: np.abs(v).mean()),
                             rho_max=('rho_err', lambda v: np.abs(v).max()))


def intercept_fit(s, keys, col='inc_meas'):
    """Per group, least-squares inc = a + b*w; returns a, its standard error, b."""
    out = []
    for k, g in s.groupby(keys):
        X = np.column_stack([np.ones(len(g)), g.w])
        y = g[col].values
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ c
        se = np.sqrt(r @ r / (len(y) - 2) * np.linalg.inv(X.T @ X)[0, 0])
        out.append(dict(key=k, a=c[0], se=se, b=c[1]))
    return pd.DataFrame(out).set_index('key')


def slope_volumes(s):
    """CO2 volume implied by the measured increments with each isotherm's
    intercept removed, set against four models' volumes at the same points.
    An intercept is an offset between gas-free and loaded runs or a loading
    dependence of the volume; these data cannot tell which."""
    from validation.model_comparison import brine_models
    mods = brine_models()
    M2 = gas_mw('CO2')
    Wb = 1000.0 + W_SOLIDS
    fit = intercept_fit(s, 'Tn')
    rows = []
    for Tn, g in s.groupby('Tn'):
        n2 = g.w / (1 - g.w) * Wb / M2
        rho1 = (((Wb + n2 * M2) / (1 + g.inc_meas / 100) - Wb) / n2) / g.Vimp
        Vc = (((Wb + n2 * M2) / (1 + (g.inc_meas - fit.a[Tn]) / 100) - Wb) / n2) / rho1
        r = dict(Tn=Tn, a=fit.a[Tn], t=fit.a[Tn] / fit.se[Tn], V_slope=float(np.median(Vc)))
        for k in ('route', 'plyasunov', 'ad03', 'garcia'):
            r[k] = 100 * (np.mean([mods[k](T, P, M_EQ) for T, P in zip(g['T'], g.P)]) / r['V_slope'] - 1)
        rows.append(r)
    return pd.DataFrame(rows).set_index('Tn')


def apparent_slope(s):
    """dV_app/dx per isotherm if the intercepts were read as loading dependence
    instead of a run offset; compared with the measured -117 and -42 (Table 5)."""
    M2 = gas_mw('CO2')
    x = (s.w / M2) / (s.w / M2 + (1 - s.w) / 18.01528)
    return pd.Series({Tn: np.polyfit(x[g.index], g.Vimp, 1)[0] for Tn, g in s.groupby('Tn')})


def main():
    s = score()
    pd.set_option('display.width', 160)
    for by in ('w', 'Tn'):
        t = summary(s, by)
        t['rel'] = 100 * t.mae / t.inc_meas
        print(f"\nZhang 2011, by {by} (increments and errors in % of rho1; rel = MAE/mean increment, %)")
        print(t.round(4).to_string())
    a = s.d.abs()
    print(f"\nAll {len(s)}: increment MAE {a.mean():.4f} pp, bias {s.d.mean():+.4f} pp, "
          f"mean increment {s.inc_meas.mean():.4f} pp ({100 * a.mean() / s.inc_meas.mean():.0f}%); "
          f"density MAE {s.rho_err.abs().mean():.4f}%, max {s.rho_err.abs().max():.4f}%; "
          f"above Spycher-Pruess saturation: {int(s.over.sum())}")
    u = s[~s.over]
    print(f"Below saturation ({len(u)}): increment MAE {u.d.abs().mean():.4f} pp, bias {u.d.mean():+.4f} pp")
    v = slope_volumes(s)
    print("\nIntercept a (pp) and t, slope-implied V (cm3/mol), model V minus slope-implied V (%)")
    print(v.round(3).to_string())
    print("\nIf read as loading dependence, dV_app/dx (cm3/mol per unit x):", apparent_slope(s).round(0).to_dict())
    return s, v


if __name__ == '__main__':
    s, v = main()
    if '--check' in sys.argv:
        a = s.d.abs()
        assert len(s) == 97 and int(s.over.sum()) == 0
        assert round(a.mean(), 3) == 0.067 and round(100 * a.mean() / s.inc_meas.mean()) == 13
        assert round(s.rho_err.abs().mean(), 2) == 0.07 and round(s.rho_err.abs().max(), 2) == 0.08
        assert (v.t > 10).all()
        assert (round(v.a.min(), 2), round(v.a.max(), 2)) == (0.04, 0.07)
        assert (round(v.route.min(), 1), round(v.route.max(), 1)) == (0.2, 0.7)
        assert (round(v.plyasunov.min(), 1), round(v.plyasunov.max(), 1)) == (-4.3, -3.3)
        assert (round(v.ad03.min()), round(v.ad03.max())) == (-12, -8)
        assert (round(v.garcia.min(), 1), round(v.garcia.max(), 1)) == (-5.6, 0.7)
        k = apparent_slope(s)
        assert (round(k.min(), -1), round(k.max(), -1)) == (100, 190), k
        print('  pins hold')
