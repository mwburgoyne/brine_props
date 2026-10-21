"""Song et al. (2013) CO2-loaded NaCl brine densities: independent test at 1-4 mol/kg.

Song, Zhan, Zhang, Liu & Jian, J. Chem. Eng. Data 58:3342 (2013), DOI
10.1021/je400459y; data from the NIST ThermoML record (song2013_data.csv,
provenance in its header; values checked against the paper's Table 2, Paper 67).
300 CO2-loaded rows at prepared CO2 mass fractions 0.01-0.03 in 1, 2, 3 and
4 mol/kg NaCl, five nominal isotherms 333-413 K, 10-18 MPa, plus 100 gas-free
rows from the same magnetic-suspension-balance campaign. The paper states a
combined expanded uncertainty of 0.44 kg/m3 (k=2; ThermoML lists ~1), defers the
loading procedure to Zhang et al. 2011 (Paper 68) and reports no phase check;
99 loaded rows exceed Spycher-Pruess saturation (see w_sat).

Baseline: Song's own gas-free densities, fitted per nominal isotherm as linear
in P, with a quadratic-in-T derivative (fitted over each molality's gas-free
rows) carrying the 0.1-0.5 K offsets between loaded and gas-free rows. No
parameter is fitted to the loaded rows. Found by the revision-1 data hunt
(manuscript/revision1/lit_data_hunt.md); it is the only CO2-brine set above
2.5 mol/kg with a same-apparatus gas-free baseline and known loading.

--check asserts the headline numbers quoted in the manuscript.
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

from brine_gas.brine_properties import M_NACL
from brine_gas.plyasunov_model import gas_mw
from brine_gas.vphi_route import V_phi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_DATA, 'song2013_data.csv')
T_NOM = (333, 353, 373, 393, 413)


def load():
    return pd.read_csv(DATA, comment='#')


def points():
    """Loaded rows with their baseline: (m, T, P, n2 per kg water, rho, rho1), g/cm3."""
    d = load()
    M2 = gas_mw('CO2')
    out = []
    for m in sorted(d.m_NaCl.unique()):
        R = d[d.m_NaCl == m]
        base = R[R.w_CO2 == 0]
        A = np.column_stack([np.ones(len(base)), base.T_K, base.T_K ** 2, base.P_MPa])
        g, *_ = np.linalg.lstsq(A, base.rho_kg_m3, rcond=None)
        dTd = lambda T: g[1] + 2 * g[2] * T
        for Tn in T_NOM:
            b = base[np.abs(base.T_K - Tn) < 5]
            Tb = b.T_K.mean()
            cb = np.polyfit(b.P_MPa, b.rho_kg_m3 - dTd(Tb) * (b.T_K - Tb), 1)
            L = R[(R.w_CO2 > 0) & (np.abs(R.T_K - Tn) < 5)]
            Wb = 1000.0 + m * M_NACL
            for _, r in L.iterrows():
                rho1 = (np.polyval(cb, r.P_MPa) + dTd(Tb) * (r.T_K - Tb)) / 1000.0
                n2 = r.w_CO2 / (1.0 - r.w_CO2) * Wb / M2
                out.append((float(m), r.T_K, r.P_MPa, n2, r.rho_kg_m3 / 1000.0, rho1, Tn, r.w_CO2))
    return out


def score(salt=True):
    M2 = gas_mw('CO2')
    rows = []
    for m, T, P, n2, rho, rho1, Tn, w in points():
        Wb = 1000.0 + m * M_NACL
        V = float(V_phi('CO2', T, P, m_nacl=m, salt_effect=salt))
        rmod = (Wb + n2 * M2) / (Wb / rho1 + n2 * V)
        Vimp = ((Wb + n2 * M2) / rho - Wb / rho1) / n2
        rows.append(dict(m=m, Tn=Tn, inc_meas=100 * (rho / rho1 - 1), inc_mod=100 * (rmod / rho1 - 1),
                         rho_err=100 * (rmod / rho - 1), V_ratio=V / Vimp,
                         over=w > w_sat(T, P, m)))
    return pd.DataFrame(rows)


def w_sat(T, P, m):
    """CO2 mass fraction of the whole solution at saturation, Spycher-Pruess
    (pyrestoolbox CO2_Brine_Mixture). Added 2026-10-21: 99 of the 300 prepared
    loadings exceed it, so the samples' phase state needs the paper's text."""
    from pyrestoolbox import brine
    ppm = 1e6 * m * M_NACL / (1000.0 + m * M_NACL)
    mix = brine.CO2_Brine_Mixture(pres=P * 10.0, temp=T - 273.15, ppm=ppm, metric=True)
    xc, xw = mix.x[0], mix.x[1]
    mc = xc / xw * 1000.0 / 18.01528
    M2 = gas_mw('CO2')
    return mc * M2 / (1000.0 + m * M_NACL + mc * M2)


def summary(salt=True, undersat_only=False):
    s = score(salt)
    if undersat_only:
        s = s[~s.over]
    s['d'] = s.inc_mod - s.inc_meas
    return s.groupby('m').agg(n=('d', 'size'), inc_meas=('inc_meas', 'mean'),
                              mae=('d', lambda v: np.abs(v).mean()), bias=('d', 'mean'),
                              rho_mae=('rho_err', lambda v: np.abs(v).mean()),
                              rho_max=('rho_err', lambda v: np.abs(v).max()))


def offsets():
    """Per (m, isotherm): least-squares intercept a (pp) of the measured increment
    against w, its t-statistic, and the increment MAE with that intercept removed,
    with and without the salt factor. A nonzero a is an offset between the gas-free
    and loaded runs (Song's take both signs) or a loading dependence of V."""
    w = [p[7] for p in points()]
    s1, s0 = score(True), score(False)
    s1['w'] = w
    s1['mod0'] = s0.inc_mod.values
    rows = []
    for (m, Tn), g in s1.groupby(['m', 'Tn']):
        X = np.column_stack([np.ones(len(g)), g.w])
        c, *_ = np.linalg.lstsq(X, g.inc_meas.values, rcond=None)
        r = g.inc_meas.values - X @ c
        se = np.sqrt(r @ r / (len(g) - 2) * np.linalg.inv(X.T @ X)[0, 0])
        corr = g.inc_meas - c[0]
        rows.append(dict(m=m, Tn=Tn, a=c[0], t=c[0] / se,
                         mae_salt=(g.inc_mod - corr).abs().mean(), mae_nosalt=(g.mod0 - corr).abs().mean()))
    return pd.DataFrame(rows)


def main():
    for salt in (True, False):
        print(f"\nSong 2013, route {'with' if salt else 'WITHOUT'} the salt factor "
              f"(increment rho/rho1 - 1, pp; density error %)")
        print(summary(salt).round(3).to_string())
    for salt in (True, False):
        print(f"\nSame, ONLY rows below Spycher-Pruess saturation, {'with' if salt else 'WITHOUT'} salt factor")
        print(summary(salt, undersat_only=True).round(3).to_string())
    s = score(True)
    print(f'\nrows above Spycher-Pruess saturation: {int(s.over.sum())} of {len(s)}')
    print('\nmodel / implied V_phi by isotherm (route with salt factor):')
    print(s.pivot_table(index='m', columns='Tn', values='V_ratio').round(3).to_string())
    if '--check' in sys.argv:
        a, b = summary(True), summary(False)
        assert len(score()) == 300
        assert abs(a.mae.max() - PIN_MAE_MAX) < 5e-4, a.mae.max()
        assert abs(a.rho_max.max() - PIN_RHO_MAX) < 5e-3, a.rho_max.max()
        assert abs(a.mae.mean() - PIN_MAE_MEAN) < 5e-4 and abs(b.mae.mean() - PIN_MAE_MEAN_NOSALT) < 5e-4, (a.mae.mean(), b.mae.mean())
        u, v = summary(True, True), summary(False, True)
        assert int(score().over.sum()) == 99
        # 4 mol/kg, 413 K: density change vs gas-free at 14 MPa (linear in P per
        # mass fraction), quoted in Section 6 to show no break at saturation.
        d = load(); d = d[(d.m_NaCl == 4) & (np.abs(d.T_K - 413) < 4)]
        r14 = {w: np.polyval(np.polyfit(g.P_MPa, g.rho_kg_m3, 1), 14.0) for w, g in d.groupby('w_CO2')}
        dd = [r14[w] - r14[0.0] for w in sorted(r14) if w > 0]
        assert np.allclose(dd, [-0.6, -1.0, -1.4], atol=0.05), dd
        o = offsets()
        assert int((o.t.abs() > 3).sum()) == 9 and len(o) == 20
        assert (round(o.a.min(), 3), round(o.a.max(), 3)) == (-0.077, 0.133)
        om = o.groupby('m')[['mae_salt', 'mae_nosalt']].mean().round(3)
        assert om.values.tolist() == [[0.078, 0.073], [0.056, 0.041], [0.103, 0.158], [0.077, 0.081]], om
        assert np.allclose(u.mae.values, [0.044, 0.069, 0.084, 0.050], atol=6e-4), u.mae.values
        assert np.allclose(v.mae.values, [0.039, 0.100, 0.136, 0.064], atol=6e-4), v.mae.values
        print('  pins hold')


PIN_MAE_MAX, PIN_RHO_MAX, PIN_MAE_MEAN, PIN_MAE_MEAN_NOSALT = 0.1032, 0.362, 0.0730, 0.1002

if __name__ == '__main__':
    main()
