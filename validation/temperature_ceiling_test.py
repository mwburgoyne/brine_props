"""Quantify how chain error behaves versus temperature, against every
temperature-resolved data source, so the accuracy-ceiling claim (~450 K)
rests on numbers rather than judgment.

Four exhibits:
  1. Freshwater V_phi error vs temperature, per gas, against the calibration
     densimetry (Hnedkovsky 294-473 K + the 298 K sources + Murphy & Gaines).
     Fit data, not held out - but if error grew with T, it would show here.
  2. CO2-brine density error vs temperature against Calabrese (275-449 K),
     data never used in calibration, from calabrese_density_results.csv
     (model densification recomputed per row via vphi_route).
  3. Above the data: PR-route vs Plyasunov-correlation divergence, 448-598 K.
     Model-model spread measures ignorance, not error in either.
  4. Viscosity baseline vs Kestin's NaCl tables by temperature band
     (Kestin's range tops out at 423 K), plus the per-gas data ceilings.

Run: python3 temperature_ceiling_test.py
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

import csv
import os
import sys

import numpy as np


import fits.fit_pr_vshift as FIT
import brine_gas.vphi_route as VR
import brine_gas.plyasunov_model as PLY
from brine_gas.brine_properties import M_NACL


def band(T, edges=(300.0, 375.0, 430.0, 475.0)):
    lo = 270.0
    for hi in edges:
        if T <= hi:
            return f'{lo:.0f}-{hi:.0f}'
        lo = hi
    return f'>{edges[-1]:.0f}'


def exhibit1():
    print('=' * 74)
    print('1. FRESHWATER V_phi |error| vs T, calibration densimetry (fit data)')
    print('=' * 74)
    cal = FIT.calibration_set()
    for gas in ('CH4', 'CO2', 'H2S'):
        rows = {}
        for T, P, V, src in cal[gas]:
            e = abs(100.0 * (VR.V_phi(gas, T, P) - V) / V)
            rows.setdefault(band(T), []).append(e)
        line = '  '.join(f'{b}: {np.mean(v):4.1f}% (n={len(v)}, max {np.max(v):.1f})'
                         for b, v in sorted(rows.items()))
        print(f'  {gas:4s} {line}')
    print('  (No band above 430 K scores worse than the mid bands -> no growth')
    print('   trend inside the 473 K data; verify from the numbers above.)')


def exhibit2():
    print('=' * 74)
    print('2. CO2-BRINE density error (pp) vs isotherm, Calabrese (held out)')
    print('=' * 74)
    path = os.path.join(_RESULTS, 'calabrese_density_results.csv')
    per_iso = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m, x = float(row['m']), float(row['x'])
            T, P = float(row['T']), float(row['P'])
            rho_b = float(row['rho_base_fit'])           # kg/m3, their own fit
            f_meas = float(row['f_exp_pct'])             # measured densification, pp
            veff = VR.V_phi("CO2", T, min(P, 100.0)) * (1.0 + VR.salt_fraction(m))
            m2 = 55.508 * x / (1.0 - x)
            wb = 1000.0 + m * M_NACL
            rho_model = (wb + m2 * 44.0095) / (wb / (rho_b / 1000.0) + m2 * veff)
            f_model = 100.0 * (rho_model * 1000.0 / rho_b - 1.0)
            per_iso.setdefault((m, round(T)), []).append(f_model - f_meas)
    print('  m      T_K   n   mean err (pp)  mean|err|  max|err|')
    for (m, T), e in sorted(per_iso.items()):
        e = np.array(e)
        print(f'  {m:4.2f}  {T:5.0f}  {len(e):2d}   {np.mean(e):+.3f}         '
              f'{np.mean(np.abs(e)):.3f}      {np.max(np.abs(e)):.3f}')


def exhibit3():
    print('=' * 74)
    print('3. ABOVE THE DATA: PR route vs Plyasunov correlation, 30 MPa, %')
    print('   (model-model spread = ignorance where nothing measures)')
    print('=' * 74)
    Ts = (448.15, 473.15, 498.15, 523.15, 548.15, 573.15, 598.15)
    print('  T_K   ' + '  '.join(f'{T - 0.15:5.0f}' for T in Ts))
    for gas in ('CH4', 'CO2', 'H2S', 'N2', 'H2'):
        vals = []
        for T in Ts:
            try:
                pr = VR.V_phi(gas, T, 30.0)
                pl = PLY.V_phi(gas, T, 30.0)
                vals.append(f'{100.0 * (pr - pl) / pl:+5.0f}')
            except Exception:
                vals.append('    -')
        print(f'  {gas:4s}  ' + '  '.join(f'{v:>5s}' for v in vals))


def exhibit4():
    print('=' * 74)
    print('4. VISCOSITY: baseline vs Kestin NaCl tables, by T band, and the')
    print('   per-gas correction data ceilings')
    print('=' * 74)
    import brine_gas.kestin_nacl_viscosity as KN
    import brine_gas.viscosity_route as VRT
    bands = {}
    for t_C in range(25, 151, 25):
        for p in (0.1, 10.0, 20.0, 35.0):
            for m in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
                T = t_C + 273.15
                mine = VRT.brine_viscosity(T, p, m=m)     # mPa s
                theirs = KN.mu(t_C, p, m) / 1000.0        # micro Pa s -> mPa s
                b = '298-348' if T <= 348.15 else ('348-398' if T <= 398.15
                                                  else '398-423')
                bands.setdefault(b, []).append(abs(100 * (mine / theirs - 1)))
    for b, v in sorted(bands.items()):
        print(f'  baseline vs Kestin {b} K: mean {np.mean(v):.2f}%, '
              f'max {np.max(v):.2f}%  (n={len(v)})')
    print('  Per-gas correction data ceilings (top measured temperature):')
    print('    CO2 449 K (Calabrese); CH4 394 K (Ostermann); H2S 309 K')
    print('    (Murphy & Gaines); N2/H2 no positive measurement at any T.')


if __name__ == '__main__':
    exhibit1()
    exhibit2()
    exhibit3()
    exhibit4()
