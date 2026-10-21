"""Rowe & Chou (1970) NaCl solution specific volume, Eq. 6.

SOURCE: Papers/63_RoweChou_1970_NaCl_PVTx_0-175C_350kgcm2.pdf
A. M. Rowe Jr. and J. C. S. Chou, "Pressure-volume-temperature-concentration
relation of aqueous NaCl solutions", J. Chem. Eng. Data 15(1), 61-66 (1970),
DOI 10.1021/je60044a016. Coefficients read from 450 dpi renders of pp. 63 and
65 (the text layer is unusable), 2026-10-21.

    v = A - P B - P^2 C + x D + x^2 E - x P F - x^2 P G - x P^2 H / 2

v in cm3/g, T in K, P absolute in kgf/cm2, x the NaCl weight fraction.
Stated: within 1.5 ppt over 0-150 degC, 0-25 wt% NaCl, 1-350 kgf/cm2.

Used because Yan et al. (2011) calibrated their densimeter against brine of
the same salinity whose density they computed from this correlation, so their
reported CO2-loaded densities inherit its error. Inverting them against
another baseline mixes two references inside a 0.3% increment.
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

import numpy as np

KGF_CM2_PER_MPA = 10.197162


def _A(T): return 5.916365 - 0.01035794 * T + 0.9270048e-5 * T**2 - 1127.522 / T + 100674.1 / T**2
def _B(T): return 0.5204914e-2 - 0.10482101e-4 * T + 0.8328532e-8 * T**2 - 1.1702939 / T + 102.2783 / T**2
def _C(T): return 0.118547e-7 - 0.6599143e-10 * T
def _D(T): return -2.5166 + 0.0111766 * T - 0.170552e-4 * T**2
def _E(T): return 2.84851 - 0.0154305 * T + 0.223982e-4 * T**2
def _F(T): return -0.0014814 + 0.82969e-5 * T - 0.12469e-7 * T**2
def _G(T): return 0.0027141 - 0.15391e-4 * T + 0.22655e-7 * T**2
def _H(T): return 0.62158e-6 - 0.40075e-8 * T + 0.65972e-11 * T**2


def specific_volume(T, P_MPa, x):
    P = P_MPa * KGF_CM2_PER_MPA
    return (_A(T) - P * _B(T) - P**2 * _C(T) + x * _D(T) + x**2 * _E(T)
            - x * P * _F(T) - x**2 * P * _G(T) - 0.5 * x * P**2 * _H(T))


def density(T, P_MPa, x):
    """g/cm3."""
    return 1.0 / specific_volume(T, P_MPa, x)


def yan_baseline(T, P_MPa, m_nacl):
    """Gas-free reference density (g/cm3) for inverting Yan et al. 2011: the
    Rowe-Chou brine they calibrated against for m > 0; IF97 water at m = 0."""
    from brine_gas.water_properties import rho_w
    from brine_gas.brine_properties import salinity_from_molality
    if m_nacl > 0:
        return density(T, P_MPa, salinity_from_molality(m_nacl))
    return float(rho_w(T, P_MPa)) / 1000.0


if __name__ == '__main__':
    import sys
    from brine_gas.water_properties import rho_w
    from brine_gas.brine_properties import rho_brine, salinity_from_molality
    ok = True
    print('  water (x = 0) against IF97:')
    for T, P in ((298.15, 0.101325), (323.15, 20.0), (373.15, 30.0), (413.15, 30.0)):
        rc, w = density(T, P, 0.0), float(rho_w(T, P)) / 1000.0
        d = 100 * (rc / w - 1)
        ok &= abs(d) < 0.15
        print(f'    {T:7.2f} K {P:6.2f} MPa  RC {rc:.5f}  IF97 {w:.5f}  {d:+.3f}%')
    print('  5 mol/kg NaCl against Spivey:')
    S5 = salinity_from_molality(5.0)
    for T, P in ((323.2, 20.0), (373.2, 20.0), (413.2, 20.0), (413.2, 40.0)):
        rc, sp = density(T, P, S5), rho_brine(T, P, S5)
        sp = sp / 1000 if sp > 100 else sp
        print(f'    {T:7.2f} K {P:6.2f} MPa  RC {rc:.5f}  Spivey {sp:.5f}  {100*(rc/sp-1):+.3f}%')
    if '--check' in sys.argv:
        assert ok, 'Rowe-Chou water departs from IF97 by > 0.15%'
        print('  pins hold')
