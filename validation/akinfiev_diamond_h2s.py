"""Does the Akinfiev-Diamond (2003) H2S correlation reproduce the measured
near-ambient temperature trend of V_inf(H2S)?

Context: the manuscript claims the delivered route reproduces a measured H2S
temperature dependence (Murphy & Gaines 1974: +0.69 cm3/mol between their
sub-300 K and above-305 K groups) that the incumbent correlation (Plyasunov
2020: +0.02, flat) misses. Mark asked whether the OTHER published correlation
of this quantity - Akinfiev & Diamond (2003), GCA 67:613, DOI
10.1016/S0016-7037(02)01141-9, whose H2S V_inf equation Akinfiev et al. (2016)
carry into brine by Young's rule - also misses it. This script answers that.

Model, as implemented in CHNOSZ (AD.R, verified against its source 2026-08-08):

    V_inf = (1 - xi)*V1 + xi*R*T*kappa_T + R*T*rho1*kappa_T*(a + b*(1000/T)^0.5)

with V1 the pure-water molar volume (cm3/mol), rho1 its density (g/cm3),
kappa_T its isothermal compressibility (1/bar), R = 83.14463 cm3 bar/mol/K.
Parameters (a in cm3/g, b in cm3 K^0.5/g, xi dimensionless) from CHNOSZ
OBIGT AD.csv, reference tag AD03.1 = Akinfiev & Diamond (2003) Table 1.
H2S VERIFIED against the primary (Papers/52, obtained 2026-08-08): the paper
prints xi = -0.2029, a = -13.4046, b = 13.8582 exactly, and identifies them as
its PREFERRED H2S set, refitted to data below 200 degC after the global fit
showed 'noticeable' ambient discrepancy - so the wrong-sign verdict below is
against their near-ambient-optimised parameters, not a high-T-weighted fit:

    H2S: a = -13.4046, b = 13.8582, xi = -0.2029
    CO2: a =  -8.8321, b = 11.2684, xi = -0.0850   (sanity check only)

NOTE: Akinfiev et al. (2016) use a READJUSTED (SOCW-consistent) parameter set
published only inside their freeware code, so this tests the original 2003
correlation, which is what "Akinfiev-Diamond 2003" means everywhere here.

Self-checks: CO2 32.7 and H2S 34.9 cm3/mol at 298.15 K / 1 bar, against
measured 33.4 (Hnedkovsky, 20 MPa) / 33.9 (Moore) for CO2 and 35.1 +/- 0.6
(Murphy & Gaines) for H2S.
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

HERE = os.path.dirname(os.path.abspath(__file__))

from brine_gas.water_properties import rho_w
from brine_gas.plyasunov_model import V2_inf as V_ply
from brine_gas.vphi_route import V_phi as V_delivered

R_CM3_BAR = 83.14463
MW_WATER = 18.0153

AD03 = {
    'H2S': dict(a=-13.4046, b=13.8582, xi=-0.2029),
    'CO2': dict(a=-8.8321, b=11.2684, xi=-0.0850),
    'CH4': dict(a=-11.8462, b=14.8615, xi=-0.1131),
}


def v_inf_ad03(gas, T, P_mpa):
    """Akinfiev-Diamond (2003) V_inf in cm3/mol at T/K, P/MPa (IF97 water)."""
    p = AD03[gas]
    P_bar = P_mpa * 10.0
    rho1 = rho_w(T, P_mpa) / 1000.0                    # g/cm3
    # kappa_T = (1/rho) drho/dP, dP = 0.1 MPa = 1 bar; forward difference
    # near the low-pressure edge of IF97 Region 1, central elsewhere
    dP = 0.1
    if P_mpa - dP <= 0.05:
        drho = (rho_w(T, P_mpa + dP) - rho_w(T, P_mpa)) / 1000.0
        drho_dPbar = drho / (dP * 10.0)                # g/cm3 per bar
    else:
        drho = (rho_w(T, P_mpa + dP) - rho_w(T, P_mpa - dP)) / 1000.0
        drho_dPbar = drho / (2 * dP * 10.0)            # g/cm3 per bar
    kappa = drho_dPbar / rho1                          # 1/bar
    v1 = MW_WATER / rho1                               # cm3/mol
    return ((1 - p['xi']) * v1
            + p['xi'] * R_CM3_BAR * T * kappa
            + R_CM3_BAR * T * rho1 * kappa * (p['a'] + p['b'] * (1000.0 / T) ** 0.5))


def main():
    print('SANITY at 298.15 K, 0.1 MPa (measured: CO2 ~33.4-33.9, H2S 35.1 +/- 0.6):')
    for g in ('CO2', 'H2S', 'CH4'):
        print(f"  {g:4s} AD03 {v_inf_ad03(g, 298.15, 0.1):6.2f} cm3/mol")

    # The Murphy & Gaines test, identical grouping to fit_pr_vshift.py
    from fits.fit_pr_vshift import murphy_gaines_h2s
    mg = murphy_gaines_h2s()
    lo = [r for r in mg if r[0] < 300.15]
    hi = [r for r in mg if r[0] > 305.15]
    print(f'\nMurphy & Gaines H2S, {len(lo)} points below 300 K vs {len(hi)} above 305 K:')
    rows = [('measured', None),
            ('AD03', lambda T, P: v_inf_ad03('H2S', T, P)),
            ('Plyasunov', lambda T, P: float(V_ply('H2S', T, P))),
            ('delivered', lambda T, P: V_delivered('H2S', T, P))]
    for nm, fn in rows:
        if fn is None:
            a = float(np.mean([V for _, _, V in lo]))
            c = float(np.mean([V for _, _, V in hi]))
        else:
            a = float(np.mean([fn(T, P) for T, P, _ in lo]))
            c = float(np.mean([fn(T, P) for T, P, _ in hi]))
        print(f'  {nm:<10}{a:7.2f} -> {c:6.2f}   trend {c - a:+.2f} cm3/mol')

    print('\nAD03 vs Hnedkovsky H2S densimetry (20 MPa):')
    for T, meas in ((298.15, 34.8), (323.15, 36.0), (373.15, 39.0),
                    (423.15, 43.0), (473.15, 49.2)):
        v = v_inf_ad03('H2S', T, 20.0)
        print(f'  {T:6.1f} K: AD03 {v:6.2f}  measured {meas:5.1f}  ({100*(v-meas)/meas:+.1f}%)')


if __name__ == '__main__':
    main()
