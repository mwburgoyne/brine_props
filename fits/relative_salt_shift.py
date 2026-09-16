"""The RELATIVE salt-shift law: fit, and full costing against the shipped
absolute law. Mark's direction 2026-07-30 (after the collapse test showed the
two frames statistically indistinguishable once the KOH 5 M points are set
aside): switch the shipped term to a dimensionless relative form, costed
here BEFORE implementation.

FORM:   V_eff(gas,T,P,m) = V_phi(gas,T,P) * (1 + g(m)/100)
        g(m) = -c*m/(1+d*m)  [percent], m = NaCl molality

FIT BASIS: the same seven Tiepel & Gubbins KCl points the shipped absolute
law was fitted to (direct dilatometry, no density data, no KOH involved -
which sidesteps the KOH-5 question entirely). Weighted by propagated sds.

If ADOPTED, the coefficients printed here get pinned in vphi_route +
validation.py + pyResToolbox + ResToolbox3 and this script is the generator.
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
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
from fits.salt_effect_vphi import TIEPEL
from brine_gas.vphi_route import V_phi
from brine_gas.plyasunov_model import gas_mw

# The SUPERSEDED absolute law (shipped 2026-07-25 to 2026-07-30), kept here as
# the costing baseline. The live law is vphi_route.salt_fraction.
ABS_A, ABS_B = 0.5914, 0.0416


def salt_shift(m_nacl):
    m = max(float(m_nacl), 0.0)
    return -ABS_A * m / (1.0 + ABS_B * m)

GASES = ('CH4', 'C2H6', 'C3H8', 'NC4H10', 'CO2', 'H2S', 'H2', 'N2')


M_KCL = 74.5513   # g/mol
T_TIEPEL = 298.15  # Tiepel & Gubbins Table I, 25 degC, atmospheric pressure


def kcl_molality_from_molarity(c_mol_per_L, T_K=T_TIEPEL):
    """Tiepel's KCl concentrations are MOLAR. Convert to molality with the
    Appelo KCl solution density at Tiepel's own temperature (review item Fig. 5,
    2026-09-05): c = m rho / (1 + m M_KCl/1000), rho in kg/L. 1/2/4 mol/L ->
    1.033/2.133/4.587 mol/kg."""
    from scipy.optimize import brentq
    from brine_gas.salt_route import brine_density
    f = lambda m: (m * brine_density(T_K, 0.101325, salts={'KCl': m}) / 1000.0
                   / (1.0 + m * M_KCL / 1000.0) - c_mol_per_L)
    return brentq(f, 0.5 * c_mol_per_L, 2.0 * c_mol_per_L, xtol=1e-12)


def fit_relative(molal=True):
    """molal=True (shipped since 2026-09-05) fits against molality; molal=False
    reproduces the 2026-07-30 fit that treated Tiepel's molar values as molal
    (c = 1.7009, d = 0.090684)."""
    water = {g: (v, s) for g, e, c, v, s in TIEPEL if e == 'water'}
    conv = kcl_molality_from_molarity if molal else (lambda c: c)
    pts = [(conv(c), 100 * (v - water[g][0]) / water[g][0],
            100 * float(np.hypot(s, water[g][1])) / water[g][0])
           for g, e, c, v, s in TIEPEL if e == 'KCl']
    m = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    sd = np.array([p[2] for p in pts])
    (c, d), cov = curve_fit(lambda mm, cc, dd: -cc * mm / (1 + dd * mm),
                            m, y, p0=(1.8, 0.05), sigma=sd, absolute_sigma=True,
                            bounds=([0, 0], [20, 5]), maxfev=20000)
    resid = y + c * m / (1 + d * m)
    return c, d, float(np.sqrt(np.mean(resid ** 2))), np.sqrt(np.diag(cov))


def main():
    c, d, rms, (c_sd, d_sd) = fit_relative()
    c0, d0, rms0, _ = fit_relative(molal=False)
    g = lambda m: -c * m / (1.0 + d * m)   # percent
    print('=' * 78)
    print('RELATIVE SALT-SHIFT LAW: FIT AND COSTING vs SHIPPED ABSOLUTE')
    print('=' * 78)
    print(f'\n  g(m) = -{c:.4f}*m/(1 + {d:.4f}*m) percent '
          f'(c +/- {c_sd:.3f}, d +/- {d_sd:.3f}; KCl n=7 on MOLALITY, fit RMS {rms:.2f} pp)')
    print(f'  PINS (8 sig digits, paste into vphi_route.SALT_C/SALT_D): '
          f'{c / 100:.8g}, {d:.8g}')
    print(f'  superseded molar-as-molal fit: g = -{c0:.4f}*m/(1 + {d0:.4f}*m), '
          f'RMS {rms0:.2f} pp; at 1/2.5/5 molal it gave '
          f'{-c0/(1+d0):+.3f} / {-c0*2.5/(1+2.5*d0):+.3f} / {-c0*5/(1+5*d0):+.3f} %')
    # the shipped constants must be this generator's output
    import brine_gas.vphi_route as _VR
    assert abs(_VR.SALT_C - c / 100) < 1e-9 and abs(_VR.SALT_D - d) < 1e-8, \
        f'vphi_route pins ({_VR.SALT_C}, {_VR.SALT_D}) are not this fit ({c/100:.8g}, {d:.8g})'
    print(f'  fraction at 1/2/5 molal: {g(1):+.2f} / {g(2):+.2f} / {g(5):+.2f} %')

    print('\n  1. PER-GAS SHIFT (cm3/mol), 298.15 K / 10 MPa (V_inf from route, salt-free):')
    print(f"    {'gas':<8} {'V_inf':>7} {'abs m=1':>8} {'rel m=1':>8} "
          f"{'abs m=5':>8} {'rel m=5':>8}")
    for gas in GASES:
        v = float(V_phi(gas, 298.15, 10.0))
        print(f'    {gas:<8} {v:7.2f} {salt_shift(1):8.2f} {g(1) * v / 100:8.2f} '
              f'{salt_shift(5):8.2f} {g(5) * v / 100:8.2f}')

    print('\n  2. T-DEPENDENCE the relative law introduces (CO2, 20 MPa, 5 molal):')
    for T in (298.15, 373.15, 449.0):
        v = float(V_phi('CO2', T, 20.0))
        print(f'    {T:6.1f} K: V_inf {v:5.1f} -> abs {salt_shift(5):+.2f}, '
              f'rel {g(5) * v / 100:+.2f} cm3/mol')

    print('\n  3. YAN 2011 RE-SCORED (V_phi error %, model with each law):')
    yan = pd.read_csv(os.path.join(_RESULTS, 'yan2011_results.csv'))
    vshift = np.vectorize(salt_shift)
    v_inf = yan.vphi_model - vshift(yan.m)              # strip shipped
    for name, model in (('absolute (shipped)', yan.vphi_model),
                        ('relative', v_inf * (1 + g(yan.m) / 100.0))):
        err = 100 * (model - yan.vphi_implied) / yan.vphi_implied
        by_m = [f'm={m:g}: {err[np.isclose(yan.m, m)].mean():+5.1f}'
                for m in (0, 1, 5)]
        print(f'    {name:<20} ' + '  '.join(by_m)
              + f'   RMS(all) {np.sqrt(np.mean(err ** 2)):.2f}%')

    print('\n  4. CALABRESE RE-SCORED (anchored offsets barely move; RMS by subset):')
    cal = pd.read_csv(os.path.join(_RESULTS, 'calabrese_density_results.csv'))
    v_inf_c = cal.V_ply - vshift(cal.m)
    for name, model in (('absolute (shipped)', cal.V_ply),
                        ('relative', v_inf_c * (1 + g(cal.m) / 100.0))):
        err = 100 * (model - cal.V_imp) / cal.V_imp
        by_m = [f'm={m:g}: {np.sqrt(np.mean(err[np.isclose(cal.m, m)] ** 2)):5.2f}%'
                for m in (0.77, 2.5)]
        print(f'    {name:<20} ' + '  '.join(by_m))

    print('\n  5. DELIVERED-DENSITY IMPACT (change in the GAS density effect, '
          'rel vs abs):')
    from brine_gas.brine_properties import rho_brine, salinity_from_molality
    T, P = 323.15, 10.0
    for m in (2.5, 5.0):
        S = salinity_from_molality(m)
        rho1 = rho_brine(T, P, S) / 1000.0
        parts = []
        for gas in ('CO2', 'CH4', 'H2S'):
            M2 = gas_mw(gas)
            v0 = float(V_phi(gas, T, P))          # m absent -> salt-free
            va = v0 + salt_shift(m)
            vr = v0 * (1 + g(m) / 100.0)
            ea, er = M2 - rho1 * va, M2 - rho1 * vr
            parts.append(f'{gas} {100 * (er / ea - 1):+.1f}%')
        print(f'    {m:g} molal, 323 K: ' + ',  '.join(parts))
    print('\n  6. EOS-EMBEDDED SALINITY RESPONSE (the double-count check; '
          'quoted in the manuscript):')
    import brine_gas.pr_vphi_model as _PR
    for gas in ('CO2', 'CH4', 'H2S', 'N2', 'H2'):
        v0 = _PR.V2_inf(gas, 323.15, 20.0, m_nacl=0.0)
        v1 = _PR.V2_inf(gas, 323.15, 20.0, m_nacl=1.0)
        print(f'    {gas:<5} EOS m into alpha/kij: {100 * (v1 / v0 - 1):+.3f} '
              f'%/molal  vs measured fraction {g(1):+.3f}')
    print('    -> an order of magnitude too small and linear where measurements'
          ' saturate;\n       the delivered route evaluates V_inf at freshwater'
          ' parameters and the\n       fraction above carries the whole effect.')
    print("""
  7. IMPLEMENTATION RIPPLE if adopted:
     - vphi_route.salt_shift -> relative form (V_phi route signature unchanged)
     - validation.py: 3 pinned shift values + shape checks re-pin (198 total)
     - pyResToolbox brine/vphi_route + tests; ResToolbox3 + tests (scripted port)
     - documents: Table 2 caption, salinity passages, worked examples, gate""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
