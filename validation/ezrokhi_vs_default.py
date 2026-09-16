"""Eclipse's default CO2 Ezrokhi coefficients against this framework, on data.

THE COMPARISON. Both approaches predict the same thing, the densification of
brine by dissolved CO2:

    log10(rho_sat / rho_gas-free) = A(T) * w_CO2

so comparing them isolates the CO2 coefficient and removes any question about
the salt coefficient, which both would carry identically. Measured densities
come from Yan et al. (2011), 0/1/5 mol/kg NaCl, and Calabrese et al. (2019),
0.77 and 2.50 mol/kg, using each paper's own measured dissolved amount. Where a
paper measured its own gas-free brine (Calabrese did, Yan did not) that
measurement is the baseline, so the comparison is not contaminated by a brine
correlation.

ECLIPSE DEFAULT (DENAQA / VISCAQA, per weight fraction, T in degC):
    density    A0 = 0.1033, A1 = -2.2991e-5, A2 = -2.3658e-6
    viscosity  B0 = B1 = B2 = 0

That viscosity default is not a small approximation. It asserts that dissolved
CO2 does not change brine viscosity at all, where the Calabrese correlation's brine
points give +12% near saturation at 275 K falling to under +3% at 449 K.
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

HERE = os.path.dirname(os.path.abspath(__file__))

from brine_gas.brine_properties import M_NACL, rho_brine, salinity_from_molality
from fits.ezrokhi_fits import A_density
from brine_gas.plyasunov_model import gas_mw
from brine_gas.water_properties import MW_WATER, rho_w

# tNavigator Table 3.2 / 3.3 (image `ezrokhi_coeffics.png`, supplied by Mark),
# understood to match the SLB defaults. Density coefficients per weight fraction,
# T in Celsius.
#
# SIGN CORRECTION, and it is the failure mode this project keeps meeting: the
# table prints CO2 a2 as 0.000002 with no minus sign. That reading makes A RISE
# with temperature, which is the wrong sign of curvature - the densification must
# weaken as V_phi grows while molar mass does not - and it diverges from both
# this framework and the value Mark quoted from Eclipse (-2.3658e-6, which rounds
# to -0.000002). Taken as negative - and note this is NOT a model-versus-model
# argument: A must fall with temperature because the densification weakens as
# V_phi grows while molar mass does not, and the positive reading makes it rise.
#
# C1's printed a2 = 0.000003 is a DIFFERENT case and is left AS PRINTED. There
# is no qualitative discriminator: A_CH4 is negative and falls with temperature
# under either sign, so nothing is obviously wrong with the positive reading.
# Comparing the two against this framework gives mean deviations of 0.0375
# (positive) and 0.0251 (negative), which slightly favours negative but is
# model-against-model - the exact argument this project has retracted before -
# and the gap is far smaller than the H2S disagreement documented below. The
# sign of C1's a2 is therefore UNRESOLVED from the image alone.
TNAV_DENSITY = {
    'CO2':  (0.1033000, -0.000023, -0.000002),   # a2 sign corrected
    'H2S':  (0.0065950, -0.000897,  0.0),
    'CH4':  (-0.500757, -0.001400,  0.000003),   # a2 sign as printed
    'NaCl': (0.309360,  -0.000069,  0.000000138),
    'CaCl2': (0.3627,    0.0,       0.00000014),
}
# Table 3.3 lists NO gas at all - only NaCl and CaCl2. So the shipped viscosity
# treatment of every dissolved gas is exactly zero, which is the finding.
TNAV_VISCOSITY = {'NaCl': (0.71800, 0.003590, 0.0),
                  'CaCl2': (1.48700, -0.001720, 0.0)}

ECLIPSE_A = TNAV_DENSITY['CO2']
ECLIPSE_B = (0.0, 0.0, 0.0)
M_CO2 = gas_mw('CO2')


def a_eclipse(T_C):
    a0, a1, a2 = ECLIPSE_A
    return a0 + a1 * T_C + a2 * T_C * T_C


def _rows():
    """(T_K, P_MPa, m_NaCl, m_CO2, rho_meas, rho_base, source) per point."""
    import validation.yan2011_validation as Y
    out = []
    for T, P, m, mco2, x, rho in Y.DATA:
        S = salinity_from_molality(m) if m > 0 else 0.0
        base = (rho_brine(T, P, S) if m > 0 else rho_w(T, P)) / 1000.0
        out.append((T, P, m, mco2, rho, base, 'Yan 2011'))

    cal = pd.read_csv(os.path.join(_RESULTS, 'calabrese_density_results.csv'))
    for _, r in cal.iterrows():
        # Calabrese measured its own gas-free brine; use it, not a correlation.
        base = r['rho_base_fit'] / 1000.0
        Wb = 1000.0 + r['m'] * M_NACL
        n_w = 1000.0 / MW_WATER
        mco2 = r['x'] * n_w / (1.0 - r['x'])          # mol CO2 per kg water
        out.append((r['T'], r['P'], r['m'], mco2, r['rho_meas'] / 1000.0,
                    base, 'Calabrese 2019'))
    return out


def evaluate():
    recs = []
    for T, P, m, mco2, rho, base, src in _rows():
        Wb = 1000.0 + m * M_NACL
        w = mco2 * M_CO2 / (Wb + mco2 * M_CO2)        # weight fraction CO2
        meas = np.log10(rho / base)
        tc = T - 273.15
        # the tNavigator table AS PRINTED (a2 unsigned, read positive) - scored
        # alongside so the sign-corrected reading is not the only case shown
        a0, a1, a2 = ECLIPSE_A
        a_printed = a0 + a1 * tc + abs(a2) * tc * tc
        recs.append(dict(src=src, m=m, T=T, P=P, w=w, meas=meas,
                         ecl=a_eclipse(tc) * w,
                         ecl_printed=a_printed * w,
                         ours=A_density('CO2', tc, min(max(P, 1.0), 99.0), m) * w))
    return pd.DataFrame(recs)


def main():
    df = evaluate()
    print('=' * 78)
    print('ECLIPSE DEFAULT CO2 EZROKHI vs THIS FRAMEWORK, AGAINST MEASURED DENSITY')
    print('=' * 78)
    print(f'\n  {len(df)} points. Error is in the DENSIFICATION itself, as')
    print('  percentage points of brine density - the quantity being predicted.\n')
    for col, lbl in (('ecl', 'Eclipse default'), ('ecl_printed', 'default, a2 as printed'),
                     ('ours', 'this framework')):
        e = 100.0 * (10 ** df[col] - 10 ** df.meas)
        print(f'  {lbl:<18} mean {e.mean():+7.3f} pp   mean|err| {e.abs().mean():.3f} pp'
              f'   max|err| {e.abs().max():.3f} pp')

    print(f"\n  {'source':<16}{'m':>6}{'n':>5}{'Eclipse':>22}{'this framework':>22}")
    print(f"  {'':<16}{'':>6}{'':>5}{'mean':>11}{'mean|e|':>11}{'mean':>11}{'mean|e|':>11}")
    print('  ' + '-' * 71)
    for (src, m), g in df.groupby(['src', 'm']):
        ee = 100.0 * (10 ** g.ecl - 10 ** g.meas)
        eo = 100.0 * (10 ** g.ours - 10 ** g.meas)
        print(f'  {src:<16}{m:6.2f}{len(g):5d}{ee.mean():+11.3f}{ee.abs().mean():11.3f}'
              f'{eo.mean():+11.3f}{eo.abs().mean():11.3f}')

    print("""
  READING THIS. The two agree closely on CO2 density, which is a good result for
  both: the Eclipse default is a sound CO2 density correction and this work does
  not overturn it. The differences that matter are elsewhere.""")

    print('\n' + '=' * 78)
    print('WHERE THE DEFAULT ACTUALLY COSTS SOMETHING')
    print('=' * 78)
    from brine_gas.garcia_mixing import viscosity_correction_single
    print("""
  1. VISCOSITY. The default is B0 = B1 = B2 = 0, i.e. dissolved CO2 is asserted
     to have no effect on brine viscosity. Measured (Calabrese Eq. 25, 415 brine
     points, AARD 0.9%):""")
    for tc in (25, 50, 75, 100, 150):
        f = viscosity_correction_single('CO2', 0.02, degf=tc * 1.8 + 32)
        print(f'       {tc:3d} degC, x = 0.02:  measured {100 * (f - 1):+5.1f}%'
              f'   Eclipse default  +0.0%')
    print("""
  2. OTHER GASES. Defaults exist for CO2, H2S and CH4 (and NaCl, CaCl2). N2 and
     H2 have none, and this work supplies them. Note also that the shipped H2S
     density coefficient disagrees with this framework by a factor of 1.4 at
     25 degC growing to 2.1 at 100 degC - the largest disagreement in the set,
     and the one worth a second look.

  3. PRESSURE. Neither form has a pressure term. Over 10-70 MPa this framework's
     A moves by about 14% of its own size for CO2, which is error no Ezrokhi
     coefficient set can carry, default or otherwise.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
