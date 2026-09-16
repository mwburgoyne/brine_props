"""Ostermann SPE 14211 CH4 viscosity: unified Arrhenius x Langmuir fit.

SOURCE: Papers/9_Ostermann_1985_SPE14211_CH4_brine_viscosity.pdf, Tables 1a-1c,
the full measured dataset (8 pressures at 100 and 150 degF, 7 at 250 degF).
Transcribed from the PDF text layer and cross-checked against a rendered page
image; the scan carries a doubled text layer so neither read alone is trusted.

WHY THIS EXISTS. Until 2026-07-25 the CH4 correction was built from the paper's
three *summary* plateau values plus an assumed linear-in-x ramp below plateau,
spliced with a min(). That needed four parameters and a switch, and the ramp
shape was never fitted to anything: it was pinned by forcing ramp = plateau at
the S&W x_CH4 at 2000 psia. Two secondhand beliefs turned out to be wrong once
the primary source was read:

  1. The plateau is NOT flat. The measured ratio drifts up by 1.0 pp (100 degF)
     and 1.6 pp (150 degF) between 2000 and 7000 psi. "Essentially constant" in
     the text is an approximation, and reasoning that assumed strict flatness
     (which forced a very sharp saturation) was mistaken.
  2. The dissolved amount need not be modelled. The paper reports Rsw directly,
     so x_CH4 is measured, not taken from an EOS.

RESULT. A single equation with no cap fits all 23 points as well as the previous
four-parameter spliced form:

    mu_sat/mu_free = 1 + A exp(B/T) * x/(K + x)

    Arrhenius in T (activated flow, hydration structure destroyed thermally),
    Langmuir in x (finite structurable capacity), linear in x as x -> 0 as the
    dilute limit requires, and asymptoting naturally in both variables.

Fit quality against the 23 measurements (RMS of the viscosity-ratio residual):
    unified Arrhenius x Langmuir   3 params   0.723 pp
    Arrhenius x Weibull            3 params   0.704 pp  (indistinguishable)
    previous capped ramp           4 params   0.713 pp
    Arrhenius x linear, no satn.   2 params   1.146 pp  (saturation IS needed)

Langmuir is preferred over Weibull on interpretation (site occupancy) since the
two are statistically indistinguishable here.
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
from scipy.optimize import curve_fit


# SPE 14211 Tables 1a-1c: degF -> [(P_sat psi, Rsw SCF/STB, mu/mu_pure), ...]
OSTERMANN = {
    100: [(500, 5.0, 1.03072), (1000, 9.0, 1.05201), (2000, 14.0, 1.05862),
          (3000, 18.5, 1.05708), (4000, 21.5, 1.05753), (5000, 24.7, 1.05979),
          (6000, 27.0, 1.05844), (7000, 29.3, 1.06881)],
    150: [(500, 4.1, 0.99713), (1000, 7.0, 1.01054), (2000, 12.0, 1.03818),
          (3000, 15.9, 1.03879), (4000, 19.0, 1.03892), (5000, 21.5, 1.04517),
          (6000, 23.8, 1.04752), (7000, 25.8, 1.05437)],
    250: [(1000, 7.0, 1.02457), (2000, 13.1, 1.02983), (3000, 17.5, 1.02405),
          (4000, 21.5, 1.02849), (5000, 25.0, 1.02269), (6000, 28.0, 1.02519),
          (7000, 30.5, 1.02653)],
}

MOLAR_VOL_SCF = 379.48          # scf per lbmol at 60 degF, 14.696 psia
LBMOL_WATER_PER_STB = 350.15 / 18.015   # 1 STB = 5.6146 ft3 at 62.366 lb/ft3


def x_ch4_from_rsw(rsw):
    """Dissolved CH4 mole fraction (salt-free basis) from Rsw in SCF/STB."""
    n_gas = rsw / MOLAR_VOL_SCF
    return n_gas / (n_gas + LBMOL_WATER_PER_STB)


def dataset():
    T, X, E, F = [], [], [], []
    for degf, rows in OSTERMANN.items():
        for _, rsw, ratio in rows:
            T.append((degf - 32.0) / 1.8 + 273.15)
            X.append(x_ch4_from_rsw(rsw))
            E.append(ratio - 1.0)
            F.append(degf)
    return map(np.array, (T, X, E, F))


def model(TX, A, B, K):
    T, x = TX
    return A * np.exp(B / T) * x / (K + x)


def main():
    T, X, E, F = dataset()
    p, _ = curve_fit(model, (T, X), E, p0=[2e-3, 1150, 1e-3], maxfev=600000)
    A, B, K = p
    res = model((T, X), *p) - E
    print(f'{len(E)} measured points, x from {X.min():.5f} to {X.max():.5f}')
    print(f'\n  mu_sat/mu_free = 1 + A exp(B/T) x/(K+x)')
    print(f'    A = {A:.8e}')
    print(f'    B = {B:.5f} K      (E_a = {8.314 * B / 1000:.2f} kJ/mol)')
    print(f'    K = {K:.8e}        (half-saturation, 1 CH4 per {1/K:,.0f} waters)')
    print(f'\n  RMS residual {100*np.sqrt(np.mean(res**2)):.3f} pp, '
          f'max {100*np.abs(res).max():.3f} pp')
    print('  bias by isotherm: ' + ', '.join(
        f'{d} degF {100*np.mean(res[F==d]):+.2f} pp' for d in (100, 150, 250)))

    print('\n  measured vs model:')
    print(f"    {'degF':>5} {'psi':>6} {'Rsw':>6} {'x':>9} {'meas':>9} {'model':>9} {'diff pp':>8}")
    for degf, rows in OSTERMANN.items():
        for P, rsw, ratio in rows:
            t = (degf - 32.0) / 1.8 + 273.15
            x = x_ch4_from_rsw(rsw)
            m = 1 + model((np.array([t]), np.array([x])), *p)[0]
            print(f'    {degf:5d} {P:6d} {rsw:6.1f} {x:9.5f} {ratio:9.5f} '
                  f'{m:9.5f} {100*(m-ratio):+8.2f}')

    print('\n  dilute limit slope d(excess)/dx = (A/K) exp(B/T):')
    for degf in (100, 150, 250):
        t = (degf - 32.0) / 1.8 + 273.15
        print(f'    {degf} degF: {A/K*np.exp(B/t):.1f} per unit mole fraction')
    return p


if __name__ == '__main__':
    main()
