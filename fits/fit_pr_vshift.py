"""Fit the dimensionless VSHIFT of `pr_vphi_model` to DIRECT VOLUMETRIC data only.

Run this to regenerate `pr_vphi_model.VSHIFT`. It prints the fitted values, the
in-sample fit, a held-out comparison against Plyasunov, and the H2S temperature
trend that motivated the whole exercise.

THE INCLUSION RULE, and why it is drawn here. A volume shift calibrated on bad
V_phi is a bad volume shift, so only measurements that determined a volume
DIRECTLY are used:

  INCLUDED (direct volumetric)
    Hnedkovsky 1996  vibrating-tube densimetry, CH4/CO2/H2S, 298-473 K
                     (Paper 2, digitised from the printed table)
    Moore 1982       direct densimetry at 298.15 K (Paper 22)
    Tiepel 1972      Horiuti dilatometry at 298.15 K (Paper 26)
    Bignell 1984     densities of gas-saturated water, N2 correlation (Paper 24)
    Murphy & Gaines  float-method densities, H2S, re-derived from their Table I
                     (Paper 10, via murphy_gaines_h2s_refit.vphi_from_table_I)
    Zhou & Battino   dilatometric, 298.15 K (Paper 4), except their CH4 value
                     which literature-density.md rejects as inconsistent with
                     every other source

  EXCLUDED (indirect - the quantity was inferred, not measured)
    Enns 1965        equilibrium gas pressure versus applied hydrostatic pressure
    O'Sullivan 1970  solubility versus pressure (Krichevsky-Kasarnovsky slope)
    Heusler 1972     reversible cell voltage versus pressure
    Torin-Ollarves   KK fitting parameter; the authors state it reads low
    Lauder 1959      float method, but its two CO2 values (44 and 28 cm3/mol at
                     273 K) are ONE measurement at two loadings, i.e. finite-
                     concentration, not the infinite-dilution quantity fitted here

  EXCLUDED (V_phi inverted from brine density)
    Calabrese 2019, Yan 2011 - these carry a baseline uncertainty of about
    +/-0.3 cm3/mol amplified by dividing by x, and they scatter five to seven
    times wider than direct densimetry (see the c2 scatter work). They are
    validation, never calibration.

The INDIRECT set is scored below as a held-out check (`HELD_OUT`). The
brine-inverted set is NOT scored here - Calabrese and Yan are scored on
delivered density by `calabrese_density_validation.py` and
`yan2011_validation.py`, which is where they belong: they test the whole chain,
not the volume shift alone.
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
from scipy.optimize import minimize_scalar, minimize

HERE = os.path.dirname(os.path.abspath(__file__))

from brine_gas.plyasunov_model import V2_inf as V_plyasunov
import brine_gas.pr_vphi_model as PR

# ---------------------------------------------------------------- CALIBRATION
# Hnedkovsky 1996, vibrating-tube densimetry. (T/K, P/MPa, V_phi)
HNEDKOVSKY = {
    'CH4': [(298.15, 28, 37.2), (298.15, 35, 36.3), (323.15, 28, 37.5),
            (323.15, 35, 37.1), (373.15, 28, 40.7), (373.15, 35, 40.3),
            (423.15, 28, 46.0), (423.15, 35, 45.8), (473.15, 28, 54.4),
            (473.15, 35, 53.8)],
    'CO2': [(298.15, 20, 33.4), (298.15, 35, 33.5), (323.15, 20, 33.8),
            (323.15, 35, 33.7), (373.15, 20, 37.8), (373.15, 35, 37.2),
            (423.15, 20, 42.7), (423.15, 35, 41.9), (473.15, 20, 50.0),
            (473.15, 35, 48.4)],
    'H2S': [(298.15, 20, 34.8), (298.15, 35, 35.0), (323.15, 20, 36.0),
            (323.15, 35, 35.8), (373.15, 20, 39.0), (373.15, 35, 38.4),
            (423.15, 20, 43.0), (423.15, 35, 42.5), (473.15, 20, 49.2),
            (473.15, 35, 47.7)],
}

# Direct volumetric determinations at 298.15 K, 0.1 MPa
AT_298 = {
    'CH4':  [('Moore 1982 densimetry', 34.5), ('Tiepel 1972 dilatometry', 37.42)],
    'CO2':  [('Moore 1982 densimetry', 33.9)],
    'H2S':  [],
    'N2':   [('Moore 1982 densimetry', 35.7), ('Zhou 2001 dilatometry', 33.1)],
    'H2':   [('Moore 1982 densimetry', 26.7), ('Tiepel 1972 dilatometry', 25.20),
             ('Zhou 2001 dilatometry', 23.1)],
    'C2H6': [('Moore 1982 densimetry', 52.9), ('Tiepel 1972 dilatometry', 53.27),
             ('Zhou 2001 dilatometry', 49.6)],
}

# Bignell 1984 N2 correlation V/(cm3/mol) = 34.49 - 0.00822 t, t in degC.
# A correlation rather than raw points, so it is included at three temperatures
# only and flagged as such. Bignell MEASURED 3-21 degC only (stated in the
# paper), so the correlation is sampled INSIDE that window; the pre-2026-08-08
# sampling (0, 25, 50) extrapolated his fit line 29 K beyond data and gave
# s(N2) = -0.155288. In-window sampling adopted 2026-08-08 (Mark's call).
BIGNELL_N2_T = (3.0, 12.0, 21.0)


def bignell_n2(t_c):
    return 34.49 - 0.00822 * t_c


def murphy_gaines_h2s():
    """The 21 H2S volumes re-derived from Murphy & Gaines Table I densities."""
    import fits.murphy_gaines_h2s_refit as MG
    return [(t + 273.15, max(p_atm * MG.ATM, 0.101325), V)
            for t, p_atm, rho, rstar, x, V in MG.vphi_from_table_I()]


# Barbero, McCurdy & Tremaine 1982 (Paper 62), Can. J. Chem. 60:1872, Table 1:
# standard-state (infinite-dilution) H2S volumes from vibrating-tube densimetry
# of 0.03-0.08 mol/kg solutions at atmospheric pressure, checked against the
# rendered page. Printed error limits (sd of the mean) 2.33 / 0.11 / 0.13.
# Added to the calibration 2026-09-06 (Mark: "incorporate"); the shift moved
# -0.079740 -> -0.079416, 0.009 cm3/mol.
BARBERO_H2S = [(283.15, 0.101325, 34.04), (298.15, 0.101325, 34.92),
               (313.15, 0.101325, 35.60)]


def calibration_set():
    """(gas -> [(T, P, V_phi, source)]) using direct volumetric data only."""
    out = {g: [] for g in PR.SUPPORTED}
    for g, pts in HNEDKOVSKY.items():
        for T, P, V in pts:
            out[g].append((T, float(P), V, 'Hnedkovsky 1996'))
    for g, items in AT_298.items():
        for nm, V in items:
            out[g].append((298.15, 0.1, V, nm))
    for t in BIGNELL_N2_T:
        out['N2'].append((t + 273.15, 0.1, bignell_n2(t), 'Bignell 1984 (correlation)'))
    for T, P, V in murphy_gaines_h2s():
        out['H2S'].append((T, P, V, 'Murphy & Gaines 1974'))
    for T, P, V in BARBERO_H2S:
        out['H2S'].append((T, P, V, 'Barbero 1982'))
    # McBride-Wright 2015 (Paper 21): 98 CO2-water densities at three MEASURED
    # loadings, 274.7-449 K, 15-100 MPa, reduced against IAPWS water per row
    # (mcbridewright2015_density_check.py). Fresh-water inversion at measured
    # composition, the same class as Murphy & Gaines; ADMITTED 2026-09-06
    # (Mark: "adopt the point-weighted refit"). Point-weighted like every
    # other source.
    import validation.mcbridewright2015_density_check as MW
    for x, T, P, rho, V, _ in MW.reduce():
        out['CO2'].append((float(T), float(min(P, 100.0)), float(V), 'McBride-Wright 2015'))
    return out


# ------------------------------------------------------------------ HELD OUT
HELD_OUT = {
    'CO2': [(298.15, 0.1, 34.80, 'Enns 1965 (indirect)'),
            (273.15, 0.1, 44.0, 'Lauder 1959 dilute (finite loading)'),
            (273.15, 0.1, 28.0, 'Lauder 1959 loaded (finite loading)')]
           # Hebach 2004 (Paper 23): 203 CO2-saturated water densities, x from
           # the flash (not measured), so held out by the inclusion rule.
           + __import__('validation.hebach2004_density_check', fromlist=['_']).held_out_points(),
    'CH4': [(324.65, 0.1, 37.10, "O'Sullivan 1970 (indirect)")],
    'N2':  [(298.15, 0.1, 33.30, 'Enns 1965 (indirect)'),
            (324.65, 0.1, 34.05, "O'Sullivan 1970 (indirect)"),
            # Kennan & Pollack 1990 (Paper 56): modified Van Slyke, the
            # average over their seven N2 points at 44.6-115.8 atm (mean
            # 7.9 MPa), +/-2 cm3/mol. Their own Krichevsky-Kasarnovsky
            # refit of the same solubilities gives 36, and the Alvarez &
            # Fernandez-Prini Comment (Paper 57) 35.5 with gamma = 1; the
            # two readings bracket the route. Held out, never fitted.
            (298.15, 7.9, 31.0, 'Kennan & Pollack 1990 (Van Slyke)')],
    # Bignell 1987 (Paper 61): 50 H2 density differences at 4-25 degC, reduced
    # with the flash solubility in bignell1987_h2_check.py (pinned there). The
    # route sits 5-16% above them; below all three direct 298 K values. Held
    # out, never fitted (2026-09-06).
    'H2':  [(298.15, 0.1, 24.5, 'Heusler 1972 (indirect)')]
           + __import__('validation.bignell1987_h2_check', fromlist=['_']).held_out_points(),
    'H2S': [], 'C2H6': [],
}


def _score(pts, fn):
    e = np.array([fn(T, P) - V for T, P, V, _ in pts])
    v = np.array([V for _, _, V, _ in pts])
    return dict(n=len(pts), mae=float(np.mean(np.abs(e))),
                rms=float(np.sqrt(np.mean(e ** 2))),
                map_=float(np.mean(np.abs(100 * e / v))),
                mx=float(np.max(np.abs(100 * e / v))))


def _fmt(r):
    return f"{r['mae']:5.2f}{r['rms']:6.2f}{r['map_']:6.1f}{r['mx']:7.1f}"


def main():
    cal = calibration_set()

    print('=' * 88)
    print('CALIBRATION SET (direct volumetric measurements only)')
    print('=' * 88)
    print(f"\n  {'gas':<6}{'n':>4}  {'T range / K':>16}  sources")
    for g in PR.SUPPORTED:
        pts = cal[g]
        if not pts:
            # C3H8 and NC4H10 shifts are set from single-source 298 K values
            # outside this fit (see pr_vphi_model.VSHIFT comments).
            print(f'  {g:<6}   0  set outside this fit (single-source 298 K)')
            continue
        Ts = [p[0] for p in pts]
        srcs = sorted({p[3] for p in pts})
        print(f'  {g:<6}{len(pts):4d}  {min(Ts):7.1f} to {max(Ts):5.1f}  '
              + ', '.join(srcs))

    print('\n' + '=' * 88)
    print('FIT: one dimensionless shift s per gas')
    print('=' * 88)
    fitted, two_par = {}, {}
    for g in PR.SUPPORTED:
        pts = cal[g]
        if not pts:
            continue
        o = minimize_scalar(
            lambda s: _score(pts, lambda T, P: PR.V2_inf(g, T, P, s=s))['rms'],
            bounds=(-1.0, 1.0), method='bounded', options={'xatol': 1e-9})
        fitted[g] = float(o.x)
        # two-parameter s(T) = s0 + s1*1000/T, fitted only to show it overfits
        o2 = minimize(
            lambda p: _score(pts, lambda T, P: PR.V2_inf_raw(g, T, P)
                             - (p[0] + p[1] * 1000.0 / T) * PR.b_covolume(g))['rms'],
            [fitted[g], 0.0], method='Nelder-Mead',
            options={'maxiter': 20000, 'fatol': 1e-12})
        two_par[g] = tuple(float(x) for x in o2.x)

    print(f"\n  {'gas':<6}{'s (1 par)':>12}{'b / cm3.mol-1':>15}"
          f"{'c = s.b':>10}   {'s0, s1 (2 par, for comparison)':>32}")
    for g in PR.SUPPORTED:
        if g not in fitted:
            continue
        b = PR.b_covolume(g)
        print(f'  {g:<6}{fitted[g]:12.6f}{b:15.2f}{fitted[g]*b:10.2f}   '
              f'{two_par[g][0]:14.4f}{two_par[g][1]:10.4f}')

    print('\n  Paste into pr_vphi_model.VSHIFT:')
    print('    VSHIFT = {')
    for g in PR.SUPPORTED:
        if g not in fitted:
            continue
        print(f"        {g!r:<8}: {fitted[g]:.6f},")
    print('    }')

    print('\n' + '=' * 88)
    print('IN-SAMPLE fit, and Plyasunov on the same points')
    print('=' * 88)
    print(f"\n  {'gas':<6}{'n':>4}   {'Plyasunov':>26}   {'PR + s':>26}")
    print(f"  {'':<6}{'':>4}   {'MAE   RMS   MA%   max%':>26}   "
          f"{'MAE   RMS   MA%   max%':>26}")
    print('  ' + '-' * 70)
    for g in PR.SUPPORTED:
        pts = cal[g]
        if not pts:
            continue
        pl = _score(pts, lambda T, P: float(V_plyasunov(g, T, P)))
        pr = _score(pts, lambda T, P: PR.V2_inf(g, T, P, s=fitted[g]))
        print(f'  {g:<6}{len(pts):4d}   {_fmt(pl):>26}   {_fmt(pr):>26}')
    print('\n  CAVEAT: Hnedkovsky is the data Plyasunov was fitted to, so this is')
    print('  an even-handed comparison (both calibrated on it), not a clean test.')

    print('\n' + '=' * 88)
    print('HELD OUT: measurements excluded from the fit')
    print('=' * 88)
    print(f"\n  {'gas':<6}{'n':>4}   {'Plyasunov':>26}   {'PR + s':>26}")
    print('  ' + '-' * 70)
    for g in PR.SUPPORTED:
        pts = HELD_OUT.get(g, [])
        if not pts or g not in fitted:
            continue
        pl = _score(pts, lambda T, P: float(V_plyasunov(g, T, P)))
        pr = _score(pts, lambda T, P: PR.V2_inf(g, T, P, s=fitted[g]))
        print(f'  {g:<6}{len(pts):4d}   {_fmt(pl):>26}   {_fmt(pr):>26}')
    print('\n  The CO2 row is dominated by Lauder\'s 44 and 28 cm3/mol, which are')
    print('  one measurement at two loadings; no infinite-dilution model can')
    print('  satisfy both, so that row bounds nothing.')

    print('\n' + '=' * 88)
    print('THE H2S TEMPERATURE TREND - the defect this route was built to fix')
    print('=' * 88)
    mg = murphy_gaines_h2s()
    lo = [r for r in mg if r[0] < 300.15]
    hi = [r for r in mg if r[0] > 305.15]
    print(f'\n  Murphy & Gaines, {len(lo)} points below 300 K vs {len(hi)} above 305 K:')
    for nm, fn in (
            ('measured', None),
            ('Plyasunov', lambda T, P: float(V_plyasunov('H2S', T, P))),
            ('PR + s', lambda T, P: PR.V2_inf('H2S', T, P, s=fitted['H2S']))):
        if fn is None:
            a = float(np.mean([V for _, _, V in lo]))
            c = float(np.mean([V for _, _, V in hi]))
        else:
            a = float(np.mean([fn(T, P) for T, P, _ in lo]))
            c = float(np.mean([fn(T, P) for T, P, _ in hi]))
        print(f'    {nm:<11}{a:7.2f} -> {c:6.2f}   trend {c - a:+.2f} cm3/mol')
    print('\n  validation.py pins Plyasunov as FLAT here against a measured rise.')
    print('  This route recovers most of the trend, which is the main reason it')
    print('  is worth carrying alongside the incumbent rather than instead of it.')

    print('\n' + '=' * 88)
    print('PARAMETER COST')
    print('=' * 88)
    print("""
  Plyasunov     35 p_in numbers per gas at 8 significant digits (7 digits costs
                1.09% on H2), plus square-well sigma/epsilon/lambda for B12, plus
                IAPWS-IF97 for rho1* and kappa_T.
  PR + VSHIFT   ONE number per gas. Tc, Pc, omega, the S&W water alpha and
                kij_aq are already required for the solubility side, so the
                marginal cost is one number and one cubic root solve.""")


if __name__ == '__main__':
    main()
