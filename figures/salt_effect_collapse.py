"""Does the salt effect on gas V_phi collapse to ONE trend - and in which
representation, absolute shift (cm3/mol) or relative shift (dV/V0)?

Mark's question 2026-07-30. The record so far tested cross-gas similarity only
PER CELL (same electrolyte, same concentration: salt_effect_vphi.py Q1, CV_abs
0.028 vs CV_rel 0.261 at KCl 2 M). This script attempts the GLOBAL collapse:
every direct salting-out measurement, both representations, against IONIC
STRENGTH (so the 2:1 CaCl2 can fold onto the 1:1 salts), with a weighted
common saturating curve fitted to each representation and the residual
scatter compared as a fraction of signal.

Data are IMPORTED from salt_effect_vphi.py (never retyped). Tiepel is
molarity, O'Sullivan/Enns molal; the ~10% basis difference is smaller than
the scatter and is flagged, not converted. O'Sullivan and Enns state no
per-point sd; they are assigned +/-0.5 cm3/mol (comparable to Tiepel's
propagated shift sds of 0.6-1.6) and that assignment is printed.

Output: figures/salt_collapse.png + printed metrics.
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
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
from fits.salt_effect_vphi import (TIEPEL, SALTING_OUT, OSULLIVAN,
                              ENNS_O2_WATER, ENNS_O2_SEAWATER, SEAWATER_I)
from brine_gas.vphi_route import V_phi

OUT = os.path.join(_FIGOUT, 'salt_collapse.png')

# ionic strength per unit concentration
I_FACTOR = {'KCl': 1.0, 'KI': 1.0, 'KOH': 1.0, 'CaCl2': 3.0, 'NaCl': 1.0}


def build_points():
    """(gas, salt, I, dV, dV_sd, V0) for every direct salting-out point."""
    pts = []
    water = {g: (v, s) for g, e, c, v, s in TIEPEL if e == 'water'}
    for g, e, c, v, s in TIEPEL:
        if e in SALTING_OUT:
            v0, s0 = water[g]
            pts.append((g, e, I_FACTOR[e] * c, v - v0, float(np.hypot(s, s0)), v0))
    for g, d in OSULLIVAN.items():
        v0 = d[0.0]
        for m in (1.0, 4.0):
            pts.append((g, 'NaCl', m, d[m] - v0, 0.5, v0))
    ew = float(np.mean(ENNS_O2_WATER))
    pts.append(('O2', 'seawater', SEAWATER_I, ENNS_O2_SEAWATER - ew, 0.5, ew))
    return pts


def sat(I, a, b):
    return -a * I / (1.0 + b * I)


def fit_and_score(I, y, sd, label, unit):
    (a, b), _ = curve_fit(sat, I, y, p0=(0.5, 0.1), sigma=sd,
                          absolute_sigma=True, bounds=([0, 0], [50, 10]),
                          maxfev=20000)
    r = y - sat(I, a, b)
    chi2 = float(np.sum((r / sd) ** 2) / (len(y) - 2))
    wm = float(np.average(np.abs(y), weights=1 / np.asarray(sd) ** 2))
    rms = float(np.sqrt(np.mean(r ** 2)))
    print(f'  {label:<28} -{a:.3f}*I/(1+{b:.3f}*I)  chi2/dof {chi2:4.2f}  '
          f'RMS resid {rms:.3f} {unit} = {100 * rms / wm:.0f}% of mean |shift|')
    return (a, b), rms / wm, chi2


def main():
    pts = build_points()
    gas = np.array([p[0] for p in pts])
    salt = np.array([p[1] for p in pts])
    I = np.array([p[2] for p in pts], float)
    dv = np.array([p[3] for p in pts], float)
    sd = np.array([p[4] for p in pts], float)
    v0 = np.array([p[5] for p in pts], float)
    rel = 100.0 * dv / v0
    rel_sd = 100.0 * sd / v0

    print('=' * 78)
    print('GLOBAL COLLAPSE TEST: absolute vs relative, against ionic strength')
    print('=' * 78)
    print(f'\n  {len(pts)} direct salting-out points, {len(set(gas))} gases, '
          f'{len(set(salt))} electrolytes.')
    print("  O'Sullivan/Enns assigned +/-0.5 cm3/mol (no stated sd); "
          'Tiepel molarity, others molal.\n')
    fa, fr = {}, {}
    fa['p'], fa['frac'], fa['chi2'] = fit_and_score(I, dv, sd,
                                                    'ABSOLUTE dV (cm3/mol)', 'cm3/mol')
    fr['p'], fr['frac'], fr['chi2'] = fit_and_score(I, rel, rel_sd,
                                                    'RELATIVE dV/V0 (%)', '%')

    print('\n  Same fits EXCLUDING NaCl (the shallow outlier salt in both):')
    m = salt != 'NaCl'
    fa2 = fit_and_score(I[m], dv[m], sd[m], 'ABSOLUTE, no NaCl', 'cm3/mol')
    fr2 = fit_and_score(I[m], rel[m], rel_sd[m], 'RELATIVE, no NaCl', '%')

    print("""
  The discriminator cells (same salt, same concentration, V0 varying):
    KCl 2 M,  V0 31.7 -> 53.3 (1.7x): abs -1.11/-1.05/-1.09 (CV 0.028)
                                      rel -3.5/-2.8/-2.0%   (CV 0.26)
    KOH 2 M,  V0 25.2 -> 30.4 (1.2x): abs -1.11/-1.37 (CV 0.15)
                                      rel -4.4/-4.5%  (CV 0.02)
  The only cell with a LARGE V0 lever (C2H6, 1.7x) is absolute-conserved;
  the relative-conserved cell has a 1.2x lever inside +/-1 sigma errors.""")

    # what the choice means for the undata'd big gases
    T, P = 298.15, 10.0
    for g in ('C3H8', 'NC4H10'):
        vg = float(V_phi(g, T, P))
        d_abs = sat(2.0, *fa['p'])
        d_rel = sat(2.0, *fr['p']) / 100.0 * vg
        print(f'  At I = 2, {g} (V0 = {vg:.1f}): absolute law {d_abs:+.2f}, '
              f'relative law {d_rel:+.2f} cm3/mol ({d_rel / d_abs:.1f}x)')

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e8e8e6'
    SALT_COLOR = {'KCl': '#2a78d6', 'KI': '#2a78d6', 'KOH': '#2a78d6',
                  'seawater': '#2a78d6', 'NaCl': '#eb6834', 'CaCl2': '#1baf7a'}
    GAS_MARK = {'Ar': 'o', 'CH4': 's', 'C2H6': '^', 'O2': 'D', 'H2': 'v',
                'N2': 'P'}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor=SURFACE)
    Igrid = np.linspace(0, 12.5, 200)
    for ax, y, ysd, fit, ttl, ylab in (
            (axes[0], dv, sd, fa, 'Absolute shift', 'dV$_\\phi$ (cm$^3$/mol)'),
            (axes[1], rel, rel_sd, fr, 'Relative shift', 'dV$_\\phi$/V$_0$ (%)')):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.axhline(0, color=INK2, lw=0.8)
        for i in range(len(pts)):
            ax.errorbar(I[i], y[i], yerr=ysd[i], fmt=GAS_MARK[gas[i]],
                        color=SALT_COLOR[salt[i]], ms=8, mec=SURFACE, mew=1.0,
                        ecolor='#c9c8c4', elinewidth=1.2, capsize=0, zorder=4)
        ax.plot(Igrid, sat(Igrid, *fit['p']), color=INK, lw=1.8,
                ls=(0, (5, 3)), zorder=3)
        ax.set_title(f"{ttl}: residual scatter {100 * fit['frac']:.0f}% of "
                     f"mean shift, chi$^2$/dof {fit['chi2']:.1f}",
                     fontsize=11.5, color=INK, loc='left')
        ax.set_xlabel('ionic strength (mol/L Tiepel, mol/kg others)',
                      color=INK, fontsize=11)
        ax.set_ylabel(ylab, color=INK, fontsize=11)
        ax.tick_params(colors=INK2, labelsize=10)

    # shipped law on the absolute panel
    axes[0].plot(Igrid, -0.5914 * Igrid / (1 + 0.0416 * Igrid), color='#1baf7a',
                 lw=2.2, zorder=2)
    axes[0].text(9.0, -0.5914 * 9 / (1 + 0.0416 * 9) + 0.35, 'shipped law',
                 color='#1baf7a', fontsize=10, ha='center')
    axes[0].text(11.9, sat(11.9, *fa['p']) - 0.35, 'common fit', color=INK,
                 fontsize=10, ha='right', va='top')
    axes[1].text(11.9, sat(11.9, *fr['p']) - 0.6, 'common fit', color=INK,
                 fontsize=10, ha='right', va='top')

    handles = ([plt.Line2D([], [], marker=mk, ls='', color='#9b9a96', ms=8,
                           label=g) for g, mk in GAS_MARK.items()]
               + [plt.Line2D([], [], marker='o', ls='', color=c, ms=8, label=s_)
                  for s_, c in (('K salts / seawater', '#2a78d6'),
                                ('NaCl', '#eb6834'), ('CaCl2', '#1baf7a'))])
    fig.legend(handles=handles, loc='lower center', ncol=9, fontsize=9,
               frameon=False, labelcolor=INK, columnspacing=1.0,
               handletextpad=0.4)
    fig.suptitle('Direct salt-effect measurements: absolute vs relative '
                 'collapse against ionic strength', fontsize=13, color=INK,
                 x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f'\n  wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
