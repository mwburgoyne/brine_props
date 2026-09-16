"""Cubic-in-pressure parameterisation of the Ezrokhi density coefficients.

WHAT THIS SETTLES (2026-08-14). The Ezrokhi form has no pressure term, so the
manuscript previously quoted the density sets at three fixed pressures
(15/30/45 MPa) and let the user pick the nearest. This script replaces the
choice with an equation: each of a0, a1, a2 is itself fitted as a cubic in
pressure,

    a_j(p) = c0 + c1*p + c2*p^2 + c3*p^3,      p in MPa, valid 5-100 MPa,

to per-pressure refits (the quadratic-in-T fit of the analytic A of
Eq. A.2, exactly as ezrokhi_fits.py does at 30 MPa) every 5 MPa.

SELF-CHECK, run every time: on an OFF-GRID lattice (T and p between the
fitting nodes) the cubic-parameterised A(T,p) must match the exact analytic
A to within 5% of the per-pressure refit FLOOR - the error the quadratic-in-T
form carries even with coefficients refitted exactly at the evaluation
pressure. Measured 2026-08-14: the parameterisation adds under 1% to that
floor for every component (worst absolute error in A: CO2 6.9e-4,
H2 1.4e-2 on coefficients of 0.10 and 4.8), which is below 0.01% of density
at any component's dissolved weight fraction. The script exits nonzero if
that degrades.

OUTPUTS
  manuscript/tab_ezrokhi_pressure.tex   the tabular block \\input by the
                                        manuscript (caption stays in
                                        manuscript.tex for the prose gate)
  figures/fig12_ezrokhi_pressure.png    + the manuscript PDF twin via _save
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

from fits.ezrokhi_fits import A_density, quad_fit, T_C, salt_B_fit, M_FIT
from brine_gas.brine_properties import rho_brine, salinity_from_molality
from brine_gas.water_properties import rho_w
from figures.make_figures import (FIGDIR, GAS_COLOUR, GAS_LABEL, INK, MUTED, NAVY,
                          _save, plt)

GASES = ['CO2', 'CH4', 'H2S', 'N2', 'H2']
COMPS = GASES + ['NaCl']
P_GRID = np.arange(5.0, 101.0, 5.0)
M_SALT_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
COMP_COLOUR = dict(GAS_COLOUR, NaCl='#4b5563')
COMP_LABEL = dict(GAS_LABEL, NaCl='NaCl')


def salt_A_of_T(P):
    """A_NaCl(T) at pressure P: slope of log10(rho_b/rho_w) vs S (Spivey)."""
    out = []
    for t in T_C:
        T = t + 273.15
        num, den = [], []
        for m in M_SALT_GRID:
            S = salinity_from_molality(m)
            num.append(np.log10((rho_brine(T, P, S) / 1000.0)
                                / (rho_w(T, P) / 1000.0)))
            den.append(S)
        out.append(np.polyfit(den, num, 1)[0])
    return np.array(out)


def A_of_T(comp, P):
    if comp == 'NaCl':
        return salt_A_of_T(P)
    return np.array([A_density(comp, t, P, 0.0) for t in T_C])


def sweep():
    """(comp -> array [len(P_GRID), 3]) of per-pressure quad-in-T refits.
    'NaCl_B' is the salt VISCOSITY set (ezrokhi_fits.salt_B_fit, the one
    through-origin fit over 0.5-5 mol/kg) at each pressure: the only B that
    depends on pressure, through Kestin's factor (Mark, 2026-09-08: give it
    in the same cubic form as the density sets)."""
    out = {c: np.array([quad_fit(T_C, A_of_T(c, P))[:3] for P in P_GRID])
           for c in COMPS}
    out['NaCl_B'] = np.array([quad_fit(T_C, salt_B_fit(P))[:3] for P in P_GRID])
    return out


def cubic_fits(data):
    """(comp, a_j) -> ascending cubic constants (c0, c1, c2, c3) in p/MPa;
    ('NaCl_B', b_j) for the salt viscosity set."""
    out = {}
    for c in COMPS:
        for j, name in enumerate(('a0', 'a1', 'a2')):
            out[(c, name)] = np.polyfit(P_GRID, data[c][:, j], 3)[::-1]
    for j, name in enumerate(('b0', 'b1', 'b2')):
        out[('NaCl_B', name)] = np.polyfit(P_GRID, data['NaCl_B'][:, j], 3)[::-1]
    return out


def salt_B_end_to_end(cubics):
    """Off-grid score of the parameterised B_NaCl(T,p) vs the exact fit,
    against the per-pressure refit floor; viscosity consequence at 5 mol/kg.
    Returns (max|dB| param, max|dB| floor, ratio, viscosity %)."""
    T_eval = np.arange(22.5, 150.0, 5.0)
    P_eval = np.arange(7.5, 100.0, 5.0)
    w5 = salinity_from_molality(5.0)
    d_par, d_flr = [], []
    for P in P_eval:
        exact = np.interp(T_eval, T_C, salt_B_fit(P))
        b0, b1, b2, _ = quad_fit(T_C, salt_B_fit(P))
        d_flr.append(np.abs(b0 + b1 * T_eval + b2 * T_eval**2 - exact))
        c0, c1, c2 = [float(np.polyval(cubics[('NaCl_B', n)][::-1], P)) for n in ('b0', 'b1', 'b2')]
        d_par.append(np.abs(c0 + c1 * T_eval + c2 * T_eval**2 - exact))
    mp, mf = np.max(d_par), np.max(d_flr)
    return mp, mf, mp / mf, 100 * np.log(10) * mp * w5


def a_param(cubics, comp, P):
    return [float(np.polyval(cubics[(comp, n)][::-1], P))
            for n in ('a0', 'a1', 'a2')]


def end_to_end(cubics):
    """Off-grid score of parameterised A vs exact, against the refit floor."""
    T_eval = np.arange(22.5, 150.0, 5.0)
    P_eval = np.arange(7.5, 100.0, 5.0)
    # Largest dissolved weight fraction each component reaches inside the
    # envelope: gases at their ~30 MPa solubility ceilings (manuscript
    # Discussion), NaCl at 5 mol/kg. Translates dA into delivered density:
    # drho/rho = ln(10) * dA * w.
    from brine_gas.plyasunov_model import gas_mw
    W_MAX = {'NaCl': salinity_from_molality(5.0)}
    for g, x in (('CO2', 0.03), ('H2S', 0.05), ('CH4', 0.004),
                 ('N2', 0.004), ('H2', 0.004)):
        W_MAX[g] = x * gas_mw(g) / ((1 - x) * 18.015268 + x * gas_mw(g))
    rows, ok = [], True
    for comp in COMPS:
        d_par, d_flr = [], []
        for P in P_eval:
            a_t = A_of_T(comp, P)
            if comp == 'NaCl':
                exact = np.interp(T_eval, T_C, a_t)
            else:
                exact = np.array([A_density(comp, t, P, 0.0) for t in T_eval])
            a0, a1, a2, _ = quad_fit(T_C, a_t)
            d_flr.append(np.abs(a0 + a1 * T_eval + a2 * T_eval**2 - exact))
            c0, c1, c2 = a_param(cubics, comp, P)
            d_par.append(np.abs(c0 + c1 * T_eval + c2 * T_eval**2 - exact))
        mp, mf = np.max(d_par), np.max(d_flr)
        drho = 100 * np.log(10) * mp * W_MAX[comp]
        rows.append((comp, mp, mf, mp / mf, drho))
        ok &= mp <= 1.05 * mf and drho < 0.05
    return rows, ok


def salt_form_quality(cubics):
    """How well NaCl follows the two polynomial forms, in property terms at
    5 mol/kg (the worst case; every error scales with w): worst quad-in-T
    residual of A (density %) and of B (viscosity %) over P_GRID, and the
    cubic-in-p additions (density %, viscosity %). Mark, 2026-09-08."""
    w5 = salinity_from_molality(5.0)
    ea = max(quad_fit(T_C, A_of_T('NaCl', P))[3] for P in P_GRID)
    eb = max(quad_fit(T_C, salt_B_fit(P))[3] for P in P_GRID)
    rows, _ = end_to_end(cubics)
    drho_cubic = [r[4] for r in rows if r[0] == 'NaCl'][0]
    dmu_cubic = salt_B_end_to_end(cubics)[3]
    return dict(A_quad=100 * np.log(10) * ea * w5, A_cubic=drho_cubic,
                B_quad=100 * np.log(10) * eb * w5, B_cubic=dmu_cubic)


def emit_table(cubics, path):
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Component & & $c_0$ & $c_1$ & $c_2$ & $c_3$ \\', r'\midrule']

    def sci(v):
        m, e = f'{v:+.4e}'.split('e')
        return f'${m} \\times 10^{{{int(e)}}}$'

    for comp in COMPS:
        label = COMP_LABEL[comp]
        for j, name in enumerate(('a0', 'a1', 'a2')):
            c = cubics[(comp, name)]
            head = label if j == 0 else ''
            lines.append(f'{head} & $a_{j}$ & ' +
                         ' & '.join(sci(v) for v in c) + r' \\')
        if comp != COMPS[-1]:
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(path, 'w') as f:
        f.write('% GENERATED by code/ezrokhi_pressure_fit.py - do not edit\n')
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)
    # the salt VISCOSITY set, the one B with a pressure dependence, as its own
    # table (Mark, 2026-09-08) in the same directory
    nacl = [r'\begin{tabular}{lrrrr}', r'\toprule',
            r' & $c_0$ & $c_1$ & $c_2$ & $c_3$ \\', r'\midrule']
    for j, name in enumerate(('b0', 'b1', 'b2')):
        c = cubics[('NaCl_B', name)]
        nacl.append(f'$b_{j}$ & ' + ' & '.join(sci(v) for v in c) + r' \\')
    nacl += [r'\bottomrule', r'\end{tabular}']
    path_b = os.path.join(os.path.dirname(path), 'tab_ezrokhi_nacl_b.tex')
    with open(path_b, 'w') as f:
        f.write('% GENERATED by code/ezrokhi_pressure_fit.py - do not edit\n')
        f.write('\n'.join(nacl) + '\n')
    print('wrote', path_b)


def draw_figure(data, cubics):
    fig, axes = plt.subplots(len(COMPS), 3, figsize=(7.4, 9.6), sharex=True)
    pd = np.linspace(P_GRID[0], P_GRID[-1], 200)
    for i, comp in enumerate(COMPS):
        col = COMP_COLOUR[comp]
        for j, name in enumerate(('a0', 'a1', 'a2')):
            ax = axes[i, j]
            ax.plot(pd, [np.polyval(cubics[(comp, name)][::-1], p)
                         for p in pd], color=col, lw=1.6, zorder=2)
            ax.plot(P_GRID, data[comp][:, j], 'o', mfc='white', mec=INK,
                    mew=0.8, ms=3.2, zorder=3)
            ax.tick_params(labelsize=7)
            ax.yaxis.get_offset_text().set_fontsize(6)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
            if i == 0:
                ax.set_title(f'$a_{j}$', fontsize=10)
            if j == 0:
                ax.set_ylabel(COMP_LABEL[comp], fontsize=9, rotation=0,
                              ha='right', va='center', labelpad=16)
            if i == len(COMPS) - 1:
                ax.set_xlabel('$p$, MPa', fontsize=8, color=MUTED)
    fig.align_ylabels()
    fig.tight_layout()
    _save(fig, 'fig12_ezrokhi_pressure')


def main():
    data = sweep()
    cubics = cubic_fits(data)

    print('CUBIC-IN-PRESSURE CONSTANTS  a_j(p) = c0 + c1 p + c2 p^2 + c3 p^3')
    print(f"{'comp':<5} {'coef':<3} {'c0':>13} {'c1':>13} {'c2':>13} {'c3':>13}")
    for comp in COMPS:
        for name in ('a0', 'a1', 'a2'):
            c = cubics[(comp, name)]
            print(f'{comp:<5} {name:<3} ' + ' '.join(f'{v:>13.4e}' for v in c))
    for name in ('b0', 'b1', 'b2'):
        c = cubics[('NaCl_B', name)]
        print(f'{"NaClB":<5} {name:<3} ' + ' '.join(f'{v:>13.4e}' for v in c))
    q = salt_form_quality(cubics)
    print(f"NaCl form quality at 5 mol/kg: A quad-in-T {q['A_quad']:.3f}% density, cubic-in-p {q['A_cubic']:.3f}%; "
          f"B quad-in-T {q['B_quad']:.2f}% viscosity, cubic-in-p {q['B_cubic']:.2f}%")
    assert q['A_quad'] < 0.06 and q['B_quad'] < 0.25, q
    mp, mf, r, dmu = salt_B_end_to_end(cubics)
    print(f'NaCl viscosity set, off-grid: max|dB| param {mp:.2e}, floor {mf:.2e}, ratio {r:.3f}, '
          f'viscosity {dmu:.3f}% at 5 mol/kg')
    # B_NaCl(p) has a kink at 35 MPa (Kestin's factor is held at its boundary
    # value above it), which a cubic smooths: the floor ratio is therefore not
    # the test here; the viscosity consequence is (pinned 0.33% at 5 mol/kg,
    # against the 4% linear-in-w cost and the 1.8% Kestin error)
    if not (dmu < 0.5):
        print('SELF-CHECK FAILED on the NaCl viscosity cubics')
        return 1
    assert abs(dmu - 0.33) < 0.03, dmu

    print('\nEND-TO-END, OFF-GRID: parameterised A(T,p) vs exact Eq. A.2')
    print(f"{'comp':<5} {'max|dA| param':>14} {'max|dA| floor':>14} {'ratio':>7} "
          f"{'density%':>9}")
    rows, ok = end_to_end(cubics)
    for comp, mp, mf, r, drho in rows:
        print(f'{comp:<5} {mp:>14.2e} {mf:>14.2e} {r:>7.3f} {drho:>8.3f}%')
    print('floor = per-pressure quad-in-T refit, the form\'s irreducible error;')
    print('density% = the full residual at the component\'s largest dissolved')
    print('weight fraction inside the envelope (gases at their ~30 MPa')
    print('solubility ceilings, NaCl at 5 mol/kg).')
    if not ok:
        print('SELF-CHECK FAILED: >1.05x floor, or >0.05% delivered density')
        return 1
    print('self-check passed: within 5% of the floor and 0.05% of density.')

    emit_table(cubics, os.path.join(_TABLES, 'tab_ezrokhi_pressure.tex'))
    draw_figure(data, cubics)
    return 0


if __name__ == '__main__':
    sys.exit(main())
