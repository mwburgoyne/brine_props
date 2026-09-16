"""Per-gas freshwater V_phi PARITY panels: model vs every direct measurement
on disk (Mark's request 2026-07-30; parity form per his follow-up).

One panel per gas. x = measured V_phi, y = the DELIVERED route (vphi_route
'auto', freshwater) evaluated at each point's own (T, P), so pressure never
has to be collapsed away. The solid line is 1:1; dashed guides are +/-5%.
Filled points are the calibration set (direct volumetric only, from
fit_pr_vshift.calibration_set); open orange points are the held-out
indirect/finite-loading determinations, never fitted. Each panel carries the
in-sample MAE% and worst point.

C3H8 and NC4H10 anchors are stated in pr_vphi_model's VSHIFT comments (Moore
1982 70.7, Zhou & Battino 2001 75.0; Moore 1982 76.6) and are self-checked
against the documented 72.85 mean before plotting.

TWO renderings since 2026-08-08, one drawing routine so they cannot diverge:
  figures/vphi_fit_quality.png       all 8 gases, 2x4, with source legend -
                                     the manuscript's fig:vphi_parity
  figures/vphi_fit_quality_5gas.png  the five in-scope gases as a single
                                     strip for the event note
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
from fits.fit_pr_vshift import calibration_set, HELD_OUT
from brine_gas.vphi_route import V_phi
from brine_gas.iapws_if97 import p_sat_if97

FIGDIR = _FIGOUT

GASES_ALL = ('CH4', 'CO2', 'H2S', 'N2', 'H2', 'C2H6', 'C3H8', 'NC4H10')
GASES_SCOPE = ('CH4', 'CO2', 'H2S', 'N2', 'H2')

# direct 298.15 K anchors for the two gases absent from the calibration set
# (documented in pr_vphi_model VSHIFT comments; self-checked below)
EXTRA = {'C3H8': [(298.15, 0.1, 70.7, 'Moore 1982 densimetry'),
                  (298.15, 0.1, 75.0, 'Zhou 2001 dilatometry')],
         'NC4H10': [(298.15, 0.1, 76.6, 'Moore 1982 densimetry')]}
assert abs((70.7 + 75.0) / 2 - 72.85) < 1e-9   # the documented C3H8 mean

SURFACE, INK, INK2, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e8e8e6'
C_IN, C_OUT = '#2a78d6', '#eb6834'
MARK = {'Hnedkovsky 1996': 'o', 'Murphy & Gaines 1974': 's',
        'Moore 1982 densimetry': '^', 'Tiepel 1972 dilatometry': 'D',
        'Zhou 2001 dilatometry': 'v', 'Bignell 1984 (correlation)': 'P',
        'Barbero 1982': 'X', 'McBride-Wright 2015': '.'}
NICE = {'CH4': 'CH$_4$', 'CO2': 'CO$_2$', 'H2S': 'H$_2$S', 'N2': 'N$_2$',
        'H2': 'H$_2$', 'C2H6': 'C$_2$H$_6$', 'C3H8': 'C$_3$H$_8$',
        'NC4H10': '$n$C$_4$H$_{10}$'}


def model_at(gas, T, P):
    """Delivered freshwater route, with P clamped just above saturation."""
    return float(V_phi(gas, T, max(P, 1.10 * float(p_sat_if97(T)), 0.1)))


def draw_panel(ax, gas, cal, compact=False, report=None):
    """One parity panel. `compact` is the event-note strip styling."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)

    pts = cal.get(gas, [])
    held = HELD_OUT.get(gas, [])
    meas, mod, err = [], [], []
    for T, P, V, src in pts:
        m = model_at(gas, T, P)
        meas.append(V)
        mod.append(m)
        err.append(100 * (m - V) / V)
        ax.scatter(V, m, s=44 if compact else 54, marker=MARK.get(src, 'o'),
                   color=C_IN, edgecolors=SURFACE, linewidths=1.0, zorder=4)
    eh = []
    for T, P, V, src in held:
        m = model_at(gas, T, P)
        meas.append(V)
        mod.append(m)
        if 'finite loading' in src:
            # Lauder's two CO2 values are DIRECT determinations at finite
            # loading, not infinite-dilution checks: drawn with their own
            # symbol and left out of the held-out score (review 2026-09-05).
            ax.scatter(V, m, s=52 if compact else 62, marker='s',
                       facecolors='none', edgecolors=INK2, linewidths=1.4,
                       zorder=4)
            continue
        eh.append(abs(100 * (m - V) / V))
        ax.scatter(V, m, s=48 if compact else 58, marker='o',
                   facecolors='none', edgecolors=C_OUT, linewidths=1.8,
                   zorder=4)

    lo = min(meas + mod) * 0.94
    hi = max(meas + mod) * 1.06
    g = np.array([lo, hi])
    ax.plot(g, g, color=INK, lw=1.2, zorder=2)
    for f in (0.95, 1.05):
        ax.plot(g, g * f, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    if not compact:
        x1 = lo + 0.20 * (hi - lo)
        x2 = lo + 0.62 * (hi - lo)
        ax.text(x1, x1 * 1.052, '+5%', color=INK2, fontsize=8, rotation=45,
                ha='center', va='bottom', rotation_mode='anchor')
        ax.text(x2, x2 * 0.948, '-5%', color=INK2, fontsize=8, rotation=45,
                ha='center', va='top', rotation_mode='anchor')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')

    err = np.abs(err)
    if compact:
        ax.text(0.06, 0.945, NICE[gas], transform=ax.transAxes, fontsize=15,
                fontweight='bold', color=INK, va='top', zorder=5)
        if len(pts):
            ax.set_title(f'MAE {err.mean():.1f}%, max {err.max():.1f}%',
                         fontsize=9.5, color=INK2, loc='left', pad=6)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(colors=INK2, labelsize=8.5)
    else:
        stats = (f'MAE {err.mean():.1f}%, max {err.max():.1f}% (n={len(pts)})'
                 if len(pts) else 'no direct data')
        ax.set_title(f'{gas}   {stats}', fontsize=10.5, color=INK, loc='left')
        ax.tick_params(colors=INK2, labelsize=9)
    if eh:
        ax.text(0.06, 0.80 if compact else 0.90,
                f'held-out {np.mean(eh):.1f}% (n={len(eh)})',
                transform=ax.transAxes, fontsize=8.5 if compact else 9,
                color=C_OUT)
    if report is not None:
        report.append(f'{gas:<8} {len(pts):5d} '
                      f'{err.mean() if len(pts) else float("nan"):6.1f} '
                      f'{err.max() if len(pts) else float("nan"):6.1f} '
                      f'{len(eh):7d} '
                      f'{np.mean(eh) if eh else float("nan"):10.1f}')


def render_full(cal):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.6), facecolor=SURFACE)
    report = []
    for ax, gas in zip(axes.flat, GASES_ALL):
        draw_panel(ax, gas, cal, compact=False, report=report)
    for ax in axes[1]:
        ax.set_xlabel('reference V$_\\phi$ (cm$^3$/mol)', color=INK, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('model V$_\\phi$ at the same (T, P)', color=INK,
                      fontsize=10)

    handles = ([plt.Line2D([], [], marker=m, ls='', color='#9b9a96', ms=7,
                           label=s.replace(' densimetry', '').replace(
                               ' dilatometry', ''))
                for s, m in MARK.items()]
               + [plt.Line2D([], [], marker='o', ls='', mfc='none', mec=C_OUT,
                             ms=8, mew=1.8, label='held out (never fitted)')])
    fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=9,
               frameon=False, labelcolor=INK, columnspacing=1.0,
               handletextpad=0.4)
    fig.suptitle('Freshwater V$_\\phi$ parity, per gas: delivered route at '
                 "each point's own (T, P)\n"
                 'against every direct measurement, '
                 'held-out indirect determinations open',
                 fontsize=16, fontweight='bold', color=INK, x=0.5,
                 ha='center')
    fig.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=5.0)
    out = os.path.join(FIGDIR, 'vphi_fit_quality.png')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"{'gas':<8} {'n_cal':>5} {'MAE%':>6} {'max%':>6} {'n_held':>7} "
          f"{'held MAE%':>10}")
    print('\n'.join(report))
    print(f'wrote {out}')
    # PDF copy for the manuscript (Fig. vphi_parity)
    pdfdir = os.path.join(os.path.dirname(HERE), 'manuscript', 'figures')
    if os.path.isdir(pdfdir):
        pdf = os.path.join(pdfdir, 'vphi_fit_quality.pdf')
        fig.savefig(pdf, facecolor=SURFACE)
        print(f'wrote {pdf}')
    plt.close(fig)


def render_strip(cal):
    """The event-note strip: the five gases in scope, one row, on the page's
    own white so it does not print as a grey block."""
    fig, axes = plt.subplots(1, 5, figsize=(12.6, 3.15), facecolor='#ffffff')
    for ax, gas in zip(axes, GASES_SCOPE):
        draw_panel(ax, gas, cal, compact=True)
        ax.set_facecolor('#ffffff')
    axes[0].set_ylabel('model V$_\\phi$ (cm$^3$/mol)', color=INK, fontsize=10)
    fig.supxlabel('reference V$_\\phi$ (cm$^3$/mol), '
                  "each point at its own (T, P)", color=INK, fontsize=10,
                  y=0.015)
    fig.tight_layout(rect=(0, 0.055, 1, 1.0), w_pad=1.6)
    out = os.path.join(FIGDIR, 'vphi_fit_quality_5gas.png')
    fig.savefig(out, dpi=150, facecolor='#ffffff')
    print(f'wrote {out}')
    plt.close(fig)


def render_scope_grid(cal):
    """The manuscript parity figure since 2026-09-01 (Codex proposal, adopted):
    the five gases in scope on a 2x3 grid, legend in the sixth slot, so the
    principal validation graphic matches the declared scope. C3H8/NC4H10
    evidence stays in the text and in render_full."""
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.4), facecolor=SURFACE)
    for ax, gas in zip(axes.flat, GASES_SCOPE):
        draw_panel(ax, gas, cal, compact=False)
        ax.set_facecolor(SURFACE)
    axes[1, 2].axis('off')
    handles = ([plt.Line2D([], [], marker=m, ls='', color='#9b9a96', ms=7,
                           label=src.replace(' densimetry', '')
                                    .replace(' dilatometry', ''))
                for src, m in MARK.items()]
               + [plt.Line2D([], [], marker='o', ls='', mfc='none', mec=C_OUT,
                             ms=8, mew=1.8,
                             label='held out (never fitted, scored)'),
                  plt.Line2D([], [], marker='s', ls='', mfc='none', mec=INK2,
                             ms=8, mew=1.4,
                             label='Lauder 1959, finite loading (not scored)')])
    axes[1, 2].legend(handles=handles, loc='center', fontsize=9, frameon=False,
                      labelcolor=INK, handletextpad=0.5)
    fig.supxlabel('reference V$_\\phi$ (cm$^3$/mol), each point at its own (T, P)',
                  color=INK, fontsize=10, y=0.02)
    fig.supylabel('model V$_\\phi$ at the same (T, P)', color=INK, fontsize=10,
                  x=0.015)
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    pdfdir = os.path.join(FIGDIR, '..', 'manuscript', 'figures')
    if os.path.isdir(pdfdir):
        pdf = os.path.join(pdfdir, 'vphi_fit_quality_5gas.pdf')
        fig.savefig(pdf, facecolor=SURFACE)
        print(f'wrote {pdf}')
    out = os.path.join(FIGDIR, 'vphi_fit_quality_5gas_grid.png')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f'wrote {out}')
    plt.close(fig)


def main():
    cal = calibration_set()
    for g, pts in EXTRA.items():
        cal[g] = [(T, P, V, s) for T, P, V, s in pts]
    render_full(cal)
    render_strip(cal)
    render_scope_grid(cal)
    return 0


if __name__ == '__main__':
    sys.exit(main())
