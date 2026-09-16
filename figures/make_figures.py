"""Figure generation for the Whitson event note and the whitepaper.

Writes PNGs to ../figures/. All curves are computed live from the implemented
chain (pyResToolbox SoreideWhitson + Plyasunov/Garcia) so figures cannot drift
from the code; experimental points are transcribed from the source papers.

Palette: Okabe-Ito derived, validated for colour-vision deficiency (worst
adjacent pair deltaE 11.0 deutan, all slots >= 3:1 against the surface).
Colour follows the gas, never the plot order. Whitson navy is reserved for
ink (titles, annotations, axes), not for data.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from pyrestoolbox import brine as rtb_brine
from brine_gas.plyasunov_model import gas_mw
# Figures MUST use the shipped default route, not a hard-wired one. Before
# 2026-07-25 this imported V_phi from plyasunov_model directly, so when the
# default moved to PR+VSHIFT every figure kept showing the old route while the
# surrounding text described the new one.
from brine_gas.vphi_route import V_phi as _V_phi_default


def V_phi(gas, T, P, route='auto', m_nacl=0.0):
    return _V_phi_default(gas, T, P, route, m_nacl)
from brine_gas.brine_properties import rho_brine, salinity_from_molality, M_NACL
from brine_gas.water_properties import rho_w

FIGDIR = _FIGOUT
os.makedirs(FIGDIR, exist_ok=True)

# --- design tokens ----------------------------------------------------------
NAVY = '#0A3161'        # Whitson brand ink
INK = '#1a1a1a'
MUTED = '#6b7280'
GRID = '#e5e7eb'
SURFACE = '#ffffff'

GAS_COLOUR = {          # fixed by entity, never by series order
    'CO2': '#D55E00',
    'CH4': '#0072B2',
    'H2S': '#009E73',
    'N2':  '#8C4799',
    'H2':  '#B07A00',
}
GAS_LABEL = {'CO2': 'CO$_2$', 'CH4': 'CH$_4$', 'H2S': 'H$_2$S',
             'N2': 'N$_2$', 'H2': 'H$_2$'}

plt.rcParams.update({
    'figure.dpi': 200, 'savefig.dpi': 200,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.titlecolor': NAVY,
    'axes.linewidth': 0.8, 'axes.grid': True, 'axes.axisbelow': True,
    'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'legend.frameon': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
})


def _save(fig, name):
    """Write PNG for the Word/event-note pipeline and PDF for LaTeX."""
    png = os.path.join(FIGDIR, name + '.png')
    fig.savefig(png, bbox_inches='tight')
    pdfdir = os.path.join(FIGDIR, '..', 'manuscript', 'figures')
    if os.path.isdir(pdfdir):
        fig.savefig(os.path.join(pdfdir, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('wrote', png)


def _style(ax, title=None, xlabel=None, ylabel=None):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc='left', fontsize=10, fontweight='bold', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _sw(degc, bar, ppm, **kw):
    # framework pinned to 'default', the refresh paper's published
    # recommendation: explicit so the figures use the dissolved amounts the
    # manuscript declares even if the library default moves.
    kw.setdefault('framework', 'default')
    return rtb_brine.SoreideWhitson(pres=bar, temp=degc, ppm=ppm, metric=True, **kw)


# ===========================================================================
# FIG 1  Who densifies, who lightens, and how it fades with temperature
# ===========================================================================
def fig_density_signature():
    bar, m = 300.0, 1.0
    ppm = m * M_NACL / (1000 + m * M_NACL) * 1e6
    temps = np.arange(30, 181, 10.0)

    cases = {'CO2': dict(y_CO2=1.0), 'H2S': dict(y_H2S=1.0),
             'CH4': dict(y_CO2=0.0, sg=0.554), 'N2': dict(y_N2=1.0),
             'H2': dict(y_H2=1.0)}

    # Sized for the note's 3.9-inch side-by-side column (72% display scale),
    # which also suits the manuscript at \textwidth; fonts are set to survive
    # that reduction, and the two half-plane cues carry the sign story.
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    curves = {}
    for gas, kw in cases.items():
        eff = []
        for t in temps:
            try:
                s = _sw(t, bar, ppm, **kw)
                rg, rf = float(s.bDen[0]), float(s.bDen[1])
                eff.append(100.0 * (rg / rf - 1.0))
            except Exception:
                eff.append(np.nan)
        curves[gas] = np.array(eff)
        ax.plot(temps, curves[gas], color=GAS_COLOUR[gas], lw=2.4,
                solid_capstyle='round')

    ax.set_xlim(25, 196)
    ax.set_ylim(-1.95, 1.25)
    ax.axhline(0, color=NAVY, lw=1.2, zorder=1)
    ax.axhspan(0, 1.25, color=GAS_COLOUR['CO2'], alpha=0.045, zorder=0)
    ax.axhspan(-1.95, 0, color=GAS_COLOUR['CH4'], alpha=0.035, zorder=0)
    ax.text(193, 1.16, 'DENSER than gas-free brine', ha='right', va='top',
            fontsize=8.5, color=MUTED)
    ax.text(28, -1.86, 'LIGHTER', ha='left', va='bottom',
            fontsize=8.5, color=MUTED)
    for gas, y in curves.items():                      # direct labels, not a legend box
        i = np.isfinite(y).nonzero()[0][-1]
        ax.annotate(GAS_LABEL[gas], (temps[i], y[i]), xytext=(5, 0),
                    textcoords='offset points', color=GAS_COLOUR[gas],
                    fontsize=10.5, fontweight='bold', va='center')
    ax.annotate('gas-free brine', (temps[0], 0), xytext=(2, 4),
                textcoords='offset points', color=NAVY, fontsize=8.5)

    _style(ax, 'Density change of saturated brine',
           'Temperature (°C)', 'Change vs gas-free brine (%)')
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(10.5)
    ax.yaxis.label.set_fontsize(10.5)
    ax.tick_params(labelsize=9.5)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    fig.tight_layout()
    _save(fig, 'fig1_density_signature')
    return curves


# ===========================================================================
# FIG 2  Where the common approach goes wrong
# ===========================================================================
def fig_viscosity_contrast():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.7))

    # (a) CO2: Islam-Carlson (T-independent, in common use) vs the Calabrese
    # correlation (e1, e2 inherited from McBride-Wright's CO2-water data)
    def cal_pct(T_K, x=0.02):
        return 100.0 * (np.exp(65.560 * np.exp(-2.468 * (T_K / 142.0 - 1.0)) * x) - 1.0)
    x = 0.02
    degc = np.linspace(15.0, 160.0, 120)
    T_K = degc + 273.15
    cal = cal_pct(T_K, x)
    ic_val = 100.0 * (4.65 * x ** 1.0134)
    ic = ic_val * np.ones_like(degc)

    ax1.plot(degc, ic, color=MUTED, lw=2.0, ls='--')
    ax1.plot(degc, cal, color=GAS_COLOUR['CO2'], lw=2.2, solid_capstyle='round')
    ax1.set_xlim(10, 175)
    ax1.set_ylim(0, 12.6)
    ax1.annotate('Islam-Carlson\n(temperature-independent)', (35, ic_val),
                 xytext=(0, 10), textcoords='offset points', color=MUTED,
                 fontsize=8, ha='left')
    ax1.annotate('Calabrese 2019 correlation\n(CO$_2$ slope from McBride-Wright\nCO$_2$-water data)', (85, 6.6),
                 color=GAS_COLOUR['CO2'], fontsize=8, ha='left', fontweight='bold')
    # Ratio OF THE PLOTTED CURVES at x = 0.02, evaluated at the EXACT stated
    # temperatures (not the nearest grid point, which used to print 4.1 for
    # 378 K). 378 K is the ceiling of Islam-Carlson's own stated scope.
    for T_ann in (378.0, 423.0):
        tc = T_ann - 273.15
        c = cal_pct(T_ann, x)
        ax1.plot([tc, tc], [c, ic_val], color=NAVY, lw=0.9)
        ax1.annotate(f'{ic_val / c:.1f}× at {T_ann:.0f} K',
                     (tc, 0.5 * (c + ic_val)), xytext=(5, 0),
                     textcoords='offset points', color=NAVY, fontsize=8,
                     fontweight='bold', va='center')
    _style(ax1, 'CO$_2$: overstated at reservoir temperature',
           'Temperature (°C)', 'Viscosity increase at $x$ = 0.02 (%)')

    _density_scaling_panel(ax2)

    fig.tight_layout()
    _save(fig, 'fig2_viscosity_contrast')


# The sign test, GENERATED. One measured state per gas, in fresh water:
#   CO2  Calabrese 2019 correlation (a correlation evaluation, not one measured
#        point; its CO2 slope is McBride-Wright's) at 323.15 K, 20 MPa, x = 0.02
#   CH4  Ostermann 1985, 100 degF (311 K), 3000 psia, from the tabulated row
#   H2S  Murphy & Gaines 1974 Table IV, 30.0 degC, 21.5 atm (the one row whose
#        Burgess & Germann mole fraction needs no extrapolation)
# The density-scaled prediction carries the shipped chain's fractional density
# change at that state into a viscosity change of the same sign, with the
# constant set so CO2's scaled bar equals its measured bar. CO2 therefore
# tests nothing; CH4 and H2S carry the result. Both series are effects in %
# at the same state, so the bars share one axis. N2 is not shown: its only
# null sits at x ~ 1e-4, where both series are under 0.05%.
# Before 2026-09-02 the scaled bars were hand-typed per-x coefficients on the
# same axis as measured effects (disclosed in the caption, but a mixed axis).
def sign_test_cases():
    from brine_gas.garcia_mixing import density_change_pct, viscosity_correction_single
    from fits.ostermann_ch4_refit import OSTERMANN, x_ch4_from_rsw
    from fits.murphy_gaines_h2s_refit import TABLE_IV, x_h2s_burgess_germann
    T = 323.15
    co2 = ('CO2', T, 20.0, 0.02,
           100.0 * (viscosity_correction_single('CO2', 0.02, (T - 273.15) * 1.8 + 32) - 1.0))
    psia, rsw, ratio = OSTERMANN[100][3]
    assert psia == 3000
    ch4 = ('CH4', (100.0 - 32.0) / 1.8 + 273.15, psia * 0.006894757,
           x_ch4_from_rsw(rsw), 100.0 * (ratio - 1.0))
    tc, atm, ratio = TABLE_IV[3]
    assert tc == 30.0
    h2s = ('H2S', tc + 273.15, atm * 0.101325, x_h2s_burgess_germann(tc, atm),
           100.0 * (ratio - 1.0))
    rows = []
    for gas, T, P, x, meas in (co2, ch4, h2s):
        drho = density_change_pct(gas, x, T, P, S=0.0)[2]
        rows.append(dict(gas=gas, T=T, P=P, x=x, measured=meas, drho=drho))
    k = rows[0]['measured'] / rows[0]['drho']
    for r in rows:
        r['scaled'] = k * r['drho']
    return rows


_SCALED_COLOUR = MUTED
_MEASURED_COLOUR = GAS_COLOUR['CO2']   # one colour per SERIES; the axis names the gas


def _density_scaling_panel(ax, title='Viscosity cannot be scaled from density'):
    rows = sign_test_cases()
    xpos = np.arange(len(rows))
    w = 0.36
    scaled = [r['scaled'] for r in rows]
    meas = [r['measured'] for r in rows]
    ax.bar(xpos - w / 2 - 0.01, scaled, w, color=_SCALED_COLOUR,
           label='Scaled from the density change')
    ax.bar(xpos + w / 2 + 0.01, meas, w, color=_MEASURED_COLOUR,
           label='Measured (CO$_2$: correlation)')
    ax.axhline(0, color=NAVY, lw=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{GAS_LABEL[r['gas']]}\n{r['T'] - 273.15:.0f} °C, $x$ = {r['x']:.3f}"
                        for r in rows], fontsize=8)
    lo = min(min(scaled), 0.0)
    hi = max(meas)
    ax.set_ylim(lo - 0.15 * (hi - lo), hi * 1.45)

    # The CO2 pair is equal by construction: say so on the figure, not only
    # in the caption.
    ax.annotate('constant set here', (0, meas[0]), xytext=(0, 5),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=7.5, color=NAVY)
    # The sign disagreements sit BELOW the scaled bar, where there is room.
    for i, r in enumerate(rows):
        if r['scaled'] < 0 < r['measured']:
            ax.annotate('sign wrong', (i - w / 2 - 0.01, r['scaled']),
                        xytext=(0, -4), textcoords='offset points',
                        ha='center', va='top', fontsize=7.5, color=NAVY,
                        fontweight='bold')

    _style(ax, title, None, 'Change in viscosity (%)')
    ax.legend(fontsize=8, loc='upper right', ncol=1)


# ===========================================================================
# FIG 9  The sign test on its own - the lean single panel for the event note,
# which does not carry the Islam-Carlson benchmark.
# ===========================================================================
def fig_density_scaling_only():
    fig, ax = plt.subplots(figsize=(4.9, 3.6))
    _density_scaling_panel(ax, 'Viscosity cannot be scaled from density')
    fig.tight_layout()
    _save(fig, 'fig9_density_scaling')


# ===========================================================================
# FIG 3  Blind test against measured densities (Yan 2011)
# ===========================================================================
def _yan_data():
    src = open(os.path.join(_ROOT, 'validation', 'yan2011_validation.py')).read()
    return eval('[' + src.split('DATA = [')[1].split(']\n')[0] + ']')


def fig_validation_parity():
    M2 = gas_mw('CO2')
    pts = {0: [], 1: [], 5: []}
    for T, P, m, mco2, x, rho_meas in _yan_data():
        S = salinity_from_molality(m) if m > 0 else 0.0
        rb = (rho_brine(T, P, S) if m > 0 else rho_w(T, P)) / 1000.0
        Wb = 1000.0 + m * M_NACL
        V = V_phi('CO2', T, P, 'auto', m)
        rho_mod = (Wb + mco2 * M2) / (Wb / rb + mco2 * V)
        pts[m].append((100 * (rho_meas / rb - 1), 100 * (rho_mod / rb - 1)))

    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    lim = 1.45
    ax.plot([0, lim], [0, lim], color=NAVY, lw=1.0, zorder=1)
    ax.annotate('1:1', (lim * 0.93, lim * 0.93), xytext=(6, -8),
                textcoords='offset points', color=NAVY, fontsize=8)
    marks = {0: ('o', '0 mol/kg'), 1: ('s', '1 mol/kg'), 5: ('^', '5 mol/kg')}
    shades = {0: '#0072B2', 1: '#D55E00', 5: '#009E73'}
    for m, v in pts.items():
        a = np.array(v)
        ax.scatter(a[:, 0], a[:, 1], s=34, marker=marks[m][0],
                   facecolor=shades[m], edgecolor=SURFACE, linewidth=0.8,
                   label=f'NaCl {marks[m][1]}', zorder=3)
    _style(ax, 'Densification at measured dissolved amounts',
           'Measured increase over gas-free brine (%)',
           'Predicted increase (%)')
    ax.set_xlim(-0.16, lim); ax.set_ylim(-0.16, lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    _save(fig, 'fig3_validation_parity')


# ===========================================================================
# FIG 4  (whitepaper) V_phi against the densimetric data it is judged on
# ===========================================================================
def fig_vphi_vs_data():
    hned = {
        'CH4': [(298.15, 36.75), (323.15, 37.30), (373.15, 40.50),
                (423.15, 45.90), (473.15, 54.10), (523.15, 64.80)],
        'CO2': [(298.15, 33.45), (323.15, 33.75), (373.15, 37.50),
                (423.15, 42.30), (473.15, 49.20), (523.15, 59.75)],
        'H2S': [(298.15, 34.90), (323.15, 35.90), (373.15, 38.70),
                (423.15, 42.75), (473.15, 48.45), (523.15, 56.75)],
    }
    # O'Sullivan & Smith's 324.65 K value only: their higher-temperature table
    # is disqualified in the manuscript (Section 6.1: extrapolation to infinite
    # dilution too uncertain by the authors' own account, fitted-surface
    # columns), and only the 324.65 K point is in HELD_OUT. Dropped 2026-09-06.
    osull = {'N2': [(324.65, 34.05)]}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    T_CAL_END, T_CLAIM = 473.15, 450.0
    T = np.linspace(280, 530, 200)
    for gas in ('CH4', 'CO2', 'H2S', 'N2', 'H2'):
        v = np.array([V_phi(gas, t, 30.0) for t in T])
        # solid inside the calibration data (to 473 K), dashed extrapolation beyond
        ax.plot(T[T <= T_CAL_END] - 273.15, v[T <= T_CAL_END],
                color=GAS_COLOUR[gas], lw=2.0, zorder=2)
        ax.plot(T[T >= T_CAL_END] - 273.15, v[T >= T_CAL_END],
                color=GAS_COLOUR[gas], lw=2.0, ls=(0, (3, 2)), zorder=2)
    for gas, d in hned.items():
        a = np.array(d)
        ax.scatter(a[:, 0] - 273.15, a[:, 1], s=36, marker='o',
                   facecolor=SURFACE, edgecolor=GAS_COLOUR[gas], linewidth=1.4, zorder=3)
    for gas, d in osull.items():
        a = np.array(d)
        ax.scatter(a[:, 0] - 273.15, a[:, 1], s=40, marker='D',
                   facecolor=SURFACE, edgecolor=GAS_COLOUR[gas], linewidth=1.4, zorder=3)
    for t_line, lab in ((T_CLAIM, 'accuracy claimed to here'),
                        (T_CAL_END, 'calibration data end')):
        ax.axvline(t_line - 273.15, color=MUTED, lw=0.8, ls=':', zorder=1)
        ax.annotate(lab, (t_line - 273.15, 22.5), xytext=(3 if t_line > 460 else -3, 0),
                    textcoords='offset points', rotation=90, fontsize=7,
                    color=MUTED, ha='left' if t_line > 460 else 'right', va='bottom')
    # endpoint values pair up (CH4/N2 and H2S/CO2 land within ~2 cm3/mol),
    # so the labels take small fixed vertical offsets instead of overlapping
    nudge = {'CH4': 5, 'N2': -5, 'H2S': 4, 'CO2': -4, 'H2': 0}
    for gas in ('CH4', 'CO2', 'H2S', 'N2', 'H2'):
        y = V_phi(gas, 528.0, 30.0)
        ax.annotate(GAS_LABEL[gas], (528.0 - 273.15, y),
                    xytext=(6, nudge[gas]), textcoords='offset points',
                    color=GAS_COLOUR[gas], fontsize=9, fontweight='bold',
                    va='center')
    ax.scatter([], [], s=36, marker='o', facecolor=SURFACE, edgecolor=MUTED,
               linewidth=1.4, label='Hnedkovsky 1996 (vibrating tube, 30 MPa)')
    ax.scatter([], [], s=40, marker='D', facecolor=SURFACE, edgecolor=MUTED,
               linewidth=1.4, label="O'Sullivan & Smith 1970 (solubility-derived)")
    ax.legend(fontsize=8, loc='upper left')
    _style(ax, 'Apparent molar volume against direct and solubility-derived data',
           'Temperature (°C)', '$V_\\phi$ at 30 MPa (cm³/mol)')
    ax.set_xlim(0, 275)
    fig.tight_layout()
    _save(fig, 'fig4_vphi_vs_data')


# ===========================================================================
# FIG 5  (whitepaper) CH4 viscosity: step, ramp and the extrapolation clamp
# ===========================================================================
def fig_ch4_ramp():
    """CH4 viscosity: the unified form against all 23 Ostermann measurements."""
    from brine_gas.garcia_mixing import _ch4_viscosity_ratio, _CH4_A, _CH4_B
    from fits.ostermann_ch4_refit import OSTERMANN, x_ch4_from_rsw
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6))

    shades = {100: '#0072B2', 150: '#D55E00', 250: '#009E73'}
    xg = np.linspace(1e-5, 0.0055, 300)
    for degf, rows in OSTERMANN.items():
        c = shades[degf]
        ax1.plot(xg * 1000, 100 * (_ch4_viscosity_ratio(xg, degf) - 1),
                 color=c, lw=2.0, zorder=2)
        xs = np.array([x_ch4_from_rsw(r) for _, r, _ in rows])
        ys = np.array([100 * (v - 1) for _, _, v in rows])
        ax1.scatter(xs * 1000, ys, s=32, marker='o', facecolor=SURFACE,
                    edgecolor=c, linewidth=1.4, zorder=3)
        ax1.annotate(f'{degf} °F', (0.0053 * 1000,
                     100 * (_ch4_viscosity_ratio(0.0053, degf) - 1)),
                     xytext=(5, 0), textcoords='offset points', color=c,
                     fontsize=8, fontweight='bold', va='center')
    ax1.scatter([], [], s=32, marker='o', facecolor=SURFACE, edgecolor=MUTED,
                linewidth=1.4, label='Ostermann SPE 14211 (23 points)')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.set_xlim(0, 6.4)
    _style(ax1, 'Levels off with dissolved concentration',
           'Dissolved CH$_4$, $x$ × 1000', 'Viscosity increase (%)')

    t = np.linspace(60, 450, 200)
    raw = 1.109 - 5.98e-4 * t + 1.0933e-6 * t ** 2
    uni = np.array([_ch4_viscosity_ratio(0.003, v) for v in t])
    asym = np.array([1 + _CH4_A * np.exp(_CH4_B / ((v - 32) / 1.8 + 273.15))
                     for v in t])
    ax2.plot(t, 100 * (raw - 1), color=MUTED, lw=2.0, ls='--',
             label="Ostermann's relationship")
    ax2.plot(t, 100 * (asym - 1), color=NAVY, lw=1.2, ls=':',
             label='Levelled-off ceiling, $1 + A\\,e^{B/T}$')
    ax2.plot(t, 100 * (uni - 1), color=GAS_COLOUR['CH4'], lw=2.2,
             label='Unified form at $x$ = 0.003')
    ax2.axvspan(100, 250, color=NAVY, alpha=0.06)
    ax2.annotate('measured range', (175, 0.45), color=NAVY, fontsize=8, ha='center')
    ax2.set_ylim(0, 9.2)
    ax2.legend(fontsize=7.5, loc='upper right')
    _style(ax2, 'Temperature dependence and extrapolation',
           'Temperature (°F)', 'Viscosity increase (%)')

    fig.tight_layout()
    _save(fig, 'fig5_ch4_ramp')


# ===========================================================================
# FIG 6  (whitepaper) The solvent-basis correction
# ===========================================================================
def fig_solvent_basis():
    from brine_gas.plyasunov_model import V_phi as VP
    T, P, x2 = 323.15, 20.0, 0.015
    ms = np.linspace(0, 5.5, 60)
    M2, V = gas_mw('CO2'), VP('CO2', T, P)
    nw = 1000.0 / 18.015268
    new, old = [], []
    for m in ms:
        S = salinity_from_molality(m)
        rb = rho_brine(T, P, S) / 1000.0
        n2 = x2 * nw / (1 - x2)
        Wb = 1000.0 + m * M_NACL
        new.append(100 * ((Wb + n2 * M2) / (Wb / rb + n2 * V) / rb - 1))
        W = (1 - x2) * 18.015268
        old.append(100 * ((W + x2 * M2) / (x2 * V + W / rb) / rb - 1))
    new, old = np.array(new), np.array(old)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ax1.plot(ms, old, color=MUTED, lw=2.0, ls='--', label='Water mass only')
    ax1.plot(ms, new, color=GAS_COLOUR['CO2'], lw=2.2, label='Full gas-free brine mass')
    ax1.legend(fontsize=8, loc='upper right')
    _style(ax1, 'CO$_2$ densification vs salinity', 'NaCl (mol/kg)',
           'Increase over gas-free brine (%)')

    ax2.plot(ms, 100 * (old / new - 1), color=NAVY, lw=2.2)
    ax2.axhline(0, color=MUTED, lw=0.9)
    _style(ax2, 'Overstatement of the gas effect', 'NaCl (mol/kg)',
           'Water-mass form, excess (%)')
    for m in (1, 2, 5):
        i = int(np.argmin(abs(ms - m)))
        v = 100 * (old[i] / new[i] - 1)
        ax2.scatter([m], [v], s=30, color=NAVY, zorder=3)
        ax2.annotate(f'+{v:.0f}%', (m, v), xytext=(6, -3),
                     textcoords='offset points', color=NAVY, fontsize=8)

    fig.tight_layout()
    _save(fig, 'fig6_solvent_basis')


# ===========================================================================
# FIG 10  The delta framing: your brine, plus what the gas does to it
# ===========================================================================
def fig_workflow_delta():
    """The event-note workflow figure (Mark's messaging direction, 2026-07-31).

    The organising idea is the DELTA framing: this method never supplies the
    base brine density or viscosity, it supplies what the dissolved gas does
    to them. Two vertical lanes; in each, the dashed YOUR-CHOICE base rails
    around the solid THIS-WORK stack and meets it only at the combining step.
    The density lane carries the freshwater molar volume (one calibrated
    volume shift per gas, EOS swappable) and the salinity correction to it.
    fig8_workflow remains the whitepaper/manuscript variant.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    DENS = GAS_COLOUR['CO2']
    VISC = GAS_COLOUR['CH4']
    fig, ax = plt.subplots(figsize=(8.6, 7.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 110)
    ax.set_ylim(10.5, 112)
    ax.axis('off')

    _fitted = []

    def _claim(artist, avail):
        _fitted.append((artist, avail))
        return artist

    def _shrink_to_fit():
        fig.canvas.draw()
        px = lambda u: (ax.transData.transform((u, 0))[0]
                        - ax.transData.transform((0, 0))[0])
        for artist, avail in _fitted:
            w = artist.get_window_extent(fig.canvas.get_renderer()).width
            if w > px(avail):
                artist.set_fontsize(max(6.8,
                                        artist.get_fontsize() * px(avail) / w))

    def box(x, y, w, h, title, sub, accent, new=False, foot=None):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6',
            linewidth=1.8 if new else 1.3,
            linestyle='-' if new else (0, (4, 2.6)),
            edgecolor=accent,
            facecolor=accent + '14' if new else SURFACE, zorder=2))
        pad = w - 5.0
        n_title = title.count('\n') + 1
        _claim(ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='top',
                       fontsize=10.0, fontweight='bold', color=INK,
                       linespacing=1.25, zorder=3), pad)
        if sub:
            _claim(ax.text(x + w / 2, y + h - 2.6 - 3.6 * n_title - 1.2, sub,
                           ha='center', va='top', fontsize=8.6, color=MUTED,
                           linespacing=1.45, zorder=3), pad)
        foot = foot or ('THIS WORK' if new else None)
        if foot:
            _claim(ax.text(x + w / 2, y + 1.6, foot, ha='center', va='bottom',
                           fontsize=7.8, fontweight='bold',
                           color=accent if new else MUTED, zorder=3), pad)

    def arrow(x0, y0, x1, y1, colour=MUTED):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11,
            linewidth=1.3, color=colour, zorder=1, shrinkA=0, shrinkB=0))

    def rail(x_edge, x_rail, y_top, y_bot, colour):
        """Your-choice base skirts the this-work stack: out, down, back in."""
        ax.plot([x_edge, x_rail, x_rail], [y_top, y_top, y_bot], color=colour,
                lw=1.3, ls=(0, (4, 2.6)), zorder=1)
        ax.add_patch(FancyArrowPatch(
            (x_rail, y_bot), (x_edge, y_bot), arrowstyle='-|>',
            mutation_scale=11, linewidth=1.3, linestyle=(0, (4, 2.6)),
            color=colour, zorder=1, shrinkA=0, shrinkB=0))

    # ---------------- the strapline: the whole message
    _claim(ax.text(55, 111, 'This method does not supply the base brine '
                   'density or viscosity.', ha='center', va='top',
                   fontsize=11.5, fontweight='bold', color=INK), 106)
    _claim(ax.text(55, 106.6, 'It supplies what the dissolved gas does to '
                   'them. Keep whichever base you already trust.',
                   ha='center', va='top', fontsize=10.0, color=MUTED), 106)
    ax.plot([7, 103], [103.2, 103.2], color='#d9d9d9', lw=1.0)

    for cx, lane, col in ((28, 'DENSITY', DENS), (84, 'VISCOSITY', VISC)):
        ax.text(cx, 100.3, lane, ha='center', va='top', fontsize=9.0,
                fontweight='bold', color=col)

    XD, XV, W = 6.0, 62.0, 44.0

    # ---------------- density lane
    box(XD, 81, W, 16, 'Gas-free brine density',
        'Spivey here. Use whatever you trust.', MUTED, foot='YOUR CHOICE')
    box(XD, 56, W, 23, 'Molar volume of the dissolved\ngas in fresh water',
        'from the same EOS that supplies the\ndissolved amount (S&W here, '
        'or yours),\nvia one calibrated volume shift per gas',
        DENS, new=True)
    box(XD, 35.5, W, 17, 'Salinity correction to that volume',
        'brine squeezes the dissolved gas:\nabout -1.6% per molal, measured',
        DENS, new=True)
    box(XD, 21, W, 12, 'Mass and volume balance', None, MUTED,
        foot='BOOKKEEPING, NOT PHYSICS')
    arrow(XD + W / 2, 56, XD + W / 2, 52.5, colour=DENS)
    arrow(XD + W / 2, 35.5, XD + W / 2, 33, colour=DENS)
    rail(XD, 2.5, 89, 27, MUTED)
    arrow(XD + W / 2, 21, XD + W / 2, 17, colour=DENS)
    ax.text(XD + W / 2, 15.5, 'Gas-saturated brine density', ha='center',
            va='top', fontsize=10.5, fontweight='bold', color=DENS)

    # ---------------- viscosity lane
    box(XV, 81, W, 16, 'Gas-free brine viscosity',
        'IAPWS x salt ratio here.\nUse whatever you trust.', MUTED,
        foot='YOUR CHOICE')
    box(XV, 48, W, 20, 'Per-gas viscosity multiplier',
        'how much the dissolved gas thickens\nthe brine - calibrated to '
        'measured\nviscosities, never scaled from density',
        VISC, new=True)
    box(XV, 21, W, 12, 'Multiply', None, MUTED, foot='THE FACTORS COMPOSE')
    arrow(XV + W / 2, 48, XV + W / 2, 33, colour=VISC)
    rail(XV, 58.5, 89, 27, MUTED)
    arrow(XV + W / 2, 21, XV + W / 2, 17, colour=VISC)
    ax.text(XV + W / 2, 15.5, 'Gas-saturated brine viscosity', ha='center',
            va='top', fontsize=10.5, fontweight='bold', color=VISC)

    # No legend strip: the YOUR CHOICE / THIS WORK chips inside the boxes and
    # the caption's dashed-vs-solid sentence already carry it (cut 2026-08-08).

    _shrink_to_fit()
    _save(fig, 'fig10_workflow_delta')



# ===========================================================================
# FIG 11  The relative salinity shift (whitepaper Eq. 5) against direct data
# ===========================================================================
def fig_salt_fraction():
    """Eq. (5) support: the shipped relative fraction over every direct
    salting-out measurement, vs ionic strength. Colour = salt class (only the
    seven KCl points are the fit basis); marker = gas. Data imported from
    salt_effect_collapse - never retyped."""
    from figures.salt_effect_collapse import build_points
    from brine_gas.vphi_route import SALT_C, SALT_D

    from fits.relative_salt_shift import kcl_molality_from_molarity
    # Tiepel's KCl concentrations are molar; the fit basis is on molality since
    # 2026-09-05, so the fitted points are plotted where they were fitted. The
    # other Tiepel salts (KI, CaCl2, KOH, TMAB) stay at mol/L as reported.
    # Only the KCl fit basis (converted to molality) and the NaCl comparison
    # are plotted: the other salts (KI, KOH, CaCl2, seawater) are reported in
    # mol/L with no density to convert them, and a mol/L point on a mol/kg
    # axis is not quantitatively comparable to the curve. Their qualitative
    # result (every one below the pure-water value) is stated in Section 5.1.
    # (2026-09-07, ChatGPT R7 close-out; hollow-at-mol/L was the interim.)
    pts = [(g, salt, kcl_molality_from_molarity(I) if salt == 'KCl' else I,
            dv, sd, v0) for g, salt, I, dv, sd, v0 in build_points()
           if salt in ('KCl', 'NaCl')]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.grid(True, color='#e8e8e6', lw=0.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color=MUTED, lw=0.8)

    CLASS = lambda salt: ('fit' if salt == 'KCl'
                          else 'nacl' if salt == 'NaCl' else 'other')
    COL = {'fit': NAVY, 'other': '#9aa2ad', 'nacl': '#D55E00'}
    MARK = {'Ar': 'o', 'CH4': 's', 'C2H6': '^', 'O2': 'D', 'H2': 'v',
            'N2': 'P'}
    # The KCl fit basis and the NaCl points are on molality. The other salts
    # (KI, KOH, seawater) are plotted HOLLOW at their reported molar
    # concentration: no density is available to convert them, and they are
    # not fitted, so they are shown as qualitative evidence only (2026-09-07,
    # review R7; the axis no longer labels them as mol/kg).
    for gas, salt, I, dv, sd, v0 in pts:
        cls = CLASS(salt)
        hollow = cls != 'fit'
        ax.errorbar(I, 100 * dv / v0, yerr=100 * sd / v0,
                    fmt=MARK[gas], color=COL[cls], ms=7,
                    mfc='none' if hollow else COL[cls],
                    mew=1.6 if hollow else 1.0,
                    mec=COL[cls] if hollow else 'white',
                    ecolor='#c9c8c4', elinewidth=1.1, capsize=0,
                    zorder=4 if cls == 'fit' else 3)

    import numpy as _np
    # g(m) solid over the adopted 0 to 5 mol/kg interval, dashed beyond it
    m = _np.linspace(0, 5.0, 100)
    ax.plot(m, -100 * SALT_C * m / (1 + SALT_D * m), color=INK, lw=1.8,
            zorder=2)
    m = _np.linspace(5.0, 6.5, 40)
    ax.plot(m, -100 * SALT_C * m / (1 + SALT_D * m), color=INK, lw=1.2,
            ls='--', zorder=2)
    ax.text(5.2, -100 * SALT_C * 5.2 / (1 + SALT_D * 5.2) - 0.9,
            'g(m), dashed beyond 5 mol/kg', fontsize=9, color=INK, ha='left')

    ax.set_xlabel('molality (mol/kg)')
    ax.set_ylabel('relative shift in V$_\\phi$ (%)')
    shown = {g for g, *_ in pts}
    handles = ([plt.Line2D([], [], marker=mk, ls='', color='#9aa2ad', ms=7,
                           label=g) for g, mk in MARK.items() if g in shown]
               + [plt.Line2D([], [], marker='o', ls='', color=NAVY, ms=7,
                             label='KCl (fit basis)'),
                  plt.Line2D([], [], marker='o', ls='', mfc='none',
                             mec='#D55E00', mew=1.6, ms=7, label='NaCl')])
    ax.legend(handles=handles, loc='upper right', fontsize=7.5, ncol=2,
              frameon=False)
    _save(fig, 'fig11_salt_fraction')



# ===========================================================================
# FIG 6 (manuscript, 2026-09-07)  The measured evidence behind the three
# fitted viscosity factors, as parity of calculated against measured
# increase. Replaces the Islam-Carlson contrast + scaling-bar figure (that
# comparison stays in the text; the sign test survives in the event note's
# fig_density_scaling_only). Data imported from the generators, never retyped.
# ===========================================================================
def fig_viscosity_evidence():
    import csv
    from brine_gas.garcia_mixing import viscosity_correction_single
    from fits.ostermann_ch4_refit import dataset as _ost
    from fits.murphy_gaines_h2s_refit import viscosity_coefficients

    def degf(T_K):
        return (T_K - 273.15) * 1.8 + 32.0

    panels = []
    # CO2: McBride-Wright CO2-water (calibration of e1, e2; ratio against the
    # Mao-Duan water baseline) and Calabrese 0.77 m NaCl rows (test of the
    # salt-independence assumption; ratio against their own x = 0 rows).
    rows = list(csv.DictReader(open(os.path.join(_RESULTS, 'raw_co2_viscosity_ratios.csv'))))
    co2 = {}
    for r in rows:
        src = 'McBride-Wright, water' if r['source'].startswith('McBride') else 'Calabrese, 0.77 m NaCl'
        meas = 100.0 * (np.exp(float(r['ln_ratio'])) - 1.0)
        calc = 100.0 * (viscosity_correction_single('CO2', float(r['x_CO2']), degf(float(r['T_K']))) - 1.0)
        co2.setdefault(src, []).append((meas, calc))
    panels.append(('CO$_2$', co2, {'McBride-Wright, water': ('o', True),
                                    'Calabrese, 0.77 m NaCl': ('s', False)}))
    # CH4: the 23 Ostermann ratios at measured loading (calibration of Eq. ch4)
    T, X, E, F = _ost()
    ch4 = {'Ostermann, water': [(100.0 * e, 100.0 * (viscosity_correction_single('CH4', x, f) - 1.0))
                                 for x, e, f in zip(X, E, F)]}
    panels.append(('CH$_4$', ch4, {'Ostermann, water': ('o', True)}))
    # H2S: the five Murphy & Gaines ratios; four fitted, the 35.2 degC one
    # below the apparatus resolution and excluded
    h2s = {'Murphy-Gaines, fitted': [], 'Murphy-Gaines, unresolved': []}
    for t, p_atm, r, x, a in viscosity_coefficients():
        key = 'Murphy-Gaines, unresolved' if t > 35.0 else 'Murphy-Gaines, fitted'
        h2s[key].append((100.0 * (r - 1.0), 100.0 * (viscosity_correction_single('H2S', x) - 1.0)))
    panels.append(('H$_2$S', h2s, {'Murphy-Gaines, fitted': ('o', True),
                                   'Murphy-Gaines, unresolved': ('o', False)}))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    report = []
    for ax, (gas, series, style), gkey in zip(axes, panels, ('CO2', 'CH4', 'H2S')):
        col = GAS_COLOUR[gkey]
        allv = [v for pts in series.values() for pr in pts for v in pr]
        lo, hi = min(0.0, min(allv)) - 0.5, max(allv) + 0.8
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        title_bits = []
        for name, pts in series.items():
            mk, filled = style[name]
            m = np.array(pts)
            ax.plot(m[:, 0], m[:, 1], mk, ms=6, color=col,
                    mfc=col if filled else 'none', mew=1.4, ls='', label=name, zorder=3)
            if filled or gkey == 'CO2':
                rms = np.sqrt(np.mean((m[:, 1] - m[:, 0]) ** 2))
                title_bits.append(f'{name.split(",")[0]} n={len(m)}, RMS {rms:.2f} pp')
                report.append((gkey, name, len(m), rms))
        _style(ax, f'{gas}: ' + '; '.join(title_bits) if gkey != 'CO2' else gas,
               'measured viscosity increase (%)',
               'calculated (%)' if gkey == 'CO2' else None)
        if gkey == 'CO2':
            ax.set_title(gas + '\n' + '\n'.join(title_bits), loc='left', fontsize=9,
                         fontweight='bold', pad=6)
        ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    _save(fig, 'fig6_viscosity_evidence')
    print(f"  {'gas':<5}{'series':<30}{'n':>4}{'RMS pp':>8}")
    for g, name, n, rms in report:
        print(f'  {g:<5}{name:<30}{n:>4}{rms:>8.2f}')
    return report

if __name__ == '__main__':
    fig_workflow_delta()
    fig_salt_fraction()
    fig_density_signature()
    fig_viscosity_contrast()
    fig_viscosity_evidence()
    fig_density_scaling_only()
    fig_validation_parity()
    fig_vphi_vs_data()
    fig_ch4_ramp()
    fig_solvent_basis()
