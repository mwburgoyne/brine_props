"""Generate every number in the manuscript's worked-examples appendix.

Two full-chain walkthroughs (Mark's request 2026-08-08), field units in,
density and viscosity out, printing every intermediate:

  Example 1: pure CO2 over 1.0 mol/kg NaCl brine at 3000 psia, 175 degF.
  Example 2: 50:50 CH4/CO2 free gas over 5.0 wt% NaCl brine, same P and T
             (held at Example 1's conditions so the two are comparable).

The dissolved amounts come from the pyResToolbox Soreide-Whitson flash (the
companion refresh); everything downstream is this repo's delivered chain via
`vphi_route`, `garcia_mixing` and `viscosity_route`.

THE TABLES MUST BE WALKABLE (Mark 2026-08-16). A reader who takes each printed
value at its printed precision and does the stated arithmetic must land on the
next printed value, and the per-gas density shares must add to the printed
total. Every derived row therefore carries a `check=(fn, deps)`, and `walk()`
re-runs the whole chain from the ROUNDED printed values and asserts each row
reproduces itself. That is what sets the displayed precision: densities need
three decimals in kg/m3 because the densification is a difference of two
nearly-equal numbers. Displayed digits exceed the accuracy of the method, which
Sections 5 to 7 state; they are there so the arithmetic closes.

Per-gas density shares are EXACT, not first order: the mass and volume balance
gives (rho - rho1)/rho1 = sum_i m_i (M_i - rho1 V_i) / (rho1 V), with V the
solution volume per kg water. Same numerator as Eq. first_order, denominator
rho1*V instead of W_b, and the shares then add to the total exactly. The
first-order form divides by W_b and its terms do NOT add to the total (they
over-count by 1/(1+b), b = rho1*sum(m_i V_i)/W_b: +0.195 against +0.192 in
Example 2). Contributions add; they do not compound - the balance is a ratio of
two sums, and every V_i and M_i is referenced to the same gas-free baseline.

Run this script to regenerate the appendix; `--tex` also writes the two tables
into manuscript/ for \\input, so the printed numbers cannot drift from the code.
The deliverable gate does not cover worked examples (its documented limit).
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

HERE = os.path.dirname(os.path.abspath(__file__))

from brine_gas.vphi_route import V_phi, salt_fraction
from brine_gas.brine_properties import rho_brine, M_NACL
from brine_gas.garcia_mixing import (density_single_gas, density_mixed_gas, gas_mw,
                           viscosity_correction_single,
                           viscosity_correction_mixed, gas_saturated_viscosity)
from brine_gas.viscosity_route import brine_viscosity, pressure_factor, _composition
from brine_gas.iapws_viscosity import mu_water_TP
from brine_gas.water_properties import MW_WATER
from pyrestoolbox import brine as rtb_brine

PSIA_TO_MPA = 0.006894757
N_W = 1000.0 / MW_WATER              # moles of water per kg, 55.5084
MANUSCRIPT = _TABLES


def flash(P_mpa, T_k, ppm, **kw):
    # framework pinned to 'default', the refresh paper's published
    # recommendation (embedded delta_kij); explicit so a future library
    # default move cannot silently change the dissolved amounts.
    kw.setdefault('framework', 'default')
    return rtb_brine.SoreideWhitson(pres=P_mpa * 10.0, temp=T_k - 273.15,
                                    ppm=ppm, metric=True, **kw)


def header(t):
    print('\n' + '=' * 74 + f'\n{t}\n' + '=' * 74)


def show(label, value, unit=''):
    print(f'  {label:<52s} {value:>14} {unit}')


def rnd(x, dp):
    return float(f'{x:.{dp}f}')


class Table:
    """A worked-example table that can print itself, emit LaTeX, and prove a
    reader reproduces it from the printed values alone."""

    def __init__(self, title):
        self.title = title
        self.rows = []          # (key, label, tex_label, tex_src, dp, unit)
        self.val = {}           # key -> full-precision value
        self.checks = {}        # key -> (fn(printed) -> value, [deps])
        self.rules = set()      # keys that carry a \midrule above them

    def add(self, key, label, value, dp, unit='', tex_label=None,
            tex_src='', check=None, rule=False):
        self.rows.append((key, label, tex_label or label, tex_src, dp, unit))
        self.val[key] = value
        if check is not None:
            self.checks[key] = check
        if rule:
            self.rules.add(key)
        return value

    def printed(self):
        return {k: rnd(self.val[k], dp)
                for k, _, _, _, dp, _ in self.rows}

    def _dp(self, key):
        return dict((k, d) for k, _, _, _, d, _ in self.rows)[key]

    def _set_dp(self, key, dp):
        for i, row in enumerate(self.rows):
            if row[0] == key:
                self.rows[i] = row[:4] + (dp,) + row[5:]
                return

    def resolve(self):
        """Pick each derived row's displayed precision: the decimals nearest
        the requested one at which a reader's arithmetic still lands on the
        printed value. Rows resolve in order, so each sees its upstream rows
        already fixed. Too few decimals drops information the rows below need;
        too many expose the propagated rounding of the inputs, so the window is
        narrow, sits either side of the request, and has to be searched."""
        for key, _l, _tl, _ts, dp0, _u in list(self.rows):
            if key not in self.checks:
                continue
            fn, _deps = self.checks[key]
            for dp in (dp0, dp0 - 1, dp0 + 1, dp0 - 2, dp0 + 2, dp0 + 3):
                if dp < 0:
                    continue
                self._set_dp(key, dp)
                if rnd(fn(self.printed()), dp) == rnd(self.val[key], dp):
                    break
            else:
                self._set_dp(key, dp0)

    def walk(self, sums=()):
        """Assert the table reproduces itself from its own printed values."""
        p = self.printed()
        bad = []
        for key, (fn, _deps) in self.checks.items():
            dp = self._dp(key)
            if rnd(fn(p), dp) != p[key]:
                bad.append(f'{self.title}: row {key!r} does not walk - a '
                           f'reader gets {rnd(fn(p), dp)}, table prints '
                           f'{p[key]}')
        for parts, total in sums:
            dp = self._dp(total)
            got = rnd(sum(p[k] for k in parts), dp)
            if got != p[total]:
                bad.append(f'{self.title}: {" + ".join(parts)} = {got} does '
                           f'not match printed {total} = {p[total]}')
        if bad:
            raise AssertionError('\n'.join(bad))

    def emit(self):
        for key, label, _tl, _ts, dp, unit in self.rows:
            sign = '+' if key.startswith(('dens', 'share', 'g_', 'dc')) else ''
            show(label, f'{self.val[key]:{sign}.{dp}f}', unit)

    # plain-text units of the console listing -> their LaTeX form
    TEX_UNIT = {'': '', 'g/g': '', '%': '\\%', 'mPa s': '~mPa~s',
                'MPa/mol': '~MPa~mol$^{-1}$', 'MPa/cm3': '~MPa~cm$^{-3}$',
                'MPa mol/cm6': '~MPa~mol~cm$^{-6}$',
                'g': '~g', 'cm3': '~cm$^3$', 'kg/m3': '~kg~m$^{-3}$',
                'cm3/mol': '~cm$^3$~mol$^{-1}$',
                'mol/kg water': '~mol~kg$^{-1}$',
                '1e-4/MPa': '~$\\times 10^{-4}$~MPa$^{-1}$',
                'cm3/mol/MPa': '~cm$^3$~mol$^{-1}$~MPa$^{-1}$'}

    def tex(self, caption, tex_label):
        out = ['% GENERATED by code/worked_examples.py --tex. Do not edit.',
               '\\begin{table}[H]', '\\centering',
               f'\\caption{{{caption}}}', f'\\label{{{tex_label}}}',
               '\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{0.88}',
               '\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.37\\textwidth}lr}', '\\toprule',
               'Quantity & Source & Value \\\\', '\\midrule']
        for key, _label, tex_label_, tex_src, dp, unit in self.rows:
            if key in self.rules:
                out.append('\\midrule')
            sign = '+' if key.startswith(('dens', 'share', 'g_', 'dc')) else ''
            v = f'{self.val[key]:{sign}.{dp}f}'
            if v[0] in '+-':                    # keep the sign in math mode
                v = f'${v[0]}{v[1:]}$'
            out.append(f'{tex_label_} & {tex_src} & {v}'
                       f'{self.TEX_UNIT[unit]} \\\\')
        out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
        return '\n'.join(out)


SALINITY_ROWS = [
    # key, console label, LaTeX quantity, unit, relation, decimals
    ('m',    'molality m (input here)', 'molality $m$ (input here)',
     'mol/kg water', '$1000\\,S/[(1-S)M_s]$', 6),
    # S carries eight decimals because the TDS row multiplies it by 10^6: at
    # six, a reader's TDS lands one mg/L off the printed value.
    ('S',    'weight fraction S', 'weight fraction $S$', 'g/g',
     '$m M_s/(1000 + m M_s)$', 8),
    ('ppm',  'ppm (mg/kg)', 'ppm (mg/kg)', '', '$10^6\\,S$', 0),
    # an auxiliary property, not a salinity measure: the gas-free brine
    # density the two density-based rows below need, from the Spivey model
    ('rho',  'rho_b at 60 degF, 1 atm (Spivey)',
     'gas-free $\\rho_b$ (60~$^{\\circ}$F, 1~atm)', 'g/cm3',
     '$\\rho_{\\mathrm{Spivey}}(288.71~\\mathrm{K},\\,0.1013~\\mathrm{MPa},\\,S)$ \\cite{McCain2011}', 6),
    ('c',    'molarity c', 'molarity $c$', 'mol/L', '$10^3 \\rho_b S/M_s$', 4),
    ('tds',  'TDS', 'TDS', 'mg/L', '$10^6\\,S \\rho_b$', 0),
]

SALINITY_CHECKS = {
    'S':   lambda p: p['m'] * M_NACL / (1000.0 + p['m'] * M_NACL),
    'ppm': lambda p: 1e6 * p['S'],
    'c':   lambda p: 1000.0 * p['rho'] * p['S'] / M_NACL,
    'tds': lambda p: 1e6 * p['S'] * p['rho'],
}


def salinity_columns(molalities):
    """The salinity ledger for each example brine: every common expression of
    one NaCl salinity, at a precision that lets a reader recompute each row
    from the two above it. Standard-conditions density from Spivey, the
    paper's gas-free default (Mark 2026-08-08: Spivey, not the monograph's
    Rowe-Chou, per normal usage); Rowe-Chou (W&B Eq. 9.3) is a cross-check."""
    T_SC, P_SC = (60.0 - 32.0) / 1.8 + 273.15, 0.101325   # 60 degF, 1 atm
    cols = []
    for m in molalities:
        S = m * M_NACL / (1000.0 + m * M_NACL)
        rho_sc = rho_brine(T_SC, P_SC, S) / 1000.0        # g/cm3, Spivey
        rc = 1.0 / (1.0009 - 0.7114 * S + 0.26055 * S * S)
        assert abs(rho_sc / rc - 1.0) < 2e-3, (rho_sc, rc)
        cols.append(dict(m=m, S=S, ppm=1e6 * S, rho=rho_sc,
                         c=1000.0 * rho_sc * S / M_NACL,
                         tds=1e6 * S * rho_sc))
    dp = {k: d for k, _l, _t, _u, _r, d in SALINITY_ROWS}
    for key in [r[0] for r in SALINITY_ROWS]:
        if key not in SALINITY_CHECKS:
            continue
        fn = SALINITY_CHECKS[key]
        for cand in (dp[key], dp[key] - 1, dp[key] + 1, dp[key] + 2):
            if cand < 0:
                continue
            dp[key] = cand
            if all(rnd(fn({k: rnd(c[k], dp[k]) for k in c}), cand)
                   == rnd(c[key], cand) for c in cols):
                break
    for c in cols:
        p = {k: rnd(c[k], dp[k]) for k in c}
        for key, fn in SALINITY_CHECKS.items():
            assert rnd(fn(p), dp[key]) == p[key], (
                f'salinity ledger: row {key!r} does not walk - a reader gets '
                f'{rnd(fn(p), dp[key])}, table prints {p[key]}')
    return cols, dp


def salinity_tex(cols, dp):
    # [H]: placed where the text puts it, so the Appendix C.1 heading and its
    # own table follow on the same page (Mark, 2026-09-07); needs \usepackage{float}
    out = ['% GENERATED by code/worked_examples.py --tex. Do not edit.',
           '\\begin{table}[H]', '\\centering',
           '\\caption{Salinity measures and exact NaCl conversions '
           '($M_s = 58.443$~g~mol$^{-1}$), after Whitson and Brul\\\'e\'s '
           'Table~9.2 \\cite{WhitsonBrule2000}.}',
           '\\label{tab:salinity}',
           '\\footnotesize\\setlength{\\tabcolsep}{3pt}',
           '\\begin{tabular}{lllrr}', '\\toprule',
           'Measure & Unit & Relation or model & Example 1 & Example 2 \\\\',
           '\\midrule']
    unit_tex = {'mol/kg water': 'mol/kg water', 'g/g': 'g/g', '': '--',
                'g/cm3': 'g/cm$^3$', 'mol/L': 'mol/L', 'mg/L': 'mg/L'}
    for key, _lab, tex_q, unit, rel, _d in SALINITY_ROWS:
        vals = ' & '.join(f'{c[key]:,.{dp[key]}f}' if dp[key] == 0
                          else f'{c[key]:.{dp[key]}f}' for c in cols)
        vals = vals.replace(',', '{,}')
        out.append(f'{tex_q} & {unit_tex[unit]} & {rel} & {vals} \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def salinity_ledger(col, dp):
    for key, label, _t, unit, _r, _d in SALINITY_ROWS:
        v = (f'{col[key]:,.0f}' if dp[key] == 0
             else f'{col[key]:.{dp[key]}f}')
        show(label, v, unit)


H_P = 1.0      # MPa, central-difference step for the printed derivative rows
H_FINE = 0.01  # MPa, the cross-check step (the derivative is stable to <0.01%)


def c_brine(T, P, S):
    """Gas-free brine compressibility (1/rho) drho/dp from Spivey, 1e-4/MPa."""
    return 1e4 * (rho_brine(T, P + H_FINE, S) - rho_brine(T, P - H_FINE, S)) / (2 * H_FINE) / rho_brine(T, P, S)


def dvdp_rows(t, gas, key, T, P, g, label_gas='', tex_gas='', water_key=None):
    """The pressure derivative of the gas volume, shown step by step in closed
    form (manuscript Appendix B: Eqs. B.4 and B.5, their v_w derivatives, and
    the chain rule through the water root), then the salinity factor applied.
    A central difference with h = H_P is printed as the check row. Temperature,
    molality and dissolved composition are held fixed; the flash is NOT rerun.
    Keys are suffixed so example 2 can carry two gases."""
    from brine_gas.pr_vphi_model import _derivatives
    d = _derivatives(gas, T, P)
    fd = (V_phi(gas, T, P + H_P) - V_phi(gas, T, P - H_P)) / (2 * H_P)
    fd_fine = (V_phi(gas, T, P + H_FINE) - V_phi(gas, T, P - H_FINE)) / (2 * H_FINE)
    assert abs(d['dV2_dp'] / fd - 1) < 1e-4, (gas, d['dV2_dp'], fd)
    assert abs(d['dV2_dp'] / fd_fine - 1) < 1e-6, (gas, d['dV2_dp'], fd_fine)
    gl, gt = label_gas, tex_gas
    # the water-only quantities (root, dp/dV, d2p/dV2, dv_w/dp) are printed
    # once per table; a second gas reuses them through water_key
    wk = water_key if water_key is not None else key
    if water_key is None:
        t.add('vw' + key, f'PR water root v_w at T, p', d['v_w'], 4, 'cm3/mol',
              tex_label='PR water root $v_w$ at $(T, p)$', tex_src='Eq.~\\ref{eq:app_cubic}',
              rule=True)
        t.add('dpdv' + key, f'(dp/dV) at v_w', d['dPdV'], 4, 'MPa/cm3',
              tex_label='$(\\partial p/\\partial V)$', tex_src='Eq.~\\ref{eq:app_dpdv}')
        t.add('d2v' + key, f'(d2p/dV2)', d['d2PdV2'], 4, 'MPa mol/cm6',
              tex_label='$(\\partial^2 p/\\partial V^2)$', tex_src='Eq.~\\ref{eq:app_d2pdv2}')
        t.add('dvwdp' + key, 'dv_w/dp = 1/(dp/dV)', d['dvw_dp'], 7, 'cm3/mol/MPa',
              tex_label='$\\partial v_w/\\partial p = 1/(\\partial p/\\partial V)$', tex_src='Eq.~\\ref{eq:app_dvdp}',
              check=(lambda p, k=key: 1.0 / p['dpdv' + k], ['dpdv' + key]))
    t.add('dpdn' + key, f'(dp/dn2){gl} at v_w', d['dPdn2'], 3, 'MPa/mol',
          tex_label=f'$(\\partial p/\\partial n_2)${gt}', tex_src='Eq.~\\ref{eq:app_dpdn}',
          rule=(water_key is not None))
    t.add('d2n' + key, f'(d2p/dn2 dV){gl}', d['d2Pdn2dV'], 3, 'MPa/cm3',
          tex_label=f'$(\\partial^2 p/\\partial n_2 \\partial V)${gt}', tex_src='Eq.~\\ref{eq:app_d2pdn2dv}')
    t.add('dvdvw' + key, f'dV_inf/dv_w{gl} = -[(d2p/dn2dV)(dp/dV) - (dp/dn2)(d2p/dV2)]/(dp/dV)^2',
          d['dV2_dv'], 5, '',
          tex_label=f'$\\partial \\Vinf/\\partial v_w${gt}',
          tex_src='Eq.~\\ref{eq:app_dvdp}',
          check=(lambda p, k=key, w=wk: -(p['d2n' + k] * p['dpdv' + w] - p['dpdn' + k] * p['d2v' + w]) / p['dpdv' + w] ** 2,
                 ['d2n' + key, 'dpdv' + wk, 'dpdn' + key, 'd2v' + wk]))
    t.add('dvdp' + key, f'freshwater dV_phi/dp{gl} = (dV_inf/dv_w)(dv_w/dp)', d['dV2_dp'], 5, 'cm3/mol/MPa',
          tex_label=f'freshwater $(\\partial \\Vphi/\\partial p)_T${gt}',
          tex_src='Eq.~\\ref{eq:app_dvdp}',
          check=(lambda p, k=key, w=wk: p['dvdvw' + k] * p['dvwdp' + w], ['dvdvw' + key, 'dvwdp' + wk]))
    t.add('dvdpb' + key, f'brine dV_phi/dp{gl} = (1+g) x freshwater derivative', (1.0 + g) * d['dV2_dp'], 5, 'cm3/mol/MPa',
          tex_label=f'brine $(\\partial \\Vphi/\\partial p)_{{T,m}} = (1 + g)\\,(\\partial \\Vphi/\\partial p)_T${gt}',
          tex_src='Eq.~\\ref{eq:salt_fraction}',
          check=(lambda p, k=key: (1.0 + p['g_'] / 100.0) * p['dvdp' + k], ['g_', 'dvdp' + key]))
    return d['dV2_dp']


def c_phi_row(t, key, veff_key, label_gas='', tex_gas=''):
    """c_phi = -(1/V_eff) dV_brine/dp, 1e-4/MPa, from the printed rows."""
    c = -1e4 * t.val['dvdpb' + key] / t.val[veff_key]
    t.add('cphi' + key, f'c_phi{label_gas} = -(1/V_eff) dV_brine/dp', c, 4, '1e-4/MPa',
          tex_label=f'$c_\\phi${tex_gas} $= -(1/V_{{\\mathrm{{eff}}}})\\,(\\partial \\Vphi/\\partial p)_{{T,m}}$',
          tex_src='Eq.~\\ref{eq:compressibility}',
          check=(lambda p, k=key, v=veff_key: -1e4 * p['dvdpb' + k] / p[v], ['dvdpb' + key, veff_key]))
    return c


def example1(T, P, T_degf):
    """Pure CO2 over 1.0 mol/kg NaCl."""
    t = Table('Example 1')
    m = 1.0
    S = m * M_NACL / (1000.0 + m * M_NACL)
    M2 = gas_mw('CO2')

    t.add('m', 'molality m (input)', m, 6, 'mol/kg water',
          tex_label='molality $m$', tex_src='input')
    t.add('S', 'salt mass fraction S = m*M_s/(1000+m*M_s)', S, 6, 'g/g',
          tex_label='salt mass fraction $S = m M_s/(1000 + m M_s)$',
          tex_src='unit conversion',
          check=(lambda p: p['m'] * M_NACL / (1000.0 + p['m'] * M_NACL),
                 ['m']))

    x2 = float(flash(P, T, S * 1e6, y_CO2=1.0).x.get('CO2'))
    t.add('x2', 'dissolved CO2, salt-free mole fraction x2', x2, 8,
          tex_label='dissolved CO$_2$, salt-free mole fraction $x_2$',
          tex_src='S\\&W flash \\cite{BurgoyneRefresh}')
    t.add('m2', 'gas molality m2 = 55.5084*x2/(1-x2)',
          N_W * x2 / (1.0 - x2), 6, 'mol/kg water',
          tex_label='gas molality $m_2 = 55.5084\\,x_2/(1-x_2)$',
          tex_src='basis conversion',
          check=(lambda p: N_W * p['x2'] / (1.0 - p['x2']), ['x2']))

    v_fresh = V_phi('CO2', T, P)
    g = salt_fraction(m)
    t.add('v', 'freshwater V_phi(CO2) at T, P', v_fresh, 5, 'cm3/mol',
          tex_label='freshwater $\\Vphi$ at $(T,p)$',
          tex_src='Eq.~\\ref{eq:pmv_exact} + shift')
    t.add('g_', 'salinity fraction g(m)', 100.0 * g, 4, '%',
          tex_label='salinity fraction $g(m)$',
          tex_src='Eq.~\\ref{eq:salt_fraction}')
    t.add('veff', 'V_eff = V_phi*(1+g)', v_fresh * (1.0 + g), 4, 'cm3/mol',
          tex_label='$V_{\\mathrm{eff}} = \\Vphi (1 + g)$',
          tex_src='Eq.~\\ref{eq:salt_fraction}',
          check=(lambda p: p['v'] * (1.0 + p['g_'] / 100.0), ['v', 'g_']))

    rho1 = rho_brine(T, P, S)
    t.add('rho1', 'gas-free brine density rho1 (Spivey)', rho1, 4, 'kg/m3',
          tex_label='gas-free brine density $\\rho_1$',
          tex_src='Spivey \\cite{McCain2011}')
    t.add('Wb', 'solvent mass W_b = 1000 + m*M_NaCl', 1000.0 + m * M_NACL, 4,
          'g', tex_label='solvent mass $W_b = 1000 + m M_s$',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: 1000.0 + p['m'] * M_NACL, ['m']))
    t.add('mass', 'mass per kg water = W_b + m2*M_CO2',
          t.val['Wb'] + t.val['m2'] * M2, 4, 'g',
          tex_label='mass per kg water $= W_b + m_2 M_{\\mathrm{CO_2}}$',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: p['Wb'] + p['m2'] * M2, ['Wb', 'm2']))
    t.add('vol', 'volume per kg water = W_b/rho1 + m2*V_eff',
          t.val['Wb'] / (rho1 / 1000.0) + t.val['m2'] * t.val['veff'], 4,
          'cm3',
          tex_label='volume per kg water $= W_b/\\rho_1 + m_2 V_{\\mathrm{eff}}$',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: p['Wb'] / (p['rho1'] / 1000.0)
                 + p['m2'] * p['veff'], ['Wb', 'rho1', 'm2', 'veff']))
    t.add('rho', 'saturated density rho = mass/volume',
          t.val['mass'] / t.val['vol'] * 1000.0, 3, 'kg/m3',
          tex_label='saturated density $\\rho = $ mass/volume',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: p['mass'] / p['vol'] * 1000.0, ['mass', 'vol']))
    t.add('dens', 'densification rho/rho1 - 1',
          100.0 * (t.val['rho'] / rho1 - 1.0), 2, '%',
          tex_label='densification $\\rho/\\rho_1 - 1$',
          check=(lambda p: 100.0 * (p['rho'] / p['rho1'] - 1.0),
                 ['rho', 'rho1']))

    # the delivered entry point must agree with the table's own arithmetic
    assert abs(density_single_gas('CO2', x2, T, P, S=S) - t.val['rho']) < 1e-6

    # compressibility (Eq. compressibility, App. D), Mark 2026-09-07: the
    # pressure derivative shown step by step, then c_phi, then the mix
    dvdp_rows(t, 'CO2', '', T, P, g)
    c_phi_row(t, '', 'veff')
    t.add('cb', 'gas-free brine compressibility c_b = (1/rho1) drho1/dp', c_brine(T, P, S), 4,
          '1e-4/MPa', tex_label='gas-free compressibility $c_b$',
          tex_src='Spivey \\cite{McCain2011}, numerical')
    t.add('phig', 'gas volume fraction phi_g = m2*V_eff/volume',
          100.0 * t.val['m2'] * t.val['veff'] / t.val['vol'], 4, '%',
          tex_label='gas volume fraction $\\phi_g = m_2 V_{\\mathrm{eff}}/V$',
          tex_src='Eq.~\\ref{eq:compressibility}',
          check=(lambda p: 100.0 * p['m2'] * p['veff'] / p['vol'], ['m2', 'veff', 'vol']))
    t.add('c', 'saturated compressibility c = (1-phi_g)*c_b + phi_g*c_phi',
          (1 - t.val['phig'] / 100.0) * t.val['cb'] + t.val['phig'] / 100.0 * t.val['cphi'], 4,
          '1e-4/MPa', tex_label='saturated compressibility $c$',
          tex_src='Eq.~\\ref{eq:compressibility}',
          check=(lambda p: (1 - p['phig'] / 100.0) * p['cb'] + p['phig'] / 100.0 * p['cphi'],
                 ['phig', 'cb', 'cphi']))
    t.add('dc', 'compressibility change c/c_b - 1', 100.0 * (t.val['c'] / t.val['cb'] - 1.0), 2, '%',
          tex_label='compressibility change $c/c_b - 1$',
          check=(lambda p: 100.0 * (p['c'] / p['cb'] - 1.0), ['c', 'cb']))

    mu_w = float(mu_water_TP(T, P)) * 1000.0
    mu_b = brine_viscosity(T, P, m=m)
    f_p = pressure_factor(T, P, _composition(m=m))
    r_salt = (mu_b / mu_w) / f_p
    fac = viscosity_correction_single('CO2', x2, degf=T_degf)
    mu_sat = gas_saturated_viscosity(T, P, gas_dict={'CO2': x2}, m=m)
    t.add('mu_w', 'water viscosity mu_w (IAPWS-2008)', mu_w, 4, 'mPa s',
          tex_label='water viscosity $\\mu_w$',
          tex_src='IAPWS-2008 \\cite{Huber2009}', rule=True)
    t.add('r_salt', 'ion-additive salt ratio r_salt', r_salt, 4,
          tex_label='ion-additive salt ratio $r_{salt}$',
          tex_src='Section~\\ref{subsec:visc_base}')
    t.add('f_p', 'Kestin pressure factor f_p', f_p, 4,
          tex_label='Kestin pressure factor $f_p$',
          tex_src='Section~\\ref{subsec:visc_base}')
    t.add('mu_b', 'gas-free brine mu_b = mu_w*r_salt*f_p', mu_b, 4, 'mPa s',
          tex_label='gas-free brine $\\mu_b = \\mu_w\\,r_{salt}\\,f_p$',
          tex_src='Eq.~\\ref{eq:visc_chain}',
          check=(lambda p: p['mu_w'] * p['r_salt'] * p['f_p'],
                 ['mu_w', 'r_salt', 'f_p']))
    t.add('f_co2', 'CO2 factor exp[e1*exp(-e2*(T/T0-1))*x2]', fac, 4,
          tex_label='CO$_2$ factor $\\exp[e_1 e^{-e_2(T/T_0-1)} x_2]$',
          tex_src='Eq.~\\ref{eq:calabrese25}')
    t.add('mu', 'saturated viscosity mu = mu_b*factor', mu_sat, 4, 'mPa s',
          tex_label='saturated viscosity $\\mu = \\mu_b \\times$ factor',
          tex_src='Eq.~\\ref{eq:visc_top}',
          check=(lambda p: p['mu_b'] * p['f_co2'], ['mu_b', 'f_co2']))
    t.resolve()
    t.walk()
    return t


def example2(T, P, T_degf):
    """Equimolar CH4/CO2 free gas over 5.0 wt% NaCl."""
    t = Table('Example 2')
    S = 0.05
    m = 1000.0 * S / ((1.0 - S) * M_NACL)
    M_c, M_d = gas_mw('CH4'), gas_mw('CO2')

    t.add('S', 'salt mass fraction S (input)', S, 6, 'g/g',
          tex_label='salt mass fraction $S$', tex_src='input')
    t.add('m', 'molality m = 1000*S/((1-S)*M_NaCl)', m, 6, 'mol/kg water',
          tex_label='molality $m = 1000\\,S/[(1-S) M_s]$',
          tex_src='unit conversion',
          check=(lambda p: 1000.0 * p['S'] / ((1.0 - p['S']) * M_NACL),
                 ['S']))

    mix = flash(P, T, S * 1e6, y_CO2=0.5, sg=0.5539)
    x_c, x_d = float(mix.x.get('CH4')), float(mix.x.get('CO2'))
    xt = x_c + x_d
    t.add('xc', 'dissolved CH4, x_CH4', x_c, 9,
          tex_label='dissolved CH$_4$, $x_{\\mathrm{CH_4}}$',
          tex_src='S\\&W flash of the equimolar gas')
    t.add('xd', 'dissolved CO2, x_CO2', x_d, 9,
          tex_label='dissolved CO$_2$, $x_{\\mathrm{CO_2}}$',
          tex_src='S\\&W flash')
    t.add('mc', 'gas molality m_CH4 = 55.5084*x_CH4/(1-x_t)',
          N_W * x_c / (1.0 - xt), 6, 'mol/kg water',
          tex_label='$m_{\\mathrm{CH_4}} = 55.5084\\,x_{\\mathrm{CH_4}}/(1-x_t)$',
          tex_src='basis conversion',
          check=(lambda p: N_W * p['xc'] / (1.0 - p['xc'] - p['xd']),
                 ['xc', 'xd']))
    t.add('md', 'gas molality m_CO2 = 55.5084*x_CO2/(1-x_t)',
          N_W * x_d / (1.0 - xt), 6, 'mol/kg water',
          tex_label='$m_{\\mathrm{CO_2}} = 55.5084\\,x_{\\mathrm{CO_2}}/(1-x_t)$',
          tex_src='basis conversion',
          check=(lambda p: N_W * p['xd'] / (1.0 - p['xc'] - p['xd']),
                 ['xc', 'xd']))

    v_c, v_d = V_phi('CH4', T, P), V_phi('CO2', T, P)
    g = salt_fraction(m)
    t.add('vc', 'freshwater V_phi(CH4)', v_c, 5, 'cm3/mol',
          tex_label='freshwater $\\Vphi$(CH$_4$)',
          tex_src='Eq.~\\ref{eq:pmv_exact} + shift')
    t.add('vd', 'freshwater V_phi(CO2)', v_d, 5, 'cm3/mol',
          tex_label='freshwater $\\Vphi$(CO$_2$)',
          tex_src='Eq.~\\ref{eq:pmv_exact} + shift')
    t.add('g_', 'salinity fraction g(m)', 100.0 * g, 4, '%',
          tex_label='salinity fraction $g(m)$',
          tex_src='Eq.~\\ref{eq:salt_fraction}')
    t.add('vce', 'V_eff(CH4) = V_phi*(1+g)', v_c * (1.0 + g), 4, 'cm3/mol',
          tex_label='$V_{\\mathrm{eff}}$(CH$_4$) $= \\Vphi(1+g)$',
          tex_src='Eq.~\\ref{eq:salt_fraction}',
          check=(lambda p: p['vc'] * (1.0 + p['g_'] / 100.0), ['vc', 'g_']))
    t.add('vde', 'V_eff(CO2) = V_phi*(1+g)', v_d * (1.0 + g), 4, 'cm3/mol',
          tex_label='$V_{\\mathrm{eff}}$(CO$_2$) $= \\Vphi(1+g)$',
          tex_src='Eq.~\\ref{eq:salt_fraction}',
          check=(lambda p: p['vd'] * (1.0 + p['g_'] / 100.0), ['vd', 'g_']))

    rho1 = rho_brine(T, P, S)
    t.add('rho1', 'gas-free brine density rho1 (Spivey)', rho1, 4, 'kg/m3',
          tex_label='gas-free brine density $\\rho_1$',
          tex_src='Spivey \\cite{McCain2011}')
    t.add('Wb', 'solvent mass W_b = 1000 + m*M_NaCl', 1000.0 + m * M_NACL, 4,
          'g', tex_label='solvent mass $W_b = 1000 + m M_s$',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: 1000.0 + p['m'] * M_NACL, ['m']))
    t.add('mass', 'mass per kg water = W_b + sum m_i*M_i',
          t.val['Wb'] + t.val['mc'] * M_c + t.val['md'] * M_d, 4, 'g',
          tex_label='mass per kg water $= W_b + \\sum_i m_i M_i$',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: p['Wb'] + p['mc'] * M_c + p['md'] * M_d,
                 ['Wb', 'mc', 'md']))
    t.add('vol', 'volume per kg water = W_b/rho1 + sum m_i*V_i',
          t.val['Wb'] / (rho1 / 1000.0) + t.val['mc'] * t.val['vce']
          + t.val['md'] * t.val['vde'], 3, 'cm3',
          tex_label='volume per kg water $= W_b/\\rho_1 + \\sum_i m_i V_i$',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: p['Wb'] / (p['rho1'] / 1000.0)
                 + p['mc'] * p['vce'] + p['md'] * p['vde'],
                 ['Wb', 'rho1', 'mc', 'md', 'vce', 'vde']))
    t.add('rho', 'saturated density rho = mass/volume',
          t.val['mass'] / t.val['vol'] * 1000.0, 3, 'kg/m3',
          tex_label='saturated density $\\rho = $ mass/volume',
          tex_src='Eq.~\\ref{eq:garcia_pkw}',
          check=(lambda p: p['mass'] / p['vol'] * 1000.0, ['mass', 'vol']))
    t.add('dens', 'densification rho/rho1 - 1',
          100.0 * (t.val['rho'] / rho1 - 1.0), 3, '%',
          tex_label='densification $\\rho/\\rho_1 - 1$',
          check=(lambda p: 100.0 * (p['rho'] / p['rho1'] - 1.0),
                 ['rho', 'rho1']))
    # EXACT per-gas shares: same numerator as the first-order criterion,
    # denominator rho1*V rather than W_b, so the two shares add to the total.
    t.add('share_c', '  of which CH4, m_i(M_i - rho1*V_i)/(rho1*V)',
          100.0 * t.val['mc'] * (M_c - rho1 / 1000.0 * t.val['vce'])
          / (rho1 / 1000.0 * t.val['vol']), 3, '%',
          tex_label='\\quad of which CH$_4$, '
                    '$m_i(M_i - \\rho_1 V_i)/(\\rho_1 V)$',
          tex_src='Eq.~\\ref{eq:exact_share}',
          check=(lambda p: 100.0 * p['mc']
                 * (M_c - p['rho1'] / 1000.0 * p['vce'])
                 / (p['rho1'] / 1000.0 * p['vol']),
                 ['mc', 'rho1', 'vce', 'vol']))
    t.add('share_d', '  of which CO2',
          100.0 * t.val['md'] * (M_d - rho1 / 1000.0 * t.val['vde'])
          / (rho1 / 1000.0 * t.val['vol']), 3, '%',
          tex_label='\\quad of which CO$_2$',
          tex_src='Eq.~\\ref{eq:exact_share}',
          check=(lambda p: 100.0 * p['md']
                 * (M_d - p['rho1'] / 1000.0 * p['vde'])
                 / (p['rho1'] / 1000.0 * p['vol']),
                 ['md', 'rho1', 'vde', 'vol']))

    assert abs(density_mixed_gas({'CH4': x_c, 'CO2': x_d}, T, P, S=S)
               - t.val['rho']) < 1e-6

    # compressibility, per-gas derivatives and volume fractions (App. D),
    # Mark 2026-09-07: each gas's derivative shown step by step
    dvdp_rows(t, 'CH4', '_c', T, P, g, label_gas='(CH4)', tex_gas='(CH$_4$)')
    c_phi_row(t, '_c', 'vce', label_gas='(CH4)', tex_gas='(CH$_4$)')
    dvdp_rows(t, 'CO2', '_d', T, P, g, label_gas='(CO2)', tex_gas='(CO$_2$)', water_key='_c')
    c_phi_row(t, '_d', 'vde', label_gas='(CO2)', tex_gas='(CO$_2$)')
    t.add('cb', 'gas-free brine compressibility c_b', c_brine(T, P, S), 4, '1e-4/MPa',
          tex_label='gas-free compressibility $c_b$',
          tex_src='Spivey \\cite{McCain2011}, numerical')
    t.add('phi_c', 'volume fraction phi(CH4) = m_i*V_i/V',
          100.0 * t.val['mc'] * t.val['vce'] / t.val['vol'], 4, '%',
          tex_label='volume fraction $\\phi_i = m_i V_i/V$, CH$_4$',
          tex_src='Eq.~\\ref{eq:compressibility}',
          check=(lambda p: 100.0 * p['mc'] * p['vce'] / p['vol'], ['mc', 'vce', 'vol']))
    t.add('phi_d', 'volume fraction phi(CO2)',
          100.0 * t.val['md'] * t.val['vde'] / t.val['vol'], 4, '%',
          tex_label='\\quad CO$_2$', tex_src='Eq.~\\ref{eq:compressibility}',
          check=(lambda p: 100.0 * p['md'] * p['vde'] / p['vol'], ['md', 'vde', 'vol']))
    t.add('c', 'saturated compressibility c = (1-sum phi_i)*c_b + sum phi_i*c_phi,i',
          (1 - (t.val['phi_c'] + t.val['phi_d']) / 100.0) * t.val['cb']
          + t.val['phi_c'] / 100.0 * t.val['cphi_c'] + t.val['phi_d'] / 100.0 * t.val['cphi_d'], 4,
          '1e-4/MPa',
          tex_label='saturated compressibility $c$',
          tex_src='Eq.~\\ref{eq:compressibility}',
          check=(lambda p: (1 - (p['phi_c'] + p['phi_d']) / 100.0) * p['cb']
                 + p['phi_c'] / 100.0 * p['cphi_c'] + p['phi_d'] / 100.0 * p['cphi_d'],
                 ['phi_c', 'phi_d', 'cb', 'cphi_c', 'cphi_d']))
    t.add('dc', 'compressibility change c/c_b - 1', 100.0 * (t.val['c'] / t.val['cb'] - 1.0), 2, '%',
          tex_label='compressibility change $c/c_b - 1$',
          check=(lambda p: 100.0 * (p['c'] / p['cb'] - 1.0), ['c', 'cb']))

    mu_w = float(mu_water_TP(T, P)) * 1000.0
    mu_b = brine_viscosity(T, P, m=m)
    f_p = pressure_factor(T, P, _composition(m=m))
    r_salt = (mu_b / mu_w) / f_p
    f_c = viscosity_correction_single('CH4', x_c, degf=T_degf)
    f_d = viscosity_correction_single('CO2', x_d, degf=T_degf)
    f_mix = viscosity_correction_mixed({'CH4': x_c, 'CO2': x_d}, degf=T_degf)
    mu_sat = gas_saturated_viscosity(T, P, gas_dict={'CH4': x_c, 'CO2': x_d},
                                     m=m)
    t.add('mu_w', 'water viscosity mu_w (IAPWS-2008)', mu_w, 4, 'mPa s',
          tex_label='water viscosity $\\mu_w$',
          tex_src='IAPWS-2008 \\cite{Huber2009}', rule=True)
    t.add('r_salt', 'ion-additive salt ratio r_salt', r_salt, 4,
          tex_label='ion-additive salt ratio $r_{salt}$',
          tex_src='Section~\\ref{subsec:visc_base}')
    t.add('f_p', 'Kestin pressure factor f_p', f_p, 4,
          tex_label='Kestin pressure factor $f_p$',
          tex_src='Section~\\ref{subsec:visc_base}')
    t.add('mu_b', 'gas-free brine mu_b = mu_w*r_salt*f_p', mu_b, 4, 'mPa s',
          tex_label='gas-free brine $\\mu_b = \\mu_w\\,r_{salt}\\,f_p$',
          tex_src='Eq.~\\ref{eq:visc_chain}',
          check=(lambda p: p['mu_w'] * p['r_salt'] * p['f_p'],
                 ['mu_w', 'r_salt', 'f_p']))
    t.add('f_ch4', 'CH4 factor 1 + A*exp(B/T)*x/(K+x)', f_c, 4,
          tex_label='CH$_4$ factor $1 + A e^{B/T} x/(K+x)$',
          tex_src='Eq.~\\ref{eq:ch4}')
    t.add('f_co2', 'CO2 factor', f_d, 4,
          tex_label='CO$_2$ factor', tex_src='Eq.~\\ref{eq:calabrese25}')
    t.add('f_mix', 'combined factor (multiplicative)', f_mix, 4,
          tex_label='combined factor (multiplicative)',
          tex_src='Section~\\ref{subsec:visc_mixtures}',
          check=(lambda p: p['f_ch4'] * p['f_co2'], ['f_ch4', 'f_co2']))
    t.add('mu', 'saturated viscosity', mu_sat, 4, 'mPa s',
          tex_label='saturated viscosity',
          tex_src='Eq.~\\ref{eq:visc_top}',
          check=(lambda p: p['mu_b'] * p['f_mix'], ['mu_b', 'f_mix']))
    t.resolve()
    t.walk(sums=[(('share_c', 'share_d'), 'dens')])
    return t


def main(write_tex=False):
    P_psia, T_degf = 3000.0, 175.0
    P = P_psia * PSIA_TO_MPA               # MPa
    T = (T_degf - 32.0) / 1.8 + 273.15     # K
    header('CONDITIONS (both examples)')
    show('pressure', f'{P_psia:.0f} psia = {P:.3f}', 'MPa')
    show('temperature', f'{T_degf:.0f} degF = {T:.2f}', 'K')

    cols, sal_dp = salinity_columns([1.0, 1000.0 * 0.05 / (0.95 * M_NACL)])
    header('SALINITY LEDGER, Example 1 brine (m = 1.0 mol/kg NaCl)')
    salinity_ledger(cols[0], sal_dp)
    header('SALINITY LEDGER, Example 2 brine (S = 5.0 wt% NaCl)')
    salinity_ledger(cols[1], sal_dp)

    header('EXAMPLE 1: pure CO2, m = 1.0 mol/kg NaCl')
    t1 = example1(T, P, T_degf)
    t1.emit()

    header('EXAMPLE 2: equimolar CH4/CO2 free gas, S = 5.0 wt% NaCl')
    t2 = example2(T, P, T_degf)
    t2.emit()
    show('  first-order sum (W_b denominator, does NOT close)',
         f'{first_order_sum(T, P):+.2f}', '%')

    print('\nWalk check passed: every derived row above reproduces itself '
          'from the printed values, and the two shares add to the total.')

    if write_tex:
        # closure and digit count are stated once, in the appendix text
        cap1 = ('Worked example 1: CO$_2$-saturated 1.0~mol~kg$^{-1}$ NaCl '
                'brine at 20.684~MPa, 352.59~K.')
        cap2 = ('Worked example 2: brine (5.0~wt\\% NaCl) saturated with an '
                'equimolar CH$_4$/CO$_2$ gas at 20.684~MPa, 352.59~K. The two '
                'per-gas shares are exact and add to the total densification.')
        for t, cap, lab, fn in ((t1, cap1, 'tab:ex1', 'tab_ex1.tex'),
                                (t2, cap2, 'tab:ex2', 'tab_ex2.tex')):
            path = os.path.join(MANUSCRIPT, fn)
            with open(path, 'w') as fh:
                fh.write(t.tex(cap, lab))
            print(f'  wrote {path}')
        path = os.path.join(MANUSCRIPT, 'tab_salinity.tex')
        with open(path, 'w') as fh:
            fh.write(salinity_tex(cols, sal_dp))
        print(f'  wrote {path}')


def first_order_sum(T, P):
    """The first-order criterion's per-gas terms summed, for Example 2. Kept
    to show what the W_b denominator costs: it over-counts, because it omits
    the volume the dissolved gas itself adds."""
    S = 0.05
    m = 1000.0 * S / ((1.0 - S) * M_NACL)
    mix = flash(P, T, S * 1e6, y_CO2=0.5, sg=0.5539)
    x_c, x_d = float(mix.x.get('CH4')), float(mix.x.get('CO2'))
    xt = x_c + x_d
    g = salt_fraction(m)
    rho1 = rho_brine(T, P, S) / 1000.0
    W_b = 1000.0 + m * M_NACL
    tot = 0.0
    for gname, x in (('CH4', x_c), ('CO2', x_d)):
        m_i = N_W * x / (1.0 - xt)
        V_i = V_phi(gname, T, P) * (1.0 + g)
        tot += m_i * (gas_mw(gname) - rho1 * V_i) / W_b
    return 100.0 * tot


if __name__ == '__main__':
    main(write_tex='--tex' in sys.argv)
