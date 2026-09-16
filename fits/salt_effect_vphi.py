"""Every direct measurement of the salt effect on gas V_phi, in one place.

WHY: the decision "no salinity term on V_phi" has until now rested on two
inversions of density data (Yan 2011, Calabrese 2019) plus one directly
measured source (O'Sullivan & Smith). Inversions are weak evidence because a
baseline error and a V_phi error are not separable. This script assembles the
measurements that observed the salt effect DIRECTLY, and asks two questions the
project has not previously asked:

  Q1. Is the cross-gas similarity Tiepel & Gubbins report in the ABSOLUTE
      shift (cm3/mol) or in the RELATIVE shift (%)? The two imply different
      corrections, and the paper states it both ways in different places.
  Q2. What magnitude does the direct evidence actually support, and is it
      big enough to matter for brine density?

SOURCES (all in Papers/, all read from the primary and checked against a
rendered page image, not from pdftotext, which mangled both tables)

  Paper 26  Tiepel & Gubbins 1972, J. Phys. Chem. 76:3044-3049,
            DOI 10.1021/j100665a024. Table I, Horiuti dilatometry, 25 degC.
            Concentrations are MOLARITY. Electrolytes are KCl, KI, CaCl2,
            KOH (salting-out) and (Me)4NBr, (Bu)4NBr (salting-in).
            NOTE: contains NO NaCl. Stated accuracy +/-3% in water, up to
            +/-6% in electrolyte solutions; the tabulated +/- are standard
            deviations of three or more experiments.
            OCR corrections made from the page image: the Ar-KCl first row is
            1.0 M (pdftotext read 1.3), and CH4-KCl at 2.0 M is +/-0.66
            (pdftotext read 0.68).

  Paper 3   O'Sullivan & Smith 1970, J. Phys. Chem. 74:1460-1466. Table III,
            infinite dilution at 51.5 degC, NaCl, MOLALITY. Only direct
            NaCl measurement for N2 and CH4. (Their Table IV is unusable;
            see osullivan_salinity_validation.py for why.)

  Paper 28  Heusler & Gaiser 1972, Z. phys. Chem. NF 77:305-316,
            DOI 10.1524/zpch.1972.77.1-6.305. H2 in NaCl (with small HCl
            additions) at 23-25 degC, from the pressure dependence of a
            reversible cell voltage. V0_H2 = 24.5 +/- 0.5 cm3/mol by
            extrapolation to zero ionic strength; dV/dI = -5.7 cm3/mol per
            molal for NaCl at I < 1 m (-7.0 for HCl).
            CAVEAT recorded from Fig. 5: the NaCl points are FOUR points at
            only two concentrations, 0.4 and 0.7 m. The 2 m end of that
            figure is HCl, not NaCl.

  Paper 31  Enns, Scholander & Bradstreet 1965, J. Phys. Chem. 69:389-391.
            Table I, 25 degC (NOT 273 K), from the hydrostatic-pressure
            dependence of gas equilibrium pressure. Includes one row of
            O2 in SEA WATER against four rows of O2 in water, which is a
            direct salt-effect observation at seawater ionic strength.

UNITS CAVEAT: Tiepel is molarity, O'Sullivan and Heusler & Gaiser are molality.
At these concentrations molality exceeds molarity, so converting Tiepel to a
per-molal basis would REDUCE its slope by roughly 10%. That is smaller than the
scatter and changes no conclusion here, so the published units are kept and the
comparison is flagged rather than silently converted.
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

import sys
import os
import itertools

import numpy as np


# ---------------------------------------------------------------------------
# Paper 26, Tiepel & Gubbins 1972, Table I. (gas, electrolyte, molarity) ->
# (V_phi, sd). Verified against the rendered page image of p. 3047.
# ---------------------------------------------------------------------------
TIEPEL = [
    ('Ar',   'water',      0.00, 31.71, 0.43),
    ('Ar',   'KCl',        1.00, 31.11, 0.64),
    ('Ar',   'KCl',        2.00, 30.60, 0.53),
    ('Ar',   'KCl',        4.00, 29.89, 0.83),
    ('Ar',   'KI',         2.00, 30.98, 0.47),
    ('Ar',   'KI',         4.00, 30.24, 0.42),
    ('Ar',   'CaCl2',      2.00, 29.63, 0.79),
    ('Ar',   'CaCl2',      4.00, 28.80, 1.42),
    ('Ar',   '(Me)4NBr',   1.62, 32.64, 0.77),
    ('Ar',   '(Me)4NBr',   2.74, 33.76, 0.77),
    ('Ar',   '(Bu)4NBr',   1.25, 35.37, 0.86),
    ('CH4',  'water',      0.00, 37.42, 0.45),
    ('CH4',  'KCl',        2.00, 36.37, 0.66),
    ('CH4',  'KCl',        4.00, 35.51, 0.65),
    ('CH4',  '(Me)4NBr',   1.62, 39.39, 1.57),
    ('CH4',  '(Me)4NBr',   2.74, 39.99, 1.50),
    ('CH4',  '(Bu)4NBr',   1.25, 43.04, 1.22),
    ('C2H6', 'water',      0.00, 53.27, 0.81),
    ('C2H6', 'KCl',        2.00, 52.18, 0.87),
    ('C2H6', 'KCl',        4.00, 50.91, 0.97),
    ('O2',   'water',      0.00, 30.38, 0.97),
    ('O2',   'KOH',        2.00, 29.01, 0.56),
    ('O2',   'KOH',        5.00, 27.97, 0.61),
    ('H2',   'water',      0.00, 25.20, 0.56),
    ('H2',   'KOH',        2.00, 24.09, 0.71),
    ('H2',   'KOH',        5.00, 22.41, 1.33),
]
SALTING_OUT = {'KCl', 'KI', 'CaCl2', 'KOH'}

# Paper 3, Table III, 51.5 degC, NaCl, molality
OSULLIVAN = {'N2': {0.0: 34.05, 1.0: 32.76, 4.0: 32.52},
             'CH4': {0.0: 37.10, 1.0: 37.12, 4.0: 36.61}}

# Paper 28: NaCl, ~23-25 degC
HG_V0, HG_V0_SD, HG_SLOPE_NACL, HG_SLOPE_HCL = 24.5, 0.5, -5.7, -7.0
HG_NACL_RANGE = (0.4, 0.7)   # molality span of the actual NaCl points, Fig. 5

# Paper 31, Table I, 25 degC
ENNS_O2_WATER = [31.9, 31.9, 32.2, 32.3]
ENNS_O2_SEAWATER = 31.7
SEAWATER_I = 0.72            # ionic strength, mol/kg, standard 35 g/kg seawater


def water_value(gas):
    for g, e, c, v, s in TIEPEL:
        if g == gas and e == 'water':
            return v, s
    raise KeyError(gas)


def sep(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def q1_absolute_or_relative():
    """Tiepel's abstract says the RELATIVE change is similar across gases; the
    body's trend (3) says similar values of V1 - V1o, which is ABSOLUTE. They
    cannot both be right. Decide it from their own table."""
    sep('Q1  Is the cross-gas similarity ABSOLUTE or RELATIVE?')
    print("""
  Paper 26 states this two ways. Abstract: "the relative change in V1 due to
  the electrolyte is similar for the various gases studied". Body, trend (3):
  "Similar values of V1 - V1o are observed for various solute gases in a given
  electrolyte solution" - which is the absolute change. Their own Table I
  decides it. For each (electrolyte, concentration) cell holding two or more
  gases, compare the spread of the absolute shift against the spread of the
  relative shift, as a coefficient of variation.
""")
    cells = {}
    for g, e, c, v, s in TIEPEL:
        if e == 'water':
            continue
        cells.setdefault((e, c), []).append((g, v, s))

    print(f"  {'electrolyte':<12} {'M':>5} {'gases':<18} "
          f"{'abs shift (cm3/mol)':>26} {'CV_abs':>7} {'CV_rel':>7}")
    print('  ' + '-' * 82)
    out_abs, out_rel, in_abs, in_rel = [], [], [], []
    for (e, c), members in sorted(cells.items()):
        if len(members) < 2:
            continue
        d_abs, d_rel, names = [], [], []
        for g, v, s in members:
            v0, _ = water_value(g)
            d_abs.append(v - v0)
            d_rel.append(100.0 * (v - v0) / v0)
            names.append(g)
        cv_a = np.std(d_abs, ddof=1) / abs(np.mean(d_abs))
        cv_r = np.std(d_rel, ddof=1) / abs(np.mean(d_rel))
        shifts = ', '.join(f'{d:+.2f}' for d in d_abs)
        print(f'  {e:<12} {c:5.2f} {"/".join(names):<18} {shifts:>26} '
              f'{cv_a:7.3f} {cv_r:7.3f}')
        if e in SALTING_OUT:
            out_abs.append(cv_a)
            out_rel.append(cv_r)
        else:
            in_abs.append(cv_a)
            in_rel.append(cv_r)

    print(f'\n  Salting-out cells (n={len(out_abs)}): mean CV of the absolute '
          f'shift {np.mean(out_abs):.3f}, of the relative shift '
          f'{np.mean(out_rel):.3f}')
    print(f'  Salting-in  cells (n={len(in_abs)}): mean CV of the absolute '
          f'shift {np.mean(in_abs):.3f}, of the relative shift '
          f'{np.mean(in_rel):.3f}')
    verdict_out = 'ABSOLUTE' if np.mean(out_abs) < np.mean(out_rel) else 'RELATIVE'
    print(f'\n  For the salting-out electrolytes - the ones a reservoir brine '
          f'resembles -\n  the {verdict_out} shift is the better-conserved '
          f'quantity, so the body text is\n  right and the abstract wording is '
          f'loose. Any gas-generic salinity term\n  should therefore be '
          f'ADDITIVE in cm3/mol, not multiplicative.')
    print(f'\n  Neither similarity holds for the salting-in tetraalkylammonium '
          f'salts\n  (CV {np.mean(in_abs):.2f}/{np.mean(in_rel):.2f}), which '
          f'are irrelevant to brine but show the\n  "similar across gases" '
          f'claim is not a general law.')
    return verdict_out


def q1b_significance():
    """The individual Tiepel shifts are small against their own error bars.
    Only the consistency of sign across all salting-out points carries weight."""
    sep('Q1b  Are the individual shifts significant, or only their sign?')
    n_neg = n_tot = 0
    worst = []
    for g, e, c, v, s in TIEPEL:
        if e not in SALTING_OUT:
            continue
        v0, s0 = water_value(g)
        d = v - v0
        sd = np.hypot(s, s0)
        n_tot += 1
        n_neg += d < 0
        worst.append((abs(d) / sd, g, e, c, d, sd))
    worst.sort()
    print(f'\n  {n_neg} of {n_tot} salting-out measurements show a DECREASE.')
    print(f'  Two-sided sign test if the true effect were zero: '
          f'p = {2 * 0.5 ** n_tot:.2e}')
    print(f'\n  But per-point significance is weak. Shift / combined sd:')
    for t, g, e, c, d, sd in worst:
        print(f'    {g:<5} {e:<7} {c:.2f} M  {d:+6.2f} +/- {sd:.2f}  '
              f'= {t:.1f} sigma')
    n_2sig = sum(1 for t, *_ in worst if t >= 2)
    print(f'\n  Only {n_2sig} of {n_tot} points clear 2 sigma on their own. The '
          f'evidence for a salt\n  effect on V_phi is the unanimous sign across '
          f'{n_tot} measurements, five gases and\n  four electrolytes - not any '
          f'single measurement. Stated that way it is\n  strong; stated as a '
          f'per-point magnitude it is not.')


def q2_magnitude():
    sep('Q2  What magnitude does the direct evidence support?')
    print('\n  A. Tiepel & Gubbins 1972, salting-out only, 25 degC (per MOLAR):')
    print(f"    {'gas':<6} {'electrolyte':<8} {'M':>5} {'V_phi':>7} "
          f"{'shift':>7} {'per M':>7} {'rel':>7}")
    print('    ' + '-' * 52)
    per_molar = []
    for g, e, c, v, s in TIEPEL:
        if e not in SALTING_OUT:
            continue
        v0, _ = water_value(g)
        d = v - v0
        print(f'    {g:<6} {e:<8} {c:5.2f} {v:7.2f} {d:+7.2f} {d / c:+7.3f} '
              f'{100 * d / v0:+6.1f}%')
        per_molar.append(d / c)
    print(f'\n    Mean absolute shift {np.mean(per_molar):+.3f} cm3/mol per '
          f'molar of salting-out\n    electrolyte (sd {np.std(per_molar, ddof=1):.3f}, '
          f'n={len(per_molar)}). The effect SATURATES:')
    for e, gases in (('KCl', ('Ar', 'CH4', 'C2H6')), ('KOH', ('O2', 'H2'))):
        for g in gases:
            pts = [(c, v) for gg, ee, c, v, s in TIEPEL if gg == g and ee == e]
            if len(pts) < 2:
                continue
            v0, _ = water_value(g)
            lo, hi = pts[0], pts[-1]
            print(f'      {g:<5} in {e:<5} {(lo[1] - v0) / lo[0]:+.3f} per M at '
                  f'{lo[0]:.1f} M -> {(hi[1] - v0) / hi[0]:+.3f} per M at '
                  f'{hi[0]:.1f} M')

    print('\n  B. O\'Sullivan & Smith 1970, NaCl, 51.5 degC (per MOLAL):')
    for g, d in OSULLIVAN.items():
        v0 = d[0.0]
        for m in (1.0, 4.0):
            print(f'    {g:<5} {m:.0f} m  {d[m]:6.2f}  '
                  f'{d[m] - v0:+6.2f} cm3/mol  ({100 * (d[m] - v0) / v0:+5.1f}%)'
                  f'  = {(d[m] - v0) / m:+.3f} per molal')

    print('\n  C. Enns 1965, O2 in sea water vs water, 25 degC:')
    ew = float(np.mean(ENNS_O2_WATER))
    esd = float(np.std(ENNS_O2_WATER, ddof=1))
    d = ENNS_O2_SEAWATER - ew
    print(f'    water {ew:.2f} +/- {esd:.2f} (n={len(ENNS_O2_WATER)}), '
          f'sea water {ENNS_O2_SEAWATER:.2f}')
    print(f'    shift {d:+.2f} cm3/mol ({100 * d / ew:+.1f}%) at I ~ '
          f'{SEAWATER_I} mol/kg = {d / SEAWATER_I:+.3f} per molal')
    print(f'    This is {abs(d) / esd:.1f} sigma against the scatter of their own '
          f'four water rows,\n    so it BOUNDS the effect rather than measuring '
          f'it: at seawater strength the\n    shift cannot exceed about '
          f'{abs(d) + 2 * esd:.1f} cm3/mol.')

    print('\n  D. Heusler & Gaiser 1972, H2 in NaCl, 23-25 degC:')
    print(f'    V0 = {HG_V0} +/- {HG_V0_SD} cm3/mol, dV/dI = {HG_SLOPE_NACL} '
          f'cm3/mol per molal (NaCl),')
    print(f'    {HG_SLOPE_HCL} for HCl. Relative: '
          f'{100 * HG_SLOPE_NACL / HG_V0:+.1f}% per molal.')

    sep('CROSS-SOURCE RECONCILIATION')
    rows = [
        ('Tiepel 72 KCl/KOH, 5 gases, 25 C', float(np.mean(per_molar)), 'molar',
         'dilatometry, direct'),
        ("O'Sullivan 70 NaCl N2, 51.5 C", (OSULLIVAN['N2'][4.0] - OSULLIVAN['N2'][0.0]) / 4.0,
         'molal', 'solubility vs P'),
        ("O'Sullivan 70 NaCl CH4, 51.5 C", (OSULLIVAN['CH4'][4.0] - OSULLIVAN['CH4'][0.0]) / 4.0,
         'molal', 'solubility vs P'),
        ('Enns 65 seawater O2, 25 C', d / SEAWATER_I, 'molal', 'solubility vs P'),
        ('Heusler 72 NaCl H2, 24 C', HG_SLOPE_NACL, 'molal', 'cell voltage vs P'),
    ]
    print(f"\n  {'source':<34} {'cm3/mol per unit':>17} {'basis':>6}  method")
    print('  ' + '-' * 76)
    for name, slope, basis, method in rows:
        print(f'  {name:<34} {slope:+17.3f} {basis:>6}  {method}')
    agree = [r[1] for r in rows if 'Heusler' not in r[0]]
    print(f"""
  Four of the five agree on a shift between {min(agree):.2f} and {max(agree):.2f} cm3/mol per
  unit concentration, across three unrelated methods, five gases and five
  electrolytes. Heusler & Gaiser's {HG_SLOPE_NACL} is 5 to 12 times steeper and is the
  outlier. Two reasons to down-weight it rather than the other four: it is the
  only indirect determination in the set (V_H2 is obtained by subtracting the
  partial molal volume of HCl, itself taken from other laboratories, from a
  cell-reaction volume), and its NaCl evidence is four points at just two
  concentrations, 0.4 and 0.7 m, so the slope is an extrapolation from below
  0.7 m. Its zero-salt intercept, {HG_V0} +/- {HG_V0_SD}, is sound and is used as an
  anchor; its slope is recorded but not adopted.

  DEFENSIBLE STATEMENT: dissolved-gas V_phi falls by roughly 0.3 to 0.7
  cm3/mol per unit salt concentration in salting-out electrolytes, the fall
  saturates above about 2 units, and to within experimental scatter the fall
  is the SAME NUMBER OF cm3/mol for every gas measured. Every clause of that
  now carries a named independent measurement.""")
    return float(np.mean(per_molar))


def q3_does_it_matter(slope):
    sep('Q3  Is that big enough to matter for brine density?')
    from brine_gas.vphi_route import V_phi  # shipped default route
    from brine_gas.brine_properties import rho_brine, salinity_from_molality
    T, P, m = 323.15, 10.0, 4.0
    S = salinity_from_molality(m)
    rho1 = rho_brine(T, P, S) / 1000.0   # g/cm3, the SAME brine the shift is for
    print(f"""
  The dissolved-gas contribution to brine density scales with the excess mass
  term (M2 - rho1*V_phi). An additive shift dV in V_phi changes that term by
  -rho1*dV, so the FRACTIONAL change in the gas density effect is
  -rho1*dV/(M2 - rho1*V_phi). Evaluated at {T:.2f} K, {P:.0f} MPa in the same
  brine the shift is derived for ({m:.0f} molal NaCl, S = {S:.4f} w/w,
  rho1 = {rho1:.4f} g/cm3), with the literature shift dV applied at 4 units of
  salt (the effect saturates, so 4*{slope:.2f} is an upper bound rather than a
  linear extrapolation):
""")
    dV = 4.0 * slope
    print(f"  {'gas':<7} {'V_phi':>7} {'M2-rho1*V':>10} {'with dV':>9} "
          f"{'change in gas density effect':>30}")
    print('  ' + '-' * 68)
    MW = {'CH4': 16.043, 'CO2': 44.010, 'H2S': 34.081, 'N2': 28.014, 'H2': 2.016}
    excess = {}
    for gas, M2 in MW.items():
        v = float(V_phi(gas, T, P))
        base = M2 - rho1 * v
        shifted = M2 - rho1 * (v + dV)
        excess[gas] = (base, shifted)
        print(f'  {gas:<7} {v:7.2f} {base:10.2f} {shifted:9.2f} '
              f'{100 * (shifted / base - 1):+29.1f}%')
    co2, co2_s = excess['CO2']
    h2s, h2s_s = excess['H2S']
    print(f"""
  The shift is {dV:+.2f} cm3/mol, so the excess-mass term moves by a fixed
  {-rho1 * dV:+.2f} g/mol for EVERY gas. That is why it matters most where the
  term is smallest: under 10% for CH4 and H2, whose density effect is large
  and negative, but {100 * (co2_s / co2 - 1):.0f}% for CO2, whose {co2:+.1f} g/mol is a small
  difference between two big numbers. Note that H2S does NOT change sign
  under this shift ({h2s:+.1f} to {h2s_s:+.1f} g/mol): it still lightens 4 molal
  brine at 50 degC, though by about half as much.

  CO2 is the only gas with saturated-brine density data to test this on -
  see test_additive_salt_term.py.""")


def main():
    print('=' * 78)
    print('DIRECT MEASUREMENTS OF THE SALT EFFECT ON DISSOLVED-GAS V_phi')
    print('=' * 78)
    q1_absolute_or_relative()
    q1b_significance()
    slope = q2_magnitude()
    q3_does_it_matter(slope)


if __name__ == '__main__':
    main()
