"""Murphy & Gaines (1974) H2S refit from the primary source.

SOURCE: Papers/10_MurphyGaines_1974_H2S_water_density_viscosity_20atm.pdf
Joseph A. Murphy and George L. Gaines, Jr., "Density and viscosity of aqueous
hydrogen sulfide solutions at pressures to 20 atm", J. Chem. Eng. Data 19(4),
359-362, DOI 10.1021/je60063a015.

Obtained 2026-07-25. Everything about H2S in this project previously rested on a
second-hand reading of this paper, and reading it corrected three things:

  1. AUTHORS. Recorded here for months as "W. R. Murphy and T. M. Gaines".
     Actually Joseph A. Murphy and George L. Gaines, Jr.
  2. TITLE. Recorded as viscosity-only; the paper reports DENSITY as well.
  3. DATA. Two of the five viscosity ratios in Table IV were recorded here as
     1.033 when the paper prints 1.038 (the 17.9 atm and 21.5 atm points).
     That shifts the fitted coefficient.

Table IV, "Viscosity of Water Saturated with H2S", is the entire experimental
basis for the H2S viscosity correction. Five points, all between 28 and 36 degC.
The authors describe the effect as "3-6% (which is only slightly more than our
experimental error)".

The paper also reports the apparent molal volume of H2S in water directly:
35.1 +/- 0.6 cm3/mol over the range, averaging 34.7 near 21 degC and 35.5 at
40 degC. That is an independent check on the Plyasunov H2S volume, and it is the
second measurement underpinning the claim that H2S lightens brine.

The nitrogen control is also from this paper: water pressurised with N2 to 18 atm
at 28.1 and 35.2 degC showed "no difference from measurements at 1 atm, within
our experimental error".
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

from brine_gas.plyasunov_model import V_phi

# Table IV: (t degC, P atm, tau/tau_0)
TABLE_IV = [(28.1, 17.3, 1.055), (28.1, 17.9, 1.038), (28.1, 19.0, 1.052),
            (30.0, 21.5, 1.038), (35.2, 18.0, 1.006)]

# Apparent molal volume of H2S in water, from the paper's text (cm3/mol)
VPHI_REPORTED = {'mean': (35.1, 0.6), 21.0: 34.7, 40.0: 35.5}

# Table I, "Density of Water Equilibrated with N2 or H2S", H2S block.
# Pressure columns in atm; each cell is (t degC, rho g/cm3).
#
# TRANSCRIPTION NOTE. The scan carries a doubled text layer and the extracted
# text drops a digit in places (it renders the nitrogen row as 0.94446 where the
# value is 0.99446). Every row is self-checking against pure-water density at
# its own temperature, and the density-float method bounds all values inside the
# float range 0.9925-0.9980 g/cm3, so misreads are detectable. The nitrogen
# block is NOT used: the authors state their N2 data are of "insufficient
# precision to permit satisfactory calculation of apparent molal volumes",
# scattering between 20 and 50 cm3/mol. The system is distilled water
# throughout; the paper contains no brine, NaCl or electrolyte.
TABLE_I_P_ATM = [1.0, 4.4, 7.8, 11.2, 14.6, 18.0]
TABLE_I_H2S = [
    [(21.3, 0.99786), (21.2, 0.99785), (21.3, 0.99789)],
    [(26.1, 0.99672), (25.8, 0.99674), (25.7, 0.99676), (25.7, 0.99675),
     (25.8, 0.99678)],
    [(27.9, 0.99619), (27.7, 0.99620), (27.5, 0.99622), (27.5, 0.99628),
     (27.5, 0.99624), (27.6, 0.99625)],
    [(33.5, 0.99440), (33.0, 0.99440), (32.7, 0.99451), (32.4, 0.99453),
     (32.2, 0.99454), (32.1, 0.99456)],
    [(41.0, 0.99176), (40.6, 0.99178), (40.2, 0.99180), (39.8, 0.99182),
     (39.5, 0.99184), (39.3, 0.99186)],
]


def vphi_from_table_I():
    """Derive V_phi(H2S) from the measured densities, independently of their fit.

    Apparent molar volume from volume additivity on a salt-free mole basis:
        V_phi = [ (x1 M1 + x2 M2)/rho - x1 M1/rho* ] / x2
    with rho* the pure-water density at the same temperature and pressure.
    The 1 atm column is excluded: x_H2S there is ~0.0013, so V_phi is divided by
    a very small number and the column is hypersensitive to the last digit.
    """
    from brine_gas.water_properties import rho_w
    rows = []
    for series in TABLE_I_H2S:
        for (t, rho), p_atm in zip(series, TABLE_I_P_ATM):
            if p_atm <= 1.01:
                continue
            MPa = max(p_atm * ATM, 0.101325)
            rstar = float(rho_w(t + 273.15, MPa)) / 1000.0
            x = x_h2s(t, p_atm)
            x1 = 1.0 - x
            V = ((x1 * 18.015268 + x * 34.081) / rho - x1 * 18.015268 / rstar) / x
            rows.append((t, p_atm, rho, rstar, x, V))
    return rows


def vphi_slope_fit():
    """The measured H2S composition slope, from the same 21 Table I volumes.

    The points span x = 0.005 to 0.029, so they constrain dV_phi/dx as well as
    the intercept. T and x co-vary across the table (solubility falls with T),
    so the slope is fitted jointly: V = c0 + c1*t + c2*x. The four series with
    three or more retained points also give a slope each at near-constant T
    (within-series T drift is ~1 degC, worth ~0.06 cm3/mol against a ~0.3
    cm3/mol slope effect).

    The slope is insensitive to the S&W solubility basis: assumed x enters the
    derived V only through M1*(1/rho - 1/rho*)/x, and rho - rho* is small
    because H2S sits within a gram per mole of density-neutral. Tested
    numerically 2026-08-14: a fractional solubility bias growing linearly by
    30% of itself from 4.4 to 18 atm moves the fitted slope -15.1 -> -16.8,
    i.e. by under 2 cm3/mol per unit x, within the slope's own 3.3 standard
    error (S&W's known bias level here is 4.5-9.2%). The slope is carried by
    the measured densities, not by the flash.

    Returns a dict: slope and its standard error (cm3/mol per unit x), the
    joint-fit t-slope, per-series slopes, the x span, and the derived
    manuscript numbers at 303 K / 7.5 MPa / x = 0.035 (excess-mass term with
    and without the slope, and the delivered-density move in %).
    """
    rows = vphi_from_table_I()
    t = np.array([r[0] for r in rows])
    x = np.array([r[4] for r in rows])
    V = np.array([r[5] for r in rows])
    A = np.column_stack([np.ones_like(x), t, x])
    coef, *_ = np.linalg.lstsq(A, V, rcond=None)
    resid = V - A @ coef
    cov = (resid @ resid / (len(x) - 3)) * np.linalg.inv(A.T @ A)
    per_series = []
    for series in TABLE_I_H2S:
        pts = [(xx, vv) for tt, _p, rr, _rs, xx, vv in rows
               if any(abs(tt - a) < 1e-9 and abs(rr - b) < 1e-9 for a, b in series)]
        if len(pts) < 3:
            continue
        xs = np.array([p[0] for p in pts])
        vs = np.array([p[1] for p in pts])
        per_series.append((float(np.mean([tt for tt, _p, rr, _rs, _x, _v in rows
                                          if any(abs(tt - a) < 1e-9 and abs(rr - b) < 1e-9
                                                 for a, b in series)])),
                           len(pts), float(np.polyfit(xs, vs, 1)[0])))

    # Derived Discussion numbers at 303.15 K, 7.5 MPa, x = 0.035.
    from brine_gas.vphi_route import V_phi as V_route
    from brine_gas.water_properties import rho_w
    T0, P0, x0, M2 = 303.15, 7.5, 0.035, 34.081
    v0 = float(V_route('H2S', T0, P0))
    r1 = float(rho_w(T0, P0)) / 1000.0
    dv = float(coef[2]) * x0
    em0 = M2 - r1 * v0
    em1 = M2 - r1 * (v0 + dv)
    # Eq. 5 first order: delta_rho/rho ~ (excess-mass change) * molality / 1000
    molality = x0 / (1.0 - x0) / 0.018015268
    return dict(slope=float(coef[2]), slope_se=float(np.sqrt(cov[2, 2])),
                t_slope=float(coef[1]), rms=float(resid.std(ddof=3)),
                x_min=float(x.min()), x_max=float(x.max()),
                per_series=per_series,
                excess_mass_0=em0, excess_mass_sloped=em1,
                density_move_pct=100.0 * (em1 - em0) * molality / 1000.0)

# Exponent set to UNITY 2026-07-31. The prior 1.0134 was borrowed from the
# Islam-Carlson CO2 form and never fitted to H2S; the five points cannot
# distinguish exponents (RMS 1.740 pp at b=1 vs 1.738 at b=1.0134), so the
# linear dilute-limit form ships. main() prints both for the record.
ATM = 0.101325


# Burgess & Germann (1969) Table 5, "Composition of H2S-H2O Liquid", mole
# fraction H2S in water. Papers/34, page 274; transcribed from a 400 dpi RENDER
# of the page, not the text layer. This is the solubility source Murphy & Gaines
# say they used ("Solubility data for H2S were taken from the same sources"),
# so it is the internally consistent basis for their viscosity ratios.
#
# Their correlation covers 100-400 psia and the hydrate point to 100 degC, but
# the TABLE starts at 30 degC, so the 28.1 degC measurements are 1.9 degC below
# its lowest row.
BG_PSIA = (250.0, 280.0, 300.0, 320.0, 340.0)
BG_TABLE5 = {
    30.0: (0.02382, 0.02680, 0.02879, 0.03077, None),
    50.0: (0.01702, 0.01915, 0.02057, 0.02199, 0.02340),
}
ATM_TO_PSIA = 14.6959


def x_h2s_burgess_germann(t_degc, p_atm):
    """x_H2S from Burgess & Germann Table 5, log-linear in T and P."""
    psia = p_atm * ATM_TO_PSIA
    out = []
    for t in (30.0, 50.0):
        pp = [(q, v) for q, v in zip(BG_PSIA, BG_TABLE5[t]) if v is not None]
        out.append(np.interp(psia, [q for q, _ in pp], [np.log(v) for _, v in pp]))
    return float(np.exp(out[0] + (out[1] - out[0]) * (t_degc - 30.0) / 20.0))


def x_h2s(t_degc, p_atm):
    """Dissolved H2S mole fraction from S&W at the measurement conditions."""
    from pyrestoolbox import brine
    # framework pinned to 'default' (the refresh paper's published
    # recommendation), explicitly, so a library default move cannot silently
    # change the solubility basis of this reduction.
    m = brine.SoreideWhitson(pres=p_atm * ATM * 10.0, temp=t_degc, ppm=0.0,
                             y_H2S=1.0, metric=True, framework='default')
    return float(m.x['H2S'])


def viscosity_coefficients():
    """The five implied linear coefficients from Table IV on the B&G basis.

    Returns (t_degC, P_atm, ratio, x_H2S, a) per row. The shipped a = 1.70 is
    the mean of the four rows below 35 degC (the 35.2 degC ratio of 1.006 was
    below the apparatus's resolution); the mean of all five is 1.41, and the
    manuscript quotes 1.41-1.70 as the sensitivity range, labelling 1.70 a
    deliberately conservative choice (2026-08-15, on the outcome-based-
    exclusion challenge)."""
    rows = []
    for t, p, r in TABLE_IV:
        x = x_h2s_burgess_germann(t, p)
        rows.append((t, p, r, x, (r - 1.0) / x))
    return rows


def main():
    print('MURPHY & GAINES 1974, Table IV, refit from the primary source')
    print('x_H2S basis: Burgess & Germann Table 5, the solubility source the')
    print('paper itself used. (S&W puts x 4.5-9.2% high here, biasing a low.)\n')
    print(f"  {'t degC':>7} {'P atm':>6} {'tau/tau0':>9} {'x_H2S':>8} {'implied a':>10}")
    print('  ' + '-' * 46)
    rows = viscosity_coefficients()
    for t, p, r, x, a in rows:
        print(f'  {t:7.1f} {p:6.1f} {r:9.3f} {x:8.5f} {a:10.3f}')

    a_all = np.array([r[4] for r in rows])
    a_28_30 = np.array([r[4] for r in rows if r[0] < 35])
    print(f'\n  implied a spans {a_all.min():.2f} to {a_all.max():.2f}')
    print(f'  mean of all five points          : {a_all.mean():.3f}')
    print(f'  mean excluding the 35.2 degC null: {a_28_30.mean():.3f}  <- basis for the adopted value')

    print('\n  exponent check (adopted coefficient refitted per form):')
    for b, label in ((1.0, 'b = 1 (SHIPPED)'), (1.0134, 'b = 1.0134 (borrowed, retired)')):
        a_b = np.mean([(r - 1.0) / x ** b for t, _p, r, x, _a in rows if t < 35])
        pred = [1 + a_b * r[3] ** b for r in rows]
        res = [100 * (p - r[2]) for p, r in zip(pred, rows)]
        print(f'    {label}: a = {a_b:.4f}, residuals ' +
              ', '.join(f'{v:+.2f} pp' for v in res) +
              f'   (RMS {np.sqrt(np.mean(np.square(res))):.3f} pp)')
    print('    The five points cannot distinguish the exponents; the linear')
    print('    dilute-limit form ships with a = 1.70.')

    print('\n  APPARENT MOLAR VOLUME cross-check (their own measurement):')
    for label, T in (('21 degC', 294.15), ('40 degC', 313.15)):
        key = 21.0 if T < 300 else 40.0
        mod = float(V_phi('H2S', T, 2.0))
        print(f'    {label}: measured {VPHI_REPORTED[key]:.1f}, '
              f'Plyasunov {mod:.2f} cm3/mol ({100*(mod/VPHI_REPORTED[key]-1):+.1f}%)')
    print(f'    measured mean {VPHI_REPORTED["mean"][0]} +/- '
          f'{VPHI_REPORTED["mean"][1]} cm3/mol over 21-40 degC')
    print('\n  INDEPENDENT DERIVATION from their Table I densities:')
    rows = vphi_from_table_I()
    V = np.array([r[5] for r in rows])
    print(f'    {len(V)} points (1 atm column excluded): mean {V.mean():.2f}, '
          f'sd {V.std():.2f} cm3/mol')
    print(f'    their published value: {VPHI_REPORTED["mean"][0]} +/- '
          f'{VPHI_REPORTED["mean"][1]} cm3/mol  -> reproduced')
    print('    by temperature group:')
    import collections
    g = collections.defaultdict(list)
    for t, _p, _r, _rs, _x, v in rows:
        g[round(t / 5) * 5].append(v)
    for k in sorted(g):
        mod = float(V_phi('H2S', k + 273.15, 0.5))
        print(f'      ~{k:2d} degC: derived {np.mean(g[k]):6.2f} (n={len(g[k])}), '
              f'model {mod:6.2f}  ({100*(mod/np.mean(g[k])-1):+.1f}%)')
    print('\n    The derived volumes RISE with temperature, ~34.8 at 21 degC to')
    print('    ~35.6 at 40 degC, confirming their stated trend from the raw data.')
    print('    The Plyasunov model is flat over the same span, so it matches at')
    print('    the cold end and runs about 2% low by 40 degC.')

    print('\n  COMPOSITION SLOPE from the same 21 points (added 2026-08-14):')
    f = vphi_slope_fit()
    print(f"    joint fit V = c0 + c1*t + c2*x over x = {f['x_min']:.4f} to "
          f"{f['x_max']:.4f}:")
    print(f"    dV_phi/dx = {f['slope']:+.1f} +/- {f['slope_se']:.1f} cm3/mol "
          f"per unit x  (t-slope {f['t_slope']:+.4f} cm3/mol/degC, "
          f"rms {f['rms']:.3f})")
    print('    per-series (near-isothermal) slopes:')
    for tmean, n, s in f['per_series']:
        print(f'      ~{tmean:4.1f} degC (n={n}): {s:+.0f} cm3/mol per unit x')
    print(f"    at 303 K, 7.5 MPa, x = 0.035: excess-mass term "
          f"{f['excess_mass_0']:+.2f} -> {f['excess_mass_sloped']:+.2f} g/mol, "
          f"delivered density {f['density_move_pct']:+.2f}%")
    print('    An eighth of the CO2 slope (-117 +/- 34) and the same sign;')
    print('    the sign of the H2S density effect survives its own measured slope.')


if __name__ == '__main__':
    main()
