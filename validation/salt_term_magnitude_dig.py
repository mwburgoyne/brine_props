"""Dig on the ~3x under-correction of the V_phi salt term: WHERE does the
steep slope live?

CONTEXT. Shipped is dV = -0.5914*m/(1+0.0416*m) (Tiepel dilatometry, 25 degC).
The paired Calabrese comparison says -1.736 +/- 0.120 cm3/mol per molal pooled
over 275-449 K. Every DIRECT measurement (-0.3 to -0.7 per unit, saturating)
sits at 25-51.5 degC; the steep number comes from an inversion POOLED across
temperature. Nobody has cut the paired slope BY temperature. If the slope is
~-0.6 near ambient and steepens with T, the direct data and the inversion
agree with each other and the shipped term is wrong in FORM (no T dependence),
not just scale. If the slope is flat at -1.7 across T, the inversion flatly
contradicts the dilatometry in the window where both exist.

Also cut by pressure (same logic), and re-score candidate laws on the CURRENT
route for both inversions, split by molality subset.
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

CAL = os.path.join(_RESULTS, 'calabrese_density_results.csv')
YAN = os.path.join(_RESULTS, 'yan2011_results.csv')

DVDX = -117.0          # measured composition slope of V_phi, cm3/mol per unit x
SALT_A, SALT_B = 0.5914, 0.0416   # shipped


def shipped(m):
    return -SALT_A * np.asarray(m, float) / (1.0 + SALT_B * np.asarray(m, float))


def pairs_df(df, m_lo, m_hi, vcol='V_imp', tcol='T', pcol='P', xcol='x'):
    """All (lo, hi) row pairs matched to <1.5 K and <1.5 MPa."""
    lo = df[np.isclose(df.m, m_lo, atol=0.01)]
    hi = df[np.isclose(df.m, m_hi, atol=0.01)]
    rows = []
    for _, r in lo.iterrows():
        cand = hi[(np.abs(hi[tcol] - r[tcol]) < 1.5) & (np.abs(hi[pcol] - r[pcol]) < 1.5)]
        if len(cand):
            c = cand.iloc[(cand[pcol] - r[pcol]).abs().argsort().iloc[0]]
            dv = c[vcol] - r[vcol]
            dv_corr = dv - DVDX * (c[xcol] - r[xcol])   # remove loading part
            rows.append(dict(T=r[tcol], P=r[pcol], dv=dv, dv_corr=dv_corr,
                             dx=c[xcol] - r[xcol]))
    out = pd.DataFrame(rows)
    out['slope'] = out.dv / (m_hi - m_lo)
    out['slope_corr'] = out.dv_corr / (m_hi - m_lo)
    return out


def binned(p, col, edges, labels, vcol='slope_corr'):
    print(f"    {'bin':<14} {'n':>4} {'raw slope':>12} {'loading-corr':>13} {'se':>6}")
    print('    ' + '-' * 54)
    for (a, b), lbl in zip(zip(edges[:-1], edges[1:]), labels):
        msk = (p[col] >= a) & (p[col] < b)
        if not msk.any():
            continue
        s, sc = p.slope[msk], p.slope_corr[msk]
        print(f'    {lbl:<14} {msk.sum():4d} {s.mean():+12.3f} {sc.mean():+13.3f} '
              f'{sc.std(ddof=1) / np.sqrt(msk.sum()):6.3f}')
    # linear trend of the corrected slope in the binning variable
    z = np.polyfit(p[col], p[vcol], 1)
    pred = np.polyval(z, p[col])
    r2 = 1 - np.sum((p[vcol] - pred) ** 2) / np.sum((p[vcol] - p[vcol].mean()) ** 2)
    print(f'    linear trend: d(slope)/d({col}) = {z[0]:+.4f} per unit {col}, '
          f'R2 = {r2:.3f}')
    return z


def main():
    cal = pd.read_csv(CAL)
    yan = pd.read_csv(YAN)

    print('=' * 78)
    print('1. CALABRESE PAIRED SLOPE (0.77 -> 2.50 m), CUT BY TEMPERATURE')
    print('=' * 78)
    p = pairs_df(cal, 0.77, 2.50)
    print(f'\n  {len(p)} pairs. Pooled: raw {p.slope.mean():+.3f}, '
          f'loading-corrected {p.slope_corr.mean():+.3f} '
          f'+/- {p.slope_corr.std(ddof=1) / np.sqrt(len(p)):.3f} cm3/mol/molal\n')
    zt = binned(p, 'T', [270, 320, 360, 400, 455],
                ['270-320 K', '320-360 K', '360-400 K', '400-455 K'])

    print('\n  The direct-measurement window is 298-325 K. Shipped initial slope')
    print(f'  is {shipped(0.01) / 0.01:+.3f}; the corrected paired slope in 270-320 K is above.')

    print('\n' + '=' * 78)
    print('2. SAME PAIRS, CUT BY PRESSURE')
    print('=' * 78 + '\n')
    binned(p, 'P', [0, 30, 60, 105], ['15-30 MPa', '30-60 MPa', '60-100 MPa'])

    print('\n' + '=' * 78)
    print('3. YAN PAIRED LEGS (Spivey-baselined), CUT BY TEMPERATURE')
    print('=' * 78)
    yy = yan.rename(columns={'T_K': 'T', 'P_MPa': 'P', 'x_exp': 'x',
                             'vphi_implied': 'V_imp'})
    for m_lo, m_hi in ((0.0, 1.0), (1.0, 5.0)):
        py = pairs_df(yy, m_lo, m_hi)
        print(f'\n  {m_lo:g} -> {m_hi:g} molal: {len(py)} pairs, pooled corrected '
              f'{py.slope_corr.mean():+.3f} +/- '
              f'{py.slope_corr.std(ddof=1) / np.sqrt(len(py)):.3f}')
        for T in sorted(py['T'].unique()):
            msk = np.isclose(py['T'], T)
            print(f'    {T:6.1f} K  n={msk.sum():3d}  raw {py.slope[msk].mean():+.3f}  '
                  f'corr {py.slope_corr[msk].mean():+.3f}')

    print('\n' + '=' * 78)
    print('4. CANDIDATE LAWS RE-SCORED ON THE CURRENT ROUTE')
    print('=' * 78)
    # model columns already CONTAIN the shipped shift; strip it, apply candidate
    laws = {
        'shipped (-0.6 sat)': shipped,
        'linear -1.736*m': lambda m: -1.736 * np.asarray(m, float),
        'shipped x3': lambda m: 3.0 * shipped(m),
    }
    print(f"\n  {'law':<20} " + ''.join(f'{v:>9}' for v in
          ['Cal 0.77', 'Cal 2.5', 'Yan m=0', 'Yan m=1', 'Yan m=5', 'pooled']))
    print('  ' + '-' * 76)
    for name, law in laws.items():
        cols = []
        c_after = cal.V_ply - shipped(cal.m) + law(cal.m)
        c_err = 100 * (c_after - cal.V_imp) / cal.V_imp
        for m in (0.77, 2.50):
            msk = np.isclose(cal.m, m)
            cols.append(np.sqrt(np.mean(c_err[msk] ** 2)))
        y_after = yan.vphi_model - shipped(yan.m) + law(yan.m)
        y_err = 100 * (y_after - yan.vphi_implied) / yan.vphi_implied
        for m in (0.0, 1.0, 5.0):
            msk = np.isclose(yan.m, m)
            cols.append(np.sqrt(np.mean(y_err[msk] ** 2)))
        pooled = np.sqrt((np.sum(c_err ** 2) + np.sum(y_err ** 2))
                         / (len(cal) + len(yan)))
        cols.append(pooled)
        print(f'  {name:<20} ' + ''.join(f'{v:9.2f}' for v in cols))
    print('\n  (RMS V_phi error %, per molality subset. Yan m=0 is salt-free and')
    print('   identical across laws by construction; it is the route error floor.)')

    print('\n' + '=' * 78)
    print('5. THE ABSOLUTE ANCHOR AT ~296 K')
    print('=' * 78)
    print("""
  V_inf(CO2, 298 K) = 33.4 cm3/mol is direct dilatometry, and the route
  reproduces it salt-free to <0.1%. A salting-out term can only LOWER V_phi,
  so implied-minus-expected residuals near 296 K cannot be positive under ANY
  candidate salt law. Residual = V_imp - (route(T,P,m) + dV/dx * x); the
  loading slope is Calabrese-derived (-117 +/- 34), so the residual is also
  shown with it off.""")
    from brine_gas.vphi_route import V_phi
    offs = {}
    print(f"\n    {'m':>5} {'n':>4} {'resid (loading on)':>19} "
          f"{'(loading off)':>14}")
    for m in (0.77, 2.50):
        sub = cal[(np.isclose(cal.m, m)) & (cal['T'] > 293) & (cal['T'] < 313)]
        r_on, r_off = [], []
        for _, r in sub.iterrows():
            vr = float(V_phi('CO2', r['T'], min(r.P, 100.0), 'auto', r.m))
            r_on.append(r.V_imp - (vr + DVDX * r.x))
            r_off.append(r.V_imp - vr)
        r_on, r_off = np.array(r_on), np.array(r_off)
        offs[m] = r_on.mean()
        print(f'    {m:5.2f} {len(sub):4d} {r_on.mean():+12.2f} +/- '
              f'{r_on.std(ddof=1) / np.sqrt(len(sub)):4.2f} '
              f'{r_off.mean():+9.2f} +/- {r_off.std(ddof=1) / np.sqrt(len(sub)):4.2f}')
    diff = offs[2.50] - offs[0.77]
    print(f"""
  Both residuals are POSITIVE: the implied volumes are inflated at BOTH
  molalities where the absolute value is best known. Their differential,
  {diff:+.2f} cm3/mol, is the same size as the "missing" slope
  ({(-2.042 - (shipped(2.5) - shipped(0.77)) / 1.73) * 1.73:+.2f}); the steep
  paired slope is the difference of two per-molality systematics.""")

    print('=' * 78)
    print('6. DE-BIASED PAIRED SLOPE: remove each molality\'s 296 K offset')
    print('=' * 78)
    p2 = p.copy()
    p2['slope'] = p2.slope - diff / 1.73
    p2['slope_corr'] = p2.slope_corr - diff / 1.73
    print(f'\n  pooled de-biased slope: {p2.slope_corr.mean():+.3f} +/- '
          f'{p2.slope_corr.std(ddof=1) / np.sqrt(len(p2)):.3f} cm3/mol/molal '
          f'(shipped effective over this interval: '
          f'{(shipped(2.5) - shipped(0.77)) / 1.73:+.3f})\n')
    binned(p2, 'T', [270, 320, 360, 400, 455],
           ['270-320 K', '320-360 K', '360-400 K', '400-455 K'])
    print("""
  CAVEAT: the de-bias assumes each molality's offset is T-independent; it is
  measured only at 293-313 K. What is NOT assumption: the offsets exist, are
  positive, and no salt term can explain their sign.""")

    print('=' * 78)
    print('7. ERROR-MODEL DISCRIMINATION: is the offset fixed, or does it scale?')
    print('=' * 78)
    print("""
  Calabrese measures five things: tube period (-> density, water+vacuum
  calibrated per state point), T, P, a CO2 filling pressure into a known
  volume (-> n_CO2 via Span-Wagner), and pump displacement (-> n_brine).
  x is DERIVED, u(x) = 0.0004 for all mixtures. Candidate systematics and
  their V_imp fingerprints:
    A  proportional dissolved-CO2 shortfall f:  dV = f*(M2/rho - V)
       -> flat in x, DECLINES ~55% from 296 to 449 K
    C  per-fill baseline bias / bubbles / fixed CO2 deficit:
       -> scales as (1-x)/x across the loading series (2-3x range)
    B  additive offset, mechanism unknown: flat in both.""")
    from validation.calabrese_density_validation import brine_molar_mass
    from brine_gas.plyasunov_model import gas_mw
    M2 = gas_mw('CO2')
    from brine_gas.vphi_route import V_phi as _V
    cal2 = cal.copy()
    vr, shA = [], []
    for _, r in cal2.iterrows():
        v = float(_V('CO2', r['T'], min(r.P, 100.0), 'auto', r.m))
        vr.append(v)
        shA.append(M2 / (r.rho_meas / 1000.0) - v)
    cal2['vr'] = vr
    cal2['shapeA'] = shA
    cal2['off'] = cal2.V_imp - (cal2.vr + DVDX * cal2.x)

    print('\n  Offset by loading x (all T) - the 1/x family predicts a 2-3x fall:')
    for m in (0.77, 2.50):
        s = cal2[np.isclose(cal2.m, m)]
        g = s.groupby('x')['off'].agg(['mean', 'sem', 'size'])
        line = '   '.join(f'x={x:.4f}: {r["mean"]:+5.2f}+/-{r["sem"]:.2f}'
                          for x, r in g.iterrows())
        print(f'    m={m:<5} {line}')
    print('  NOT 1/x at either molality (2.5 m would need 7.7:4.2:2.9);')
    print('  instead each FILL carries its own offset, +/-1-2 cm3/mol about the core.')

    print('\n  Offset by T vs model shapes fitted in the 293-313 K window:')
    for m in (0.77, 2.50):
        s = cal2[np.isclose(cal2.m, m)].copy()
        ref = s[(s['T'] > 293) & (s['T'] < 313)]
        f_A = (ref.off / ref.shapeA).mean()
        b_B = ref.off.mean()
        s['Tbin'] = pd.cut(s['T'], [270, 313, 365, 415, 455])
        g = s.groupby('Tbin', observed=True).agg(
            off=('off', 'mean'), predA=('shapeA', lambda v: f_A * v.mean()))
        g['predB'] = b_B
        rA = np.sqrt(np.mean((g.off - g.predA) ** 2))
        rB = np.sqrt(np.mean((g.off - g.predB) ** 2))
        print(f'    m={m}: RMS misfit  A(prop-shortfall) {rA:.2f}  '
              f'B(constant) {rB:.2f} cm3/mol '
              f'{"-> B wins" if rB < rA else "-> A wins"}')

    print('\n' + '=' * 78)
    print('8. THE SLOPE BAND UNDER ADMISSIBLE ERROR MODELS')
    print('=' * 78 + '\n')
    for model in ('B', 'A'):
        c = cal2.copy()
        for m in (0.77, 2.50):
            msk = np.isclose(c.m, m)
            ref = msk & (c['T'] > 293) & (c['T'] < 313)
            if model == 'B':
                c.loc[msk, 'V_deb'] = c.V_imp[msk] - c.off[ref].mean()
            else:
                f = (c.off[ref] / c.shapeA[ref]).mean()
                c.loc[msk, 'V_deb'] = c.V_imp[msk] - f * c.shapeA[msk]
        pm = pairs_df(c, 0.77, 2.50, vcol='V_deb')
        print(f'  model {model}: de-biased paired slope '
              f'{pm.slope_corr.mean():+.3f} +/- '
              f'{pm.slope_corr.std(ddof=1) / np.sqrt(len(pm)):.3f} cm3/mol/molal')
    print(f"""
  VERDICT. Calabrese, under any error model consistent with its own x- and
  T-trends, supports a salt slope of -0.6 to -0.8 cm3/mol/molal over
  0.77-2.5 m. Direct measurement brackets -0.1 to -0.7 (O'Sullivan NaCl
  shallow, Tiepel KCl steep). Shipped ({(shipped(2.5) - shipped(0.77)) / 1.73:+.2f} effective over this
  interval) sits inside both. Nothing supports -1.7; the "3x
  under-correction" was the difference of two per-fill systematics.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
