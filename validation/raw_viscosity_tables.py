"""Coordinate-parse the two RAW CO2 viscosity tables, plus a baseline check.

WHY THIS FILE EXISTS. The shared-VCVIS viscosity work previously took its CO2
leg from Calabrese eq 25 evaluated on a grid - a CORRELATION standing in for
data, carrying his functional form (in particular an exponential temperature
decay fitted to the CO2 + H2O system) into a fit that was then used to judge
whether a different functional form works. This module replaces that with the
measured numbers:

  * Calabrese et al. (2019) Table 8   - [xCO2 + (1-x)NaCl(aq)], m = 0.77 mol/kg,
    x = 0, 0.0122, 0.0159; 274-448 K; 1.4-100 MPa; u(eta) = 0.015*eta.
    Carries its OWN gas-free baseline (the x = 0 rows), so ratios need no model.
  * McBride-Wright et al. (2015) Table 7 - [(1-x)H2O + xCO2], x = 0.0086,
    0.0168, 0.0271; 294-449 K; 15-96 MPa; u(eta) = 0.007*eta. No x = 0 rows, so
    the baseline is Mao-Duan pure water, validated below against Calabrese's
    measured x = 0 brine.

WHY COORDINATES AND NOT READING ORDER. Both tables are three side-by-side
(p, eta) column pairs with a fresh triple of temperature headers every few rows,
and the blocks under one header have UNEQUAL row counts. A reading-order parse
slides values from a short block's neighbours into the wrong column - it puts
0.228 (the 423 K column) at 448 K, where the table prints 0.197. Every number is
therefore located by its x/y position, and spot-checked against a 400 dpi RENDER
of the page rather than against the text layer.

Both tables also sit side by side with another table on the same page, so words
are filtered to one page column BEFORE being grouped into rows.
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
PAPERS = os.path.join(os.path.dirname(HERE), 'Papers')

# (pdf, zero-based page, caption prefix, x-window of the page column)
CALABRESE_T8 = (os.path.join(
    PAPERS, '13_Calabrese_2019_CO2_brine_visc_dens_to449K_100MPa.pdf'),
    11, 'Table 8.', (0.0, 310.0))
MCBRIDEWRIGHT_T7 = (os.path.join(
    PAPERS, '21_McBrideWright_2015_CO2_water_visc_dens_274-449K_100MPa.pdf'),
    7, 'Table 7.', (310.0, 1e4))

U_ETA_CALABRESE = 0.015          # their footnote, relative
U_ETA_MCBRIDEWRIGHT = 0.007
M_CALABRESE = 0.77               # mol/kg NaCl


# ------------------------------------------------------------------ page words
def _words(page, xlo, xhi):
    return sorted([(w[0], w[1], w[4]) for w in page.get_text('words')
                   if xlo <= w[0] < xhi], key=lambda r: (round(r[1], 1), r[0]))


def _rows(words, ytol=2.0):
    out, cur, y0 = [], [], None
    for x, y, t in words:
        if y0 is None or abs(y - y0) <= ytol:
            cur.append((x, t))
            if y0 is None:
                y0 = y
        else:
            out.append((y0, sorted(cur)))
            cur, y0 = [(x, t)], y
    if cur:
        out.append((y0, sorted(cur)))
    return out


def _isnum(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def parse_table(spec):
    """Return [(x_CO2, T_K, p_MPa, eta_mPas), ...] for one (p, eta) table.

    In the public repository the tables are read from data/ (extracted by
    this parser in the working repo and checked against rendered pages);
    the PDF branch below is kept for the record and needs PyMuPDF.
    """
    _csv = {"Table 8.": "calabrese2019_table8_viscosity.csv",
            "Table 7.": "mcbridewright2015_table7_viscosity.csv"}.get(spec[2])
    if _csv is not None:
        import csv as _csvmod
        with open(_os.path.join(_DATA, _csv)) as _f:
            return [(float(r["x_CO2"]), float(r["T_K"]), float(r["p_MPa"]),
                     float(r["eta_mPas"])) for r in _csvmod.DictReader(_f)]
    import fitz
    pdf, page_no, caption, (xlo, xhi) = spec
    page = fitz.open(pdf)[page_no]
    rows = _rows(_words(page, xlo, xhi))

    i0 = i1 = None
    for i, (_, ws) in enumerate(rows):
        txt = ' '.join(t for _, t in ws)
        if i0 is None and txt.startswith(caption):
            i0 = i
        elif i0 is not None and txt.startswith('aStandard'):
            i1 = i
            break
    if i0 is None or i1 is None:
        raise RuntimeError(f'{caption} caption or footnote not found in {pdf}')
    body = rows[i0:i1]

    # Locate the three (p, eta) column pairs ONCE, from the 'p/MPa' header row.
    # Using fixed centres (rather than each temperature-header row) keeps the
    # assignment right when a header row carries only two temperatures.
    colx = None
    for _, ws in body:
        ps = [x for x, t in ws if t == 'p/MPa']
        if len(ps) >= 2:
            colx = sorted(ps)
            break
    if colx is None:
        raise RuntimeError(f'p/MPa header row not found in {caption}')

    records, xcur, tcur = [], None, []
    for _, ws in body:
        toks = list(ws)
        txt = ' '.join(t for _, t in toks)

        if txt.startswith('x = '):
            xcur = float(toks[2][1])
            continue
        if txt.startswith('T = '):
            tcur, i = [], 0
            while i < len(toks):
                if toks[i][1] == 'T':
                    tcur.append(float(toks[i + 2][1]))
                    i += 4
                else:
                    i += 1
            continue
        if xcur is None or not tcur:
            continue

        nums = [(x, float(t)) for x, t in toks if _isnum(t)]
        if len(nums) < 2 or len(nums) % 2:
            continue
        for j in range(0, len(nums), 2):
            xp, p = nums[j]
            _, eta = nums[j + 1]
            k = int(np.argmin([abs(xp - c) for c in colx]))
            if k < len(tcur):
                records.append((xcur, tcur[k], p, eta))
    return records


# ------------------------------------------------------- image-verified checks
SPOT_CHECKS = {
    'calabrese': [                       # read off a 400 dpi render of page 12
        (0.0000, 274.65, 1.4, 1.748),
        (0.0000, 398.18, 100.0, 0.273),
        (0.0000, 423.31, 100.0, 0.228),
        (0.0000, 448.27, 100.0, 0.197),  # reading order wrongly puts 0.228 here
        (0.0122, 274.64, 99.9, 1.813),
        (0.0122, 448.33, 30.3, 0.183),
        (0.0159, 274.63, 15.2, 1.961),
        (0.0159, 373.29, 100.0, 0.351),
    ],
    'mcbridewright': [                   # read off a 400 dpi render of page 8
        (0.0086, 294.30, 15.1, 1.013),
        (0.0086, 448.93, 96.3, 0.178),
        (0.0168, 373.03, 30.1, 0.2932),
        (0.0168, 448.29, 96.2, 0.179),
        (0.0271, 294.31, 30.3, 1.096),
        (0.0271, 448.71, 96.4, 0.179),
    ],
}


def check_spots(name, recs, verbose=True):
    ok = True
    for x, T, p, e in SPOT_CHECKS[name]:
        hit = [r for r in recs if r[0] == x and abs(r[1] - T) < 0.1
               and abs(r[2] - p) < 0.05]
        got = hit[0][3] if hit else None
        good = hit and abs(got - e) < 1e-9
        ok &= bool(good)
        if verbose:
            print(f"  {'OK ' if good else 'FAIL'} x={x:.4f} T={T:7.2f} p={p:5.1f}"
                  f'  printed {e}  parsed {got}')
    return ok


# --------------------------------------------------------------- baseline model
def water_visc(T_K, p_MPa, m_NaCl=0.0):
    """Mao-Duan brine viscosity, mPa.s (== cP). m_NaCl = 0 gives pure water."""
    from pyrestoolbox.brine import brine_props
    w = m_NaCl * 0.058443 / (1.0 + m_NaCl * 0.058443)      # NaCl mass fraction
    psi = min(p_MPa, 99.99) * 145.0377377   # IF97 Region 1 stops exactly at 100 MPa
    _, _, visc, _, _ = brine_props(p=psi, degf=(T_K - 273.15) * 1.8 + 32.0,
                                   wt=100.0 * w, ch4_sat=0)
    return float(visc)


# ------------------------------------------------------------------- assembly
def calabrese_ratios(recs):
    """ln(eta/eta0) against Calabrese's OWN measured x = 0 brine, interpolated in p."""
    base = {}
    for x, T, p, e in recs:
        if x == 0.0:
            base.setdefault(round(T), []).append((p, e))
    for k in base:
        base[k].sort()

    out = []
    for x, T, p, e in recs:
        if x == 0.0:
            continue
        key = min(base, key=lambda k: abs(k - T))
        if abs(key - T) > 2.0:
            continue
        ps = np.array([r[0] for r in base[key]])
        es = np.array([r[1] for r in base[key]])
        if p < ps.min() - 0.3 or p > ps.max() + 0.3:
            continue                            # never extrapolate the baseline
        e0 = float(np.interp(p, ps, es))
        out.append(dict(gas='CO2', T=T, p=p, x=x, eta=e, eta0=e0,
                        y=float(np.log(e / e0)), sigma=U_ETA_CALABRESE * np.sqrt(2),
                        src='Calabrese T8 (0.77 m NaCl)'))
    return out


def mcbridewright_ratios(recs):
    """ln(eta/eta_water) against Mao-Duan pure water (no x = 0 rows in the table)."""
    out = []
    for x, T, p, e in recs:
        e0 = water_visc(T, p, 0.0)
        out.append(dict(gas='CO2', T=T, p=p, x=x, eta=e, eta0=e0,
                        y=float(np.log(e / e0)),
                        sigma=np.sqrt(U_ETA_MCBRIDEWRIGHT ** 2 + 0.010 ** 2),
                        src='McBride-Wright T7 (pure water)'))
    return out


def load_all(verbose=False):
    """Both CO2 legs as one list of dicts."""
    c = parse_table(CALABRESE_T8)
    m = parse_table(MCBRIDEWRIGHT_T7)
    if not (check_spots('calabrese', c, verbose)
            and check_spots('mcbridewright', m, verbose)):
        raise RuntimeError('spot checks failed - extraction is not trustworthy')
    return calabrese_ratios(c), mcbridewright_ratios(m), c, m


# ------------------------------------------------------------------------ main
def main():
    print('=' * 78)
    print('RAW CO2 VISCOSITY TABLES - coordinate extraction and baseline check')
    print('=' * 78)

    print('\nCalabrese (2019) Table 8, spot checks vs a 400 dpi page render:')
    cal = parse_table(CALABRESE_T8)
    ok1 = check_spots('calabrese', cal)
    print('\nMcBride-Wright (2015) Table 7, spot checks vs a 400 dpi page render:')
    mbw = parse_table(MCBRIDEWRIGHT_T7)
    ok2 = check_spots('mcbridewright', mbw)
    if not (ok1 and ok2):
        print('\n  SPOT CHECKS FAILED - do not use this extraction')
        return 1

    # --- baseline validation: Mao-Duan against Calabrese's MEASURED x = 0 brine
    print('\n' + '-' * 78)
    print('BASELINE CHECK: Mao-Duan vs Calabrese measured gas-free 0.77 m brine')
    print('-' * 78)
    dev = []
    for x, T, p, e in cal:
        if x != 0.0:
            continue
        md = water_visc(T, p, M_CALABRESE)
        dev.append(100.0 * (md / e - 1.0))
    dev = np.array(dev)
    print(f'  n = {len(dev)}   mean {dev.mean():+.2f}%   '
          f'mean|dev| {np.abs(dev).mean():.2f}%   max|dev| {np.abs(dev).max():.2f}%')
    print('  (this is why a Mao-Duan pure-water baseline is acceptable for the')
    print('   McBride-Wright leg, which prints no x = 0 rows of its own)')

    ratios_c = calabrese_ratios(cal)
    ratios_m = mcbridewright_ratios(mbw)
    n_loaded = sum(1 for r in cal if r[0] > 0)
    print(f'\n  Calabrese:      {n_loaded} loaded points -> {len(ratios_c)} ratios '
          f'({n_loaded - len(ratios_c)} dropped outside the baseline pressure range)')
    print(f'  McBride-Wright: {len(ratios_m)} ratios')

    # --- the shape question: is the CO2 response linear in x, or saturating?
    print('\n' + '-' * 78)
    print('SHAPE TEST: ratio of ln(eta/eta0) between compositions at matched T')
    print('  linear in x  -> this ratio equals the ratio of x')
    print('  saturating   -> this ratio is much closer to 1')
    print('-' * 78)
    for label, rows, pair in (('Calabrese  ', ratios_c, (0.0122, 0.0159)),
                              ('McBride-Wr.', ratios_m, (0.0086, 0.0271))):
        print(f'\n  {label}  x = {pair[0]} vs {pair[1]}  (x ratio '
              f'{pair[1] / pair[0]:.2f})')
        for T in sorted({round(r['T']) for r in rows}):
            a = [r['y'] for r in rows if round(r['T']) == T and r['x'] == pair[0]]
            b = [r['y'] for r in rows if round(r['T']) == T and r['x'] == pair[1]]
            if a and b:
                print(f'    T = {T:4d} K   y {np.mean(a):7.4f} -> {np.mean(b):7.4f}'
                      f'   ratio {np.mean(b) / np.mean(a):6.2f}')

    out = os.path.join(_RESULTS, 'raw_co2_viscosity_ratios.csv')
    with open(out, 'w') as f:
        f.write('source,T_K,p_MPa,x_CO2,eta_mPas,eta0_mPas,ln_ratio,sigma\n')
        for r in ratios_c + ratios_m:
            f.write(f"{r['src']},{r['T']},{r['p']},{r['x']},{r['eta']},"
                    f"{r['eta0']:.5f},{r['y']:.6f},{r['sigma']:.4f}\n")
    print(f'\n  written {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
