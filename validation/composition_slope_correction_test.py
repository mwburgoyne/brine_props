"""Would folding the measured composition slopes into V_phi help? Tested: NO.

THE QUESTION (Mark, 2026-08-15). The model holds V_phi at its x = 0 value and
takes no composition trend. The measured slopes exist (-117 +/- 34 for CO2 from
the Calabrese 0.77 m inversion, -15.1 +/- 3.3 for H2S from Murphy & Gaines), so
why not ship V_phi(x) = V_inf + (dV/dx) * x, zero for the other gases?

THE ANSWER, from the held end-to-end data (run this script):

  1. Bolted on without a refit, the CO2 slope DEGRADES the Calabrese density
     validation: 0.77 m mean +0.205 -> +0.297 pp, max 0.394 -> 0.494; 2.50 m
     mean +0.029 -> +0.073 pp. The shipped V_inf already sits BELOW the implied
     volumes (-3.1 cm3/mol at 0.77 m above 300 K), and the neglected negative
     slope partially cancels that at finite loading; the correction breaks the
     cancellation.
  2. Refitting V_inf jointly with the slope consumes the only end-to-end
     validation (the slope IS a Calabrese quantity), and the offsets the refit
     demands are molality-inconsistent: +4.7 cm3/mol at 0.77 m, +2.0 at 2.50 m.
  3. Yan 2011, the only data reaching x ~ 0.03 (freshwater rows to x = 0.0274),
     shows no -117 signature: a persisting slope would put ~0.46% of density at
     its top loadings; the observed m = 0 errors are 0.158 pp max, and the
     residual-vs-x trend is small and positive (opposite sign).
  4. H2S: the shift is already fitted to the M&G Table I volumes at finite x
     (0.005-0.029), so adding -15 * x on top double-counts the very window the
     slope was measured in; it would need a joint refit of shift + slope for a
     term worth ~0.1% of density at x = 0.035.

So the measured slopes are DIAGNOSTIC BOUNDS (they mark the composition
ceiling), not calibrated model pieces. Recorded in investigations-closed.

Inputs: code/calabrese_density_results.csv (from calabrese_density_validation.py)
and code/yan2011_results.csv (from yan2011_validation.py); re-run those first if
missing. Self-checks: shipped-variant stats must reproduce the manuscript's
+0.205/+0.029 pp means, and the degradation must reproduce.
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

from validation.salt_term_magnitude_dig import DVDX          # -117.0, the one in-code copy
from brine_gas.brine_properties import M_NACL
from brine_gas.water_properties import MW_WATER

M_CO2 = 44.0095
CAL = os.path.join(_RESULTS, 'calabrese_density_results.csv')
YAN = os.path.join(_RESULTS, 'yan2011_results.csv')


def brine_molar_mass(m):
    return (1000.0 + m * M_NACL) / (1000.0 / MW_WATER + m)


def densification_err_pp(d, vphi_of_row):
    """Model minus measured densification, pp of brine density (Calabrese Eq. 18)."""
    out = []
    for _, r in d.iterrows():
        Mb = brine_molar_mass(r.m)
        rb = r.rho_base_fit / 1000.0
        V = vphi_of_row(r)
        rho_mod = (r.x * M_CO2 + (1 - r.x) * Mb) / (r.x * V + (1 - r.x) * Mb / rb)
        out.append(100.0 * (rho_mod / rb - 1.0) - r.f_exp_pct)
    return pd.Series(out, index=d.index)


def main():
    d = pd.read_csv(CAL)
    print('Calabrese loadings (their gas+brine x basis):')
    print(d.groupby('m').x.describe()[['count', 'min', 'mean', 'max']]
          .round(4).to_string())

    variants = {'shipped (V_inf, no slope)': lambda r: r.V_ply,
                'plus slope, no refit': lambda r: r.V_ply + DVDX * r.x}
    for m, g in d.groupby('m'):
        c = (g.V_imp - (g.V_ply + DVDX * g.x)).mean()
        print(f'joint-refit offset the slope would demand at m={m}: '
              f'{c:+.2f} cm3/mol')

    stats = {}
    print('\nDensification error (pp of brine density):')
    for name, fn in variants.items():
        e = densification_err_pp(d, fn)
        for m, g in d.groupby('m'):
            em = e[g.index]
            stats[(name, m)] = (em.mean(), em.abs().max())
            print(f'  {name:<28} m={m}: mean {em.mean():+.3f}  '
                  f'max {em.abs().max():.3f}')

    y = pd.read_csv(YAN)
    y0 = y[y.m == 0]
    sig = 100.0 * y0.x_exp * abs(DVDX) * y0.x_exp / 18.0
    print(f'\nYan m=0 (x to {y0.x_exp.max():.4f}): a persisting {DVDX:.0f} slope '
          f'predicts up to {sig.max():.2f}% of density;')
    print(f'  observed |model - measured| densification max '
          f'{(y0.dfrac_model_pp - y0.dfrac_meas_pp).abs().max():.3f} pp')

    # ---- self-checks ----
    ok = True

    def chk(label, cond):
        nonlocal ok
        print(('  PASS: ' if cond else '  FAIL: ') + label)
        ok &= cond

    print('\nSELF-CHECKS')
    chk('shipped stats reproduce manuscript (+0.152 / -0.008 pp means; 2026-09-06 CO2 refit)',
        abs(stats[('shipped (V_inf, no slope)', 0.77)][0] - 0.152) < 0.003
        and abs(stats[('shipped (V_inf, no slope)', 2.50)][0] + 0.008) < 0.003)
    chk('bolt-on slope degrades both molalities (mean error grows)',
        stats[('plus slope, no refit', 0.77)][0]
        > stats[('shipped (V_inf, no slope)', 0.77)][0]
        and stats[('plus slope, no refit', 2.50)][0]
        > stats[('shipped (V_inf, no slope)', 2.50)][0])
    chk('Yan m=0 shows no slope signature (max error << predicted)',
        (y0.dfrac_model_pp - y0.dfrac_meas_pp).abs().max() < 0.5 * sig.max())
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
