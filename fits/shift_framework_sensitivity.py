"""How specific are the volume shifts to the equation-of-state parameter set?

Refits the one-per-gas shift on the shipped calibration set (fit_pr_vshift)
with the companion framework's 'mc3' binary interaction set in place of the
pinned 'default' set, everything else unchanged. Backs the Section 2.3
statement that within the Soreide-Whitson family the shifts move by at most
1.6% (under 0.05 cm3/mol at 298 K), whereas another EOS needs its own refit.
--check asserts the quoted numbers.
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
from scipy.optimize import minimize_scalar

import brine_gas.pr_vphi_model as PR
import fits.fit_pr_vshift as F

GASES = ('CH4', 'CO2', 'H2S', 'N2', 'H2', 'C2H6')


def refit(framework):
    orig = PR._SW.get_kij_aq
    PR._SW.get_kij_aq = lambda g, T, m, framework_=framework, **_: orig(g, T, m, framework=framework_)
    try:
        cal = F.calibration_set()
        out = {}
        for g in GASES:
            pts = cal[g]
            o = minimize_scalar(lambda s: F._score(pts, lambda T, P: PR.V2_inf(g, T, P, s=s))['rms'],
                                bounds=(-1.0, 1.0), method='bounded', options={'xatol': 1e-9})
            out[g] = float(o.x)
        return out
    finally:
        PR._SW.get_kij_aq = orig


def main():
    d, m = refit('default'), refit('mc3')
    rel, dv = {}, {}
    for g in GASES:
        rel[g] = 100.0 * (m[g] / d[g] - 1.0)
        dv[g] = (m[g] - d[g]) * PR._ab(g, 298.15)[1]
        print(f'  {g:<5} default {d[g]:+.6f}  mc3 {m[g]:+.6f}  {rel[g]:+.2f}%  {dv[g]:+.3f} cm3/mol at 298 K')
    if '--check' in sys.argv:
        assert all(abs(d[g] - PR.VSHIFT[g]) < 1e-6 for g in GASES), 'default refit no longer reproduces VSHIFT'
        assert abs(max(abs(v) for v in rel.values()) - 1.61) < 0.01, rel
        assert max(abs(v) for v in dv.values()) < 0.05, dv
        print('  pins hold')


if __name__ == '__main__':
    main()
