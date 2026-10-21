"""H2S solubility near the Murphy & Gaines conditions: Burgess & Germann (1969)
and the S&W companion flash against Lee & Mather (1977) and Selleck et al.
(1952), both as compiled in IUPAC SDS-32 (Fogg & Young 1988), pages 9 and 15,
text layer checked against rendered page images 2026-10-21.
Promoted to code/ 2026-10-21 (FPE R1.4): backs the manuscript statement that
the evaluated solubility data favour the companion model over Burgess & Germann
at the Murphy & Gaines states. 303.2 K values re-checked against SDS-32 PDF
p. 31 the same day. --check asserts the quoted numbers."""

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

import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
import fits.murphy_gaines_h2s_refit as m

ATM = 101.325  # kPa
# (total P / kPa, x_H2S)
LM_303 = [(1659.5, .02453), (1758.9, .02582), (1900.5, .02806), (1945.3, .02873), (1955.9, .02833),
          (1997.0, .02956), (2034.3, .02993), (2059.1, .03054), (2079.1, .03063), (2106.7, .03077),
          (2169.0, .03195), (2262.1, .03275), (2265.5, .03292)]
LM_313 = [(1469.9, .01810), (1534.6, .01950), (1583.1, .01979), (1609.1, .01941), (1636.8, .01975),
          (1671.4, .02013), (1671.5, .02061), (1707.2, .02077), (1788.3, .02126), (1892.8, .02282),
          (1934.1, .02334), (1981.9, .02347), (1983.1, .02398), (2052.2, .02408), (2059.9, .02498),
          (2061.3, .02517), (2100.7, .02526), (2100.8, .02546), (2128.3, .02529), (2227.0, .02612),
          (2417.4, .02826), (2452.9, .02822), (2458.9, .02926), (2479.4, .02873), (2561.6, .02983)]
SELLECK_311 = [(1724.0, .0207), (2068.0, .0250)]  # 250 and 300 psia at 310.93 K


def score(name, T, pts, quiet=False):
    sw, bg = [], []
    for P, x in pts:
        pa = P / ATM
        sw.append(m.x_h2s(T - 273.15, pa) / x - 1)
        if 250 <= pa * 14.6959 <= 340:  # inside B&G Table 5 pressure columns
            bg.append(m.x_h2s_burgess_germann(T - 273.15, pa) / x - 1)
    if not quiet:
        print(f"{name}: S&W {100*np.mean(sw):+.1f}% mean (n={len(sw)}, {100*min(sw):+.1f} to {100*max(sw):+.1f}); "
          f"B&G {100*np.mean(bg):+.1f}% mean (n={len(bg)}, {100*min(bg):+.1f} to {100*max(bg):+.1f})")
    return np.array(sw), np.array(bg)


def at_mg_points(quiet=False):
    out = []
    f303 = np.polyfit(*zip(*LM_303), 1)
    f313 = np.polyfit(*zip(*LM_313), 1)
    for t, pa, _ratio in m.TABLE_IV:
        P, T = pa * ATM, t + 273.15
        x = np.exp(np.log(np.polyval(f303, P))
                   + (np.log(np.polyval(f313, P)) - np.log(np.polyval(f303, P))) * (T - 303.2) / 10.0)
        xb, xs = m.x_h2s_burgess_germann(t, pa), m.x_h2s(t, pa)
        out.append((xb / x - 1, xs / x - 1))
        if not quiet:
            print(f"M&G {t} degC {pa} atm: Lee-Mather interp {x:.5f} | B&G {xb:.5f} ({100*(xb/x-1):+.1f}%) "
                  f"| S&W {xs:.5f} ({100*(xs/x-1):+.1f}%)")
    return np.array(out)


if __name__ == '__main__':
    score('Lee & Mather 303.2 K', 303.2, LM_303)
    score('Lee & Mather 313.2 K', 313.2, LM_313)
    score('Selleck 310.93 K', 310.93, SELLECK_311)
    at_mg_points()
    for T in (298.15, 301.25, 303.15, 308.35):
        x = np.exp(-24.912 + 3477.1 / T + 0.3993 * np.log(T) + 0.015700 * T)
        print(f"IUPAC SDS-32 x_H2S at 1.013 bar partial pressure, {T} K: {x:.6f}")
    if '--check' in sys.argv:
        sw, bg = score('', 303.2, LM_303, quiet=True)
        mg = at_mg_points(quiet=True)
        assert abs(100 * sw.mean() - 0.4) < 0.05 and abs(100 * bg.mean() + 5.8) < 0.05, (sw.mean(), bg.mean())
        assert abs(100 * mg[:, 0].min() + 7.1) < 0.05 and abs(100 * mg[:, 0].max() + 4.9) < 0.05, mg[:, 0]
        assert abs(100 * mg[:, 1].min() + 1.8) < 0.05 and abs(100 * mg[:, 1].max() - 1.7) < 0.05, mg[:, 1]
        print('  pins hold')
