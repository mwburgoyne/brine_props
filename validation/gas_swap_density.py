"""The event note's gas-swap bullet, generated and pinned.

The note's closing bullet: at 150 degC, swapping a depleted gas reservoir's
gas from pure CH4 to a 50:50 molar CH4/CO2 mix adds +0.34% (3.3 kg/m3) to the
saturated brine density at the contact, and sinking into the gas-free aquifer
needs the CO2 fraction to push the saturated brine past the gas-free line,
around 57% at these conditions. The manuscript has no counterpart, so this
script is the bullet's only test. It runs before the note renders.

State: 150 degC, 30 MPa, 1 mol/kg NaCl. Dissolved amounts from the
companion Soreide-Whitson flash (framework='default', pinned); the density
from the repo's own chain (garcia_mixing.density_mixed_gas over vphi_route)
and cross-checked against the library's delivered density, which implements
the same chain.

Run:  python3 code/gas_swap_density.py          prints the table
      python3 code/gas_swap_density.py --check  also asserts the note's
                                               printed numbers (exit 1 on
                                               mismatch)
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
import re
import sys

import numpy as np

from pyrestoolbox import brine as rtb_brine
from brine_gas.brine_properties import M_NACL, salinity_from_molality
from brine_gas.garcia_mixing import density_mixed_gas, density_change_pct  # noqa: F401
from brine_gas.brine_properties import rho_brine

DEGC, P_MPA, M_NACL_MOLAL = 150.0, 30.0, 1.0
SG_CH4 = 0.5539            # the hydrocarbon leg of the flash; App. C uses the same
NOTE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'whitson_event_note.md')


def flash(y_co2):
    ppm = M_NACL_MOLAL * M_NACL / (1000.0 + M_NACL_MOLAL * M_NACL) * 1e6
    return rtb_brine.SoreideWhitson(pres=P_MPA * 10.0, temp=DEGC, ppm=ppm,
                                    y_CO2=y_co2, sg=SG_CH4, metric=True,
                                    framework='default')


def saturated_change_pct(y_co2):
    """Density change of brine saturated with the y_co2 : (1 - y_co2) CH4/CO2
    gas, % of the gas-free density, from the repo chain; returns (pct, rho_sat,
    rho_free, x_dict, library_pct)."""
    s = flash(y_co2)
    T = DEGC + 273.15
    S = salinity_from_molality(M_NACL_MOLAL)
    x = {g: float(s.x[g]) for g in ('CH4', 'CO2') if float(s.x.get(g, 0.0)) > 0}
    rho1 = rho_brine(T, P_MPA, S)
    rho = density_mixed_gas(x, T, P_MPA, rho1=rho1, S=S)
    lib = 100.0 * (float(s.bDen[0]) / float(s.bDen[1]) - 1.0)
    return 100.0 * (rho / rho1 - 1.0), rho, rho1, x, lib


def crossover_y_co2():
    """CO2 mole fraction in the free gas at which saturated brine density
    equals the gas-free density (bisection; the change is monotone in y)."""
    lo, hi = 0.0, 1.0
    f_lo = saturated_change_pct(lo)[0]
    assert f_lo < 0 and saturated_change_pct(hi)[0] > 0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if saturated_change_pct(mid)[0] < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute():
    ch4 = saturated_change_pct(0.0)
    mix = saturated_change_pct(0.5)
    swap_pct = mix[0] - ch4[0]
    swap_kgm3 = mix[1] - ch4[1]
    return dict(ch4=ch4, mix=mix, swap_pct=swap_pct, swap_kgm3=swap_kgm3,
                y_star=crossover_y_co2())


def main(check=False):
    r = compute()
    print(f'State: {DEGC:.0f} degC, {P_MPA:.0f} MPa, {M_NACL_MOLAL:.0f} mol/kg NaCl; '
          f'gas-free brine {r["ch4"][2]:.2f} kg/m3')
    print(f'{"free gas":<18}{"x_CH4":>9}{"x_CO2":>9}{"rho_sat":>10}{"change %":>10}'
          f'{"library %":>11}')
    for lab, (pct, rho, _, x, lib) in (('pure CH4', r['ch4']),
                                       ('50:50 CH4/CO2', r['mix'])):
        print(f'{lab:<18}{x.get("CH4", 0):9.5f}{x.get("CO2", 0):9.5f}{rho:10.2f}'
              f'{pct:+10.3f}{lib:+11.3f}')
    print(f'\nswap CH4 -> 50:50: {r["swap_pct"]:+.3f}% ({r["swap_kgm3"]:+.2f} kg/m3)')
    print(f'sinking crossover: y_CO2 = {100 * r["y_star"]:.1f}% of the free gas')

    ok = True
    # repo chain and the delivered library agree
    for key in ('ch4', 'mix'):
        if abs(r[key][0] - r[key][4]) > 0.005:
            ok = False
            print(f'MISMATCH: repo chain vs library on {key}: '
                  f'{r[key][0]:+.4f} vs {r[key][4]:+.4f}')
    if check:
        note = open(NOTE).read()
        want = [f'adds +{r["swap_pct"]:.2f}% ({r["swap_kgm3"]:.1f} kg/m3)',
                f'around {100 * r["y_star"]:.0f}% here']
        for w in want:
            if w not in note:
                ok = False
                print(f'NOTE MISMATCH: expected "{w}" in whitson_event_note.md')
        # the bullet's opening claim is that the swap densifies
        if r['swap_pct'] <= 0:
            ok = False
        print('note check:', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(check='--check' in sys.argv) else 1)
