"""
NaCl brine density from the modular Pitzer construction, and its acceptance test.

Assembles the three verified pieces:
    water     : IAPWS-IF97 Region 1              (`iapws_if97`)
    A_V       : Bradley & Pitzer 1979            (`bradley_pitzer_dielectric`)
    salt      : Rogers & Pitzer 1982 Eq. (22)    (`rogers_pitzer_nacl`)

A_V and the U parameters now come from the same pair of papers that were fitted
together (Rogers & Pitzer cite Bradley & Pitzer as their reference [7] for
exactly this quantity), so the same-source rule is satisfied.

ACCEPTANCE CRITERION, set before building: match or beat Spivey against
Calabrese's 96 measured gas-free brine densities. Spivey scores 0.027% mean and
0.093% max against a stated measurement uncertainty of 0.07%, so it is already
agreeing with the data better than the data knows itself. **A tie is a pass** -
the win being bought here is multi-salt capability and auditability, not accuracy.

Run: python3 pitzer_brine_density.py
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

import numpy as np
import pandas as pd

from brine_gas.iapws_if97 import rho_if97, rho_and_kappa
from brine_gas.bradley_pitzer_dielectric import A_V, T_MIN as BP_T_MIN, T_MAX as BP_T_MAX
import brine_gas.rogers_pitzer_nacl as rp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, 'Papers', 'Calabrese_Table6.xlsx')

# Rogers & Pitzer resolved these two against their own Table A-1; applying them
# here so the module is usable without running their __main__ first.
rp.U11_DENOM_OFFSET = 227.0
rp.U19_SET_II = 3.0669940e-5


def brine_density(T_K, P_MPa, m_nacl, set_name=None):
    """
    Gas-free NaCl brine density in kg/m3, from the additive Pitzer construction.

    Parameters:
        T_K     : temperature in K (273.15-623.15, IF97 Region 1)
        P_MPa   : pressure in MPa (to 100)
        m_nacl  : NaCl molality, mol/kg water
        set_name: 'I' (Rogers & Pitzer low-temperature fit), 'II' (overall), or
                  None to pick automatically - Set I at or below 50 degC, which
                  is the range the paper fitted it for, Set II above.

    Returns:
        density in kg/m3
    """
    if not BP_T_MIN <= T_K <= BP_T_MAX:
        raise ValueError(
            f"T = {T_K} K is outside the Bradley & Pitzer range "
            f"({BP_T_MIN}-{BP_T_MAX} K), so A_V is undefined")

    P_bar = P_MPa * 10.0
    rho_w, kappa_per_MPa = rho_and_kappa(T_K, P_MPa)
    v_w = 1000.0 / rho_w                      # cm3/g
    beta_w = kappa_per_MPa / 10.0             # 1/bar

    if set_name is None:
        set_name = 'I' if T_K <= 323.15 else 'II'

    av = A_V(T_K, P_bar, rho_w=rho_w / 1000.0, beta_w=beta_w)
    v = rp.specific_volume(T_K, P_bar, m_nacl, v_w, av, set_name)
    return 1000.0 / v                          # cm3/g -> kg/m3


def _spivey_density(T_K, P_MPa, m_nacl):
    """Shipped Spivey brine density in kg/m3, via pyrestoolbox, for comparison."""
    from pyrestoolbox.brine import brine_props
    w = m_nacl * 0.058443 / (1.0 + m_nacl * 0.058443)
    _, sg, _, _, _ = brine_props(p=min(P_MPa, 99.99) * 145.0377377,
                                 degf=(T_K - 273.15) * 1.8 + 32.0,
                                 wt=100.0 * w, ch4_sat=0)
    # pyrestoolbox returns specific gravity on a 1000 kg/m3 basis (verified
    # empirically: its freshwater limit reproduces IF97 exactly, because the
    # shipped Spivey path is IF97 water x Spivey's salt RATIO).
    return float(sg) * 1000.0


def main():
    print("=" * 78)
    print("Modular Pitzer brine density - acceptance test")
    print("  IF97 water + Bradley & Pitzer A_V + Rogers & Pitzer salt")
    print("=" * 78)

    # ------------------------------------------------------- sanity anchors
    print("\n1. Pure water limit (m -> 0 must return IF97 exactly):")
    for T, P in ((298.15, 0.1), (373.15, 30.0), (423.15, 50.0)):
        got = brine_density(T, P, 0.0)
        ref = rho_if97(T, P)
        print(f"   T={T:7.2f} K P={P:5.1f} MPa: {got:9.4f} vs IF97 {ref:9.4f} "
              f"({(got - ref) / ref * 100:+.6f}%)")

    # ------------------------------------- against the measured Calabrese set
    print("\n2. Against Calabrese's measured CO2-FREE brine densities:")
    raw = pd.read_excel(XLSX)
    raw.columns = ['m', 'x', 'T', 'P', 'rho']
    base = raw[raw.x == 0.0].copy()
    print(f"   {len(base)} gas-free rows at m = "
          f"{sorted(base.m.unique())} mol/kg, "
          f"T {base['T'].min():.1f}-{base['T'].max():.1f} K, "
          f"P {base['P'].min():.1f}-{base['P'].max():.1f} MPa")

    rows = []
    for _, r in base.iterrows():
        try:
            pit = brine_density(r['T'], r['P'], r['m'])
        except ValueError:
            continue
        rows.append((r['m'], r['T'], r['P'], r['rho'], pit))

    arr = np.array([(a[3], a[4]) for a in rows])
    dev = (arr[:, 1] - arr[:, 0]) / arr[:, 0] * 100.0

    print(f"\n   {'':22} {'mean |dev|':>11} {'max |dev|':>11} {'bias':>9}")
    print(f"   {'Pitzer construction':22} {np.mean(np.abs(dev)):10.4f}% "
          f"{np.max(np.abs(dev)):10.4f}% {np.mean(dev):+8.4f}%")
    print(f"   {'Spivey (shipped)':22} {0.027:10.4f}% {0.093:10.4f}% "
          f"{'':>9}  <- benchmark from project-status")
    print(f"   {'Calabrese stated u':22} {0.07:10.4f}%")

    # ------------------------------------------------- where the error sits
    print("\n3. By molality:")
    print(f"   {'m':>6} {'n':>4} {'mean |dev|':>11} {'max |dev|':>11} {'bias':>9}")
    for m in sorted(set(a[0] for a in rows)):
        sub = np.array([((a[4] - a[3]) / a[3] * 100.0) for a in rows if a[0] == m])
        print(f"   {m:6.2f} {len(sub):4d} {np.mean(np.abs(sub)):10.4f}% "
              f"{np.max(np.abs(sub)):10.4f}% {np.mean(sub):+8.4f}%")

    print("\n4. By temperature band (the Set I / Set II handover is at 50 degC):")
    print(f"   {'band':>16} {'n':>4} {'mean |dev|':>11} {'max |dev|':>11}")
    for lo, hi, lab in ((270, 323.15, "<= 50 degC (I)"), (323.15, 400, "50-127 degC"),
                        (400, 460, "127-187 degC")):
        sub = np.array([((a[4] - a[3]) / a[3] * 100.0)
                        for a in rows if lo < a[1] <= hi])
        if len(sub):
            print(f"   {lab:>16} {len(sub):4d} {np.mean(np.abs(sub)):10.4f}% "
                  f"{np.max(np.abs(sub)):10.4f}%")

    # ------------------------------------------------------ the actual prize
    print("\n5. Past Spivey's validated range - 5 molal, where the Yan gap sits.")
    print("   No measured data on disk here, so this is a CAPABILITY check,")
    print("   not an accuracy check: does the construction stay physical?")
    print(f"   {'T degC':>8} {'m=0':>9} {'m=1':>9} {'m=3':>9} {'m=5':>9}")
    for T_C in (25, 100, 175):
        T = T_C + 273.15
        vals = [brine_density(T, 30.0, m) for m in (0.0, 1.0, 3.0, 5.0)]
        print(f"   {T_C:8d} " + " ".join(f"{v:9.2f}" for v in vals))


if __name__ == "__main__":
    main()
