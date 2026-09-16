"""
What the IAPWS-2008 water-viscosity leg actually changes.

The shipped brine viscosity is Mao-Duan (2009): a 10-coefficient polynomial in
T and pure-water density for the water leg, multiplied by a salt ratio
ur(T, m). This script measures the water leg alone against IAPWS-2008, so the
decision to swap it rests on a number rather than on the word "cleanliness".

Both legs are evaluated at the SAME IF97 Region 1 density, so what is compared
is the viscosity correlation, not a density difference leaking in.

Run: python3 iapws_viscosity_benchmark.py
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


import numpy as np

from brine_gas.iapws_if97 import rho_if97, p_sat_if97
from brine_gas.iapws_viscosity import mu_iapws2008, mu_water_TP, mu_liquid_0p1MPa

# Mao-Duan (2009) pure-water viscosity coefficients, Table 2 of the primary
# (Papers/54; d1 corrected from the McCain-lineage 2885310 on 2026-08-08).
_MAODUAN_D = [0, 2885317, -11072.577, -9.0834095, 0.030925651, -0.0000274071,
              -1928385.1, 5621.6046, 13.82725, -0.047609523, 0.000035545041]


def mu_maoduan_water(T_K, rho_gcm3):
    """Mao-Duan pure-water viscosity in Pa s. rho in g/cm3, as the fit expects."""
    d = _MAODUAN_D
    ln_mu = sum(d[i] * T_K ** (i - 3) for i in range(1, 6))
    ln_mu += sum(rho_gcm3 * d[i] * T_K ** (i - 8) for i in range(6, 11))
    return np.exp(ln_mu)


def grid(T_lo=273.15, T_hi=450.0, nT=25, P_list=(0.1, 1.0, 10.0, 30.0, 50.0, 100.0)):
    """Reservoir-relevant (T, P) points inside IF97 Region 1 and above P_sat."""
    pts = []
    for T in np.linspace(T_lo, T_hi, nT):
        psat = p_sat_if97(T)
        for P in P_list:
            if P >= max(psat, 0.0):
                pts.append((float(T), float(P)))
    return pts


def main():
    print("=" * 78)
    print("IAPWS-2008 vs Mao-Duan, pure-water viscosity leg")
    print("=" * 78)

    # ---------------------------------------------------------------- sanity
    print("\n1. Is the Mao-Duan leg reproduced correctly here?")
    print("   (cross-check against pyrestoolbox.brine, which ships it)")
    try:
        from pyrestoolbox.brine import brine_props
        for T, P in ((298.15, 0.101325), (373.15, 10.0), (423.15, 30.0)):
            rho = rho_if97(T, P)
            mine = mu_maoduan_water(T, rho / 1000.0) * 1e3        # cP
            _, _, theirs, _, _ = brine_props(
                p=P * 145.0377377, degf=(T - 273.15) * 1.8 + 32.0,
                wt=0.0, ch4_sat=0)
            print(f"   T={T:7.2f} K P={P:6.2f} MPa  local {mine:.6f} cP  "
                  f"pyrestoolbox {float(theirs):.6f} cP  "
                  f"diff {(mine - float(theirs)) / float(theirs) * 100:+.4f}%")
    except ImportError:
        print("   pyrestoolbox not importable; skipping cross-check")

    # ------------------------------------------------- the comparison itself
    print("\n2. Deviation over the reservoir envelope "
          "(273-450 K, 0.1-100 MPa, Region 1):")
    pts = grid()
    devs = []
    for T, P in pts:
        rho = rho_if97(T, P)
        iap = mu_iapws2008(T, rho)
        md = mu_maoduan_water(T, rho / 1000.0)
        devs.append((T, P, (md - iap) / iap * 100.0))

    d = np.array([r[2] for r in devs])
    print(f"   n = {len(d)} points")
    print(f"   Mao-Duan vs IAPWS-2008: mean |dev| {np.mean(np.abs(d)):.4f}%, "
          f"max |dev| {np.max(np.abs(d)):.4f}%, "
          f"mean signed {np.mean(d):+.4f}%")
    worst = max(devs, key=lambda r: abs(r[2]))
    print(f"   worst at T={worst[0]:.2f} K, P={worst[1]:.1f} MPa: "
          f"{worst[2]:+.4f}%")

    # ----------------------------------------------------- where it lives
    print("\n3. Where the difference sits (mean signed dev by isotherm):")
    print(f"   {'T (K)':>8} {'T (degC)':>9} {'n':>4} {'mean dev %':>12} "
          f"{'max dev %':>11}")
    for T in (273.15, 298.15, 323.15, 373.15, 423.15, 450.0):
        sub = [r[2] for r in devs if abs(r[0] - T) < 4.0]
        if not sub:
            continue
        print(f"   {T:8.2f} {T - 273.15:9.2f} {len(sub):4d} "
              f"{np.mean(sub):+12.4f} {max(sub, key=abs):+11.4f}")

    # ------------------------------------------------ independent yardstick
    print("\n4. Both legs against the ONE independent value in the source:")
    print("   ISO/IAPWS reference, water at 20 degC and 0.101325 MPa "
          "= 1001.6 microPa s")
    T, P = 293.15, 0.101325
    rho = rho_if97(T, P)
    print(f"   IAPWS-2008 : {mu_iapws2008(T, rho) * 1e6:9.4f} microPa s  "
          f"({(mu_iapws2008(T, rho) * 1e6 - 1001.6) / 1001.6 * 100:+.4f}%)")
    print(f"   Mao-Duan   : {mu_maoduan_water(T, rho / 1000.0) * 1e6:9.4f} "
          f"microPa s  "
          f"({(mu_maoduan_water(T, rho / 1000.0) * 1e6 - 1001.6) / 1001.6 * 100:+.4f}%)")

    # ------------------------------------------------- does it reach delivered?
    print("\n5. Does this reach the delivered brine viscosity?")
    print("   The salt ratio ur(T, m) multiplies the water leg, so a relative")
    print("   change in the water leg passes through unchanged. Scale:")
    print(f"     water leg swap          : {np.mean(np.abs(d)):.3f}% mean, "
          f"{np.max(np.abs(d)):.3f}% max")
    print("     whole chain vs Calabrese: 0.64% mean, 2.83% max")
    print("   CORRECTED 2026-07-26: that 0.64% is NOT the salt term. Backing")
    print("   the ratio out of each measured point puts the salt term within")
    print("   0.06% of the models above 300 K; the whole 2.83% is four points")
    print("   at 274.65 K. See salt_viscosity_benchmark.py and")
    print("   investigations-closed A0 - the salt term is not the constraint.")


if __name__ == "__main__":
    main()
