"""
The multi-salt leg against MEASUREMENT: Krumgalz & Millero (1982) Dead Sea waters.

Reference:
    Krumgalz, B.S. and Millero, F.J. (1982). "Physico-chemical study of Dead Sea
    waters. II. Density measurements and equation of state of Dead Sea waters at
    1 atm." Marine Chemistry 11(5), 477-492.
    DOI 10.1016/0304-4203(82)90012-3   (Papers/49)

This is the test the whole multi-salt front was built for. Everything before it
scored the leg against PHREEQC, which is a reproduction check; this scores it
against a densimeter.

WHY THIS DATASET IS THE RIGHT ONE
  - **25 artificial solutions of KNOWN composition**, prepared by weight from
    analytical-grade salts, given in Table III (p. 483) directly in
    **mol/1000 g H2O** - molality, no conversion needed.
  - Four salts at once: NaCl + KCl + MgCl2 + CaCl2. Ionic strength **8.293 to
    9.600**, far beyond anything else this project has tested.
  - Densities in Table IV (p. 484) to **seven significant figures**, flow
    densimeter, at 20/30/40 degC (solutions 1-16) and 25/30/35 degC (17-25).
  - **NO SULFATE.** The paper states: "The Br- and SO4(2-) anions in these
    artificial Dead Sea water patterns were replaced by the Cl- anions in
    gram-equivalent amounts." So the CaSO4 ion-pairing limit of the free-ion
    construction does NOT apply here - these are pure chloride brines, which is
    exactly the regime `salt_route` was verified in against PHREEQC.

CROSS-CHECK ON THE TRANSCRIPTION: solution 1 here (NaCl 1.8453, KCl 0.1831,
MgCl2 1.6380, CaCl2 0.4502) reproduces Krumgalz (2000) Table 10 row 1
(Na 1.8453, K 0.1831, Mg 1.6380, Ca 0.4502, Cl 6.2048) exactly - the chloride
sums to 6.2048. Two independent papers, same numbers.

UNIT NOTE: Table IV densities are printed in g/ml. The paper converts to g/cm3
with the historical litre factor, quoting rho_max = 0.999972 g/cm3 at 4 degC.
That is 28 ppm (0.0028%) and is applied here because the target precision is
finer than it.

Run: python3 deadsea_validation.py
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

from brine_gas.salt_route import brine_density, pairing_warning, ions_from_salts

P_ATM_MPA = 0.101325
ML_TO_CM3 = 0.999972          # paper's own conversion, 28 ppm

# Table III, p. 483 - mol/1000 g H2O
#   solution: (NaCl, KCl, MgCl2, CaCl2)
TABLE_III = {
    1:  (1.8453, 0.1831,  1.6380, 0.4502),
    2:  (1.8453, 0.1831,  2.0133, 0.4502),
    3:  (1.8453, 0.1831,  1.6380, 0.4802),
    4:  (1.8453, 0.1831,  2.0133, 0.4802),
    5:  (1.8994, 0.1831,  1.6380, 0.4502),
    6:  (1.8994, 0.1831,  2.0133, 0.4502),
    7:  (1.8994, 0.1831,  1.6380, 0.4802),
    8:  (1.8994, 0.1831,  2.0133, 0.4802),
    9:  (1.8453, 0.2200,  1.6380, 0.4502),
    10: (1.8453, 0.2200,  2.0133, 0.4502),
    11: (1.8453, 0.2200,  1.6380, 0.4802),
    12: (1.8453, 0.2200,  2.0133, 0.4802),
    13: (1.8994, 0.2200,  1.6380, 0.4502),
    14: (1.8994, 0.2200,  2.0133, 0.4502),
    15: (1.8994, 0.2200,  1.6380, 0.4802),
    16: (1.8994, 0.2200,  2.0133, 0.4802),
    17: (1.8724, 0.20155, 1.9195, 0.4652),
    18: (1.8724, 0.20155, 1.7319, 0.4652),
    19: (1.8724, 0.20155, 1.8257, 0.4727),
    20: (1.8724, 0.20155, 1.8257, 0.4577),
    21: (1.8859, 0.20155, 1.8257, 0.4652),
    22: (1.8589, 0.20155, 1.8257, 0.4652),
    23: (1.8724, 0.2108,  1.8257, 0.4652),
    24: (1.8724, 0.1923,  1.8257, 0.4652),
    25: (1.8724, 0.20155, 1.8257, 0.4652),
}

# Table IV, p. 484 - measured density in g/ml, {solution: {T_degC: rho}}
TABLE_IV = {
    1:  {20: 1.208082, 30: 1.203686, 40: 1.199027},
    2:  {20: 1.228241, 30: 1.223841, 40: 1.219174},
    3:  {20: 1.210041, 30: 1.205650, 40: 1.200948},
    4:  {20: 1.230180, 30: 1.225746, 40: 1.221061},
    5:  {20: 1.209545, 30: 1.205158, 40: 1.200452},
    6:  {20: 1.229762, 30: 1.225352, 40: 1.220621},
    7:  {20: 1.211512, 30: 1.207093, 40: 1.202391},
    8:  {20: 1.231662, 30: 1.227207, 40: 1.222494},
    9:  {20: 1.209203, 30: 1.204796, 40: 1.200121},
    10: {20: 1.229413, 30: 1.224982, 40: 1.220299},
    11: {20: 1.211161, 30: 1.206763, 40: 1.202056},
    12: {20: 1.231307, 30: 1.226846, 40: 1.222156},
    13: {20: 1.210689, 30: 1.206282, 40: 1.201583},
    14: {20: 1.230818, 30: 1.226374, 40: 1.221678},
    15: {20: 1.212668, 30: 1.208259, 40: 1.203530},
    16: {20: 1.232747, 30: 1.228315, 40: 1.223601},
    17: {25: 1.223330, 30: 1.221078, 35: 1.218811},
    18: {25: 1.213347, 30: 1.211088, 35: 1.208817},
    19: {25: 1.218808, 30: 1.216566, 35: 1.214296},
    20: {25: 1.217916, 30: 1.215675, 35: 1.213400},
    21: {25: 1.218754, 30: 1.216490, 35: 1.214229},
    22: {25: 1.218024, 30: 1.215773, 35: 1.213510},
    23: {25: 1.218717, 30: 1.216468, 35: 1.214182},
    24: {25: 1.218053, 30: 1.215791, 35: 1.213544},
    25: {25: 1.218383, 30: 1.216146, 35: 1.213859},
}


def salts_for(sol):
    n, k, mg, ca = TABLE_III[sol]
    return {'NaCl': n, 'KCl': k, 'MgCl2': mg, 'CaCl2': ca}


def main():
    print("=" * 78)
    print("Multi-salt leg vs MEASUREMENT - Krumgalz & Millero (1982) Dead Sea")
    print("=" * 78)

    # sanity: no composition should trip the ion-pairing guard (no sulfate)
    tripped = [s for s in TABLE_III
               if pairing_warning(ions_from_salts(salts_for(s)))]
    print(f"\n  ion-pairing guard trips on {len(tripped)} of {len(TABLE_III)} "
          f"compositions (expected 0 - the paper replaced SO4 and Br with Cl)")

    I = [sum(0.5 * m * z * z for m, z in
             ((ions_from_salts(salts_for(s))[i],
               {'Na+': 1, 'K+': 1, 'Mg+2': 2, 'Ca+2': 2, 'Cl-': -1}[i])
              for i in ions_from_salts(salts_for(s))))
         for s in TABLE_III]
    print(f"  ionic strength {min(I):.3f} to {max(I):.3f} mol/kg "
          f"(paper states 8.293 to 9.600)")

    rows = []
    for sol, byT in TABLE_IV.items():
        for T_C, rho_ml in byT.items():
            meas = rho_ml * ML_TO_CM3 * 1000.0          # g/ml -> kg/m3
            calc = brine_density(T_C + 273.15, P_ATM_MPA,
                                 salts=salts_for(sol))
            rows.append((sol, T_C, meas, calc, (calc - meas) / meas * 100.0))

    dev = np.array([r[4] for r in rows])
    print(f"\n  {len(rows)} measured densities, {len(TABLE_IV)} compositions, "
          f"20-40 degC, 1 atm\n")
    print(f"  {'':22} {'mean |dev|':>12} {'max |dev|':>12} {'bias':>10}")
    print(f"  {'Appelo multi-salt':22} {np.mean(np.abs(dev)):11.4f}% "
          f"{np.max(np.abs(dev)):11.4f}% {np.mean(dev):+9.4f}%")
    print(f"  {'Krumgalz (2000) own fit':22} {0.017:11.4f}% "
          f"{'':>12} {'':>10}  <- his Table 10, same waters")

    print(f"\n  by temperature:")
    for T_C in sorted({r[1] for r in rows}):
        d = np.array([r[4] for r in rows if r[1] == T_C])
        print(f"    {T_C:3d} degC  n={len(d):3d}  mean {np.mean(np.abs(d)):.4f}%  "
              f"bias {np.mean(d):+.4f}%")

    worst = max(rows, key=lambda r: abs(r[4]))
    print(f"\n  worst: solution {worst[0]} at {worst[1]} degC - "
          f"measured {worst[2]:.2f}, calc {worst[3]:.2f} ({worst[4]:+.4f}%)")

    print("\n  Densimeter measurement, not a model. This is the multi-salt")
    print("  claim standing or falling on data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
