"""
IAPWS-2008 viscosity of ordinary water substance, implemented from scratch.

Reference:
    Huber, M.L. et al. (2009). "New International Formulation for the Viscosity
    of H2O." J. Phys. Chem. Ref. Data 38(2), 101-125.  (Papers/41)

What is implemented: the IAPWS recommendation for industrial use, Huber Eq. (36),

    mu = mu_0(Tbar) * mu_1(Tbar, rhobar) * mu*

which is Eq. (2) with the critical enhancement mu_2 set to 1. Coefficients are
Huber Table 2 (mu_0, 4 terms) and Table 3 (mu_1, 21 terms), both transcribed from
rendered page images rather than the PDF text layer.

What is NOT implemented, and why: the critical enhancement mu_2 (Eqs. 20-33)
contributes more than the 2% uncertainty of the correlation ONLY inside
645.91 K < T < 650.77 K and 245.8 < rho < 405.3 kg/m3 (Huber Eq. 34). That box
lies 200 K above this project's vouched accuracy ceiling of 450 K and entirely
outside IAPWS-IF97 Region 1 (T_max 623.15 K), so no state this code can reach
needs it. It is also unverifiable here: Huber Table 7 requires (d rhobar/d pbar)
from IAPWS-95 at both T and 1.5*T_c, which no dependency-free module on disk
supplies. Shipping unverified code was the worse option.

Also provides Huber Eq. (37) with Table 8 (Patek et al.), the simplified
correlation for liquid water at 0.1 MPa over 253.15-383.15 K.

No external dependencies beyond math, so this ports directly to Dart/Flutter.

Units: T in K, rho in kg/m3, P in MPa, viscosity returned in Pa s.
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


import math

# ============================================================================
# Reference constants - Huber Eq. (1), p. 104
# ============================================================================

T_STAR = 647.096      # K, critical temperature
RHO_STAR = 322.0      # kg/m3, critical density
P_STAR = 22.064       # MPa, critical pressure
MU_STAR = 1.0e-6      # Pa s

# Range over which the formulation is defined for the liquid, from Huber Sec. 4.
# The project uses it only inside IF97 Region 1, which is narrower.
T_MIN = 253.15
T_MAX = 1173.15

# ============================================================================
# Huber Table 2 (p. 109) - coefficients H_i in Eq. (11) for mu_0(Tbar)
# ============================================================================

_H = [
    1.677_52,
    2.204_62,
    0.636_656_4,
    -0.241_605,
]

# ============================================================================
# Huber Table 3 (p. 111) - coefficients H_ij in Eq. (12) for mu_1(Tbar, rhobar).
# 21 non-zero terms; every H_ij omitted from the table is identically zero.
# Stored as (i, j, H_ij) in the paper's own row order so the table can be
# checked against the page image line by line.
# ============================================================================

_HIJ = [
    (0, 0,  5.200_94e-1),
    (1, 0,  8.508_95e-2),
    (2, 0, -1.083_74),
    (3, 0, -2.895_55e-1),
    (0, 1,  2.225_31e-1),
    (1, 1,  9.991_15e-1),
    (2, 1,  1.887_97),
    (3, 1,  1.266_13),
    (5, 1,  1.205_73e-1),
    (0, 2, -2.813_78e-1),
    (1, 2, -9.068_51e-1),
    (2, 2, -7.724_79e-1),
    (3, 2, -4.898_37e-1),
    (4, 2, -2.570_40e-1),
    (0, 3,  1.619_13e-1),
    (1, 3,  2.573_99e-1),
    (0, 4, -3.253_72e-2),
    (3, 4,  6.984_52e-2),
    (4, 5,  8.721_02e-3),
    (3, 6, -4.356_73e-3),
    (5, 6, -5.932_64e-4),
]

# ============================================================================
# Huber Table 8 (p. 116) - coefficients a_i, b_i in Eq. (37), liquid water at
# 0.1 MPa (Patek et al.). Tbar here is T/(300 K), not T/T_c.
# ============================================================================

_PATEK_AB = [
    (280.68, -1.9),
    (511.45, -7.7),
    (61.131, -19.6),
    (0.459_03, -40.0),
]

PATEK_T_MIN = 253.15
PATEK_T_MAX = 383.15


# ============================================================================
# The correlation
# ============================================================================

def mu0_bar(T_K):
    """
    Dimensionless viscosity in the zero-density limit, Huber Eq. (11):

        mu_0 = 100 * sqrt(Tbar) / sum_{i=0}^{3} H_i / Tbar^i

    Function of temperature only.
    """
    Tbar = T_K / T_STAR
    denom = 0.0
    Tpow = 1.0
    for Hi in _H:
        denom += Hi / Tpow
        Tpow *= Tbar
    return 100.0 * math.sqrt(Tbar) / denom


def mu1_bar(T_K, rho):
    """
    Dimensionless residual (density-dependent) contribution, Huber Eq. (12):

        mu_1 = exp[ rhobar * sum_{i=0}^{5} (1/Tbar - 1)^i
                                 * sum_{j=0}^{6} H_ij (rhobar - 1)^j ]
    """
    Tbar = T_K / T_STAR
    rhobar = rho / RHO_STAR

    dT = 1.0 / Tbar - 1.0
    dR = rhobar - 1.0

    total = 0.0
    for i, j, Hij in _HIJ:
        total += Hij * (dT ** i) * (dR ** j)

    return math.exp(rhobar * total)


def mu_iapws2008(T_K, rho):
    """
    Viscosity of water from temperature and density, Huber Eq. (36)
    (the IAPWS recommendation for industrial use, mu_2 = 1).

    Parameters:
        T_K: temperature in K
        rho: density in kg/m3

    Returns:
        viscosity in Pa s
    """
    return mu0_bar(T_K) * mu1_bar(T_K, rho) * MU_STAR


def mu_water_TP(T_K, P_MPa):
    """
    Viscosity of pure liquid water from temperature and pressure, taking the
    density from IAPWS-IF97 Region 1.

    Huber Sec. 3.6 recommends exactly this pairing: IF97 supplies the density
    when the state is fixed by (T, P). Region 1 bounds (273.15-623.15 K, to
    100 MPa) therefore apply and are enforced by iapws_if97.

    Parameters:
        T_K: temperature in K
        P_MPa: pressure in MPa

    Returns:
        viscosity in Pa s
    """
    from brine_gas.iapws_if97 import rho_if97
    return mu_iapws2008(T_K, rho_if97(T_K, P_MPa))


def mu_liquid_0p1MPa(T_K):
    """
    Viscosity of liquid water at 0.1 MPa, Huber Eq. (37) with Table 8.

    A 4-term simplified correlation, stated uncertainty 1% in the stable liquid
    region, valid 253.15-383.15 K and explicitly not to be extrapolated. Used
    only as an independent cross-check on the full formulation at atmospheric
    pressure; the delivered path is mu_water_TP.

    Returns:
        viscosity in Pa s
    """
    if not PATEK_T_MIN <= T_K <= PATEK_T_MAX:
        raise ValueError(
            f"Eq. (37) is valid over {PATEK_T_MIN}-{PATEK_T_MAX} K and Huber "
            f"states it must not be extrapolated; got {T_K} K"
        )
    Ttilde = T_K / 300.0
    return sum(a * Ttilde ** b for a, b in _PATEK_AB) * MU_STAR


# ============================================================================
# Verification data, transcribed from rendered page images of Papers/41.
# These exist in the paper specifically so an implementation can be checked
# digit for digit; they are the acceptance test for this module.
# ============================================================================

# Huber Table 6 (p. 116): sample points for computer-program verification of
# Eq. (2) with mu_2 = 1 - i.e. exactly what mu_iapws2008 computes.
#   (T [K], rho [kg/m3], mu [microPa s])
TABLE6 = [
    (298.15,  998.0,  889.735_100),
    (298.15, 1200.0, 1437.649_467),
    (373.15, 1000.0,  307.883_622),
    (433.15,    1.0,   14.538_324),
    (433.15, 1000.0,  217.685_358),
    (873.15,    1.0,   32.619_287),
    (873.15,  100.0,   35.802_262),
    (873.15,  600.0,   77.430_195),
    (1173.15,   1.0,   44.217_245),
    (1173.15, 100.0,   47.640_433),
    (1173.15, 400.0,   64.154_608),
]

# Huber Table 7 (p. 116): sample points near the critical point, which DO carry
# the critical enhancement. Retained as a record of what this module does not
# compute - mu2_bar here is the factor omitted, peaking at 1.0919 on the
# critical isochore. Every entry lies far outside IF97 Region 1.
#   (T [K], rho [kg/m3], xi [nm], mu2_bar, mu [microPa s])
TABLE7 = [
    (647.35, 122.0,  0.309_247, 1.000_002_89, 25.520_677),
    (647.35, 222.0,  1.571_405, 1.003_751_20, 31.337_589),
    (647.35, 272.0,  5.266_522, 1.034_167_89, 36.228_143),
    (647.35, 322.0, 16.590_209, 1.091_904_40, 42.961_579),
    (647.35, 372.0,  5.603_768, 1.036_658_71, 45.688_204),
    (647.35, 422.0,  1.876_244, 1.005_963_32, 49.436_256),
]

# The value Huber Sec. 3.3 states the regression was constrained to reproduce:
# water at 20 degC and 0.101325 MPa. Independent of Tables 6 and 7.
ISO_REFERENCE_20C = 1001.6e-6  # Pa s


def verify_table6(tol_ppm=1.0, verbose=True):
    """
    Check mu_iapws2008 against every point of Huber Table 6.

    Returns (n_pass, n_total, max_relative_deviation_ppm).
    """
    worst = 0.0
    npass = 0
    for T, rho, mu_printed in TABLE6:
        mu_calc = mu_iapws2008(T, rho) * 1e6      # microPa s
        dev_ppm = abs(mu_calc - mu_printed) / mu_printed * 1e6
        worst = max(worst, dev_ppm)
        ok = dev_ppm <= tol_ppm
        npass += ok
        if verbose:
            print(f"  {'OK  ' if ok else 'FAIL'} T={T:8.2f} K  rho={rho:7.1f}  "
                  f"printed {mu_printed:12.6f}  calc {mu_calc:12.6f}  "
                  f"dev {dev_ppm:8.4f} ppm")
    return npass, len(TABLE6), worst


if __name__ == "__main__":
    print("=== IAPWS-2008 viscosity: verification against Huber (2009) ===\n")

    print("Table 6 - Eq. (2) with mu_2 = 1 (Papers/41 p. 116):")
    npass, ntot, worst = verify_table6()
    print(f"\n  {npass}/{ntot} points within 1 ppm; worst deviation "
          f"{worst:.4f} ppm\n")

    print("Constrained reference value, water at 20 degC / 0.101325 MPa:")
    mu20 = mu_water_TP(293.15, 0.101_325) * 1e6
    print(f"  full formulation via IF97 density: {mu20:.4f} microPa s "
          f"(stated 1001.6)")
    mu20p = mu_liquid_0p1MPa(293.15) * 1e6
    print(f"  Eq. (37) simplified correlation:   {mu20p:.4f} microPa s\n")

    print("Eq. (37) against the full formulation at 0.1 MPa:")
    worst_pct = 0.0
    for T in (273.15, 283.15, 293.15, 313.15, 333.15, 353.15, 373.15, 383.15):
        full = mu_water_TP(T, 0.1) * 1e6
        simp = mu_liquid_0p1MPa(T) * 1e6
        d = (simp - full) / full * 100.0
        worst_pct = max(worst_pct, abs(d))
        print(f"  T={T - 273.15:6.2f} degC   full {full:9.4f}   "
              f"Eq.37 {simp:9.4f}   {d:+7.4f}%")
    print(f"\n  worst {worst_pct:.4f}%, against Eq. (37)'s stated 1% uncertainty")
