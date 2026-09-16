"""
Rogers & Pitzer (1982) volumetric equation of state for NaCl(aq).

Reference:
    Rogers, P.S.Z. and Pitzer, K.S. (1982). "Volumetric Properties of Aqueous
    Sodium Chloride Solutions." J. Phys. Chem. Ref. Data 11(1), 15-81.
    (Papers/38)

This is the salt leg of the modular brine-volume construction: a Pitzer
ion-interaction treatment of the apparent molal volume, valid 0-300 degC,
1-1000 bar, to 5.5 molal. It is wanted for two things Spivey cannot do -
run past 2.5 molal with support, and sit in an additive construction that
extends to other salts.

MODEL (paper's own equation numbers)

    Eq. (22), specific volume of the solution:

      v = (m / (1000 + m*M2)) * { V(m1)/m1 + (1000/m - Mw*Y) * v_w
                                  + nu*|zM zX| * A_V * [h(I) - h(I1)]
                                  + 2*nuM*nuX*R*T * [ m*BV(I) - m1*BV(I1)
                                                      + (nuM*zM)*(m^2 - m1^2)*CV ] }

    Eq. (13):  h(I) = ln(1 + b*sqrt(I)) / (2b),   b = 1.2 (kg/mol)^1/2

    Rogers & Pitzer set (d beta1/dP)_T = 0, so BV_MX = (d beta0/dP)_T carries NO
    ionic-strength dependence (Sec. 4.2). Eq. (14)'s beta1 term therefore drops
    and BV(I) = BV(I1); the bracket collapses to (m - m1)*BV. That is a
    structural simplification of the paper's own making, not an approximation
    added here.

    Eqs. (23)-(25) give V(m1), BV_MX and 2*CV_MX as functions of T (K) and
    P (bar), with P0 = 1.01325 bar. Coefficients U1..U28, Table 2, come in TWO
    sets: Set I (low-temperature fit, 0-50 degC) and Set II (overall fit).

REFERENCE CONCENTRATION
    Y = 10 water molecules per mole of salt, so m1 = 1000/(Y*Mw) = 5.550825
    molal - deliberately at the top of the data range, because V(m1) varies far
    less with temperature than the infinite-dilution volume does.

TWO TRANSCRIPTION AMBIGUITIES IN THE SOURCE, both resolved numerically against
the paper's own Table A-1 by resolve_ambiguities() rather than guessed:
    1. Eq. (24) prints U11/(T - 277) while every other factor in Eqs. (24)-(25)
       is (T - 227), and the prose beneath names only 1/(T - 227) and
       1/(680 - T). Verified at 600 dpi: the page really does print 277.
    2. U19 (Set II) has a broken exponent glyph, readable as 10^-5 or 10^-3.

A_V, the Debye-Huckel limiting slope for volume, is NOT computed here. Rogers &
Pitzer take it from Bradley & Pitzer, which is not on disk, and the project rule
is that A_V must come from the same source as the parameters. It is therefore
supplied from the paper's own tabulation (Table A-1, "D-H Slope" column) via
A_V_TABLE below.

Units: T in K, P in bar, m in mol/kg water. Volumes in cm3.
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
# Constants - Rogers & Pitzer Sec. 4.1, p. 19
# ============================================================================

M2 = 58.4428          # g/mol, NaCl
MW_WATER = 18.01534   # g/mol, water (the paper's value, not IAPWS's 18.015268)
NU_M = 1.0
NU_X = 1.0
NU = 2.0
Z_M = 1.0
Z_X = 1.0
R_CM3_BAR = 83.1440   # cm3 bar mol^-1 K^-1
B_DH = 1.2            # (kg/mol)^1/2, fixed for all 1-1 electrolytes
ALPHA = 2.0           # (kg/mol)^1/2, unused: beta1 has no pressure dependence

Y_HYDRATION = 10.0
M1 = 1000.0 / (Y_HYDRATION * MW_WATER)   # 5.550825 mol/kg
P0 = 1.01325                              # bar

# ============================================================================
# Table 2 (p. 19) - fitting parameters U1..U28, indexed 1-based as U[1]..U[28].
# Transcribed from a 700 dpi render of the page image.
# ============================================================================

_U_SET_I = [None,  # index 0 unused so U[i] matches the paper
    1.0837195e+3, -2.4749323e-1,  1.2442861e-3,  0.0,
   -7.7222249e-2,  3.2423439e-4, -5.7917599e-7,  3.3254437e-6,
    0.0,          -2.1451068e-5,  2.2324909e-3, -6.4950599e-8,
    2.4503020e-10, 0.0,           1.0033371e-7, -1.2784026e-6,
   -4.6468063e-10, 5.7054131e-13, 0.0,           0.0,
    1.3581172e-10, 0.0,           0.0,          -6.8152430e-6,
   -2.5382945e-4,  6.2480692e-8, -1.0731284e-10, 0.0,
]

_U_SET_II = [None,
    1.0249125e+3,  2.7796679e-1, -3.0203919e-4,  1.4977178e-6,
   -7.2002329e-2,  3.1453130e-4, -5.9795994e-7, -6.6596010e-6,
    3.0407621e-8,  5.3699517e-5,  2.2020163e-3, -2.6538013e-7,
    8.6255554e-10,-2.6829310e-2, -1.1173488e-7, -2.6249802e-7,
    3.4926500e-10,-8.3571924e-13, 3.0669940e-5,  1.9767979e-11,
   -1.9144105e-10, 3.1387857e-14,-9.6461948e-9,  2.2902837e-5,
   -4.3314252e-4, -9.0550901e-8,  8.6926600e-11, 5.1904777e-4,
]

# The two readings under test. Defaults are set by resolve_ambiguities().
U11_DENOM_OFFSET = 277.0   # Eq. (24) as literally printed
U19_SET_II = 3.0669940e-5  # exponent -5 reading


def _U(set_name):
    if set_name == 'I':
        return list(_U_SET_I)
    if set_name == 'II':
        u = list(_U_SET_II)
        u[19] = U19_SET_II
        return u
    raise ValueError(f"set_name must be 'I' or 'II', got {set_name!r}")


def V_m1(T_K, P_bar, set_name='II', U=None):
    """Eq. (23): total volume of solution containing 1 kg water at m1, cm3."""
    U = U or _U(set_name)
    dP = P_bar - P0
    return (U[1] + U[2] * T_K + U[3] * T_K ** 2 + U[4] * T_K ** 3
            + dP * (U[5] + U[6] * T_K + U[7] * T_K ** 2)
            + dP ** 2 * (U[8] + U[9] * T_K))


def BV_MX(T_K, P_bar, set_name='II', U=None, u11_offset=None):
    """Eq. (24): BV_MX in kg mol^-1 bar^-1. No ionic-strength dependence."""
    U = U or _U(set_name)
    off = U11_DENOM_OFFSET if u11_offset is None else u11_offset
    dP = P_bar - P0
    return (U[10] + U[11] / (T_K - off) + U[12] * T_K + U[13] * T_K ** 2
            + U[14] / (680.0 - T_K)
            + dP * (U[15] + U[16] / (T_K - 227.0) + U[17] * T_K
                    + U[18] * T_K ** 2 + U[19] / (680.0 - T_K))
            + dP ** 2 * (U[20] + U[21] / (T_K - 227.0) + U[22] * T_K
                         + U[23] / (680.0 - T_K)))


def two_CV_MX(T_K, set_name='II', U=None):
    """
    Eq. (25): 2*CV_MX in kg2 mol^-2 bar^-1. Pressure-independent by
    construction, which Table A-1 confirms (identical columns at 1 and 200 bar).
    """
    U = U or _U(set_name)
    return (U[24] + U[25] / (T_K - 227.0) + U[26] * T_K + U[27] * T_K ** 2
            + U[28] / (680.0 - T_K))


def h_I(I):
    """Eq. (13): h(I) = ln(1 + b*sqrt(I)) / (2b)."""
    return math.log(1.0 + B_DH * math.sqrt(I)) / (2.0 * B_DH)


def specific_volume(T_K, P_bar, m, v_w, A_V, set_name='II'):
    """
    Eq. (22): specific volume of the NaCl solution in cm3/g.

    Parameters:
        T_K, P_bar : state
        m          : NaCl molality, mol/kg water
        v_w        : specific volume of PURE water at (T,P), cm3/g
        A_V        : Debye-Huckel limiting slope for volume,
                     (cm3/mol)(kg/mol)^1/2
        set_name   : 'I' (low-temperature fit) or 'II' (overall fit)

    Returns:
        specific volume in cm3/g. Density in g/cm3 is 1/v.
    """
    if m <= 0:
        return v_w

    U = _U(set_name)
    I, I1 = m, M1                      # 1-1 electrolyte: I = m
    BV = BV_MX(T_K, P_bar, U=U)        # I-independent, so BV(I) == BV(I1)
    CV = two_CV_MX(T_K, U=U) / 2.0

    inner = (V_m1(T_K, P_bar, U=U) / M1
             + (1000.0 / m - MW_WATER * Y_HYDRATION) * v_w
             + NU * abs(Z_M * Z_X) * A_V * (h_I(I) - h_I(I1))
             + 2.0 * NU_M * NU_X * R_CM3_BAR * T_K
               * ((m - M1) * BV + (NU_M * Z_M) * (m ** 2 - M1 ** 2) * CV))

    return (m / (1000.0 + m * M2)) * inner


def density(T_K, P_bar, m, v_w, A_V, set_name='II'):
    """Solution density in g/cm3."""
    return 1.0 / specific_volume(T_K, P_bar, m, v_w, A_V, set_name)


# ============================================================================
# Table A-1 (pp. 25-26), transcribed from rendered page images.
# Columns: T(degC), P(bar), v_w(cm3/g), A_V, V2_0(cm3/mol),
#          BV_MX(kg/mol/bar), 2CV_MX(kg2/mol2/bar)
# The first block of each pressure is the Set I (low-temperature) fit, the
# second block Set II - which is why 50 degC appears twice with different V2_0.
# ============================================================================

TABLE_A1_SET_I = [
    # T,   P,   v_w,      A_V,      V2_0,    BV_MX,     2CV_MX
    (  0,   1, 1.000171, 1.504e+00, 1.327e+01, 2.746e-05, -3.26e-06),
    ( 10,   1, 1.000259, 1.643e+00, 1.506e+01, 1.956e-05, -2.25e-06),
    ( 20,   1, 1.001771, 1.793e+00, 1.625e+01, 1.431e-05, -1.56e-06),
    ( 25,   1, 1.002947, 1.875e+00, 1.668e+01, 1.234e-05, -1.29e-06),
    ( 30,   1, 1.004365, 1.962e+00, 1.702e+01, 1.069e-05, -1.07e-06),
    ( 40,   1, 1.007851, 2.153e+00, 1.750e+01, 8.152e-06, -7.19e-07),
    ( 50,   1, 1.012115, 2.372e+00, 1.774e+01, 6.366e-06, -4.71e-07),
    (  0, 200, 0.990367, 1.462e+00, 1.452e+01, 2.525e-05, -3.26e-06),
    ( 10, 200, 0.991052, 1.587e+00, 1.607e+01, 1.801e-05, -2.25e-06),
    ( 20, 200, 0.992910, 1.724e+00, 1.711e+01, 1.317e-05, -1.56e-06),
    ( 25, 200, 0.994196, 1.799e+00, 1.749e+01, 1.133e-05, -1.29e-06),
    ( 30, 200, 0.995690, 1.879e+00, 1.780e+01, 9.792e-06, -1.07e-06),
    ( 40, 200, 0.999244, 2.055e+00, 1.824e+01, 7.404e-06, -7.19e-07),
    ( 50, 200, 1.003486, 2.255e+00, 1.846e+01, 5.717e-06, -4.71e-07),
]

TABLE_A1_SET_II = [
    ( 50,   1, 1.012115, 2.372e+00, 1.782e+01, 5.733e-06, -3.32e-07),
    ( 60,   1, 1.017087, 2.622e+00, 1.791e+01, 4.415e-06, -2.00e-07),
    ( 70,   1, 1.022724, 2.909e+00, 1.781e+01, 3.513e-06, -1.22e-07),
    ( 80,   1, 1.028999, 3.238e+00, 1.754e+01, 2.925e-06, -7.97e-08),
    ( 90,   1, 1.035897, 3.615e+00, 1.710e+01, 2.577e-06, -6.02e-08),
    (100,   1, 1.043414, 4.050e+00, 1.649e+01, 2.408e-06, -5.46e-08),
    (110,   1, 1.051530, 4.550e+00, 1.571e+01, 2.368e-06, -5.59e-08),
    (120,   2, 1.060271, 5.127e+00, 1.475e+01, 2.412e-06, -5.87e-08),
    (130,   3, 1.069653, 5.795e+00, 1.360e+01, 2.498e-06, -5.87e-08),
    (140,   4, 1.079700, 6.572e+00, 1.226e+01, 2.587e-06, -5.23e-08),
    (150,   5, 1.090444, 7.477e+00, 1.070e+01, 2.637e-06, -3.65e-08),
    (160,   6, 1.101926, 8.536e+00, 8.911e+00, 2.606e-06, -8.63e-09),
    (170,   8, 1.114196, 9.779e+00, 6.863e+00, 2.448e-06,  3.36e-08),
    (180,  10, 1.127316, 1.125e+01, 4.523e+00, 2.112e-06,  9.24e-08),
    (190,  13, 1.141359, 1.299e+01, 1.849e+00, 1.542e-06,  1.70e-07),
    (200,  16, 1.156413, 1.506e+01, -1.215e+00, 6.729e-07, 2.69e-07),
    (210,  19, 1.172584, 1.756e+01, -4.742e+00, -5.691e-07, 3.91e-07),
    (220,  23, 1.190001, 2.058e+01, -8.826e+00, -2.272e-06, 5.38e-07),
    (230,  28, 1.208817, 2.425e+01, -1.360e+01, -4.538e-06, 7.15e-07),
    (240,  33, 1.229223, 2.878e+01, -1.923e+01, -7.494e-06, 9.24e-07),
    (250,  40, 1.251452, 3.440e+01, -2.596e+01, -1.129e-05, 1.17e-06),
    (260,  47, 1.275795, 4.149e+01, -3.414e+01, -1.612e-05, 1.45e-06),
    (270,  55, 1.302623, 5.052e+01, -4.426e+01, -2.221e-05, 1.79e-06),
    (280,  64, 1.332417, 6.224e+01, -5.704e+01, -2.987e-05, 2.18e-06),
    (290,  74, 1.365815, 7.775e+01, -7.360e+01, -3.951e-05, 2.63e-06),
    (300,  86, 1.403691, 9.873e+01, -9.508e+01, -5.167e-05, 3.17e-06),
    ( 50, 200, 1.003486, 2.255e+00, 1.852e+01, 5.187e-06, -3.32e-07),
    ( 60, 200, 1.008358, 2.484e+00, 1.858e+01, 4.005e-06, -2.00e-07),
    ( 70, 200, 1.013825, 2.745e+00, 1.848e+01, 3.224e-06, -1.22e-07),
    ( 80, 200, 1.019865, 3.043e+00, 1.822e+01, 2.746e-06, -7.97e-08),
    ( 90, 200, 1.026463, 3.383e+00, 1.780e+01, 2.502e-06, -6.02e-08),
    (100, 200, 1.033614, 3.772e+00, 1.723e+01, 2.434e-06, -5.46e-08),
    (110, 200, 1.041317, 4.217e+00, 1.650e+01, 2.497e-06, -5.59e-08),
    (120, 200, 1.049580, 4.728e+00, 1.561e+01, 2.651e-06, -5.87e-08),
    (130, 200, 1.058416, 5.315e+00, 1.456e+01, 2.859e-06, -5.87e-08),
    (140, 200, 1.067844, 5.991e+00, 1.333e+01, 3.086e-06, -5.23e-08),
    (150, 200, 1.077890, 6.773e+00, 1.192e+01, 3.296e-06, -3.65e-08),
]


def resolve_ambiguities(verbose=True):
    """
    Settle the two source ambiguities against Table A-1's own BV_MX column,
    by scoring every combination and reporting the residuals.

    Table A-1 prints BV_MX to 4 significant figures, so a correct reading
    should land within roughly 0.05% and a wrong one should be obvious.
    """
    global U11_DENOM_OFFSET, U19_SET_II

    results = {}
    for off in (227.0, 277.0):
        for u19_exp in (-5, -3):
            u19 = 3.0669940 * 10.0 ** u19_exp

            worst_I = 0.0
            for T_C, P, _, _, _, bv, _ in TABLE_A1_SET_I:
                got = BV_MX(T_C + 273.15, P, set_name='I', u11_offset=off)
                worst_I = max(worst_I, abs(got - bv) / abs(bv) * 100)

            worst_II = 0.0
            u = list(_U_SET_II)
            u[19] = u19
            for T_C, P, _, _, _, bv, _ in TABLE_A1_SET_II:
                got = BV_MX(T_C + 273.15, P, U=u, u11_offset=off)
                worst_II = max(worst_II, abs(got - bv) / abs(bv) * 100)

            results[(off, u19_exp)] = (worst_I, worst_II)

    if verbose:
        print("  Worst |deviation| from Table A-1's BV_MX column, %")
        print(f"  {'U11 denom':>10} {'U19 exp':>8} {'Set I':>12} {'Set II':>12}")
        for (off, e), (wi, wii) in sorted(results.items()):
            print(f"  {f'(T-{off:.0f})':>10} {f'10^{e}':>8} "
                  f"{wi:11.4f}% {wii:11.4f}%")

    # Set I never touches U19, so it discriminates the denominator alone.
    best_off = min((227.0, 277.0), key=lambda o: results[(o, -5)][0])
    best_exp = min((-5, -3), key=lambda e: results[(best_off, e)][1])
    return best_off, best_exp, results


if __name__ == "__main__":
    print("=" * 74)
    print("Rogers & Pitzer (1982) NaCl volumetric model - source verification")
    print("=" * 74)

    print(f"\nReference concentration m1 = 1000/(Y*Mw) = {M1:.6f} molal "
          f"(paper states 5.550825)")

    print("\n1. Resolving the two transcription ambiguities against Table A-1:")
    off, exp, results = resolve_ambiguities()
    print(f"\n   -> U11 denominator: (T - {off:.0f})")
    print(f"   -> U19 (Set II)   : 10^{exp}")

    U11_DENOM_OFFSET = off
    U19_SET_II = 3.0669940 * 10.0 ** exp

    print("\n2. Full check against Table A-1 with the resolved readings:")
    for label, rows, sname in (("Set I ", TABLE_A1_SET_I, 'I'),
                               ("Set II", TABLE_A1_SET_II, 'II')):
        wb = wc = 0.0
        for T_C, P, _, _, _, bv, cv2 in rows:
            T = T_C + 273.15
            wb = max(wb, abs(BV_MX(T, P, sname) - bv) / abs(bv) * 100)
            wc = max(wc, abs(two_CV_MX(T, sname) - cv2) / abs(cv2) * 100)
        print(f"   {label}: {len(rows):2d} rows   "
              f"worst BV_MX {wb:6.3f}%   worst 2CV_MX {wc:6.3f}%")

    print("\n3. Structural check - is 2CV_MX really pressure-independent?")
    same = all(
        abs(c1 - c2) < 1e-15
        for (t1, p1, _, _, _, _, c1) in TABLE_A1_SET_I
        for (t2, p2, _, _, _, _, c2) in TABLE_A1_SET_I
        if t1 == t2 and p1 != p2
    )
    print(f"   Table A-1's 1 bar and 200 bar 2CV_MX columns identical: {same}")
    print("   (Eq. (25) has no pressure term, so this is a consistency check "
          "on the transcription)")
