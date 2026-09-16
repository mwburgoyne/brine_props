"""
Bradley & Pitzer (1979) dielectric constant of water and Debye-Huckel slopes.

Reference:
    Bradley, D.J. and Pitzer, K.S. (1979). "Thermodynamics of electrolytes. 12.
    Dielectric properties of water and Debye-Huckel parameters to 350 degC and
    1 kbar." J. Phys. Chem. 83(12), 1599-1603.  DOI 10.1021/j100475a009
    (Papers/44_BradleyPitzer_1979_dielectric_DebyeHuckel_slopes.pdf)

This supplies A_V, the Debye-Huckel limiting slope for volume, which is the one
input Rogers & Pitzer's NaCl volumetric model (Papers/38, `rogers_pitzer_nacl`)
does not carry itself - they cite this paper for it as their reference [7].
Taking it from here rather than from Archer or IAPWS keeps A_V and the fitted
U parameters from the SAME source, which is the project rule.

MODEL - dielectric constant, Eqs. (1)-(4), Table I. T in K, P in bar.

    D      = D_1000 + C * ln((B + P) / (B + 1000))          (1)
    D_1000 = U1 * exp(U2*T + U3*T^2)                        (2)
    C      = U4 + U5 / (U6 + T)                             (3)
    B      = U7 + U8/T + U9*T                               (4)

The pressure derivative is analytic, and the paper confirms it in the captions
to Figs. 2 and 3:  (dD/dP)_T = C/(B+P)  and  -(d2D/dP2)_T = C/(B+P)^2.

Stated validity: 0-350 degC, to 2000 bar below 70 degC and 5000 bar above.
That covers the whole of IF97 Region 1, so A_V is defined wherever the rest of
the chain is.

DEBYE-HUCKEL SLOPES (p. 1601)

    A_phi = (1/3) * (2*pi*N0*rho_w/1000)^(1/2) * (e^2/(D*k*T))^(3/2)
    A_V   = -2*A_phi*R*T * [ beta_w - 3*(dlnD/dP)_T ]

**THE PRINTED A_V EQUATION HAS A SIGN ERROR.** The paper prints

    A_V = -2*A_phi*R*T * [ 3*(dlnD/dP)_T + beta_w ]      <- as printed, WRONG

verified at 700 dpi, so it is the paper's error and not a scan artefact. It
cannot be right: both beta_w and (dlnD/dP)_T are positive, so the printed form
makes A_V negative, whereas the slope is positive and Rogers & Pitzer tabulate
it as such. Differentiating their own definition of A_phi settles it -

    dln(A_phi)/dP = (1/2)*beta_w - (3/2)*(dlnD/dP)
    A_V = -4*R*T*(dA_phi/dP)_T                      [Rogers & Pitzer Eq. (18)]
        = -2*A_phi*R*T*[ beta_w - 3*(dlnD/dP) ]

- and the correction is confirmed numerically by `verify_against_table_A1()`,
which scores both readings against Rogers & Pitzer's own tabulated A_V column.
Do not "restore" the printed sign.

Units: T in K, P in bar, rho_w in g/cm3, beta_w in 1/bar, A_V in
(cm3/mol)(kg/mol)^1/2. CGS electrostatic units internally.
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

from brine_gas.iapws_if97 import rho_and_kappa

# ============================================================================
# Table I (p. 1600) - constants U1..U9 in Eqs. (2)-(4)
# ============================================================================

U1 = 3.4279e2
U2 = -5.0866e-3
U3 = 9.4690e-7
U4 = -2.0525
U5 = 3.1159e3
U6 = -1.8289e2
U7 = -8.0325e3
U8 = 4.2142e6
U9 = 2.1417

# ============================================================================
# Physical constants. CGS electrostatic, as the A_phi expression requires.
# Values are the CODATA set current when the paper was written; the modern
# values move A_phi by under 1e-5 relative, checked in __main__.
# ============================================================================

N_AVOGADRO = 6.022045e23      # 1/mol
E_CHARGE_ESU = 4.803242e-10   # esu
K_BOLTZMANN = 1.380662e-16    # erg/K
R_CM3_BAR = 83.1440           # cm3 bar mol^-1 K^-1 (Rogers & Pitzer's value)

T_MIN, T_MAX = 273.15, 623.15  # the paper's 0-350 degC


def dielectric_constant(T_K, P_bar):
    """Static dielectric constant of water, Bradley & Pitzer Eq. (1)."""
    D1000 = U1 * math.exp(U2 * T_K + U3 * T_K ** 2)
    C = U4 + U5 / (U6 + T_K)
    B = U7 + U8 / T_K + U9 * T_K
    return D1000 + C * math.log((B + P_bar) / (B + 1000.0))


def dD_dP(T_K, P_bar):
    """(dD/dP)_T in 1/bar. Analytic; the paper states it in the Fig. 2 caption."""
    C = U4 + U5 / (U6 + T_K)
    B = U7 + U8 / T_K + U9 * T_K
    return C / (B + P_bar)


def dlnD_dP(T_K, P_bar):
    """(d ln D/dP)_T in 1/bar."""
    return dD_dP(T_K, P_bar) / dielectric_constant(T_K, P_bar)


def A_phi(T_K, P_bar, rho_w=None):
    """
    Debye-Huckel slope for the osmotic coefficient, (kg/mol)^1/2.

    rho_w in g/cm3; taken from IF97 Region 1 if not supplied.
    """
    if rho_w is None:
        rho_w = rho_and_kappa(T_K, P_bar / 10.0)[0] / 1000.0
    D = dielectric_constant(T_K, P_bar)
    term1 = math.sqrt(2.0 * math.pi * N_AVOGADRO * rho_w / 1000.0)
    term2 = (E_CHARGE_ESU ** 2 / (D * K_BOLTZMANN * T_K)) ** 1.5
    return term1 * term2 / 3.0


def A_V(T_K, P_bar, rho_w=None, beta_w=None, printed_sign=False):
    """
    Debye-Huckel limiting slope for volume, (cm3/mol)(kg/mol)^1/2.

    beta_w is the isothermal compressibility of water in 1/bar; taken from IF97
    if not supplied.

    printed_sign=True reproduces the paper's printed (erroneous) equation, and
    exists only so verify_against_table_A1() can score it. Never use it.
    """
    if rho_w is None or beta_w is None:
        rho, kappa_per_MPa = rho_and_kappa(T_K, P_bar / 10.0)
        if rho_w is None:
            rho_w = rho / 1000.0
        if beta_w is None:
            beta_w = kappa_per_MPa / 10.0        # 1/MPa -> 1/bar

    aphi = A_phi(T_K, P_bar, rho_w)
    dlnD = dlnD_dP(T_K, P_bar)

    if printed_sign:
        return -2.0 * aphi * R_CM3_BAR * T_K * (3.0 * dlnD + beta_w)
    return -2.0 * aphi * R_CM3_BAR * T_K * (beta_w - 3.0 * dlnD)


# ============================================================================
# Verification
# ============================================================================

def verify_A_phi_table_II(verbose=True):
    """
    Against Bradley & Pitzer's OWN Table II (p. 1601), A_phi on a T,P grid.
    Their values are printed to 3 significant figures.
    Rows sampled across the range; P = 'SAT' rows are omitted because the
    saturation pressure is not tabulated with them.
    """
    # (T_degC, P_bar, A_phi printed)
    TABLE_II = [
        (0, 100, 3.75e-1), (0, 400, 3.70e-1), (0, 1000, 3.61e-1),
        (25, 100, 3.90e-1), (25, 400, 3.84e-1), (25, 1000, 3.74e-1),
        (50, 100, 4.08e-1), (50, 400, 4.02e-1), (50, 1000, 3.91e-1),
        (100, 100, 4.57e-1), (100, 400, 4.48e-1), (100, 1000, 4.33e-1),
        (150, 100, 5.25e-1), (150, 400, 5.11e-1), (150, 1000, 4.88e-1),
        (200, 100, 6.15e-1), (200, 400, 5.91e-1), (200, 1000, 5.54e-1),
        (250, 100, 7.42e-1), (250, 400, 6.95e-1), (250, 1000, 6.34e-1),
        (300, 100, 9.52e-1), (300, 400, 8.40e-1), (300, 1000, 7.26e-1),
    ]
    worst = 0.0
    for T_C, P, printed in TABLE_II:
        got = A_phi(T_C + 273.15, P)
        dev = abs(got - printed) / printed * 100
        worst = max(worst, dev)
        if verbose and dev > 0.5:
            print(f"    T={T_C:4d} P={P:5d}  printed {printed:.3e}  "
                  f"got {got:.4e}  {dev:+.3f}%")
    return worst, len(TABLE_II)


# Archer & Wang (1990) Table 7, p. 399 (Papers/45) - A_V on a T,p grid computed
# from THEIR dielectric equation, which is the one Archer 1992 uses. This is the
# independent yardstick for A_V: it is a different dielectric formulation, so
# the spread between it and Bradley & Pitzer is the honest A_V uncertainty.
# Columns are p in MPa; blank cells (below the 0.1 MPa boiling point) are None.
_AW_P_MPA = [0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
ARCHER_WANG_AV = {
    263.15: [1.4769, 1.4747, 1.4650, 1.4533, 1.4310, 1.4099, 1.3897, 1.3704, 1.3520, 1.3343],
    273.15: [1.5066, 1.5050, 1.4980, 1.4894, 1.4727, 1.4566, 1.4410, 1.4257, 1.4108, 1.3961],
    283.15: [1.6057, 1.6038, 1.5955, 1.5855, 1.5662, 1.5477, 1.5300, 1.5128, 1.4961, 1.4798],
    298.15: [1.8305, 1.8275, 1.8143, 1.7984, 1.7682, 1.7398, 1.7130, 1.6875, 1.6633, 1.6401],
    323.15: [2.3601, 2.3542, 2.3284, 2.2974, 2.2390, 2.1849, 2.1346, 2.0875, 2.0435, 2.0021],
    348.15: [3.0979, 3.0878, 3.0437, 2.9909, 2.8918, 2.8008, 2.7168, 2.6390, 2.5668, 2.4995],
    373.15: [4.1156, 4.0990, 4.0274, 3.9418, 3.7826, 3.6375, 3.5046, 3.3827, 3.2702, 3.1662],
    423.15: [None,   7.5351, 7.3458, 7.1226, 6.7161, 6.3551, 6.0326, 5.7425, 5.4804, 5.2423],
    433.15: [None,   8.5874, 8.3547, 8.0315, 7.5869, 7.1509, 6.7638, 6.4177, 6.1066, 5.8253],
}


def verify_against_archer_wang(verbose=True):
    """
    Score this Bradley & Pitzer A_V against Archer & Wang (1990) Table 7.

    These are two INDEPENDENT dielectric formulations, so this is not a
    transcription check - it measures how much A_V actually depends on which
    dielectric equation you believe. Returns (mean_abs_pct, max_abs_pct, n).
    """
    devs = []
    for T, row in ARCHER_WANG_AV.items():
        if not T_MIN <= T <= T_MAX:
            continue
        for p_MPa, aw in zip(_AW_P_MPA, row):
            if aw is None:
                continue
            got = A_V(T, p_MPa * 10.0)
            d = (got - aw) / aw * 100.0
            devs.append((abs(d), d, T, p_MPa, aw, got))
    devs.sort(reverse=True)
    if verbose:
        print(f"    {'T':>8} {'p MPa':>7} {'A&W':>9} {'B&P':>9} {'dev %':>8}")
        for _, d, T, p, aw, got in devs[:4]:
            print(f"    {T:8.2f} {p:7.1f} {aw:9.4f} {got:9.4f} {d:+8.3f}")
    mean = sum(x[0] for x in devs) / len(devs)
    return mean, devs[0][0], len(devs)


def verify_against_table_A1(verbose=True):
    """
    Score both readings of the A_V equation against Rogers & Pitzer Table A-1's
    tabulated "D-H Slope" column - which they computed FROM this paper, so it is
    exactly the right arbiter.

    Returns (worst_correct_pct, worst_printed_pct).
    """
    from brine_gas.rogers_pitzer_nacl import TABLE_A1_SET_I, TABLE_A1_SET_II

    rows = list(TABLE_A1_SET_I) + list(TABLE_A1_SET_II)
    worst_ok = worst_bad = 0.0
    for T_C, P, v_w, av_printed, _, _, _ in rows:
        T = T_C + 273.15
        if not T_MIN <= T <= T_MAX:
            continue
        # Use Rogers & Pitzer's OWN v_w so the water model cannot contaminate
        # the comparison; beta_w still comes from IF97.
        rho_w = 1.0 / v_w
        beta_w = rho_and_kappa(T, P / 10.0)[1] / 10.0
        ok = A_V(T, P, rho_w=rho_w, beta_w=beta_w)
        bad = A_V(T, P, rho_w=rho_w, beta_w=beta_w, printed_sign=True)
        worst_ok = max(worst_ok, abs(ok - av_printed) / abs(av_printed) * 100)
        worst_bad = max(worst_bad, abs(bad - av_printed) / abs(av_printed) * 100)
    return worst_ok, worst_bad


if __name__ == "__main__":
    print("=" * 76)
    print("Bradley & Pitzer (1979) - dielectric constant and Debye-Huckel slopes")
    print("=" * 76)

    print("\n1. Dielectric constant against values quoted in the paper's text:")
    D25 = dielectric_constant(298.15, 1.0)
    print(f"   D(25 degC, 1 bar)  = {D25:.3f}   (accepted value ~78.4)")
    print(f"   dD/dP(25 degC)     = {dD_dP(298.15, 1.0):.5f} /bar   "
          f"(Fig. 2 shows ~0.004 at low T)")

    print("\n2. A_phi against Bradley & Pitzer's own Table II (3 sig figs):")
    worst, n = verify_A_phi_table_II()
    print(f"   worst deviation {worst:.3f}% over {n} sampled grid points")

    print("\n3. THE SIGN QUESTION - both readings of the A_V equation scored")
    print("   against Rogers & Pitzer Table A-1's tabulated D-H Slope column:")
    ok, bad = verify_against_table_A1()
    print(f"   [beta_w - 3 dlnD/dP]  (derived) : worst {ok:8.3f}%")
    print(f"   [3 dlnD/dP + beta_w]  (printed) : worst {bad:8.3f}%")
    print(f"\n   -> the printed equation is wrong by a sign; use the derived form")

    print("\n4. Spot value everything hinges on, A_V at 25 degC / 1 bar:")
    rho_w = 1.0 / 1.002947            # Rogers & Pitzer Table A-1 v_w
    beta_w = rho_and_kappa(298.15, 0.1)[1] / 10.0
    print(f"   A_phi = {A_phi(298.15, 1.0, rho_w):.5f}   "
          f"(Bradley & Pitzer Table II, 25 degC SAT: 3.91E-01)")
    print(f"   A_V   = {A_V(298.15, 1.0, rho_w, beta_w):.4f}   "
          f"(Rogers & Pitzer Table A-1: 1.875)")

    print("\n5. Sensitivity to the physical-constant vintage "
          "(1979 CODATA vs modern):")
    a_old = A_phi(298.15, 1.0, rho_w)
    _N, _E, _K = N_AVOGADRO, E_CHARGE_ESU, K_BOLTZMANN
    N_AVOGADRO, E_CHARGE_ESU, K_BOLTZMANN = 6.02214076e23, 4.80320471e-10, 1.380649e-16
    a_new = A_phi(298.15, 1.0, rho_w)
    N_AVOGADRO, E_CHARGE_ESU, K_BOLTZMANN = _N, _E, _K
    print(f"   A_phi 1979 constants {a_old:.6f}, modern {a_new:.6f}, "
          f"shift {(a_new - a_old) / a_old * 100:+.5f}%")
