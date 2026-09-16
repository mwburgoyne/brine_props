"""Kestin, Khalifa & Correia (1981) NaCl-solution viscosity, JPCRD 10, 71.

Papers/50. The reference correlation of oscillating-disk measurements over
20-150 degC, 0.1-35 MPa, 0-6 molal NaCl, stated to reproduce the underlying
experimental results to +/-0.5% standard deviation.

WHY THIS FILE EXISTS. The salt leg of the shipped viscosity chain (Mao-Duan)
had been scored only against Calabrese's measured gas-free brine at a SINGLE
molality (0.77 mol/kg). A salt term cannot be judged on one concentration.
Kestin supplies the molality axis: a measurement-backed yardstick from 0 to
6 molal at reservoir pressures.

WHAT IT IS AND IS NOT. These are tabulated values of a correlation, not raw
points. It is a smoothed representation of measured data, which is the best
NaCl-viscosity reference available and is how JPCRD publishes such data. Note
that Mao-Duan may have been fitted to the same measurements, so agreement
between the two is not independent evidence.

EQUATIONS, transcribed from a 300 dpi RENDER of pp. 72-73 (the OCR text layer
scrambles every exponent and drops minus signs):

    (1)  mu(p,t,m)   = mu0(t,m) * [1 + beta(t,m) * p]
    (2)  log10[mu0(t,m)/muw0(t)] = A(m) + B(m)*log10[muw0(t)/muw0(20 degC)]
    (3)  log10[muw0(t)/muw0(20 degC)]
                      = {sum_i=1..4 alpha_i*[(20-t)/degC]^i} / [(96+t)/degC]
    (4)  A(m)         = sum_i=1..3 a_i * m^i
    (5)  B(m)         = sum_i=1..3 b_i * m^i
    (6)  beta(t,m)    = betaEs(t) * betastar(m/ms) + betaw(t)
    (7)  betaw(t)     = sum_i=0..4 beta_i * (t/degC)^i          [1/GPa]
    (8)  betaEs(t)    = gamma0 + gamma1*(t/degC) - betaw(t)     [1/GPa]
    (9)  ms(t)        = sum_i=0..2 m_i * (t/degC)^i             [mol/kg]
    (10) betastar(r)  = sum_i=1..3 betastar_i * r^i

ONE TRANSCRIPTION AMBIGUITY, settled against the paper's own tables. The
exponent in (4) and (5) prints as a glyph that reads as `2` at 300 dpi, but
the sums run i = 1..3, which would make three coefficients collinear if the
exponent were fixed. Reading it as `^i` reproduces the printed tables to
0.015% worst over 18 values at 0, 3 and 6 molal; reading it as a literal `^2`
misses by up to 6478%. See `_check_vs_tables()`.

Pressure enters as GPa in Eqs. (7)-(8); the public functions take MPa.
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

# Eq. (3) - water, Kestin, Sokolov & Wakeham (1978)
_ALPHA = (1.2378, -1.303e-3, 3.06e-6, 2.55e-8)   # i = 1..4
MU_W0_20C = 1002.0          # micro Pa s, zero-pressure water at 20 degC

_A_COEF = (3.324e-2, 3.624e-3, -1.879e-4)        # Eq. (4), i = 1..3
_B_COEF = (-3.96e-2, 1.02e-2, -7.02e-4)          # Eq. (5), i = 1..3
_BETA_W = (-1.297, 5.74e-2, -6.97e-4, 4.47e-6, -1.05e-8)   # Eq. (7), i = 0..4
_GAMMA = (0.545, 2.8e-3)                         # Eq. (8)
_MS = (6.044, 2.8e-3, 3.6e-5)                    # Eq. (9), i = 0..2
_BETA_STAR = (2.5, -2.0, 0.5)                    # Eq. (10), i = 1..3

T_MIN_C, T_MAX_C = 20.0, 150.0
P_MAX_MPA = 35.0
M_MAX = 6.0


def mu_w0(t_C):
    """Zero-pressure pure-water viscosity, micro Pa s. Eq. (3)."""
    num = sum(a * (20.0 - t_C) ** (i + 1) for i, a in enumerate(_ALPHA))
    return MU_W0_20C * 10.0 ** (num / (96.0 + t_C))


def _A(m):
    return sum(a * m ** (i + 1) for i, a in enumerate(_A_COEF))


def _B(m):
    return sum(b * m ** (i + 1) for i, b in enumerate(_B_COEF))


def mu0(t_C, m):
    """Zero-pressure solution viscosity, micro Pa s. Eqs. (2), (4), (5)."""
    muw = mu_w0(t_C)
    return muw * 10.0 ** (_A(m) + _B(m) * math.log10(muw / MU_W0_20C))


def beta_w(t_C):
    """Pressure coefficient of water, 1/GPa. Eq. (7)."""
    return sum(b * t_C ** i for i, b in enumerate(_BETA_W))


def m_saturation(t_C):
    """NaCl saturation molality, mol/kg. Eq. (9), Seidell."""
    return sum(c * t_C ** i for i, c in enumerate(_MS))


def beta(t_C, m):
    """Pressure coefficient of the solution, 1/GPa. Eqs. (6), (8), (10)."""
    bw = beta_w(t_C)
    beta_Es = _GAMMA[0] + _GAMMA[1] * t_C - bw
    r = m / m_saturation(t_C)
    beta_star = sum(b * r ** (i + 1) for i, b in enumerate(_BETA_STAR))
    return beta_Es * beta_star + bw


def mu(t_C, p_MPa, m):
    """Dynamic viscosity of aqueous NaCl, micro Pa s. Eq. (1).

    t_C in degC, p_MPa in MPa, m in mol/kg NaCl. Correlated range
    20-150 degC, 0.1-35 MPa, 0-6 molal; extrapolation is the caller's problem.
    """
    return mu0(t_C, m) * (1.0 + beta(t_C, m) * p_MPa / 1000.0)


def salt_ratio(t_C, p_MPa, m):
    """mu(brine)/mu(water) at the same t and p - the salt term alone."""
    return mu(t_C, p_MPa, m) / mu(t_C, p_MPa, 0.0)


# ---------------------------------------------------------------- verification
# Values read off 300 dpi RENDERS of the paper's own Tables 1-13 (dynamic
# viscosity, micro Pa s). These are the author's tabulated output of the
# equations above, so they are the arbiter for the transcription.
# (table, t_C, p_MPa, m, printed mu)
# The p* column is 0.1 MPa or the vapour pressure, whichever is higher, so it
# is only used below 100 degC where p* = 0.1 MPa.
_TABLE_SPOTS = [
    # Table 1, p. 75, m = 0.0
    (1, 20.0, 0.1, 0.0, 1002.0),
    (1, 25.0, 0.1, 0.0, 890.1),
    (1, 50.0, 10.0, 0.0, 548.9),
    (1, 100.0, 20.0, 0.0, 287.2),
    (1, 150.0, 35.0, 0.0, 190.7),
    (1, 75.0, 30.0, 0.0, 385.8),
    # Table 7, p. 78, m = 3.0
    (7, 20.0, 0.1, 3.0, 1343.2),
    (7, 25.0, 0.1, 3.0, 1199.8),
    (7, 50.0, 10.0, 3.0, 758.7),
    (7, 100.0, 20.0, 3.0, 407.6),
    (7, 150.0, 35.0, 3.0, 273.5),
    (7, 75.0, 30.0, 3.0, 542.2),
    # Table 13, p. 81, m = 6.0
    (13, 20.0, 0.1, 6.0, 1950.8),
    (13, 25.0, 0.1, 6.0, 1737.5),
    (13, 50.0, 10.0, 6.0, 1086.8),
    (13, 100.0, 20.0, 6.0, 574.1),
    (13, 150.0, 35.0, 6.0, 379.9),
    (13, 75.0, 30.0, 6.0, 769.9),
]


def _check_vs_tables(spots=None, verbose=True):
    """Score the implementation against the printed tables."""
    spots = _TABLE_SPOTS if spots is None else spots
    if not spots:
        raise RuntimeError('no table spot values loaded')
    worst = 0.0
    for tab, t, p, m, printed in spots:
        got = mu(t, p, m)
        d = 100.0 * (got / printed - 1.0)
        worst = max(worst, abs(d))
        if verbose:
            print(f'  Table {tab:>2}  t={t:5.0f} C  p={p:5.1f} MPa  m={m:4.1f}  '
                  f'printed {printed:8.1f}  got {got:8.1f}  {d:+7.3f}%')
    if verbose:
        print(f'  worst |dev| = {worst:.3f}% over {len(spots)} printed values')
    return worst


if __name__ == '__main__':
    print('Kestin, Khalifa & Correia (1981), NaCl viscosity - self check\n')
    print(f'  mu_w0(20 C)  = {mu_w0(20.0):.1f} micro Pa s   (printed 1002.0)')
    print(f'  mu(20 C, 0.1 MPa, 0 m) = {mu(20.0, 0.1, 0.0):.1f}')
    print(f'  m_sat(20 C)  = {m_saturation(20.0):.3f} mol/kg')
    print('\n  salt ratio mu(m)/mu(water):')
    for m in (0.5, 1.0, 2.0, 4.0, 6.0):
        print(f'    m = {m:.1f}:  20 C {salt_ratio(20.0, 0.1, m):.4f}   '
              f'75 C {salt_ratio(75.0, 20.0, m):.4f}   '
              f'150 C {salt_ratio(150.0, 35.0, m):.4f}')
    if _TABLE_SPOTS:
        print('\n  against the printed tables:')
        _check_vs_tables()
