"""Kestin, Khalifa & Correia (1981) KCl-solution viscosity, JPCRD 10, 57.

Papers/51, the companion to the NaCl paper (`Papers/50`, `kestin_nacl_viscosity`).
Measurements by Grimes, Kestin & Khalifa, oscillating disk, **+/-1%** (twice the
NaCl paper's uncertainty), over **25-150 degC, 0.1-35 MPa, 0-5 molal**.

WHY THIS FILE EXISTS. The ion-additive Jones-Dole leg is verified against
PHREEQC but was never scored against measurement for any ion except Na+. This
supplies a second one: K+ as a single salt. It does not test ion MIXING, which
remains unvalidated because no measured mixed-brine viscosity was found.

THE CORRELATION differs in form from the NaCl paper - the salt effect is a
double polynomial on the RATIO rather than a log-log construction:

    (1)  mu(p,t,m) = mu0(t,m) * [1 + beta(t,m)*p]
    (2)  mu0(t,m)/muw0(t) = 1 + sum_i=0..2 sum_j=0..2 f_ij * m^(j+1) * t^i
    (3)  muw0(t) - the same Kestin, Sokolov & Wakeham water correlation the
         NaCl paper uses, so it is imported rather than re-transcribed
    (4)  beta(t,m)   = betaEs(t)*betastar(m/ms) + betaw(t)
    (5)  betaw(t)    - identical constants to the NaCl paper
    (6)  betaEs(t)   = 0.241 + 0.478e-2*(t/degC) - betaw(t)      [1/GPa]
    (7)  ms(t)       = 3.825 + 0.394e-1*t - 0.197e-4*t^2         [mol/kg]
    (8)  betastar(r) = 3.25*r - 3.5*r^2 + 1.25*r^3

NOTE THE INDEX ORDER IN (2): `i` runs over TEMPERATURE and `j` over MOLALITY,
which is the opposite of the natural reading, and the printed f_ij table is laid
out with rows = j and columns = i. Reproducing Table 11 (5 molal) at 25 degC to
four decimals is what confirms it. The text layer also renders f[2][0] as
`0.480e-2`; the 300 dpi page says **0.450e-2**.
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

from brine_gas.kestin_nacl_viscosity import beta_w, mu_w0

# Eq. (2), f_ij as printed: rows j = 0..2, columns i = 0..2
_F = (
    (0.113e-1, 0.537e-3, 0.436e-6),      # j = 0
    (-0.235e-1, 0.525e-3, -0.230e-5),    # j = 1
    (0.450e-2, -0.868e-4, 0.344e-6),     # j = 2
)

_MS = (3.825, 0.394e-1, -0.197e-4)       # Eq. (7)
_GAMMA = (0.241, 0.478e-2)               # Eq. (6)
_BETA_STAR = (3.25, -3.5, 1.25)          # Eq. (8)

T_MIN_C, T_MAX_C = 25.0, 150.0
P_MAX_MPA = 35.0
M_MAX = 5.0


def mu0(t_C, m):
    """Zero-pressure solution viscosity, micro Pa s. Eq. (2)."""
    s = 0.0
    for j, row in enumerate(_F):
        for i, f in enumerate(row):
            s += f * m ** (j + 1) * t_C ** i
    return mu_w0(t_C) * (1.0 + s)


def m_saturation(t_C):
    """KCl saturation molality, mol/kg. Eq. (7)."""
    return sum(c * t_C ** i for i, c in enumerate(_MS))


def beta(t_C, m):
    """Pressure coefficient of the solution, 1/GPa. Eqs. (4), (6), (8)."""
    bw = beta_w(t_C)
    beta_Es = _GAMMA[0] + _GAMMA[1] * t_C - bw
    r = m / m_saturation(t_C)
    beta_star = sum(b * r ** (i + 1) for i, b in enumerate(_BETA_STAR))
    return beta_Es * beta_star + bw


def mu(t_C, p_MPa, m):
    """Dynamic viscosity of aqueous KCl, micro Pa s. Eq. (1)."""
    return mu0(t_C, m) * (1.0 + beta(t_C, m) * p_MPa / 1000.0)


def salt_ratio(t_C, p_MPa, m):
    """mu(KCl solution)/mu(water) at the same t and p."""
    return mu(t_C, p_MPa, m) / mu(t_C, p_MPa, 0.0)


# ---------------------------------------------------------------- verification
# Read off 300 dpi renders of the paper's own Tables 1, 3 and 11. The p* column
# is 0.1 MPa below 100 degC.
# (table, t_C, p_MPa, m, printed mu)
_TABLE_SPOTS = [
    (1, 25.0, 0.1, 0.0, 890.1),
    (1, 150.0, 35.0, 0.0, 190.7),
    (3, 25.0, 0.1, 1.0, 904.1),
    (3, 50.0, 10.0, 1.0, 569.7),
    (3, 100.0, 20.0, 1.0, 308.2),
    (3, 150.0, 35.0, 1.0, 209.6),
    (11, 25.0, 0.1, 5.0, 1021.7),
    (11, 25.0, 35.0, 5.0, 1034.6),
    (11, 50.0, 10.0, 5.0, 686.4),
    (11, 75.0, 30.0, 5.0, 514.1),
    (11, 100.0, 20.0, 5.0, 401.9),
    (11, 150.0, 35.0, 5.0, 281.8),
]


def _check_vs_tables(verbose=True):
    """Score the implementation against the printed tables. Returns worst %."""
    worst = 0.0
    for tab, t, p, m, printed in _TABLE_SPOTS:
        got = mu(t, p, m)
        d = 100.0 * (got / printed - 1.0)
        worst = max(worst, abs(d))
        if verbose:
            print(f'  Table {tab:>2}  t={t:5.0f} C  p={p:5.1f} MPa  m={m:4.1f}  '
                  f'printed {printed:8.1f}  got {got:8.1f}  {d:+7.3f}%')
    if verbose:
        print(f'  worst |dev| = {worst:.3f}% over {len(_TABLE_SPOTS)} values')
    return worst


if __name__ == '__main__':
    print('Kestin, Khalifa & Correia (1981), KCl viscosity - self check\n')
    _check_vs_tables()
    print('\n  KCl salt ratio mu(m)/mu(water), and NaCl for contrast:')
    import brine_gas.kestin_nacl_viscosity as nacl
    print(f"    {'m':>4} {'KCl 25 C':>10} {'NaCl 25 C':>11} {'KCl 150 C':>11} "
          f"{'NaCl 150 C':>12}")
    for m in (1.0, 2.0, 3.0, 4.0, 5.0):
        print(f'    {m:4.1f} {salt_ratio(25.0, 0.1, m):10.4f} '
              f'{nacl.salt_ratio(25.0, 0.1, m):11.4f} '
              f'{salt_ratio(150.0, 35.0, m):11.4f} '
              f'{nacl.salt_ratio(150.0, 35.0, m):12.4f}')
