"""Ion-additive brine viscosity: the modified Jones-Dole equation of PHREEQC.

Appelo & Parkhurst's viscosity model, implemented from scratch and dependency
free so it ports to Dart alongside the rest of the chain. Parameters are read
straight out of PHREEQC's `pitzer.dat` (public domain, USGS), where each ion
carries a `-viscosity` line.

WHY THIS EXISTS. The shipped salt viscosity term is Mao-Duan's NaCl-only ratio.
The volume leg went multi-salt in the seventeenth block (`appelo_volumes`), so
the viscosity leg is the remaining NaCl-only piece, and it is the larger error
of the two viscosity terms.

WHERE THE MODEL COMES FROM. `pitzer.dat`'s footer states the form and cites
"Appelo and Parkhurst in prep., for parameters see subroutine viscosity in
transport.cpp". The paper does not exist yet, so the REFERENCE IMPLEMENTATION
is the specification: PHREEQC 3.8.6 `src/transport.cpp`, `Phreeqc::viscosity`.
That is also the arbiter - `jones_dole_benchmark.py` scores this module against
PHREEQC's own output, so any disagreement is our bug rather than a difference
of opinion.

THE EQUATIONS, as coded in transport.cpp:

    mu/mu_0 = 1 + A*sqrt(eq/2/kg) / mu_0  +  fan * (sum_i B_i*m_i + D-terms)

    B_i  = b0 + b1*exp(-b2*tc)                          Jones_Dole[0..2]
    D_i  = d1*exp(-d2*tc)                               Jones_Dole[3,4]
    D-term_i = D_i * m_i * ( I^d3*(1 + fI) + (m_i*f_z)^d3 ) / (2 + fI)
    d3   = Jones_Dole[5],  f_z = (z^2 + |z|)/2
    fI   = I/3/d3  (d3 >= 1);  -0.8/d3  (0.4 < d3 < 1);  -1 otherwise
    fan  = 2 - V_an/V_Cl-,  the anion-volume correction (1 for pure chloride)
    A    = the Falkenhagen-Dole electrostatic term, built from the ion tracer
           diffusion coefficients (`-dw` in pitzer.dat) and the Bradley &
           Pitzer dielectric constant.

    tc is capped at 200 degC, as PHREEQC does.

THE WATER LEG IS IAPWS-2008. PHREEQC computes `viscos_0` from Huber et al.
(2009) - the same correlation as `iapws_viscosity`, which this project already
verified to <0.034 ppm against Huber's own Table 6. The A term scales with it,
so pairing this salt leg with the Mao-Duan water leg would not reproduce
PHREEQC.

CI- IS THE REFERENCE SOLUTE: its `-viscosity` line is all zeros, so in a pure
NaCl brine the B and D terms come from Na+ alone and `fan` is exactly 1.
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
import os
import re

from brine_gas.appelo_volumes import CHARGE, Vm_ion, ionic_strength, parameters
from brine_gas.bradley_pitzer_dielectric import dielectric_constant
from brine_gas.iapws_if97 import rho_if97
from brine_gas.iapws_viscosity import mu_iapws2008

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pitzer.dat")

# PHREEQC's hard-coded reference value, transport.cpp: pure-water viscosity at
# 25 degC in mPa s. Used only to scale the tracer diffusion coefficients.
VISCOS_0_25 = 0.8900239182946

TC_MAX = 200.0          # transport.cpp caps tc for the viscosity model


# ============================================================================
# Parameter parsing
# ============================================================================

def _parse_numbers(text):
    return [float(x) for x in re.findall(
        r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text.split('#')[0])]


def species_name(reaction):
    """
    The species a SOLUTION_SPECIES reaction DEFINES, which is its first product.

    PHREEQC names a species by the right-hand side, not the left. Getting this
    wrong is silent: `H2O = OH- + H+` keys OH-'s parameters under 'H2O', and
    `CO3-2 + H+ = HCO3-` keys bicarbonate's under 'CO3-2 + H+'. The primary
    ions are unaffected because their reactions read `Na+ = Na+`, which is why
    the PHREEQC benchmark passed while bicarbonate would have silently
    contributed nothing.
    """
    rhs = reaction.split('=', 1)[1]
    first = rhs.split('#')[0].split(' + ')[0].strip()
    parts = first.split()
    if len(parts) > 1 and re.fullmatch(r'\d+(\.\d+)?', parts[0]):
        first = ' '.join(parts[1:])          # drop a stoichiometric coefficient
    return first


def load_parameters(database=None):
    """
    Parse the `-viscosity` and `-dw` lines out of pitzer.dat.

    Returns {species: {'jd': [10 floats], 'dw': Dw25 (m2/s), 'dw_t': K}}.
    Jones_Dole slots follow transport.cpp: 0-2 = b0,b1,b2; 3-4 = d1,d2;
    5 = d3; 6 = the anion factor; 7,8 = optional 25 degC overrides for b0,d1.
    """
    path = database or _DEFAULT_DB
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pitzer.dat not found at {path}. Install PHREEQC - see the recipe "
            f"in code/phreeqc_benchmark.py")

    out, species = {}, None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if not line[0].isspace() and '=' in stripped:
                species = species_name(stripped)
                continue
            if species is None:
                continue
            if stripped.startswith('-viscosity'):
                nums = _parse_numbers(stripped[len('-viscosity'):])
                jd = (nums + [0.0] * 10)[:10]
                out.setdefault(species, {})['jd'] = jd
            elif stripped.startswith('-dw'):
                nums = _parse_numbers(stripped[len('-dw'):])
                if nums:
                    rec = out.setdefault(species, {})
                    rec['dw'] = nums[0]
                    rec['dw_t'] = nums[1] if len(nums) > 1 else 0.0

    return out


_CACHE = {}


def viscosity_parameters(database=None):
    key = database or _DEFAULT_DB
    if key not in _CACHE:
        _CACHE[key] = load_parameters(database)
    return _CACHE[key]


# ============================================================================
# The model
# ============================================================================

def mu_water(T_K, P_MPa):
    """Pure-water viscosity in mPa s - IAPWS-2008, as PHREEQC uses."""
    return mu_iapws2008(T_K, rho_if97(T_K, P_MPa)) * 1e3


def _B_and_D(jd, tc, m_i, I, z):
    """The B*m and D*m terms for one ion, transport.cpp lines 6183-6195."""
    b0, b1, b2, d1, d2, d3 = jd[0], jd[1], jd[2], jd[3], jd[4], jd[5]

    # optional: values found at 25 degC held in slots 7, 8
    if jd[7] or jd[8]:
        b0 = jd[7] - b1 * math.exp(-b2 * 25.0)
        d1 = jd[8] / math.exp(-d2 * 25.0)

    B_term = (b0 + b1 * math.exp(-b2 * tc)) * m_i

    az = abs(z)
    f_z = (az * az + az) / 2.0 if az else (I / m_i)

    if d3 >= 1.0:
        fI = I / 3.0 / d3
    elif d3 > 0.4:
        fI = -0.8 / d3
    else:
        fI = -1.0

    D_term = ((d1 * math.exp(-d2 * tc)) * m_i
              * (I ** d3 * (1.0 + fI) + (m_i * f_z) ** d3) / (2.0 + fI))
    if D_term < -1e-5:
        D_term = 0.0
    return B_term, D_term


def _falkenhagen_dole_A(T_K, P_MPa, composition, params, mu_0):
    """
    The electrostatic term, returned already multiplied by mu_0 (mPa s), so
    that mu = mu_0 + A*sqrt(equivalents/2) as in transport.cpp.
    """
    eps_r = dielectric_constant(T_K, P_MPa * 10.0)
    scale = VISCOS_0_25 / mu_0

    m_plus = m_min = eq_plus = eq_min = eq_dw_plus = eq_dw_min = 0.0
    for sp, m_i in composition.items():
        z = CHARGE.get(sp, 0)
        if not z or m_i <= 0.0:
            continue
        rec = params.get(sp, {})
        Dw = rec.get('dw', 0.0)
        if not Dw:
            continue
        Dw *= scale
        dw_t = rec.get('dw_t', 0.0)
        if dw_t:
            Dw *= math.exp(dw_t / T_K - dw_t / 298.15)
        eq = m_i * abs(z)
        if z < 0:
            m_min += m_i
            eq_min += eq
            eq_dw_min += eq / Dw
        else:
            m_plus += m_i
            eq_plus += eq
            eq_dw_plus += eq / Dw

    if not (m_plus and m_min and eq_dw_plus and eq_dw_min):
        return 0.0

    z1, z2 = eq_plus / m_plus, eq_min / m_min
    D1, D2 = eq_plus / eq_dw_plus, eq_min / eq_dw_min

    t1 = ((D1 - D2)
          / (math.sqrt(D1 * z1 + D2 * z2) + math.sqrt((D1 + D2) * (z1 + z2))))
    psi = (D1 * z2 + D2 * z1) / 4.0 - z1 * z2 * t1 * t1
    A = (4.3787e-14 * T_K ** 1.5
         / (math.sqrt(eps_r * (z1 + z2) / max(z1, z2)) * (D1 * D2)) * psi)
    return A * math.sqrt((eq_plus + eq_min) / 2.0)


def _anion_factor(T_K, P_MPa, composition, params, I):
    """fan = 2 - V_an/V_Cl-, averaged over anions weighted by their molality."""
    P_bar = P_MPa * 10.0
    vm = parameters()
    V_an = m_an = 0.0
    V_Cl = None

    for sp, m_i in composition.items():
        if m_i <= 0.0:
            continue
        z = CHARGE.get(sp, 0)
        if z > 0:
            continue
        tan = params.get(sp, {}).get('jd', [0.0] * 10)[6]
        if sp == 'Cl-':
            V_Cl = Vm_ion('Cl-', T_K, P_bar, I, vm)
            V_an += V_Cl * m_i
            m_an += m_i
        elif tan and sp in vm:
            V_an += Vm_ion(sp, T_K, P_bar, I, vm) * tan * m_i
            m_an += m_i

    if not m_an:
        return 1.0
    if V_Cl is None:
        V_Cl = Vm_ion('Cl-', T_K, P_bar, I, vm)
    if not V_Cl:
        return 1.0
    return 2.0 - (V_an / m_an) / V_Cl


def viscosity_ratio(T_K, P_MPa, composition, database=None):
    """
    mu(solution)/mu(pure water) at the same T and P.

    composition: {ion: molality}, e.g. {'Na+': 3.0, 'Cl-': 3.0} or
                 {'Na+': 1.0, 'Ca+2': 1.0, 'Cl-': 3.0}.
    """
    params = viscosity_parameters(database)
    tc = min(T_K - 273.15, TC_MAX)
    I = ionic_strength(composition)
    mu_0 = mu_water(T_K, P_MPa)

    Bc = Dc = 0.0
    for sp, m_i in composition.items():
        if m_i <= 1e-9:
            continue
        if sp not in params or 'jd' not in params[sp]:
            # RAISE rather than silently contribute nothing - the same rule
            # salt_route follows for the volume leg. An ion with no
            # parameterisation is a gap in the model, not a zero.
            raise ValueError(
                f"{sp!r} carries no -viscosity parameters in pitzer.dat, so "
                f"this leg cannot represent it. Parameterised: "
                f"{', '.join(sorted(s for s, v in params.items() if 'jd' in v))}")
        jd = params[sp]['jd']
        if not (jd[0] or jd[1] or jd[3]):
            continue                      # Cl-, the reference solute (B = 0)
        B_term, D_term = _B_and_D(jd, tc, m_i, I, CHARGE.get(sp, 0))
        Bc += B_term
        Dc += D_term
    if Dc < 0.0:
        Dc = 0.0

    A_term = _falkenhagen_dole_A(T_K, P_MPa, composition, params, mu_0)
    fan = _anion_factor(T_K, P_MPa, composition, params, I)

    return 1.0 + A_term / mu_0 + fan * (Bc + Dc)


def brine_viscosity(T_K, P_MPa, composition, database=None):
    """Gas-free brine viscosity in mPa s (== cP)."""
    return mu_water(T_K, P_MPa) * viscosity_ratio(T_K, P_MPa, composition,
                                                  database)


def nacl_ratio(T_K, P_MPa, m):
    """Salt viscosity ratio for a pure NaCl brine of molality m."""
    if m <= 0.0:
        return 1.0
    return viscosity_ratio(T_K, P_MPa, {'Na+': m, 'Cl-': m})


# ============================================================================
# Reference values, captured from PHREEQC 3.8.6 itself (USER_PUNCH VISCOS and
# VISCOS_0). PHREEQC prints six significant figures, so the ratios below are
# good to about 1e-5 relative. `jones_dole_benchmark.py` runs the full 145-case
# sweep; these eight pin the module without needing the binary installed.
# (T_degC, P_MPa, composition, mu/mu_0)
# ============================================================================
PHREEQC_RATIOS = [
    (25.0, 1.0, {'Na+': 1.0, 'Cl-': 1.0}, 1.093460),
    (25.0, 1.0, {'Na+': 5.0, 'Cl-': 5.0}, 1.722441),
    (100.0, 35.0, {'Na+': 3.0, 'Cl-': 3.0}, 1.426921),
    (150.0, 100.0, {'Na+': 6.0, 'Cl-': 6.0}, 1.979729),
    (60.0, 20.0, {'Na+': 2.0, 'Ca+2': 1.0, 'Cl-': 4.0}, 1.678617),
    (100.0, 50.0, {'Na+': 2.0, 'Mg+2': 1.0, 'Cl-': 4.0}, 1.791507),
    (25.0, 1.0, {'Na+': 3.0, 'Cl-': 1.0, 'SO4-2': 1.0}, 1.653332),
    (120.0, 35.0, {'Na+': 2.0, 'K+': 0.3, 'Mg+2': 0.8, 'Ca+2': 0.5,
                   'Cl-': 4.9}, 1.955956),
]


def verify_against_phreeqc(verbose=True):
    """Score the module against PHREEQC's own output. Returns (worst%, n)."""
    worst = 0.0
    for T_C, P, comp, ref in PHREEQC_RATIOS:
        got = viscosity_ratio(T_C + 273.15, P, comp)
        d = 100.0 * (got / ref - 1.0)
        worst = max(worst, abs(d))
        if verbose:
            ions = '+'.join(f'{m:g} {s}' for s, m in comp.items())
            print(f'  {T_C:5.0f} C {P:6.1f} MPa  {ions:<40} '
                  f'{got:.6f} vs {ref:.6f}  {d:+.4f}%')
    return worst, len(PHREEQC_RATIOS)


if __name__ == '__main__':
    p = viscosity_parameters()
    n_visc = sum(1 for v in p.values() if 'jd' in v)
    print(f'parsed {n_visc} species with -viscosity parameters from pitzer.dat')
    print(f"  Na+ : {p['Na+']['jd'][:7]}")
    print(f"  Cl- : {p['Cl-']['jd'][:7]}  (the reference solute)")

    print('\nNaCl salt ratio mu(brine)/mu(water):')
    print(f"{'m':>5}  {'25 C':>9} {'50 C':>9} {'100 C':>9} {'150 C':>9}")
    for m in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        row = ''.join(f'{nacl_ratio(t + 273.15, 20.0, m):10.4f}'
                      for t in (25, 50, 100, 150))
        print(f'{m:5.1f} {row}')

    print('\nAgainst PHREEQC (the reference implementation):')
    worst, n = verify_against_phreeqc()
    print(f'  worst {worst:.4f}% over {n} cases')

    print('\nA mixed brine, 1 m NaCl + 1 m CaCl2 at 60 degC, 20 MPa:')
    r = viscosity_ratio(333.15, 20.0, {'Na+': 1.0, 'Ca+2': 1.0, 'Cl-': 3.0})
    print(f'  ratio {r:.4f}, viscosity '
          f'{brine_viscosity(333.15, 20.0, {"Na+": 1.0, "Ca+2": 1.0, "Cl-": 3.0}):.4f} mPa s')
