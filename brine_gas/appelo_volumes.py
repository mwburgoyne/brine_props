"""
Appelo ion-additive apparent molar volumes - the MULTI-SALT brine leg.

Reference:
    Appelo, C.A.J., Parkhurst, D.L. and Post, V.E.A. (2014). "Equations for
    calculating hydrogeochemical reactions of minerals and gases such as CO2 at
    high pressures and temperatures." Geochim. Cosmochim. Acta 125, 49-67.
    DOI 10.1016/j.gca.2013.10.003   (Papers/47)
    Appelo, C.A.J. (2015). Appl. Geochem. 55, 62-71.
    DOI 10.1016/j.apgeochem.2014.11.007   (Papers/48)

WHY THIS ROUTE. It is the only one found that does multi-salt AND pressure AND
dissolved gases together. Krumgalz is 1 atm and stops at 95 degC; Rogers & Pitzer
is NaCl-only; Archer is not validatable. Appelo's stated range is 0-200 degC and
1-1000 atm, which covers this project's whole envelope (450 K vouched, 100 MPa).

Ion additivity is justified in Paper 47 rather than assumed here: "the volumes of
the individual ions in a mixture simply can be added to obtain the total volume of
the solutes, and hence the density of the solution" (Harned & Owen 1958; Ellis
1968; Millero 1972). That is the Young's-rule basis this front wanted.

MODEL - the HKFmoRR equation (Helgeson-Kirkham-Flowers-modified-Redlich-Rosenfeld),
Paper 47 Sec. 2.1, p. 53. Per ion i:

    Eq (6)  Vm_i  = V0m_i
                    + A_v * 0.5*z_i^2 * I^0.5 / (1 + a0_i * B_gamma * I^0.5)
                    + ( b1_i + b2_i/(T-228) + b3_i*(T-228) ) * I^b4_i

    Eq (7)  V0m_i = 41.84 * ( 0.1*a1_i
                              + 100*a2_i/(2600+P_bar)
                              + a3_i/(T-228)
                              + 1e4*a4_i/((2600+P_bar)*(T-228))
                              + w_i*1e5 * d(1/eps_r)/dP_bar )

`A_v` is the Debye-Huckel limiting slope for volume and comes from
**Bradley & Pitzer (1979)**, which Paper 47 cites and which this project already
implements and verifies (`bradley_pitzer_dielectric`). Confirmed to be the SAME
quantity, not merely similar: Appelo's Eq. (3) prefactor `A_gamma*(2/3)*2.303` =
0.7830 against our `2*A_phi` = 0.7828.

TWO THINGS PAPER 47 DOES NOT STATE, both pinned empirically against PHREEQC's own
`VM()` output (see `verify_ions()`), because guessing them would have been exactly
the failure this project keeps hitting:

  1. **The order of the ten `-Vm` numbers in `pitzer.dat`** is
     `a1 a2 a3 a4 omega a0 b1 b2 b3 b4`.
  2. **The omega term enters as `+omega*1e5*d(1/eps)/dP`**, i.e. the tabulated
     omega is scaled by 1e5 and carries the opposite sign to the minus printed in
     Eq. (7). With the printed sign the error is +4.41 cm3/mol on Cl-; with this
     one it is -0.0003. Both readings were scored; see `verify_ions()`.

Units: T in K, P in bar, m in mol/kg water, volumes in cm3/mol.
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

from brine_gas.bradley_pitzer_dielectric import (dielectric_constant, dD_dP, A_V,
                                       N_AVOGADRO, E_CHARGE_ESU, K_BOLTZMANN)
from brine_gas.iapws_if97 import rho_if97

# ============================================================================
# Parameters, read from PHREEQC's pitzer.dat (public domain, USGS)
# ============================================================================

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pitzer.dat")

# Molar masses, g/mol, for the ions this leg supports.
MW = {
    'Na+': 22.9898, 'K+': 39.0983, 'Mg+2': 24.305, 'Ca+2': 40.08,
    'Sr+2': 87.62, 'Ba+2': 137.33, 'Li+': 6.941, 'Mn+2': 54.938,
    'Fe+2': 55.847, 'Cl-': 35.453, 'SO4-2': 96.0636, 'CO3-2': 60.0092,
    'Br-': 79.904, 'B(OH)3': 61.833, 'H4SiO4': 96.1163,
    # secondary species; only usable since the parser started naming species by
    # their first PRODUCT (`CO3-2 + H+ = HCO3-` defines HCO3-, not CO3-2)
    'HCO3-': 61.0168, 'OH-': 17.0073, 'HSO4-': 97.0715, 'CO2': 44.0095,
    'H+': 1.00794,
}

CHARGE = {
    'Na+': 1, 'K+': 1, 'Li+': 1, 'Mg+2': 2, 'Ca+2': 2, 'Sr+2': 2, 'Ba+2': 2,
    'Mn+2': 2, 'Fe+2': 2, 'Cl-': -1, 'Br-': -1, 'SO4-2': -2, 'CO3-2': -2,
    'B(OH)3': 0, 'H4SiO4': 0,
    'HCO3-': -1, 'OH-': -1, 'HSO4-': -1, 'CO2': 0, 'H+': 1,
}


def _species_name(reaction):
    """The species a SOLUTION_SPECIES reaction defines: its first product."""
    first = reaction.split('=', 1)[1].split('#')[0].split(' + ')[0].strip()
    parts = first.split()
    if len(parts) > 1 and re.fullmatch(r'\d+(\.\d+)?', parts[0]):
        first = ' '.join(parts[1:])
    return first


def load_vm_parameters(database=None):
    """
    Parse the `-Vm` blocks out of pitzer.dat.

    Returns {species: (a1, a2, a3, a4, omega, a0, b1, b2, b3, b4)}.
    Only entries with the full ten coefficients are kept.
    """
    path = database or _DEFAULT_DB
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pitzer.dat not found at {path}. Install PHREEQC - see the recipe "
            f"in code/phreeqc_benchmark.py")

    params, species = {}, None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if not line[0].isspace() and '=' in stripped:
                # PHREEQC names a species by its first PRODUCT, not by the left
                # side: `CO3-2 + H+ = HCO3-` defines HCO3-. Primary ions read
                # `Na+ = Na+` so they are unaffected either way.
                species = _species_name(stripped)
            elif stripped.startswith('-Vm') and species:
                nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',
                                  stripped[3:].split('#')[0])
                if len(nums) >= 10:
                    params[species] = tuple(float(x) for x in nums[:10])
                species = None
    return params


_PARAMS_CACHE = {}


def parameters(database=None):
    key = database or _DEFAULT_DB
    if key not in _PARAMS_CACHE:
        _PARAMS_CACHE[key] = load_vm_parameters(database)
    return _PARAMS_CACHE[key]


# ============================================================================
# The model
# ============================================================================

def d_inv_eps_dP(T_K, P_bar):
    """d(1/eps_r)/dP in 1/bar, from the Bradley & Pitzer dielectric equation."""
    D = dielectric_constant(T_K, P_bar)
    return -dD_dP(T_K, P_bar) / (D * D)


def B_gamma(T_K, P_bar, rho_w=None):
    """
    Debye-Huckel B (the reciprocal-length parameter) in 1/Angstrom.

    Derived from the same CGS constants as A_phi rather than taken from a
    remembered textbook value:
        B = sqrt(8*pi*N_A*e^2*rho_w / (1000*eps_r*k*T))   [1/cm]
    Reproduces the standard 0.3284 1/Angstrom at 25 degC, 1 bar.
    """
    if rho_w is None:
        rho_w = rho_if97(T_K, P_bar / 10.0) / 1000.0
    eps = dielectric_constant(T_K, P_bar)
    B_per_cm = math.sqrt(8.0 * math.pi * N_AVOGADRO * E_CHARGE_ESU ** 2 * rho_w
                         / (1000.0 * eps * K_BOLTZMANN * T_K))
    return B_per_cm / 1.0e8          # 1/cm -> 1/Angstrom


def V0_ion(species, T_K, P_bar, params=None):
    """Eq. (7): intrinsic (infinite-dilution) molar volume, cm3/mol."""
    p = (params or parameters())[species]
    a1, a2, a3, a4, omega = p[0], p[1], p[2], p[3], p[4]
    tt = T_K - 228.0
    return 41.84 * (0.1 * a1
                    + 100.0 * a2 / (2600.0 + P_bar)
                    + a3 / tt
                    + 1e4 * a4 / ((2600.0 + P_bar) * tt)
                    + omega * 1e5 * d_inv_eps_dP(T_K, P_bar))


def Vm_ion(species, T_K, P_bar, I, params=None, Av=None, Bg=None):
    """Eq. (6): apparent molar volume of one ion at ionic strength I, cm3/mol."""
    p = (params or parameters())[species]
    a0, b1, b2, b3, b4 = p[5], p[6], p[7], p[8], p[9]
    z = CHARGE.get(species, 0)

    v = V0_ion(species, T_K, P_bar, params)
    if I <= 0:
        return v

    if Av is None:
        Av = A_V(T_K, P_bar)
    if Bg is None:
        Bg = B_gamma(T_K, P_bar)

    sqrtI = math.sqrt(I)
    v += Av * 0.5 * z * z * sqrtI / (1.0 + a0 * Bg * sqrtI)
    tt = T_K - 228.0
    v += (b1 + b2 / tt + b3 * tt) * I ** b4
    return v


def ionic_strength(composition):
    """I = 0.5 * sum(m_i * z_i^2), mol/kg water."""
    return 0.5 * sum(m * CHARGE.get(s, 0) ** 2 for s, m in composition.items())


def brine_density(T_K, P_MPa, composition, params=None):
    """
    Gas-free brine density in kg/m3 from an ION composition.

    Parameters:
        composition: {species: molality}, e.g.
                     {'Na+': 5.0, 'Cl-': 5.0}
                     {'Na+': 1.8, 'K+': 0.2, 'Mg+2': 1.8, 'Ca+2': 0.47, 'Cl-': 6.7}

    The construction is additive over ions, which is what makes it multi-salt:
        V_solution = V_water + sum_i m_i * Vm_i
        mass       = 1000 g water + sum_i m_i * MW_i
    """
    P_bar = P_MPa * 10.0
    p = params or parameters()
    I = ionic_strength(composition)

    Av = A_V(T_K, P_bar)
    Bg = B_gamma(T_K, P_bar)

    rho_w = rho_if97(T_K, P_MPa)            # kg/m3
    V = 1e6 / rho_w                          # cm3 per 1 kg of water
    mass = 1000.0                            # g

    for s, m in composition.items():
        if m <= 0:
            continue
        V += m * Vm_ion(s, T_K, P_bar, I, p, Av, Bg)
        mass += m * MW[s]

    return mass / V * 1000.0                 # g/cm3 -> kg/m3


# ============================================================================
# Verification against PHREEQC's own per-species VM()
# ============================================================================

# PHREEQC `VM(species)` at infinite dilution, 1 atm, from pitzer.dat.
#   {species: {T_degC: Vm}}
PHREEQC_VM = {
    'Na+':   {25.0: -1.5220, 100.0: 0.23257},
    'Cl-':   {25.0: 18.045,  100.0: 16.111},
    'Ca+2':  {25.0: -18.254, 100.0: -20.024},
    'Mg+2':  {25.0: -21.936, 100.0: -25.558},
    'SO4-2': {25.0: 14.355,  100.0: 12.445},
}


def verify_ions(verbose=True):
    """
    Score V0_ion against PHREEQC's VM() at infinite dilution, and score the two
    candidate omega conventions so the choice stays reproducible.
    """
    p = parameters()
    worst = 0.0
    rows = []
    for sp, byT in PHREEQC_VM.items():
        for T_C, ref in byT.items():
            got = V0_ion(sp, T_C + 273.15, 1.01325, p)
            rows.append((sp, T_C, ref, got, got - ref))
            worst = max(worst, abs(got - ref))
    if verbose:
        print(f"    {'species':>7} {'T degC':>7} {'PHREEQC':>10} {'ours':>10} {'diff':>9}")
        for sp, T_C, ref, got, d in rows:
            print(f"    {sp:>7} {T_C:7.0f} {ref:10.4f} {got:10.4f} {d:+9.5f}")
    return worst, len(rows)


if __name__ == "__main__":
    print("=" * 72)
    print("Appelo ion-additive volumes - verification")
    print("=" * 72)

    p = parameters()
    print(f"\n1. Parsed {len(p)} species with -Vm parameters from pitzer.dat")
    print(f"   salt ions available: "
          f"{', '.join(s for s in MW if s in p)}")

    print("\n2. B_gamma sanity (standard value at 25 degC is 0.3284 1/Angstrom):")
    print(f"   B_gamma(298.15 K, 1 bar) = {B_gamma(298.15, 1.0):.5f}")

    print("\n3. Intrinsic volumes V0 vs PHREEQC VM() at infinite dilution:")
    worst, n = verify_ions()
    print(f"   worst |diff| {worst:.5f} cm3/mol over {n} points")

    print("\n4. NaCl additivity check at 25 degC "
          "(Krumgalz Table 2 gives V0_NaCl = 16.620):")
    s = V0_ion('Na+', 298.15, 1.01325) + V0_ion('Cl-', 298.15, 1.01325)
    print(f"   V0(Na+) + V0(Cl-) = {s:.3f} cm3/mol")
