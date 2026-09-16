"""
Dispatcher for the SALT LEG of the modular brine-volume construction.

    V_solution = V_water(IAPWS-IF97) + V_salt(THIS) + n_gas * V_phi,gas(S&W PR)

DEFAULT: 'appelo' since 2026-07-26.

    'appelo'  Appelo, Parkhurst & Post (2014) ion-additive volumes.
              MULTI-SALT, 0-200 degC, 1-1000 atm. Reproduces PHREEQC (the
              reference implementation of the same model) to 0.0095% mean /
              0.0274% max over 124 cases.
    'rogers'  Rogers & Pitzer (1982). **SUPERSEDED - NaCl ONLY.** Retained
              only for reproducibility of pre-2026-07-26 results.

WHY APPELO SUPERSEDED ROGERS & PITZER. Against Calabrese's 96 measured gas-free
NaCl densities the two are a dead heat - 0.0448% mean for Appelo, 0.0443% for
Rogers & Pitzer, with an identical +0.041% bias - but Appelo is strictly better on
the max (0.146% vs 0.199%) and strictly dominates on capability:

| | Appelo | Rogers & Pitzer |
|---|---|---|
| salts | **any combination of 13 ions** | NaCl only |
| range | 0-200 degC, 1-1000 atm | 0-300 degC, to 1 kbar, 5.5 molal |
| verification | **a free reference implementation (PHREEQC)** | the paper's own Table A-1 |
| printed defects found | 0 | 2 |

THIS IS NOT THE SHIPPED GAS-FREE BRINE DENSITY. The delivered pipeline still
takes `rho1` from **Spivey**, which beats both on NaCl (0.027% mean / 0.093% max)
and is the reference a petroleum audience expects. See [[PIPELINE]]. This module
is the salt leg of the MODULAR construction, which is what makes multi-salt
possible at all.

KNOWN LIMIT - free-ion additivity. This sums FREE ion volumes; it does not
speciate. That is fine where ions stay largely unassociated, which covers
chloride and most sulfate brines (verified to 0.02% against PHREEQC at ionic
strength up to 12). **It breaks where strong ion pairs or mineral saturation
intervene: a CaSO4-rich brine deviates +1.09% at 25 degC and +1.80% at 100 degC.**
Guard against that composition or speciate first.
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


DEFAULT_ROUTE = 'appelo'
ROUTES = ('appelo', 'rogers')

# Molar masses used to turn a molality of a SALT into ion molalities.
_SALT_IONS = {
    'NaCl':  {'Na+': 1.0, 'Cl-': 1.0},
    'KCl':   {'K+': 1.0, 'Cl-': 1.0},
    'CaCl2': {'Ca+2': 1.0, 'Cl-': 2.0},
    'MgCl2': {'Mg+2': 1.0, 'Cl-': 2.0},
    'CaBr2': {'Ca+2': 1.0, 'Br-': 2.0},
    'Na2SO4': {'Na+': 2.0, 'SO4-2': 1.0},
    'SrCl2': {'Sr+2': 1.0, 'Cl-': 2.0},
    'BaCl2': {'Ba+2': 1.0, 'Cl-': 2.0},
}

# Compositions where free-ion additivity is known to fail (see module docstring).
_PAIRING_RISK = ({'Ca+2', 'SO4-2'}, {'Ba+2', 'SO4-2'}, {'Sr+2', 'SO4-2'})


def ions_from_salts(salts):
    """
    Convert {salt: molality} into {ion: molality}.

    >>> ions_from_salts({'NaCl': 3.0, 'CaCl2': 1.0})
    {'Na+': 3.0, 'Cl-': 5.0, 'Ca+2': 1.0}
    """
    out = {}
    for salt, m in salts.items():
        if salt not in _SALT_IONS:
            raise ValueError(
                f"unknown salt {salt!r}; known: {sorted(_SALT_IONS)}")
        for ion, nu in _SALT_IONS[salt].items():
            out[ion] = out.get(ion, 0.0) + nu * m
    return out


def pairing_warning(composition):
    """
    Return a warning string if the composition is one where free-ion additivity
    is known to break, else None. Cheap guard; call it, do not silently proceed.
    """
    present = {ion for ion, m in composition.items() if m > 0}
    for risky in _PAIRING_RISK:
        if risky <= present:
            return (f"composition contains {sorted(risky)}: free-ion additivity "
                    f"is unreliable here (CaSO4-rich brine deviates +1.1% at "
                    f"25 degC, +1.8% at 100 degC against PHREEQC). Speciate or "
                    f"treat the result as indicative.")
    return None


def brine_density(T_K, P_MPa, composition=None, salts=None, route=None,
                  check_pairing=True):
    """
    Gas-free brine density in kg/m3 from the modular salt leg.

    Supply EITHER `composition` as {ion: molality} OR `salts` as
    {salt: molality}; the latter is converted with `ions_from_salts`.

    route: 'appelo' (default) or 'rogers' (NaCl only, superseded).
    """
    if (composition is None) == (salts is None):
        raise ValueError("supply exactly one of composition= or salts=")
    if salts is not None:
        composition = ions_from_salts(salts)

    route = route or DEFAULT_ROUTE
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")

    if check_pairing:
        warn = pairing_warning(composition)
        if warn:
            import warnings
            warnings.warn(warn, RuntimeWarning, stacklevel=2)

    if route == 'appelo':
        from brine_gas.appelo_volumes import brine_density as _rho
        return _rho(T_K, P_MPa, composition)

    # superseded NaCl-only route
    non_nacl = {k: v for k, v in composition.items()
                if k not in ('Na+', 'Cl-') and v > 0}
    if non_nacl:
        raise ValueError(
            f"route='rogers' is NaCl only and cannot represent {sorted(non_nacl)}. "
            f"Use the default 'appelo' route.")
    from brine_gas.pitzer_brine_density import brine_density as _rho
    return _rho(T_K, P_MPa, composition.get('Na+', 0.0))


def route_used(composition, route=None):
    """Which route a call would take, for logging and figures."""
    return route or DEFAULT_ROUTE


if __name__ == "__main__":
    print("Salt-leg dispatcher - default route:", DEFAULT_ROUTE)
    print()
    print("NaCl 5 molal, 100 degC, 30 MPa:")
    for r in ROUTES:
        rho = brine_density(373.15, 30.0, salts={'NaCl': 5.0}, route=r)
        print(f"   route={r:8s} {rho:9.3f} kg/m3")

    print("\nMulti-salt (only the default can do this):")
    for salts in ({'NaCl': 3.0, 'CaCl2': 1.0},
                  {'NaCl': 2.0, 'KCl': 0.3, 'MgCl2': 0.8, 'CaCl2': 0.5},
                  {'CaBr2': 3.0}):
        rho = brine_density(373.15, 30.0, salts=salts)
        print(f"   {str(salts):<52} {rho:9.3f} kg/m3")

    print("\nThe superseded route refuses what it cannot represent:")
    try:
        brine_density(373.15, 30.0, salts={'NaCl': 3.0, 'CaCl2': 1.0},
                      route='rogers')
    except ValueError as e:
        print(f"   ValueError: {e}")

    print("\nPairing guard:")
    print("  ", pairing_warning({'Ca+2': 0.5, 'SO4-2': 1.5, 'Na+': 2.0}))
