"""
Garcia (2001) density mixing rule for gas-dissolved aqueous solutions.

Single gas (Garcia Eq. 18):
    ρ = (1 + x2·M2/(M1·x1)) / (x2·V_phi/(M1·x1) + 1/ρ1)

Mixed gas (mole-fraction-weighted):
    V_phi_eff = Σ yi·V_phi_i
    M2_eff = Σ yi·M2_i
    where yi = mole fraction of gas i among dissolved gases (Σyi = 1)
    Then apply single-gas formula with V_phi_eff, M2_eff.

Brine support - SOLVENT BASIS (corrected 2026-07-25):
    The apparent molar volume is defined by V_phi = (V_solution - V_solvent)/n_gas
    with the solvent being the WHOLE gas-free brine, so n_gas·V_phi must be added
    to the volume of water AND salt. Writing the rule per mole of salt-free basis
    (x1 mol water + x2 mol gas, x1 + x2 = 1), the gas-free brine mass is

        W = x1·M1/(1 - S)        [S = weight fraction NaCl]

    since the salt/water mass ratio is S/(1-S), and

        ρ = (W + x2·M2) / (W/ρ1 + x2·V_phi)

    with ρ1 the gas-free brine density. This reduces exactly to the pure-water
    form at S = 0 and is non-singular as x2 → 1.

    The earlier form paired M1 = MW(water) with ρ1 = brine density, which omitted
    the salt mass from both the numerator and the solvent volume and so inflated
    the dissolved-gas density effect by the factor 1/(1-S) = 1 + salt/water mass
    (+5.6% at 1 mol/kg, +11% at 2 mol/kg, +29% at 5 mol/kg NaCl).

    V_phi values remain from the Plyasunov model at freshwater conditions (Garcia
    found salinity effects on V_phi are weak and within experimental uncertainty;
    Calabrese 2019 Eq. 18 carries pure-water V_phi into brine on the same basis).

Units:
    T in K, P in MPa, S in weight fraction NaCl
    x2 = total dissolved gas mole fraction in liquid phase, salt-free basis
    ρ in kg/m³, V_phi in cm³/mol, M in g/mol
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
from brine_gas.water_properties import rho_w, MW_WATER
from brine_gas.plyasunov_model import gas_mw
from brine_gas.vphi_route import V_phi, DEFAULT_ROUTE, route_used
from brine_gas.brine_properties import rho_brine, molality_from_salinity


def _check_inputs(x2, S):
    """Validate dissolved-gas fraction and salinity before use."""
    if not 0.0 <= x2 < 1.0:
        raise ValueError(
            f"x2 must lie in [0, 1); got {x2}. x2 is the total dissolved-gas "
            "mole fraction on a salt-free basis."
        )
    if not 0.0 <= S < 0.30:
        raise ValueError(
            f"S must lie in [0, 0.30) weight fraction NaCl; got {S}. "
            "Halite saturation is about 0.26 at ambient temperature."
        )


def _garcia(x2, M2, vphi, rho1_gcc, S):
    """
    Garcia Eq. 18 on the consistent gas-free-brine solvent basis.

        ρ = (W + x2·M2) / (W/ρ1 + x2·V_phi),    W = (1-x2)·M1/(1-S)

    W is the gas-free brine mass carried by (1-x2) moles of water, so the
    dissolved-gas volume x2·V_phi is added to the full brine volume rather
    than to the water-only fraction of it. Reduces to the pure-water form at
    S = 0 and stays finite as x2 → 1.

    Units: M2 in g/mol, vphi in cm³/mol, rho1_gcc in g/cm³. Returns g/cm³.
    """
    W = (1.0 - x2) * MW_WATER / (1.0 - S)
    return (W + x2 * M2) / (W / rho1_gcc + x2 * vphi)


def density_single_gas(gas, x2, T, P, rho1=None, S=0.0, route=DEFAULT_ROUTE):
    """
    Solution density with a single dissolved gas using Garcia Eq. 18.

    Parameters:
        gas: gas name string ('CO2', 'CH4', etc.)
        x2: mole fraction of dissolved gas in liquid phase, salt-free basis
            (denominator counts water + dissolved gas only, no ions)
        T: temperature in K
        P: pressure in MPa
        rho1: gas-free solvent density in kg/m³ (computed from S if None)
        S: salinity as weight fraction NaCl (default 0 = pure water)
        route: V_phi source. 'auto' (default) is the S&W modified-PR route with
            one volume shift per gas, falling back to Plyasunov where PR is not
            calibrated. See vphi_route.py for why PR is preferred and where the
            fallback bites.

    S must be supplied whenever rho1 is a brine density: it carries the salt
    mass into the mixing rule, and omitting it inflates the dissolved-gas
    density effect by 1/(1-S).

    Returns:
        solution density in kg/m³
    """
    _check_inputs(x2, S)

    if rho1 is None:
        rho1 = rho_brine(T, P, S) if S > 0 else rho_w(T, P)

    if x2 <= 0:
        return float(rho1)

    # V_phi carries the literature-anchored salt shift, so salinity enters the
    # molar volume as well as the solvent mass.
    v = V_phi(gas, T, P, route, molality_from_salinity(S))
    rho_gcc = _garcia(x2, gas_mw(gas), v, rho1 / 1000.0, S)
    return rho_gcc * 1000.0  # g/cm³ → kg/m³


def density_mixed_gas(gas_dict, T, P, rho1=None, S=0.0, route=DEFAULT_ROUTE):
    """
    Solution density with multiple dissolved gases.

    Uses mole-fraction-weighted effective V_phi and MW, then applies
    Garcia Eq. 18 with the effective values.

    Parameters:
        gas_dict: dict of {gas_name: x2_i} where x2_i is the mole fraction
                  of that gas in the liquid phase, salt-free basis.
                  Sum of all x2_i = total x2.
        T: temperature in K
        P: pressure in MPa
        rho1: gas-free solvent density in kg/m³ (computed from S if None)
        S: salinity as weight fraction NaCl (default 0 = pure water)
        route: V_phi source, as for density_single_gas.

    Returns:
        solution density in kg/m³
    """
    x2_total = sum(gas_dict.values())
    _check_inputs(x2_total, S)

    if rho1 is None:
        rho1 = rho_brine(T, P, S) if S > 0 else rho_w(T, P)

    if x2_total <= 0:
        return float(rho1)

    # Compute mole-fraction-weighted effective properties
    # yi = x2_i / x2_total (fraction of gas i among all dissolved gases)
    vphi_eff = 0.0
    mw_eff = 0.0
    for gas, x2_i in gas_dict.items():
        yi = x2_i / x2_total
        vphi_eff += yi * V_phi(gas, T, P, route, molality_from_salinity(S))
        mw_eff += yi * gas_mw(gas)

    rho_gcc = _garcia(x2_total, mw_eff, vphi_eff, rho1 / 1000.0, S)
    return rho_gcc * 1000.0


# ============================================================================
# VISCOSITY CORRECTION
# ============================================================================
#
# Experimentally-calibrated viscosity corrections for dissolved gases.
# Each gas uses the best available experimental source:
#
#   CO2: Calabrese et al. (2019), JCED 64, 3831-3847, Eq. 25
#        ln(mu/mu_brine) = 65.560 * exp(-2.468*(T_K/142 - 1)) * x_CO2
#        T-dependent (+12% at 275 K falling below +3% at 449 K); increment
#        independent of salt type and molality (m <= 6). Supersedes the
#        T-independent Islam & Carlson (2012) form (adopted 2026-07-19).
#
#   CH4: Ostermann, Bloori & Dehghani (1985), SPE 14211
#        Confirmed by Ostermann et al. (1986), SPE 15081
#        mu_sat/mu_free = 1.109 - 5.98e-4*T + 1.0933e-6*T^2  (T in degF)
#        NOTE: SPE 14211 prints 1.0933e-5 (TYPO); verified 1.0933e-6 from
#        stated plateau values (1.060 at 100F, 1.044 at 150F, 1.028 at 250F).
#        Plateau above ~2000 psi; T-dependent, not mole-fraction-dependent.
#
#   H2S: Murphy & Gaines (1974), J. Chem. Eng. Data 19(4), 359-362
#        mu = mu_brine * (1 + 1.70 * x_H2S)
#        Calibrated from 3-6% increase near saturation at 28-30 degC.
#        x_H2S from Burgess & Germann (1969), the solubility source the paper
#        itself used. Implied 'a' spans 0.26 to 2.20 across the 5 data points
#        (large scatter; authors note the effect is "only slightly more than
#        experimental error").
#        T-dependence unresolved: the 35 degC M&G point implies near-zero, but
#        Dehaghani 2023 MD (Papers/20; see h2s_md_dehaghani_analysis.py) shows
#        the per-mole effect persisting and growing over 50-120 degC. Joint
#        exponential fit unsupportable; constant a retained as central
#        estimate.
#
#   C2H6: NO EFFECT — Ostermann et al. (1986) SPE 15081
#   N2:   NO EFFECT — Murphy & Gaines (1974) control experiment
#   H2, C3H8, nC4H10: No data — no correction applied (conservative)
#
# NOTE: A previous density-based scaling approach (a_i = a_CO2 * drho%_i/drho%_CO2)
# was found to be fundamentally wrong. It predicted large viscosity DECREASES for
# CH4 (a=-11.0) and H2S (a=-0.9), but experiments show CH4 INCREASES viscosity
# by up to 6% and H2S INCREASES by 3-6%. No mechanism is asserted: the
# corrections are calibrated to measurement and nothing else.

# H2S: refitted 2026-07-25 from the PRIMARY source, Murphy & Gaines (1974),
# DOI 10.1021/je60063a015, Table IV (Papers/10_...), and re-based the same day on
# the solubility source THAT PAPER ITSELF USED.
#
# Two corrections, in order:
#   1. The pre-2026-07-25 value of 1.50 came from a second-hand reading in which
#      two of the five viscosity ratios were recorded as 1.033 where the paper
#      prints 1.038. Reading the paper gave 1.64.
#   2. Murphy & Gaines report no mole fractions; they state their H2S solubility
#      came from Burgess & Germann (1969), AIChE J 15:272 (Papers/34). Using
#      Soreide-Whitson instead put x_H2S 4.5 to 9.2% HIGH against that source,
#      and since a scales as 1/x it biased the coefficient low by the same
#      amount. Re-basing on Burgess & Germann Table 5 (image-verified) gives the
#      implied coefficients 2.20, 1.47, 1.89, 1.25 and 0.26, and the adopted
#      basis (mean of the four below 35 degC, excluding the near-null) is 1.70.
#
# EXPONENT SET TO UNITY 2026-07-31. The prior form 1 + 1.79*x^1.0134 carried an
# exponent borrowed from Islam-Carlson's CO2 fit, never fitted to H2S. The five
# points cannot distinguish exponents (RMS 1.740 pp at b=1, 1.738 at b=1.0134;
# delivered factor moves 0.01 pp at x=0.03), so the linear dilute-limit form is
# kept and the coefficient refitted on the same basis: a = 1.70.
#
# CAVEAT ON THE RE-BASING: Burgess & Germann Table 5 starts at 30 degC, so the
# three 28.1 degC points sit 1.9 degC below its lowest row and are mildly
# extrapolated. Murphy & Gaines chose those conditions deliberately close to the
# hydrate boundary, which is presumably why the table stops where it does. The
# 9% change is immaterial against the 0.26 to 2.20 spread of the five points;
# the value follows the sources. See code/murphy_gaines_h2s_refit.py.
_H2S_A = 1.70          # Murphy & Gaines Table IV on Burgess & Germann solubility

# Calabrese et al. (2019) Eq. 25 for CO2 (JCED 64:3831, DOI 10.1021/acs.jced.9b00248):
# ln(eta/eta_brine) = e1 * exp(-e2*(T/T0 - 1)) * x_CO2
# e1, e2 inherited unchanged from McBride-Wright 2015 CO2-water (274-449 K, to
# 100 MPa); Calabrese's N=415 fit is the gas-free NaCl baseline. Salt-type and
# molality independence is their tested hypothesis ("not strongly confirmed";
# assembled model AARD 0.9% NaCl, 2.3% CaCl2, m <= 6).
# Supersedes Islam-Carlson (T-independent 1 + 4.65*x^1.0134), which matches
# Calabrese only near its 25-35 degC calibration window and overstates the
# slope 3.5x at 200 degF (adopted as standard 2026-07-19).
_CAL_E1 = 65.560
_CAL_E2 = 2.468
_CAL_T0 = 142.0        # K

# Ostermann (1985) Eq. 10 / (1986) Eq. 7 coefficients for CH4
# mu_sat/mu_free = c0 + c1*T + c2*T^2  where T in degF
# NOTE: SPE 14211 prints the quadratic coefficient as 1.0933e-5, but this
# is a TYPOGRAPHICAL ERROR. Back-calculation from the stated plateau values
# (1.060 at 100F, 1.044 at 150F, 1.028 at 250F) proves the correct
# exponent is 10^-6. SPE 15081 prints the linear term as -5.93e-4 (vs
# -5.98e-4 in SPE 14211); this minor difference does not affect results
# at the stated precision.
# CH4 - unified Arrhenius x Langmuir, fitted 2026-07-25 to the FULL Ostermann
# SPE 14211 dataset (Tables 1a-1c, 23 points, Papers/9_...). See
# code/ostermann_ch4_refit.py for the fit and the transcribed data.
#
#     mu_sat/mu_free = 1 + A * exp(B/T_K) * x/(K + x)
#
# One equation, three parameters, no cap and no min(). Arrhenius in T because
# viscous flow is activated and the hydration structure carrying the excess is
# destroyed thermally (B = 1240 K, 10.3 kJ/mol, about half a water hydrogen
# bond); Langmuir in x because the structurable capacity is finite. Linear in x
# as x -> 0, as the dilute limit requires, and asymptoting naturally in both
# variables.
#
# This REPLACES the previous four-parameter construction (a linear-in-x ramp
# spliced to a temperature-only plateau by min()). Reading the primary source
# overturned two beliefs that construction rested on: the measured plateau is
# not flat (it drifts up 1.0-1.6 pp between 2000 and 7000 psi), and Rsw is
# reported directly so x need not come from an EOS. Fit quality on the 23
# measurements: this form 0.723 pp RMS with 3 parameters, the old capped ramp
# 0.713 pp with 4, an unsaturated linear form 1.146 pp with 2 (so the
# saturation is real and required).
#
# Ostermann's printed quadratic in degF is not used at all: three coefficients
# on three summary values is an exact interpolation with no degrees of freedom,
# and it turns upward past its 273.5 degF vertex.
# Input guard (2026-09-05, independent review item 2.4): an unknown gas name
# or a mole fraction outside [0, 1] used to return 1.0 silently, so a spelling
# mistake was indistinguishable from a supported no-effect gas.
VISCOSITY_CALIBRATED = frozenset({'CO2', 'CH4', 'H2S'})
VISCOSITY_NO_EFFECT = frozenset({'N2', 'H2', 'C2H6', 'C3H8', 'NC4H10'})
VISCOSITY_GASES = VISCOSITY_CALIBRATED | VISCOSITY_NO_EFFECT

_CH4_A = 1.71739196e-03
_CH4_B = 1239.77535        # K
_CH4_K = 1.52860547e-03    # half-saturation mole fraction


def _ch4_viscosity_ratio(x2, degf):
    """
    CH4-saturated brine viscosity ratio, unified Arrhenius x Langmuir form.

    mu_sat/mu_free = 1 + A exp(B/T) x/(K + x), fitted to all 23 Ostermann
    SPE 14211 measurements. Saturates naturally in x, so no cap is applied.
    """
    T_K = (degf - 32.0) / 1.8 + 273.15
    return 1.0 + _CH4_A * np.exp(_CH4_B / T_K) * x2 / (_CH4_K + x2)


def viscosity_correction_single(gas, x2, degf=None):
    """
    Viscosity multiplier for a single dissolved gas.

    Uses experimentally-calibrated corrections:
      - CO2: Calabrese (2019) Eq. 25, T-dependent (requires degf)
      - CH4: Ostermann (1985) T-dependent plateau (requires degf)
      - H2S: Calibrated from Murphy & Gaines (1974)
      - All others: no correction (1.0)

    mu_corrected = mu_brine * viscosity_correction_single(gas, x2, degf)

    Parameters:
        gas: gas name string ('CO2', 'CH4', 'H2S', etc.)
        x2: mole fraction of dissolved gas in liquid phase
        degf: temperature in degrees Fahrenheit (required for CO2 and CH4)

    Returns:
        viscosity multiplier (>= 1.0 for all calibrated gases)
    """
    gas = gas.upper()
    if gas not in VISCOSITY_GASES:
        raise ValueError(
            f"unknown gas {gas!r}; supported: {sorted(VISCOSITY_GASES)} "
            f"(of which {sorted(VISCOSITY_NO_EFFECT)} carry no correction)")
    if not 0.0 <= x2 <= 1.0:
        raise ValueError(f"dissolved mole fraction must be in [0, 1], got {x2}")
    if x2 == 0:
        return 1.0

    if gas == 'CO2':
        if degf is None:
            raise ValueError("degf (temperature) is required for CO2 viscosity correction")
        T_K = (degf - 32.0) / 1.8 + 273.15
        return float(np.exp(_CAL_E1 * np.exp(-_CAL_E2 * (T_K / _CAL_T0 - 1.0)) * x2))

    if gas == 'H2S':
        return 1.0 + _H2S_A * x2

    if gas == 'CH4':
        if degf is None:
            raise ValueError("degf (temperature) is required for CH4 viscosity correction")
        # Unified Arrhenius x Langmuir, fitted to all 23 Ostermann SPE 14211
        # measurements. Saturates in x by construction, so no cap is needed.
        return float(_ch4_viscosity_ratio(x2, degf))

    # VISCOSITY_NO_EFFECT: C2H6 and N2 measured null at low loading (Ostermann
    # 1986, Murphy & Gaines 1974); H2, C3H8, NC4H10 unmeasured, zero assumed.
    return 1.0


def viscosity_correction_mixed(gas_dict, degf=None):
    """
    Viscosity multiplier for multiple dissolved gases.

    Applies multiplicative corrections:
        mu = mu_brine * Product_i(correction_i)

    Parameters:
        gas_dict: dict of {gas_name: x2_i}
        degf: temperature in degrees Fahrenheit (required if CO2 or CH4 present)

    Returns:
        combined viscosity multiplier
    """
    total = sum(gas_dict.values())
    if total > 1.0:
        raise ValueError(f"dissolved mole fractions sum to {total}, above 1")
    factor = 1.0
    for gas, x2 in gas_dict.items():
        factor *= viscosity_correction_single(gas, x2, degf=degf)
    return factor


def gas_saturated_viscosity(T, P, gas_dict=None, composition=None, salts=None,
                            m=None, S=None, route=None):
    """
    The whole delivered viscosity chain, in one call. Returns mPa s (== cP).

        mu = mu_water(T, P) * salt ratio * pressure factor * per-gas corrections

    The gas-free part comes from `viscosity_route` (default: IAPWS-2008 water,
    ion-additive Jones-Dole salt term, Kestin pressure factor); the per-gas
    corrections are the experimentally calibrated ones above.

    Salinity may be given as `salts={salt: molality}`, `composition={ion:
    molality}`, `m=<NaCl molality>` or `S=<NaCl weight fraction>`. **A salinity
    with no species named is NaCl.**

    Parameters:
        gas_dict: {gas: x2} mole fractions of dissolved gas, or None for
                  gas-free brine
    """
    from brine_gas.viscosity_route import brine_viscosity

    mu_b = brine_viscosity(T, P, composition=composition, salts=salts, m=m,
                           S=S, route=route)
    if not gas_dict:
        return mu_b
    degf = (T - 273.15) * 1.8 + 32.0
    return mu_b * viscosity_correction_mixed(gas_dict, degf=degf)


def density_change_pct(gas_or_dict, x2_or_none, T, P, S=0.0):
    """
    Percentage density change relative to solvent (water or brine).

    Parameters:
        gas_or_dict: either a gas name string, or a dict {gas: x2_i}
        x2_or_none: mole fraction if gas_or_dict is a string, None if dict
        T, P: temperature (K) and pressure (MPa)
        S: salinity as weight fraction NaCl (default 0 = pure water)

    Returns:
        (rho_solution, rho_solvent, percent_change)
    """
    rho1 = rho_brine(T, P, S) if S > 0 else rho_w(T, P)
    if isinstance(gas_or_dict, dict):
        rho_sol = density_mixed_gas(gas_dict=gas_or_dict, T=T, P=P, rho1=rho1, S=S)
    else:
        rho_sol = density_single_gas(gas_or_dict, x2_or_none, T, P, rho1=rho1, S=S)
    pct = 100.0 * (rho_sol - rho1) / rho1
    return rho_sol, rho1, pct


# ============================================================================
# MAIN - VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Validate against Garcia (2001) CO2 results
    # Garcia reports ~2.5% density increase at x_CO2 = 0.05
    print("=== Single Gas Density Tests ===\n")

    T_test = 298.15  # 25°C
    P_test = 10.0    # 10 MPa (representative reservoir P)

    rho1 = rho_w(T_test, P_test)
    print(f"Pure water at T={T_test}K, P={P_test}MPa: ρ = {rho1:.4f} kg/m³")
    print()

    gases = ['CO2', 'CH4', 'H2S', 'N2', 'H2', 'C2H6', 'C3H8', 'NC4H10']
    x2_test = 0.02  # 2 mol% dissolved gas

    print(f"Dissolved gas mole fraction x2 = {x2_test}")
    print(f"{'Gas':<8} {'MW':>6} {'V_phi':>8} {'ρ_sol':>10} {'Δρ':>8} {'%change':>8}")
    print("-" * 55)

    for gas in gases:
        vphi = V_phi(gas, T_test, P_test)
        rho_sol = density_single_gas(gas, x2_test, T_test, P_test, rho1=rho1)
        delta = rho_sol - rho1
        pct = 100 * delta / rho1
        mw = gas_mw(gas)
        print(f"{gas:<8} {mw:6.2f} {vphi:8.2f} {rho_sol:10.4f} {delta:+8.4f} {pct:+7.3f}%")

    # Test: CO2 at x2=0.05 should give ~2.5% increase (Garcia Fig. 3)
    print(f"\n--- CO2 at x2=0.05 (Garcia reports ~2.5% increase) ---")
    rho_co2, _, pct_co2 = density_change_pct('CO2', 0.05, T_test, P_test)
    print(f"  ρ = {rho_co2:.4f} kg/m³, Δρ/ρ = {pct_co2:+.3f}%")

    # Mixed gas test
    print("\n=== Mixed Gas Test ===\n")
    gas_mix = {'CO2': 0.01, 'CH4': 0.005, 'H2S': 0.003}
    rho_mix = density_mixed_gas(gas_mix, T_test, P_test, rho1=rho1)
    delta_mix = rho_mix - rho1
    pct_mix = 100 * delta_mix / rho1
    total_x2 = sum(gas_mix.values())
    print(f"Gas mix: {gas_mix}")
    print(f"Total x2 = {total_x2}")
    print(f"ρ_mix = {rho_mix:.4f} kg/m³, Δρ = {delta_mix:+.4f}, %change = {pct_mix:+.3f}%")

    # Brine tests
    print("\n=== Brine + Gas Tests ===\n")
    S_test = 0.10  # 10 wt% NaCl
    rho_brine_val = rho_brine(T_test, P_test, S_test)
    print(f"Brine at T={T_test}K, P={P_test}MPa, S={S_test*100}wt%: ρ = {rho_brine_val:.4f} kg/m³")
    print(f"\nDensity effect of dissolved gas in brine vs pure water (x2=0.02):")
    print(f"{'Gas':<8} {'ρ(water)':>10} {'Δρ%(w)':>8} {'ρ(brine)':>10} {'Δρ%(b)':>8}")
    print("-" * 50)
    for gas in ['CO2', 'CH4', 'H2S', 'N2']:
        rho_w_sol, rho_w_base, pct_w = density_change_pct(gas, x2_test, T_test, P_test, S=0.0)
        rho_b_sol, rho_b_base, pct_b = density_change_pct(gas, x2_test, T_test, P_test, S=S_test)
        print(f"{gas:<8} {rho_w_sol:10.4f} {pct_w:+7.3f}% {rho_b_sol:10.4f} {pct_b:+7.3f}%")

    # Temperature sweep for CO2
    print("\n=== CO2 Density vs Temperature (x2=0.02, P=30 MPa) ===\n")
    print(f"{'T(°C)':<8} {'ρ_water':>10} {'ρ_CO2aq':>10} {'Δρ%':>8}")
    print("-" * 40)
    for T_C in [25, 50, 100, 150, 200, 250]:
        T_K = T_C + 273.15
        rho_w_val = rho_w(T_K, 30.0)
        rho_sol_val = density_single_gas('CO2', 0.02, T_K, 30.0, rho1=rho_w_val)
        pct_val = 100 * (rho_sol_val - rho_w_val) / rho_w_val
        print(f"{T_C:<8} {rho_w_val:10.4f} {rho_sol_val:10.4f} {pct_val:+7.3f}%")

    # ================================================================
    # VISCOSITY VALIDATION
    # ================================================================
    print("\n=== Viscosity Correction Tests ===\n")

    # CO2: Calabrese (2019) Eq. 25 at x=0.02, T-dependent
    for degf_co2 in (100, 200, 300):
        co2_vis = viscosity_correction_single('CO2', 0.02, degf=degf_co2)
        print(f"CO2 at x=0.02, {degf_co2} degF: factor = {co2_vis:.6f}")

    # H2S: Murphy & Gaines calibrated (a=1.70, linear)
    h2s_vis_005 = viscosity_correction_single('H2S', 0.05)
    h2s_vis_020 = viscosity_correction_single('H2S', 0.20)
    print(f"H2S at x=0.05: factor = {h2s_vis_005:.6f} (expect ~+8.5% increase)")
    print(f"H2S at x=0.20: factor = {h2s_vis_020:.6f} (expect ~+34%)")

    # CH4: Ostermann plateau values
    print("\nCH4 unified Arrhenius x Langmuir, ratio vs x and T:")
    print(f"  {'T(degF)':<10} {'x=0.0005':>10} {'x=0.002':>10} {'x=0.004':>10} {'x->inf':>10}")
    print(f"  {'-'*54}")
    for degf in (100, 150, 250, 350):
        row = [viscosity_correction_single('CH4', x, degf=degf)
               for x in (0.0005, 0.002, 0.004)]
        T_K = (degf - 32.0) / 1.8 + 273.15
        asym = 1.0 + _CH4_A * np.exp(_CH4_B / T_K)
        print(f"  {degf:<10} " + " ".join(f"{v:10.4f}" for v in row) + f" {asym:10.4f}")

    # Gases with no effect
    print(f"\nC2H6 at x=0.02: factor = {viscosity_correction_single('C2H6', 0.02):.4f} (expect 1.0)")
    print(f"N2 at x=0.02:   factor = {viscosity_correction_single('N2', 0.02):.4f} (expect 1.0)")
    print(f"H2 at x=0.02:   factor = {viscosity_correction_single('H2', 0.02):.4f} (expect 1.0)")

    # Mixed gas test
    gas_mix_v = {'CO2': 0.01, 'CH4': 0.005, 'H2S': 0.003}
    mixed_vis = viscosity_correction_mixed(gas_mix_v, degf=150)
    print(f"\nMixed viscosity ({gas_mix_v}, T=150F): factor = {mixed_vis:.6f}")
