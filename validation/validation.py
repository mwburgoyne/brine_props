"""
Comprehensive validation of the Plyasunov V2∞ model and Garcia mixing rule.

Tests:
1. V2∞ at 298.15K for all 8 gases (Plyasunov reference values)
2. V2∞ vs temperature for CH4, CO2, H2S (Hnedkovsky 1996 experimental data)
3. V2∞ vs Garcia (2001) CO2 cubic polynomial
4. Density predictions vs Garcia CO2 results
5. Physical reasonableness checks
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


import sys
import numpy as np

from brine_gas.water_properties import rho_w, kappa_T, V1_star, MW_WATER
from brine_gas.plyasunov_model import V2_inf, V_phi, gas_mw, B12
from brine_gas.vphi_route import V_phi as V_phi_route, route_used, boundary_step
from brine_gas.brine_properties import rho_brine, salinity_from_molality, M_NACL
from brine_gas.garcia_mixing import (density_single_gas, density_mixed_gas, density_change_pct,
                           viscosity_correction_single, viscosity_correction_mixed,
                           _ch4_viscosity_ratio, _CH4_A, _CH4_B, _CH4_K)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


def garcia_CO2_vphi(T_C):
    """Garcia (2001) Eq. 3: V_phi for CO2 in cm³/mol, T in °C."""
    return 37.51 - 9.585e-2 * T_C + 8.740e-4 * T_C**2 - 5.044e-7 * T_C**3


# ============================================================================
# TEST 1: V2∞ at 298.15K reference conditions
# ============================================================================
def test_V2inf_298K():
    print("\n=== TEST 1: V2∞ at 298.15K, 0.1 MPa ===")

    known = {
        'H2': 26.1, 'N2': 34.7, 'CH4': 37.0, 'CO2': 34.0,
        'C2H6': 51.4, 'C3H8': 67.1, 'NC4H10': 82.8, 'H2S': 34.8,
    }

    T, P = 298.15, 0.1
    for gas, expected in known.items():
        computed = V2_inf(gas, T, P)
        pct_err = abs(100 * (computed - expected) / expected)
        check(f"{gas}: {computed:.2f} vs {expected} ({pct_err:.2f}%)",
              pct_err < 0.5,
              f"computed={computed:.4f}, expected={expected}")


# ============================================================================
# TEST 2: V2∞ vs Hnedkovsky (1996) experimental data
# ============================================================================
def test_V2inf_vs_hnedkovsky():
    print("\n=== TEST 2: V2∞ vs Hnedkovsky (1996) at 30 MPa ===")

    # Hnedkovsky data: average of 28 MPa and 35 MPa measurements
    # Format: (T_K, V2inf_avg)
    hned_CH4 = [
        (298.15, 36.75), (323.15, 37.30), (373.15, 40.50),
        (423.15, 45.90), (473.15, 54.10), (523.15, 64.80),
    ]
    hned_CO2 = [
        (298.15, 33.45), (323.15, 33.75), (373.15, 37.50),
        (423.15, 42.30), (473.15, 49.20), (523.15, 59.75),
    ]
    hned_H2S = [
        (298.15, 34.90), (323.15, 35.90), (373.15, 38.70),
        (423.15, 42.75), (473.15, 48.45), (523.15, 56.75),
    ]

    P = 30.0
    print(f"\n  {'Gas':<6} {'T(°C)':>6} {'Calc':>8} {'Expt':>8} {'%Err':>7}")
    print("  " + "-" * 40)

    for gas, data, tol in [('CH4', hned_CH4, 5.0), ('CO2', hned_CO2, 5.0), ('H2S', hned_H2S, 5.0)]:
        max_err = 0
        for T_K, v_exp in data:
            v_calc = V2_inf(gas, T_K, P)
            pct = 100 * (v_calc - v_exp) / v_exp
            max_err = max(max_err, abs(pct))
            T_C = T_K - 273.15
            print(f"  {gas:<6} {T_C:6.1f} {v_calc:8.2f} {v_exp:8.2f} {pct:+6.2f}%")
        check(f"{gas} max error {max_err:.1f}% < {tol}%", max_err < tol,
              f"max_err={max_err:.2f}%")


# ============================================================================
# TEST 3: Plyasunov CO2 V2∞ vs Garcia cubic polynomial
# ============================================================================
def test_V2inf_vs_garcia_polynomial():
    print("\n=== TEST 3: Plyasunov CO2 vs Garcia cubic polynomial ===")
    print("  Garcia Eq.3 is P-independent; Plyasunov at 20 MPa for comparison")
    print(f"\n  {'T(°C)':>6} {'Plyasunov':>10} {'Garcia':>10} {'Diff':>8}")
    print("  " + "-" * 38)

    P = 20.0
    max_diff_200 = 0
    for T_C in [25, 50, 75, 100, 125, 150, 175, 200, 250, 300]:
        T_K = T_C + 273.15
        v_plyas = V2_inf('CO2', T_K, P)
        v_garcia = garcia_CO2_vphi(T_C)
        diff = v_plyas - v_garcia
        if T_C <= 250:
            max_diff_200 = max(max_diff_200, abs(diff))
        print(f"  {T_C:6} {v_plyas:10.2f} {v_garcia:10.2f} {diff:+8.2f}")

    # Garcia polynomial was fit to data 5-300°C; at 300°C it diverges from Plyasunov
    # because Garcia's cubic doesn't capture near-critical behavior well.
    # Expect agreement within ~2 cm³/mol for 25-250°C range.
    check(f"Max difference 25-250°C: {max_diff_200:.2f} cm³/mol < 3",
          max_diff_200 < 3.0,
          f"max_diff={max_diff_200:.2f}")


# ============================================================================
# TEST 4: CO2 density predictions
# ============================================================================
def test_CO2_density():
    print("\n=== TEST 4: CO2 Density Predictions ===")

    # Garcia reports nearly linear density increase with x_CO2
    # ~2.5% at x2=0.05 at 25°C
    T, P = 298.15, 10.0

    # Test linearity: compute at x2 = 0.01, 0.02, 0.03, 0.04, 0.05
    print(f"\n  CO2 density change vs mole fraction at T=25°C, P=10 MPa:")
    print(f"  {'x2':>6} {'ρ(kg/m³)':>10} {'Δρ%':>8} {'Δρ%/x2':>8}")
    print("  " + "-" * 36)

    ratios = []
    for x2 in [0.01, 0.02, 0.03, 0.04, 0.05]:
        rho_sol, rho_w_val, pct = density_change_pct('CO2', x2, T, P)
        ratio = pct / x2 if x2 > 0 else 0
        ratios.append(ratio)
        print(f"  {x2:6.3f} {rho_sol:10.4f} {pct:+7.3f}% {ratio:8.2f}")

    # Check near-linearity: ratio should be roughly constant
    ratio_range = max(ratios) - min(ratios)
    check(f"CO2 Δρ%/x2 nearly constant (range={ratio_range:.2f})",
          ratio_range < 5.0,
          f"ratios span {ratio_range:.2f}")

    # Check ~2.5% at x2=0.05
    _, _, pct_05 = density_change_pct('CO2', 0.05, T, P)
    check(f"CO2 at x2=0.05: {pct_05:.2f}% (Garcia ~2.5%)",
          1.5 < pct_05 < 3.5,
          f"pct={pct_05:.2f}")


# ============================================================================
# TEST 5: Physical reasonableness
# ============================================================================
def test_physical_reasonableness():
    print("\n=== TEST 5: Physical Reasonableness ===")

    P = 30.0

    # V2∞ should increase monotonically with T for all gases (50-250°C at 30 MPa)
    # Note: slight dip from 25→50°C is physical at elevated P for some gases
    temps = [323.15, 373.15, 423.15, 473.15, 523.15]
    for gas in ['H2', 'N2', 'CH4', 'CO2', 'C2H6', 'C3H8', 'NC4H10', 'H2S']:
        vals = [V2_inf(gas, T, P) for T in temps]
        monotonic = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
        check(f"{gas} V2∞ monotonically increasing 50-250°C",
              monotonic,
              f"values: {[f'{v:.1f}' for v in vals]}")

    # CO2 should increase density (heavy MW, small V_phi)
    _, _, pct_co2 = density_change_pct('CO2', 0.02, 298.15, 10.0)
    check(f"CO2 increases density ({pct_co2:+.3f}%)", pct_co2 > 0)

    # H2 should decrease density most (lightest MW)
    _, _, pct_h2 = density_change_pct('H2', 0.02, 298.15, 10.0)
    check(f"H2 decreases density ({pct_h2:+.3f}%)", pct_h2 < 0)

    # H2S should be near-neutral
    _, _, pct_h2s = density_change_pct('H2S', 0.02, 298.15, 10.0)
    check(f"H2S near-neutral density effect ({pct_h2s:+.3f}%)",
          abs(pct_h2s) < 0.5)

    # Heavier hydrocarbons should have larger V_phi
    v_c1 = V2_inf('CH4', 298.15, 0.1)
    v_c2 = V2_inf('C2H6', 298.15, 0.1)
    v_c3 = V2_inf('C3H8', 298.15, 0.1)
    v_c4 = V2_inf('NC4H10', 298.15, 0.1)
    check(f"V_phi ordering: CH4({v_c1:.1f}) < C2H6({v_c2:.1f}) < C3H8({v_c3:.1f}) < nC4({v_c4:.1f})",
          v_c1 < v_c2 < v_c3 < v_c4)

    # Mixed gas: adding CH4 to CO2 solution should reduce density increase
    _, _, pct_co2_only = density_change_pct('CO2', 0.02, 298.15, 10.0)
    _, _, pct_mix = density_change_pct({'CO2': 0.01, 'CH4': 0.01}, None, 298.15, 10.0)
    check(f"CO2+CH4 mix ({pct_mix:+.3f}%) < pure CO2 ({pct_co2_only:+.3f}%)",
          pct_mix < pct_co2_only)


# ============================================================================
# TEST 6: N2 vs O'Sullivan & Smith (1970)
# ============================================================================
def test_N2_vs_osullivan():
    print("\n=== TEST 6: N2 V2∞ vs O'Sullivan & Smith (1970) ===")

    # O'Sullivan data in pure water (low-P values)
    # T(°C): 51.5, 102.5, 125.0
    osullivan_N2 = [(324.65, 34.05), (375.65, 37.7), (398.15, 43.1)]

    P = 20.0  # moderate pressure
    print(f"\n  {'T(°C)':>6} {'Calc':>8} {'O-S':>8} {'%Err':>7}")
    print("  " + "-" * 34)

    for T_K, v_exp in osullivan_N2:
        v_calc = V2_inf('N2', T_K, P)
        pct = 100 * (v_calc - v_exp) / v_exp
        T_C = T_K - 273.15
        print(f"  {T_C:6.1f} {v_calc:8.2f} {v_exp:8.2f} {pct:+6.2f}%")

    # O'Sullivan data has lower precision; expect agreement within ~10%
    max_err = max(abs(100 * (V2_inf('N2', T, P) - v) / v) for T, v in osullivan_N2)
    check(f"N2 vs O'Sullivan max error {max_err:.1f}% < 15%", max_err < 15,
          f"max_err={max_err:.1f}%")


# ============================================================================
# TEST 7: Garcia solvent basis
# ============================================================================
def test_garcia_solvent_basis():
    print("\n=== TEST 7: Garcia mixing rule solvent basis ===")

    T, P, x2 = 373.15, 30.0, 0.02

    # At zero salinity the brine and pure-water forms must be identical
    rho_fresh = density_single_gas('CO2', x2, T, P, S=0.0)
    rho_fresh_explicit = density_single_gas('CO2', x2, T, P,
                                            rho1=rho_w(T, P), S=0.0)
    check(f"S=0 identical with and without explicit rho1 ({rho_fresh:.6f})",
          abs(rho_fresh - rho_fresh_explicit) < 1e-9)

    # The dissolved-gas volume must be added to the FULL gas-free brine volume,
    # not to the water-only part of it. Checked against an independent
    # construction in mass-per-kg-of-water terms, which involves no mole
    # fractions at all:
    #     rho = (Wb + n2*M2) / (Wb/rho_b + n2*V_phi),  Wb = 1000 + m*M_NaCl
    # The superseded water-mass form is reported alongside: it inflates the
    # gas density effect, approaching 1/(1-S) as x2 -> 0.
    # Must use the SAME route AND the same salinity the model uses, or this
    # compares the algebra of one configuration against the volumes of another
    # and fails for the wrong reason. V is therefore taken per molality below.
    M2 = gas_mw('CO2')
    n_w = 1000.0 / MW_WATER                     # mol water per kg water

    print(f"\n  {'m':>5} {'S':>7} {'pp(model)':>10} {'pp(water-basis)':>16} "
          f"{'ratio':>7} {'1/(1-S)':>8}")
    print("  " + "-" * 60)
    for m in [1.0, 2.0, 5.0]:
        S = salinity_from_molality(m)
        rho1 = rho_brine(T, P, S)
        rho_sol = density_single_gas('CO2', x2, T, P, rho1=rho1, S=S)
        pp = 100.0 * (rho_sol / rho1 - 1.0)

        # Independent per-kg-water construction at the same dissolved amount
        n2 = x2 * n_w / (1.0 - x2)              # mol CO2 per kg water
        Wb = 1000.0 + m * M_NACL
        V = V_phi_route('CO2', T, P, 'auto', m)
        rho_ind = (Wb + n2 * M2) / (Wb / (rho1 / 1000.0) + n2 * V) * 1000.0

        # Superseded water-mass form
        rho_old = ((MW_WATER * (1.0 - x2) + x2 * M2)
                   / (x2 * V + MW_WATER * (1.0 - x2) / (rho1 / 1000.0)) * 1000.0)
        pp_old = 100.0 * (rho_old / rho1 - 1.0)
        ratio = pp_old / pp

        print(f"  {m:5.1f} {S:7.4f} {pp:10.4f} {pp_old:16.4f} "
              f"{ratio:7.4f} {1/(1-S):8.4f}")
        check(f"m={m}: matches independent per-kg-water construction "
              f"({rho_sol:.6f} vs {rho_ind:.6f})",
              abs(rho_sol - rho_ind) < 1e-8,
              f"model={rho_sol:.10f}, independent={rho_ind:.10f}")
        check(f"m={m}: water-basis inflates the gas effect by {ratio:.4f}, "
              f"under the 1/(1-S) = {1/(1-S):.4f} limit",
              1.0 < ratio < 1.0 / (1.0 - S))

    # Mixed-gas path must use the same basis as the single-gas path
    S = salinity_from_molality(2.0)
    rho1 = rho_brine(T, P, S)
    rho_single = density_single_gas('CO2', x2, T, P, rho1=rho1, S=S)
    rho_mixed = density_mixed_gas({'CO2': x2}, T, P, rho1=rho1, S=S)
    check(f"single-gas and mixed-gas paths agree ({rho_single:.6f})",
          abs(rho_single - rho_mixed) < 1e-9)

    # Input validation
    for bad_x in (-0.01, 1.0, 1.5):
        try:
            density_single_gas('CO2', bad_x, T, P)
            check(f"x2={bad_x} rejected", False, "no exception raised")
        except ValueError:
            check(f"x2={bad_x} rejected", True)
    try:
        density_single_gas('CO2', 0.01, T, P, S=0.6)
        check("S=0.6 rejected", False, "no exception raised")
    except ValueError:
        check("S=0.6 rejected", True)


def test_vphi_route_dispatch():
    """The default V_phi route, where it falls back, and where it extrapolates.

    Added 2026-07-25 when the delivered density moved from Plyasunov to the
    S&W modified-PR route. Revised the same day when the 473 K handover was
    removed: PR now runs to the IF97 Region 1 ceiling, so the only fallbacks
    left are gases with no fitted shift and states where V2inf is undefined.
    """
    print("\n=== TEST 10: V_phi route dispatch and fallback ===")

    from brine_gas.pr_vphi_model import (T_CALIBRATED_MAX, T_VOUCHED_MAX,
                               UNCALIBRATED_IN_T, is_extrapolated)

    for gas, T, P, want in (('CO2', 373.15, 30.0, 'pr'),
                            ('H2S', 298.15, 20.0, 'pr'),
                            ('H2', 350.0, 20.0, 'pr'),
                            ('CO2', 523.15, 30.0, 'pr'),    # NO handover here
                            ('CO2', 623.0, 30.0, 'pr'),     # still PR
                            ('C3H8', 298.15, 20.0, 'pr'),      # shift added 2026-07-25
                            ('NC4H10', 298.15, 20.0, 'pr'),    # shift added 2026-07-25
                            ('CO2', 624.0, 30.0, 'plyasunov'),  # past IF97 R1
                            ('CO2', 260.0, 20.0, 'plyasunov')):    # below 273 K
        got = route_used(gas, T, P)
        check(f"{gas} at {T:.0f} K, {P:.0f} MPa uses the {want} route", got == want,
              f"got {got}")

    # There must be NO temperature handover anywhere in the vouched envelope,
    # nor anywhere up to the arithmetic limit.
    for T in (300.0, 400.0, 450.0, 473.0, 500.0, 550.0, 600.0, 620.0):
        check(f"no route change at {T:.0f} K for CO2",
              route_used('CO2', T, 30.0) == 'pr')

    # Forcing a route must override the dispatcher in both directions.
    v_auto = V_phi_route('CO2', 373.15, 30.0)
    v_pr = V_phi_route('CO2', 373.15, 30.0, 'pr')
    v_ply = V_phi_route('CO2', 373.15, 30.0, 'plyasunov')
    check(f"route='auto' equals route='pr' ({v_auto:.4f})",
          abs(v_auto - v_pr) < 1e-12)
    check(f"route='plyasunov' really differs ({v_ply:.4f} vs {v_pr:.4f})",
          abs(v_ply - v_pr) > 1e-6)

    # The literature-anchored salt fraction, adopted 2026-07-25 as an absolute
    # term and switched to the RELATIVE form 2026-07-30 (Mark's call; costing
    # in relative_salt_shift.py, evidence in closed record section J). Fixed
    # entirely from the Tiepel KCl dilatometry (molar concentrations converted
    # to molality since 2026-09-05), so these values must not drift with any
    # fit. Values generated by relative_salt_shift.py.
    from brine_gas.vphi_route import salt_fraction
    for m, want in ((0.0, 0.0000), (1.0, -0.015183), (2.5, -0.032578),
                    (5.0, -0.052705)):
        got = salt_fraction(m)
        check(f"salt fraction at {m} molal is {100 * got:+.3f}%",
              abs(got - want) < 5e-5, f"got {got:.6f}, want {want}")
    check("salt fraction is negative and saturating",
          salt_fraction(10.0) < salt_fraction(5.0) < salt_fraction(1.0) < 0.0
          and abs(salt_fraction(10.0) - salt_fraction(5.0))
          < abs(salt_fraction(5.0) - salt_fraction(0.0)))
    v0_ref = V_phi_route('CO2', 373.15, 30.0, 'auto', 0.0)
    check("salt fraction reaches V_phi through the density path",
          abs(V_phi_route('CO2', 373.15, 30.0, 'auto', 5.0)
              - v0_ref * (1.0 -0.052705)) < 5e-4 * v0_ref)

    # The three ceilings are different things and must not be conflated.
    check(f"vouched accuracy ceiling {T_VOUCHED_MAX:.0f} K sits inside the "
          f"calibration data ({T_CALIBRATED_MAX:.0f} K)",
          T_VOUCHED_MAX < T_CALIBRATED_MAX)
    check("the shift is flagged as extrapolated above the calibration data",
          is_extrapolated('CO2', 500.0) and not is_extrapolated('CO2', 400.0))
    check("gases with no temperature-resolved calibration are still declared",
          set(UNCALIBRATED_IN_T) == {'N2', 'H2', 'C2H6', 'C3H8', 'NC4H10'},
          f"got {UNCALIBRATED_IN_T}")

    # C3H8 is the weakest entry in the table and must not drift silently: its
    # shift is the mean of two mutually inconsistent 298 K determinations, not a
    # fit to a data set. Pin the value and the bracket it came from.
    v = V_phi_route('C3H8', 298.15, 20.0, 'pr')
    check(f"C3H8 V_phi at 298 K sits at the midpoint of its two anchors "
          f"({v:.2f} vs 70.7 and 75.0)", abs(v - 72.85) < 0.02, f"got {v:.4f}")
    check("C3H8 V_phi is above BOTH independent anchors' lower bound, which the "
          "superseded fallback was not (66.99)",
          v > 70.7 and V_phi_route('C3H8', 298.15, 20.0, 'plyasunov') < 70.7)

    # Route divergence, reported not asserted above the vouched envelope, but
    # pinned INSIDE it where the method actually claims accuracy.
    print(f"\n  {'gas':<6}{'dV_phi %':>10}{'d rho kg/m3':>14}{'d rho %':>10}"
          f"   (at {T_VOUCHED_MAX:.0f} K, the vouched ceiling)")
    worst = 0.0
    for gas in ('CH4', 'CO2', 'H2S', 'N2', 'H2', 'C2H6'):
        x2 = 0.02 if gas in ('CO2', 'H2S') else 0.008
        d_kg, d_pct = boundary_step(gas, T_VOUCHED_MAX, 20.0, S=0.05, x2=x2)
        vp = V_phi_route(gas, T_VOUCHED_MAX, 20.0, 'pr')
        vl = V_phi_route(gas, T_VOUCHED_MAX, 20.0, 'plyasunov')
        print(f"  {gas:<6}{100 * (vp / vl - 1):9.1f}%{d_kg:14.3f}{d_pct:9.3f}%")
        worst = max(worst, abs(d_pct))
    check(f"inside the vouched envelope the two routes agree on density to "
          f"{worst:.3f}% (limit 0.30%)", worst < 0.30)


# ============================================================================
# TEST 8: Viscosity corrections (pinned to their calibration sources)
# ============================================================================
def test_viscosity_corrections():
    print("\n=== TEST 8: Viscosity corrections ===")

    # --- CO2: Calabrese (2019) Eq. 25, ln(mu/mu_b) = 65.560*exp(-2.468*(T/142-1))*x
    # Slope ratio against the superseded Islam-Carlson constant 4.65: 1.7x at
    # 122 degF, 3.5x at 200 degF, 4.3x at 221 degF (105 degC, the ceiling of
    # IC's own stated scope - the fair headline), 9.4x at 302 degF (beyond it).
    print(f"\n  {'degF':>6} {'CO2 slope':>10} {'IC/Cal':>8} {'expected':>9}")
    print("  " + "-" * 38)
    for degf, expected in [(122, 1.7), (200, 3.5), (221, 4.3), (302, 9.4)]:
        T_K = (degf - 32.0) / 1.8 + 273.15
        slope = 65.560 * np.exp(-2.468 * (T_K / 142.0 - 1.0))
        ratio = 4.65 / slope
        print(f"  {degf:6} {slope:10.4f} {ratio:8.2f} {expected:9.1f}")
        check(f"CO2 IC/Calabrese slope ratio at {degf} degF = {ratio:.2f}",
              abs(ratio - expected) < 0.15, f"ratio={ratio:.3f}")

    # Pinned factors at x_CO2 = 0.02
    for degf, expected in [(100, 1.072073), (200, 1.026853), (300, 1.010141)]:
        f = viscosity_correction_single('CO2', 0.02, degf=degf)
        check(f"CO2 factor at x=0.02, {degf} degF = {f:.6f}",
              abs(f - expected) < 1e-6, f"got {f:.8f}, expected {expected}")

    # --- CH4: unified Arrhenius x Langmuir, fitted to all 23 SPE 14211 points
    # (code/ostermann_ch4_refit.py). One equation, no cap.
    from fits.ostermann_ch4_refit import OSTERMANN, x_ch4_from_rsw
    res = []
    for degf, rows in OSTERMANN.items():
        for _P, rsw, ratio in rows:
            m = float(_ch4_viscosity_ratio(x_ch4_from_rsw(rsw), degf))
            res.append(m - ratio)
    res = np.array(res)
    rms, mx = 100 * np.sqrt(np.mean(res ** 2)), 100 * np.abs(res).max()
    check(f"CH4 fits all {len(res)} Ostermann points, RMS {rms:.3f} pp < 0.8",
          rms < 0.8, f"rms={rms:.4f}")
    check(f"CH4 worst Ostermann residual {mx:.3f} pp < 2.5", mx < 2.5, f"max={mx:.4f}")

    # Saturates in x rather than being capped, and is monotonic in both variables
    for degf in (100, 250):
        vals = [viscosity_correction_single('CH4', x, degf=degf)
                for x in np.linspace(1e-5, 0.05, 60)]
        check(f"CH4 monotonically increasing in x at {degf} degF",
              all(a < b for a, b in zip(vals, vals[1:])))
        T_K = (degf - 32.0) / 1.8 + 273.15
        asym = 1.0 + _CH4_A * np.exp(_CH4_B / T_K)
        check(f"CH4 approaches its {degf} degF asymptote {asym:.5f} from below",
              vals[-1] < asym and vals[-1] > 0.95 * asym, f"end={vals[-1]:.5f}")
    temps = list(range(60, 520, 10))
    tv = [viscosity_correction_single('CH4', 0.003, degf=t) for t in temps]
    check("CH4 decreases monotonically with T at fixed x",
          all(a > b for a, b in zip(tv, tv[1:])))
    check(f"CH4 still above 1.0 at 510 degF ({tv[-1]:.5f})", tv[-1] > 1.0)

    # Dilute limit must be linear in x (each solute acting independently)
    r1 = viscosity_correction_single('CH4', 1e-9, degf=150) - 1
    r2 = viscosity_correction_single('CH4', 2e-9, degf=150) - 1
    check(f"CH4 excess is linear in x as x -> 0 (ratio {r2/r1:.4f} = 2)",
          abs(r2 / r1 - 2.0) < 1e-3, f"ratio={r2/r1:.6f}")


    # --- H2S: our own fit to Murphy & Gaines Table IV, a=1.70, linear in x.
    # Re-pinned 2026-07-31 when the exponent was set to unity (1.0134 was
    # borrowed from Islam-Carlson's CO2 fit, never fitted to H2S, and the five
    # points cannot distinguish exponents: RMS 1.740 vs 1.738 pp). Previously
    # re-pinned 2026-07-25 (a=1.79 on the Burgess & Germann solubility basis).
    # Pinned by value, not by comment - a stale comment here once sat two
    # revisions behind the code.
    for x, expected in [(0.03, 1.051), (0.05, 1.085)]:
        f = viscosity_correction_single('H2S', x)
        check(f"H2S factor at x={x} = {f:.6f}", abs(f - expected) < 1e-5,
              f"got {f:.8f}, expected {expected}")

    # --- Measured nulls and no-data gases
    for gas in ('C2H6', 'N2', 'H2', 'C3H8', 'NC4H10'):
        f = viscosity_correction_single(gas, 0.02, degf=150)
        check(f"{gas} no correction (factor {f:.4f})", f == 1.0)

    # --- Mixture: multiplicative, and exactly reproduces the single-gas limit
    mix = {'CO2': 0.01, 'CH4': 0.005, 'H2S': 0.003}
    combined = viscosity_correction_mixed(mix, degf=150)
    product = 1.0
    for g, x in mix.items():
        product *= viscosity_correction_single(g, x, degf=150)
    check(f"mixed factor is the product of singles ({combined:.6f})",
          abs(combined - product) < 1e-12)
    check(f"mixed factor at 150 degF = {combined:.6f}",
          abs(combined - 1.079422) < 1e-5, f"got {combined:.8f}")

    # --- degf is mandatory for the two T-dependent corrections
    for gas in ('CO2', 'CH4'):
        try:
            viscosity_correction_single(gas, 0.01)
            check(f"{gas} requires degf", False, "no exception raised")
        except ValueError:
            check(f"{gas} requires degf", True)


# ============================================================================
# TEST 9: independent (non-Plyasunov) evidence
# ============================================================================
def test_independent_evidence():
    """
    Tests 1 and 2 are partly circular: the 298.15 K anchors are Plyasunov's own
    reported values, and Hnedkovsky is the data the model was fitted to. These
    checks use sources that are neither. Full workup in
    `independent_anchor_validation.py` and `osullivan_salinity_validation.py`.
    """
    print("\n=== TEST 9: independent densimetry and the measured salt effect ===")

    # Moore 1982 (DOI 10.1021/je00027a005), Bignell 1984 (10.1021/j150666a060)
    # and Zhou & Battino 2001 (10.1021/je000215o). Model should sit inside the
    # spread of independent determinations where more than one exists.
    # Ranges widened 2026-07-25 by Tiepel & Gubbins 1972 Table I (Paper 26,
    # dilatometry: CH4 37.42, C2H6 53.27, H2 25.20), Enns 1965 Table I
    # (Paper 31, 25 degC: N2 33.30, CO2 34.80) and Heusler & Gaiser 1972
    # (Paper 28, H2 24.5). CH4 and CO2 previously had only one independent
    # determination each and so could not be range-checked at all.
    spreads = {'H2': (23.1, 26.7), 'N2': (33.1, 35.7), 'C2H6': (49.6, 53.27),
               'CH4': (34.5, 37.42), 'CO2': (33.9, 34.8)}
    for gas, (lo, hi) in spreads.items():
        v = float(V2_inf(gas, 298.15, 0.1))
        check(f"{gas} V2inf {v:.2f} inside independent spread {lo}-{hi}",
              lo <= v <= hi, f"v={v:.2f}")

    # Every gas in the five-gas headline scope must sit inside its independent
    # spread; this is the check that would fail if a coefficient edit moved a
    # V2inf out of the measured range.
    for gas in ('H2', 'N2', 'CH4', 'CO2'):
        lo, hi = spreads[gas]
        v = float(V2_inf(gas, 298.15, 0.1))
        check(f"{gas} (headline scope) inside independent range", lo <= v <= hi,
              f"v={v:.2f} vs {lo}-{hi}")

    # Bignell's N2 correlation and O'Sullivan's measurement agree to 0.05% at
    # 51.5 degC; the model should sit close to both.
    v_n2 = float(V2_inf('N2', 51.5 + 273.15, 0.1))
    check(f"N2 V2inf at 51.5 degC {v_n2:.2f} within 5% of O'Sullivan 34.05",
          abs(v_n2 / 34.05 - 1) < 0.05, f"v={v_n2:.2f}")

    # O'Sullivan & Smith measured the SALT effect on V_phi at 51.5 degC: it is
    # small, which is what licenses carrying freshwater V_phi into brine.
    # N2 34.05 -> 32.52 at 4 m (-4.5%); CH4 37.10 -> 36.61 (-1.3%).
    for gas, w, m4, lim in (('N2', 34.05, 32.52, 6.0), ('CH4', 37.10, 36.61, 3.0)):
        eff = abs(100 * (m4 / w - 1))
        check(f"{gas} measured salt effect on V_phi at 4 m is {eff:.1f}% (< {lim}%)",
              eff < lim, f"effect={eff:.2f}%")

    # Tiepel & Gubbins 1972 Table I (Paper 26): 15 salting-out measurements
    # across five gases and four electrolytes, ALL negative. The cross-gas
    # similarity is in the ABSOLUTE shift, not the relative one - the paper's
    # abstract says "relative" but its own table says otherwise, and that
    # distinction decides whether a salinity term would be additive or
    # multiplicative. Pinned from salt_effect_vphi.py.
    tiepel_kcl_2M = {'Ar': 30.60 - 31.71, 'CH4': 36.37 - 37.42,
                     'C2H6': 52.18 - 53.27}
    shifts = list(tiepel_kcl_2M.values())
    cv_abs = float(np.std(shifts, ddof=1) / abs(np.mean(shifts)))
    rel = [tiepel_kcl_2M['Ar'] / 31.71, tiepel_kcl_2M['CH4'] / 37.42,
           tiepel_kcl_2M['C2H6'] / 53.27]
    cv_rel = float(np.std(rel, ddof=1) / abs(np.mean(rel)))
    check(f"Tiepel KCl 2M: absolute shift is better conserved across gases "
          f"(CV {cv_abs:.3f}) than relative (CV {cv_rel:.3f})",
          cv_abs < cv_rel, f"cv_abs={cv_abs:.4f}, cv_rel={cv_rel:.4f}")
    check(f"Tiepel KCl 2M mean absolute shift {np.mean(shifts):.2f} cm3/mol "
          f"is gas-generic to within 3%",
          cv_abs < 0.03, f"cv_abs={cv_abs:.4f}")

    # Enns 1965 (Paper 31) measured O2 in sea water against O2 in water at
    # 25 degC: 31.70 vs 32.08 +/- 0.21. This BOUNDS the salt effect at
    # seawater ionic strength rather than measuring it (1.8 sigma), and is
    # the third independent source agreeing the effect is small.
    enns_shift = 31.70 - float(np.mean([31.9, 31.9, 32.2, 32.3]))
    check(f"Enns sea-water O2 shift {enns_shift:.2f} cm3/mol is under "
          f"1 cm3/mol at I~0.72",
          abs(enns_shift) < 1.0, f"shift={enns_shift:.3f}")

    # Mao, Zhang & Duan 2005 Table 7 (found in the S&W refresh paper set,
    # 2026-07-25): C2H6 partial molar volume along 358 bar, 311-444 K. Both
    # their model column and the column attributed to Kobayashi & Katz are
    # KK-type derivations, not densimetry, so this is a consistency check.
    # It is the ONLY non-25-degC check C2H6 has.
    for T, mao in ((310.93, 53.81), (344.26, 54.97), (444.26, 60.45)):
        v = float(V2_inf('C2H6', T, 35.8))
        err = 100 * (v / mao - 1)
        check(f"C2H6 V_phi at {T:.0f} K, 35.8 MPa within 8% of Mao 2005 "
              f"({v:.2f} vs {mao:.2f}, {err:+.1f}%)",
              abs(err) < 8.0, f"err={err:.2f}%")

    # ---- S&W PR + VSHIFT route (pr_vphi_model), added 2026-07-25 ----------
    # An alternative V_phi route: the exact infinite-dilution partial molar
    # volume from the Soreide-Whitson modified PR EOS, plus ONE dimensionless
    # volume shift per gas fitted to direct volumetric measurements only
    # (fit_pr_vshift.py). Carried alongside Plyasunov, not instead of it.
    try:
        import brine_gas.pr_vphi_model as PRV
    except Exception as exc:                      # pyrestoolbox not importable
        check("pr_vphi_model importable (needs pyrestoolbox S&W engine)",
              False, f"{exc}")
    else:
        # Pinned VSHIFT values. These are load-bearing: regenerating them
        # requires re-running fit_pr_vshift.py and updating BOTH places.
        expected_s = {'CH4': -0.111430, 'CO2': -0.070103, 'H2S': -0.079416,
                      'N2': -0.176768, 'H2': -0.178503, 'C2H6': -0.073843}
        for gas, s in expected_s.items():
            check(f"pr_vphi_model VSHIFT[{gas}] pinned at {s:.6f}",
                  abs(PRV.VSHIFT[gas] - s) < 1e-9,
                  f"got {PRV.VSHIFT[gas]:.6f}")

        # The volume shift is EXACTLY a constant offset: V2inf = V2inf_raw - s*b.
        # If this identity ever breaks, the Peneloux algebra has been altered.
        for gas in ('CO2', 'H2S'):
            raw = PRV.V2_inf_raw(gas, 350.0, 20.0)
            shifted = PRV.V2_inf(gas, 350.0, 20.0)
            implied = (raw - shifted) / PRV.b_covolume(gas)
            check(f"{gas}: Peneloux identity V2inf = raw - s*b holds exactly",
                  abs(implied - PRV.VSHIFT[gas]) < 1e-10,
                  f"implied s={implied:.10f} vs {PRV.VSHIFT[gas]:.10f}")

        # Agreement with Hnedkovsky densimetry, the calibration basis.
        # Fitted mean absolute errors: CH4 1.4%, CO2 0.9%, H2S 0.5%.
        for gas, T, P, meas, tol in (('CH4', 373.15, 28.0, 40.7, 3.0),
                                     ('CO2', 373.15, 20.0, 37.8, 3.0),
                                     ('H2S', 373.15, 20.0, 39.0, 3.0),
                                     ('CH4', 473.15, 28.0, 54.4, 3.0),
                                     ('CO2', 473.15, 20.0, 50.0, 3.0)):
            v = PRV.V2_inf(gas, T, P)
            err = 100 * (v / meas - 1)
            check(f"PR+VSHIFT {gas} at {T:.0f} K, {P:.0f} MPa within {tol}% of "
                  f"Hnedkovsky {meas} ({v:.2f}, {err:+.1f}%)",
                  abs(err) < tol, f"err={err:.2f}%")

        # THE POINT OF THE ROUTE: it recovers the H2S temperature trend that
        # Plyasunov misses. Murphy & Gaines measured +0.69 cm3/mol between their
        # cold (<300 K) and warm (>305 K) groups; Plyasunov gives +0.02.
        t_lo, t_hi = 296.5, 310.0
        tr_pr = PRV.V2_inf('H2S', t_hi, 0.5) - PRV.V2_inf('H2S', t_lo, 0.5)
        tr_pl = float(V2_inf('H2S', t_hi, 0.5)) - float(V2_inf('H2S', t_lo, 0.5))
        check(f"PR+VSHIFT reproduces a RISING H2S V_phi trend ({tr_pr:+.2f} "
              f"cm3/mol over {t_lo:.0f}-{t_hi:.0f} K) where Plyasunov is flat "
              f"({tr_pl:+.2f})",
              tr_pr > 0.2 and abs(tr_pl) < 0.1, f"PR={tr_pr:+.3f}, ply={tr_pl:+.3f}")

        # Range guards must still bite where the quantity is undefined or the
        # arithmetic fails. NOTE the 473 K handover was REMOVED 2026-07-25:
        # 500 K is now served, not refused. T_MAX is the IF97 Region 1 ceiling.
        for T, P, why in ((624.0, 20.0, 'above T_MAX'), (250.0, 20.0, 'below T_MIN'),
                          (350.0, 150.0, 'above P_MAX'), (450.0, 0.2, 'below Psat')):
            try:
                PRV.V2_inf('CO2', T, P)
                ok = False
            except ValueError:
                ok = True
            check(f"pr_vphi_model refuses {T:.0f} K, {P:.0f} MPa ({why})", ok)
        try:
            PRV.V2_inf('C5H12', 350.0, 20.0)
            ok = False
        except ValueError:
            ok = True
        check("pr_vphi_model refuses a gas with no fitted VSHIFT", ok)

        # Butane is the only gas needing a POSITIVE shift, and its value rests
        # on a single Moore run pair that breaks his own homologous series.
        # Pinned so it cannot drift unnoticed.
        v = PRV.V2_inf('NC4H10', 298.15, 20.0)
        check(f"NC4H10 reproduces Moore's measured 76.6 ({v:.2f})",
              abs(v - 76.6) < 0.02, f"got {v:.4f}")
        check("NC4H10 is the only gas with a positive volume shift",
              PRV.VSHIFT['NC4H10'] > 0
              and all(x < 0 for g, x in PRV.VSHIFT.items() if g != 'NC4H10'))

    # Murphy & Gaines 1974 measured H2S-water DENSITY by the float method
    # (DOI 10.1021/je60063a015, Table I). Their apparent molar volumes are
    # re-derived here from those densities rather than taken from their fit:
    # 21 points give 35.11 +/- 0.37 cm3/mol against their published 35.1 +/- 0.6.
    # Independent of Plyasunov and of Hnedkovsky, and the second leg of the
    # evidence that H2S lightens rather than densifies brine.
    for tc in (21.0, 40.0):
        v = float(V2_inf('H2S', tc + 273.15, 0.5))
        check(f"H2S V_phi at {tc:.0f} degC {v:.2f} inside Murphy-Gaines 35.1 +/- 0.6",
              abs(v - 35.1) < 0.6, f"v={v:.2f}")
    # KNOWN DEVIATION, pinned so it is not forgotten: their Fig. 3 line rises
    # 34.7 -> 35.5 over 21-40 degC while the model is essentially flat (+0.03).
    # At 40 degC the model is 0.63 cm3/mol low, marginally outside their +/-0.6
    # per-point band though inside the overall range. The disagreement is a
    # temperature TREND, not a level offset.
    trend = float(V2_inf('H2S', 313.15, 0.5)) - float(V2_inf('H2S', 294.15, 0.5))
    check(f"H2S V_phi model T-trend 21-40 degC is flat ({trend:+.2f}) vs "
          f"measured +0.80 cm3/mol", abs(trend) < 0.1, f"trend={trend:.3f}")

    # Re-derive their volumes from the raw densities, as an end-to-end check on
    # both the transcription and the apparent-molar-volume inversion.
    from fits.murphy_gaines_h2s_refit import vphi_from_table_I
    mg = np.array([r[5] for r in vphi_from_table_I()])
    check(f"Murphy-Gaines Table I re-derives to {mg.mean():.2f} +/- {mg.std():.2f} "
          f"cm3/mol, reproducing their published 35.1 +/- 0.6",
          abs(mg.mean() - 35.1) < 0.3 and mg.std() < 0.6,
          f"mean={mg.mean():.3f} sd={mg.std():.3f} n={len(mg)}")
    check(f"Murphy-Gaines derived volumes rise with T "
          f"(all {len(mg)} inside the float-method range)",
          mg.min() > 33.5 and mg.max() < 37.0, f"span {mg.min():.2f}-{mg.max():.2f}")

    # COMPOSITION SLOPE (added 2026-08-14). The same 21 points span
    # x = 0.005 to 0.029, so they measure dV_phi/dx for H2S: -15.1 +/- 3.3
    # cm3/mol per unit x (T and x fitted jointly; every near-isothermal series
    # slopes negative). This is the manuscript Discussion's answer to the
    # composition-ceiling exposure: an eighth of the CO2 slope (-117 +/- 34
    # from the Calabrese inversion) and the same sign, so at x = 0.035 the
    # excess-mass term moves -0.9 -> -0.4 g/mol and the sign of the H2S
    # density effect holds. It REPLACES the retired worst-case bound that
    # borrowed CO2's slope (-0.9 -> +3.2, sign reversal).
    from fits.murphy_gaines_h2s_refit import vphi_slope_fit
    sf = vphi_slope_fit()
    check(f"Murphy-Gaines measured H2S dV_phi/dx = {sf['slope']:+.1f} +/- "
          f"{sf['slope_se']:.1f} cm3/mol per unit x",
          abs(sf['slope'] - (-15.5)) < 0.1 and abs(sf['slope_se'] - 3.4) < 0.1,
          f"slope={sf['slope']:.3f} se={sf['slope_se']:.3f}")
    check(f"every near-isothermal Murphy-Gaines series slopes negative "
          f"({', '.join(f'{s:+.0f}' for _t, _n, s in sf['per_series'])})",
          len(sf['per_series']) == 4 and all(s < 0 for _t, _n, s in sf['per_series']))
    check(f"H2S excess-mass term at 303 K, x=0.035 under the measured slope: "
          f"{sf['excess_mass_0']:+.2f} -> {sf['excess_mass_sloped']:+.2f} g/mol "
          f"(sign holds; delivered density {sf['density_move_pct']:+.2f}%)",
          abs(sf['excess_mass_0'] - (-0.93)) < 0.02
          and abs(sf['excess_mass_sloped'] - (-0.40)) < 0.02
          and sf['excess_mass_sloped'] < 0
          and abs(sf['density_move_pct']) < 0.15)

    # H2S VISCOSITY SENSITIVITY RANGE (2026-08-15, ChatGPT-review challenge:
    # excluding the near-null 35.2 degC point because it is small is outcome-
    # based). The shipped a = 1.70 stays, relabelled a deliberately conservative
    # choice; the manuscript now quotes the all-five mean 1.41 and the
    # 1.41-1.70 range as ~0.9 pp of viscosity at x = 0.03.
    from fits.murphy_gaines_h2s_refit import viscosity_coefficients
    vc = viscosity_coefficients()
    a_all5 = np.mean([r[4] for r in vc])
    a_four = np.mean([r[4] for r in vc if r[0] < 35])
    sens_pp = (a_four - a_all5) * 0.03 * 100.0
    check(f"H2S viscosity coefficients: all-five mean {a_all5:.2f}, four-point "
          f"{a_four:.2f} (shipped 1.70), range worth {sens_pp:.2f} pp at x=0.03",
          abs(a_all5 - 1.41) < 0.005 and abs(a_four - 1.70) < 0.005
          and 0.8 < sens_pp < 1.0,
          f"all5={a_all5:.3f} four={a_four:.3f} sens={sens_pp:.3f}")

    # EZROKHI DEFAULT, a2 AS PRINTED (2026-08-15). The tNavigator table prints
    # CO2 a2 unsigned; the manuscript adopts the negative reading and now also
    # scores the positive (as-printed) reading against the same 261 densities:
    # 0.380 pp mean vs 0.162 corrected - the second documentary ground.
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        from validation.ezrokhi_vs_default import evaluate as _ezd_eval
        _df = _ezd_eval()
    _ep = (100.0 * (10 ** _df.ecl_printed - 10 ** _df.meas)).abs().mean()
    _ec = (100.0 * (10 ** _df.ecl - 10 ** _df.meas)).abs().mean()
    check(f"Ezrokhi CO2 default vs 261 densities: as-printed a2 {_ep:.3f} pp mean, "
          f"sign-corrected {_ec:.3f} pp (manuscript: 0.380 vs 0.162)",
          abs(_ep - 0.380) < 0.002 and abs(_ec - 0.162) < 0.002 and len(_df) == 261,
          f"printed={_ep:.4f} corrected={_ec:.4f} n={len(_df)}")

    # COMPOSITION CEILING (2026-08-15). The Discussion carries the measured CO2
    # slope (-117, Calabrese 0.77 m inversion, loadings below x = 0.016)
    # linearly to the x = 0.05 ceiling as a worst case: near 5.9 cm3/mol on
    # V_phi, about 1.6% of density - several times the 0.40 pp worst end-to-end
    # validation error, NOT "at the size of the quoted accuracy" (the
    # superseded framing, banned in check_deliverable_numbers).
    from validation.salt_term_magnitude_dig import DVDX
    x_c = 0.05
    dV_c = abs(DVDX) * x_c
    vm_w = MW_WATER / (rho_w(323.15, 30.0) / 1000.0)
    v_soln = (1.0 - x_c) * vm_w + x_c * float(V_phi_route('CO2', 323.15, 30.0))
    err_c = 100.0 * x_c * dV_c / v_soln
    check(f"CO2 composition-ceiling worst case at x=0.05: {dV_c:.2f} cm3/mol, "
          f"{err_c:.2f}% of density (manuscript: 5.9, about 1.6%)",
          abs(dV_c - 5.85) < 0.01 and 1.55 <= err_c < 1.65,
          f"dV={dV_c:.2f} err={err_c:.3f}%")
    # ... and "more than half the densification itself": densification at
    # x = 0.05 is 2.5-2.8% over the envelope, so the ratio clears 0.5.
    dens_c = density_change_pct('CO2', x_c, 323.15, 30.0)[2]
    check(f"ceiling term vs densification at x=0.05: {err_c:.2f} / {dens_c:.2f}% "
          f"= {err_c / dens_c:.2f} (manuscript: more than half)",
          0.5 < err_c / dens_c < 0.75, f"ratio={err_c / dens_c:.3f}")

    # C3H8 is a known outlier against independent data: model 67.0 against
    # Moore 70.7 and Zhou 75.0. Pinned so the discrepancy is not forgotten.
    v_c3 = float(V2_inf('C3H8', 298.15, 0.1))
    check(f"C3H8 V2inf {v_c3:.2f} is BELOW both independent values (70.7, 75.0)",
          v_c3 < 70.7, f"v={v_c3:.2f}")


# ============================================================================
# TEST 11: IAPWS-2008 water viscosity against Huber's own verification tables
# ============================================================================
def test_iapws2008_viscosity():
    """
    Huber (2009) Tables 6 and 8 exist so an implementation can be checked
    digit for digit. This is the strongest test in the file: it is not an
    agreement check against a correlation, it is a reproduction check against
    printed output of the reference implementation.
    """
    print("\n=== TEST 11: IAPWS-2008 water viscosity (Papers/41 Huber 2009) ===")

    from brine_gas.iapws_viscosity import (verify_table6, TABLE6, mu_water_TP,
                                 mu_liquid_0p1MPa, mu_iapws2008,
                                 ISO_REFERENCE_20C)

    # -- Table 6: the correlating equation with mu_2 = 1, 11 sample points.
    npass, ntot, worst_ppm = verify_table6(verbose=False)
    check(f"Huber Table 6: {npass}/{ntot} points reproduce to <1 ppm "
          f"(worst {worst_ppm:.4f} ppm)",
          npass == ntot and worst_ppm < 1.0,
          f"npass={npass}/{ntot} worst={worst_ppm:.4f} ppm")

    # Spot-pin the two liquid-state points that this project actually uses,
    # so a coefficient typo cannot hide behind an aggregate.
    for T, rho, mu_printed in TABLE6[:3]:
        got = mu_iapws2008(T, rho) * 1e6
        check(f"  T={T:.2f} K rho={rho:.0f}: {got:.6f} vs printed {mu_printed:.6f}",
              abs(got - mu_printed) / mu_printed < 1e-6,
              f"got={got:.6f} printed={mu_printed:.6f}")

    # -- The value the regression was CONSTRAINED to reproduce (Huber Sec. 3.3),
    #    reached here through IF97 density rather than a tabulated density.
    mu20 = mu_water_TP(293.15, 0.101325)
    dev = abs(mu20 - ISO_REFERENCE_20C) / ISO_REFERENCE_20C * 100
    check(f"ISO reference 20 degC / 0.101325 MPa: {mu20 * 1e6:.4f} vs 1001.6 "
          f"microPa s ({dev:.4f}%)",
          dev < 0.01, f"got={mu20 * 1e6:.4f} dev={dev:.4f}%")

    # -- Eq. (37) is an independently-coefficiented correlation from the same
    #    paper. Agreement between it and Eq. (36) cross-checks BOTH
    #    transcriptions: a typo in either table would break this.
    worst_pct = 0.0
    for T in (273.15, 293.15, 323.15, 353.15, 373.15):
        full = mu_water_TP(T, 0.1)
        simp = mu_liquid_0p1MPa(T)
        worst_pct = max(worst_pct, abs(simp - full) / full * 100)
    check(f"Eq. (37) Table 8 agrees with Eq. (36) Tables 2+3 to "
          f"{worst_pct:.4f}% at 0.1 MPa",
          worst_pct < 0.05, f"worst={worst_pct:.4f}%")

    # -- Monotonic decline with T through the reservoir range, and rise with P
    #    at fixed T. Cheap physical guard against a sign error in a high-order
    #    H_ij term that the sample points might not exercise.
    mus = [mu_water_TP(T, 30.0) for T in (283.15, 323.15, 373.15, 423.15, 450.0)]
    check("viscosity falls monotonically with T at 30 MPa",
          all(a > b for a, b in zip(mus, mus[1:])), f"{mus}")
    at_P = [mu_water_TP(373.15, P) for P in (1.0, 30.0, 100.0)]
    check("viscosity rises monotonically with P at 100 degC",
          all(a < b for a, b in zip(at_P, at_P[1:])), f"{at_P}")

    # -- What the swap changes against the shipped Mao-Duan water leg. Pinned
    #    because it is the justification for making the swap at all: above
    #    20 degC it is cosmetic, at the cold/high-pressure corner it is not.
    from validation.iapws_viscosity_benchmark import mu_maoduan_water
    from brine_gas.iapws_if97 import rho_if97
    rho = rho_if97(273.15, 100.0)
    cold = abs(mu_maoduan_water(273.15, rho / 1000.0)
               - mu_iapws2008(273.15, rho)) / mu_iapws2008(273.15, rho) * 100
    check(f"Mao-Duan water leg is {cold:.2f}% off IAPWS-2008 at 0 degC/100 MPa "
          f"(the corner that justifies the swap)",
          1.5 < cold < 1.9, f"cold={cold:.4f}%")


# ============================================================================
# TEST 12: Bradley & Pitzer A_V, and the Pitzer brine construction
# ============================================================================
def test_pitzer_brine_leg():
    """
    Pins the salt leg of the modular construction. The load-bearing item is the
    SIGN in Bradley & Pitzer's printed A_V equation, which is wrong in the paper
    and is corrected here; if anyone "restores" it, this test fails loudly.
    """
    print("\n=== TEST 12: Bradley & Pitzer A_V + Rogers & Pitzer NaCl leg ===")

    from brine_gas.bradley_pitzer_dielectric import (dielectric_constant, A_phi, A_V,
                                           verify_A_phi_table_II,
                                           verify_against_table_A1)
    from brine_gas.pitzer_brine_density import brine_density
    import brine_gas.rogers_pitzer_nacl as _rp
    from brine_gas.iapws_if97 import rho_if97

    # -- dielectric constant against the accepted ambient value
    D25 = dielectric_constant(298.15, 1.0)
    check(f"D(water, 25 degC, 1 bar) = {D25:.3f} (accepted ~78.4)",
          abs(D25 - 78.4) < 0.1, f"D={D25:.4f}")

    # -- A_phi against Bradley & Pitzer's OWN Table II
    worst, n = verify_A_phi_table_II(verbose=False)
    check(f"A_phi reproduces Bradley & Pitzer Table II to {worst:.3f}% "
          f"over {n} grid points (table is 3 sig figs)",
          worst < 0.5, f"worst={worst:.4f}%")

    # -- THE SIGN. Scored against Rogers & Pitzer's tabulated A_V column,
    #    which they computed from Bradley & Pitzer, so it is the right arbiter.
    ok, bad = verify_against_table_A1(verbose=False)
    check(f"A_V derived sign [beta_w - 3 dlnD/dP] matches Rogers & Pitzer "
          f"Table A-1 to {ok:.3f}%",
          ok < 0.5, f"worst={ok:.4f}%")
    check(f"A_V as PRINTED in Bradley & Pitzer is wrong ({bad:.0f}% off) - "
          f"do not restore the printed sign",
          bad > 100.0, f"printed-sign worst={bad:.2f}%")

    a25 = A_V(298.15, 1.0, rho_w=1.0 / 1.002947,
              beta_w=__import__('brine_gas.iapws_if97', fromlist=['_']).rho_and_kappa(298.15, 0.1)[1] / 10.0)
    check(f"A_V(25 degC, 1 bar) = {a25:.4f} vs Rogers & Pitzer tabulated 1.875",
          abs(a25 - 1.875) < 0.01, f"A_V={a25:.5f}")

    # -- the assembled construction must return IF97 exactly at zero salt
    for T, P in ((298.15, 0.1), (423.15, 50.0)):
        got, ref = brine_density(T, P, 0.0), rho_if97(T, P)
        check(f"Pitzer construction at m=0 returns IF97 exactly "
              f"({T:.2f} K, {P} MPa)",
              abs(got - ref) < 1e-9, f"got={got:.6f} ref={ref:.6f}")

    # -- monotonic in molality, and physical at 5 molal where Spivey is unpinned
    d = [brine_density(348.15, 30.0, m) for m in (0.0, 1.0, 3.0, 5.0)]
    check("brine density rises monotonically with NaCl molality to 5 molal",
          all(a < b for a, b in zip(d, d[1:])), f"{d}")

    # -- How much does A_V depend on WHICH dielectric equation you believe?
    #    Archer & Wang (1990) is an independent formulation, so this is not a
    #    transcription check - it is the honest A_V uncertainty.
    from brine_gas.bradley_pitzer_dielectric import verify_against_archer_wang
    mean_pct, max_pct, n = verify_against_archer_wang(verbose=False)
    check(f"A_V differs from Archer & Wang by {mean_pct:.2f}% mean / {max_pct:.2f}% "
          f"max over {n} points (two different dielectric equations)",
          0.5 < mean_pct < 2.0 and max_pct < 4.0,
          f"mean={mean_pct:.3f}% max={max_pct:.3f}%")

    # -- and does that reach the delivered number? Load-bearing test: a caveat
    #    that cannot move the answer is not a caveat.
    from brine_gas.bradley_pitzer_dielectric import A_V as _AV
    from brine_gas.iapws_if97 import rho_and_kappa as _rk
    worst = 0.0
    for T, P, m in ((298.15, 10.0, 1.0), (373.15, 30.0, 1.0), (423.15, 50.0, 5.0)):
        rho, k = _rk(T, P)
        base = 1000.0 / _rp.specific_volume(
            T, P * 10.0, m, 1000.0 / rho,
            _AV(T, P * 10.0, rho_w=rho / 1000.0, beta_w=k / 10.0),
            'I' if T <= 323.15 else 'II')
        pert = 1000.0 / _rp.specific_volume(
            T, P * 10.0, m, 1000.0 / rho,
            _AV(T, P * 10.0, rho_w=rho / 1000.0, beta_w=k / 10.0) * 1.03,
            'I' if T <= 323.15 else 'II')
        worst = max(worst, abs(pert - base) / base * 100)
    check(f"a 3% A_V error moves delivered density by only {worst:.4f}% - so the "
          f"dielectric-route choice is NOT a load-bearing caveat",
          worst < 0.02, f"worst={worst:.5f}%")


# ============================================================================
# TEST 13: Appelo ion-additive salt leg (DEFAULT since 2026-07-26)
# ============================================================================
def test_appelo_salt_leg():
    """
    Pins the multi-salt leg. The strongest check here is that it REPRODUCES
    PHREEQC, the reference implementation of the same model - so unlike every
    other benchmark in this suite, disagreement would be our bug.
    """
    print("\n=== TEST 13: Appelo salt leg (Papers/47) ===")

    from brine_gas.appelo_volumes import V0_ion, B_gamma, verify_ions, parameters
    import brine_gas.salt_route as sr

    check(f"salt_route default is 'appelo' (superseded Rogers & Pitzer 2026-07-26)",
          sr.DEFAULT_ROUTE == 'appelo', f"got {sr.DEFAULT_ROUTE!r}")

    p = parameters()
    check(f"parsed {len(p)} species with -Vm from pitzer.dat",
          len(p) >= 15, f"got {len(p)}")

    bg = B_gamma(298.15, 1.0)
    check(f"Debye-Huckel B derived from first principles = {bg:.5f} /Angstrom "
          f"(standard 0.3284)",
          abs(bg - 0.3284) < 0.001, f"B={bg:.6f}")

    worst, n = verify_ions(verbose=False)
    check(f"per-ion V0 reproduces PHREEQC VM() to {worst:.5f} cm3/mol "
          f"over {n} points",
          worst < 0.02, f"worst={worst:.6f}")

    # The omega convention: the sign printed in Eq. (7) is wrong by +4.41 on Cl-.
    v_cl = V0_ion('Cl-', 298.15, 1.01325)
    check(f"Cl- V0 = {v_cl:.4f} (PHREEQC 18.045) - pins the omega sign; the "
          f"PRINTED sign gives 22.46",
          abs(v_cl - 18.045) < 0.01, f"got {v_cl:.4f}")

    # NaCl additivity against an independent compilation
    nacl = V0_ion('Na+', 298.15, 1.01325) + V0_ion('Cl-', 298.15, 1.01325)
    check(f"V0(Na+)+V0(Cl-) = {nacl:.3f} vs Krumgalz Table 2 NaCl 16.620",
          abs(nacl - 16.620) < 0.2, f"got {nacl:.3f}")

    # multi-salt is the whole point: it must run, and be monotonic in salt
    d = [sr.brine_density(373.15, 30.0, salts={'NaCl': 2.0, 'CaCl2': c})
         for c in (0.0, 0.5, 1.0, 2.0)]
    check("multi-salt density rises monotonically with CaCl2 added to NaCl",
          all(a < b for a, b in zip(d, d[1:])), f"{d}")

    # the superseded route must refuse what it cannot represent
    try:
        sr.brine_density(373.15, 30.0, salts={'NaCl': 1.0, 'CaCl2': 1.0},
                         route='rogers')
        check("route='rogers' refuses multi-salt", False, "no raise")
    except ValueError:
        check("route='rogers' refuses multi-salt rather than silently "
              "dropping ions", True)

    # the known limit must stay guarded
    warn = sr.pairing_warning({'Ca+2': 0.5, 'SO4-2': 1.5})
    check("CaSO4 pairing guard fires (free-ion additivity is +1.8% off there)",
          warn is not None and 'CaSO4' in warn, f"{warn}")

    # -- AGAINST MEASUREMENT: Krumgalz & Millero (1982) Dead Sea densimetry.
    #    75 measured densities, 25 four-salt compositions, ionic strength
    #    8.29-9.60. This is the multi-salt claim standing on data.
    import numpy as _np
    from validation.deadsea_validation import (TABLE_III, TABLE_IV, salts_for,
                                    ML_TO_CM3, P_ATM_MPA)
    dev = []
    for sol, byT in TABLE_IV.items():
        for T_C, rho_ml in byT.items():
            meas = rho_ml * ML_TO_CM3 * 1000.0
            calc = sr.brine_density(T_C + 273.15, P_ATM_MPA,
                                    salts=salts_for(sol))
            dev.append((calc - meas) / meas * 100.0)
    dev = _np.array(dev)
    check(f"Dead Sea densimetry ({len(dev)} points, 25 four-salt compositions, "
          f"I = 8.3-9.6): {_np.mean(_np.abs(dev)):.4f}% mean, "
          f"{_np.max(_np.abs(dev)):.4f}% max",
          _np.mean(_np.abs(dev)) < 0.12 and _np.max(_np.abs(dev)) < 0.20,
          f"mean={_np.mean(_np.abs(dev)):.4f}% max={_np.max(_np.abs(dev)):.4f}%")
    check(f"...and the bias is NEGATIVE ({_np.mean(dev):+.4f}%) - the missing "
          f"ion-mixing terms Krumgalz predicted, NOT an implementation error "
          f"(PHREEQC itself scores -0.090% on the same data)",
          _np.mean(dev) < -0.03, f"bias={_np.mean(dev):+.4f}%")

    # composition transcription is self-checking: ionic strength must match
    # the range the paper states
    Is = []
    for sol in TABLE_III:
        ions = sr.ions_from_salts(salts_for(sol))
        z = {'Na+': 1, 'K+': 1, 'Mg+2': 2, 'Ca+2': 2, 'Cl-': -1}
        Is.append(0.5 * sum(m * z[i] ** 2 for i, m in ions.items()))
    check(f"Table III compositions give I = {min(Is):.3f}-{max(Is):.3f}, "
          f"matching the paper's stated 8.293-9.600",
          abs(min(Is) - 8.293) < 0.002 and abs(max(Is) - 9.600) < 0.002,
          f"{min(Is):.4f}-{max(Is):.4f}")


# ============================================================================
# TEST 14: the SALT VISCOSITY term - Kestin yardstick, Jones-Dole leg
# ============================================================================
def test_salt_viscosity():
    """
    Pins the eighteenth-block viscosity work.

    The headline finding is negative and must not be quietly lost: substituting
    a measurement-backed salt ratio does NOT reduce the residual against
    Calabrese, so the salt term is not where the viscosity accuracy goes.
    """
    print("\n=== TEST 14: salt viscosity term (Papers/50) ===")

    import numpy as _np
    import brine_gas.kestin_nacl_viscosity as kv
    from brine_gas.jones_dole_viscosity import nacl_ratio, verify_against_phreeqc
    from brine_gas.salt_viscosity_benchmark import (maoduan_ratio, mu_water_iapws,
                                          mu_water_maoduan)

    # --- Kestin: does the transcription reproduce the author's own tables?
    worst = kv._check_vs_tables(verbose=False)
    check(f"Kestin reproduces his own printed Tables 1/7/13 to {worst:.3f}% "
          f"over {len(kv._TABLE_SPOTS)} values (0, 3 and 6 molal)",
          worst < 0.05, f"worst={worst:.4f}%")

    check(f"Kestin mu_w0(20 degC) = {kv.mu_w0(20.0):.1f} micro Pa s, the "
          f"printed anchor 1002.0",
          abs(kv.mu_w0(20.0) - 1002.0) < 0.1, f"{kv.mu_w0(20.0):.3f}")

    # the exponent in Eqs. (4)-(5) prints ambiguously; the tables settle it
    import math as _math

    def _mu_fixed_exponent(t, p, m):
        A = sum(a * m ** 2 for a in kv._A_COEF)
        B = sum(b * m ** 2 for b in kv._B_COEF)
        muw = kv.mu_w0(t)
        return (muw * 10.0 ** (A + B * _math.log10(muw / kv.MU_W0_20C))
                * (1.0 + kv.beta(t, m) * p / 1000.0))

    bad = max(abs(100.0 * (_mu_fixed_exponent(t, p, m) / pr - 1.0))
              for _, t, p, m, pr in kv._TABLE_SPOTS)
    check(f"...and the literal '^2' reading of Eqs. (4)-(5) is off by "
          f"{bad:.0f}%, so the exponent is ^i - settled by the tables, not by "
          f"reading harder",
          bad > 100.0, f"{bad:.1f}%")

    # --- Jones-Dole: reproduction of PHREEQC, the reference implementation
    worst_jd, n_jd = verify_against_phreeqc(verbose=False)
    check(f"Jones-Dole leg reproduces PHREEQC to {worst_jd:.4f}% over {n_jd} "
          f"cases including four mixed brines",
          worst_jd < 0.01, f"worst={worst_jd:.5f}%")

    # --- species naming: PHREEQC keys a species by its first PRODUCT
    from brine_gas.jones_dole_viscosity import species_name, viscosity_parameters
    from brine_gas.appelo_volumes import parameters as _vm_params
    check("species are named by the reaction PRODUCT: 'H2O = OH- + H+' defines "
          "OH- and 'CO3-2 + H+ = HCO3-' defines HCO3-",
          species_name('H2O = OH- + H+') == 'OH-'
          and species_name('CO3-2 + H+ = HCO3-') == 'HCO3-'
          and species_name('Na+ = Na+') == 'Na+',
          f"{species_name('H2O = OH- + H+')!r}, "
          f"{species_name('CO3-2 + H+ = HCO3-')!r}")

    vp = viscosity_parameters()
    ions = {'Na+', 'K+', 'Mg+2', 'Ca+2', 'Sr+2', 'Ba+2', 'Li+', 'Cl-', 'Br-',
            'SO4-2', 'CO3-2', 'HCO3-', 'HSO4-', 'OH-', 'H+'}
    missing = sorted(ions - {s for s, v in vp.items() if 'jd' in v})
    check(f"all {len(ions)} brine ions carry -viscosity parameters, bicarbonate "
          f"and bisulfate included",
          not missing, f"missing {missing}")
    check(f"...and the same ion set carries -Vm for the anion-volume factor",
          not sorted(ions - {'H+'} - set(_vm_params())),
          f"missing {sorted(ions - {'H+'} - set(_vm_params()))}")

    try:
        nacl_ratio  # noqa: B018 - already imported above
        from brine_gas.jones_dole_viscosity import viscosity_ratio as _vr
        _vr(333.15, 20.0, {'Fe+2': 1.0, 'Cl-': 2.0})
        check("an unparameterised ion RAISES rather than silently contributing "
              "nothing", False, "no raise")
    except ValueError:
        check("an unparameterised ion RAISES rather than silently contributing "
              "nothing (Fe+2 has -Vm but no -viscosity)", True)

    # --- the comparison that settled the front
    md, jd = [], []
    for t in range(20, 155, 10):
        for p in (0.1, 10.0, 35.0):
            for m in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
                ref = kv.salt_ratio(t, p, m)
                md.append(100.0 * (maoduan_ratio(t + 273.15, m) / ref - 1.0))
                jd.append(100.0 * (nacl_ratio(t + 273.15, p, m) / ref - 1.0))
    md_mean = _np.mean(_np.abs(md))
    jd_mean = _np.mean(_np.abs(jd))
    check(f"vs Kestin to 5 molal: Mao-Duan {md_mean:.3f}% mean, Jones-Dole "
          f"{jd_mean:.3f}% - a TIE, both inside Kestin's own +/-0.5%",
          abs(md_mean - jd_mean) < 0.15 and max(md_mean, jd_mean) < 0.6,
          f"MD={md_mean:.4f}% JD={jd_mean:.4f}%")

    # Mao-Duan carries no pressure dependence at all; Kestin's brine does
    spread = max(abs(kv.salt_ratio(20.0, 35.0, 4.0) / kv.salt_ratio(20.0, 0.1, 4.0)
                     - 1.0), 0.0) * 100.0
    check(f"Kestin's salt ratio moves {spread:.2f}% from 0.1 to 35 MPa at "
          f"20 degC / 4 molal, which Mao-Duan cannot represent (its ratio has "
          f"no pressure term)",
          spread > 0.5, f"{spread:.3f}%")

    # --- the negative finding: the residual is NOT in the salt term
    from validation.raw_viscosity_tables import CALABRESE_T8, M_CALABRESE, parse_table
    rows = sorted((T, p, e) for x, T, p, e in parse_table(CALABRESE_T8)
                  if x == 0.0)
    m_cal = M_CALABRESE
    warm = [r for r in rows if r[0] >= 300.0]
    cold = [r for r in rows if r[0] < 300.0]

    imp_warm = _np.mean([e / mu_water_iapws(T, p) for T, p, e in warm])
    md_warm = _np.mean([maoduan_ratio(T, m_cal) for T, _, _ in warm])
    check(f"above 300 K the salt ratio implied by Calabrese ({imp_warm:.4f}) "
          f"matches the models ({md_warm:.4f}) to "
          f"{abs(100 * (imp_warm / md_warm - 1)):.2f}%",
          abs(imp_warm / md_warm - 1.0) < 0.005,
          f"implied={imp_warm:.4f} model={md_warm:.4f}")

    imp_cold = _np.mean([e / mu_water_iapws(T, p) for T, p, e in cold])
    md_cold = _np.mean([maoduan_ratio(T, m_cal) for T, _, _ in cold])
    check(f"but at 274.65 K the implied ratio ({imp_cold:.4f}) sits "
          f"{100 * (imp_cold / md_cold - 1):+.2f}% below every model "
          f"({md_cold:.4f}) - four points from one source, not a salt-term error",
          imp_cold / md_cold - 1.0 < -0.015,
          f"implied={imp_cold:.4f} model={md_cold:.4f}")

    inside = [r for r in rows if r[0] <= 423.31 and r[1] <= 35.0]
    d_md = _np.mean([abs(100.0 * (mu_water_maoduan(T, p) * maoduan_ratio(T, m_cal)
                                  / e - 1.0)) for T, p, e in inside])
    d_ke = _np.mean([abs(100.0 * (mu_water_iapws(T, p)
                                  * kv.salt_ratio(T - 273.15, p, m_cal) / e - 1.0))
                     for T, p, e in inside])
    check(f"swapping Kestin's MEASURED salt ratio into the chain does not "
          f"improve the fit to Calabrese ({d_ke:.3f}% vs the shipped "
          f"{d_md:.3f}%) - the salt term is NOT the binding constraint",
          d_ke >= d_md, f"kestin={d_ke:.4f}% shipped={d_md:.4f}%")


# ============================================================================
# TEST 15: the DELIVERED viscosity chain (Mark's calls, 2026-07-26)
# ============================================================================
def test_viscosity_route():
    """
    Pins the four decisions: Jones-Dole as the default salt term, IAPWS-2008
    as the water leg, Kestin's pressure factor grafted on, and NaCl assumed
    when a salinity is given with no species.
    """
    print("\n=== TEST 15: delivered viscosity chain (Papers/50, 51) ===")

    import numpy as _np
    import brine_gas.kestin_kcl_viscosity as kcl
    import brine_gas.kestin_nacl_viscosity as kv
    import brine_gas.viscosity_route as vr
    from brine_gas.garcia_mixing import gas_saturated_viscosity
    from brine_gas.salt_viscosity_benchmark import maoduan_ratio

    check(f"default viscosity route is 'jones_dole' (Mark's call 2026-07-26)",
          vr.DEFAULT_ROUTE == 'jones_dole', f"got {vr.DEFAULT_ROUTE!r}")

    # --- Kestin KCl: the second-ion validation that gated shipping
    worst = kcl._check_vs_tables(verbose=False)
    check(f"Kestin KCl reproduces his own printed Tables 1/3/11 to {worst:.3f}% "
          f"over {len(kcl._TABLE_SPOTS)} values",
          worst < 0.05, f"worst={worst:.4f}%")

    jd_kcl, md_kcl = [], []
    for t in range(25, 155, 5):
        for p in (0.1, 10.0, 20.0, 35.0):
            for m in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
                ref = kcl.salt_ratio(t, p, m)
                jd_kcl.append(100.0 * (
                    vr.salt_ratio(t + 273.15, p, salts={'KCl': m}) / ref - 1.0))
                md_kcl.append(100.0 * (
                    maoduan_ratio(t + 273.15, m) / ref - 1.0))
    jd_mean = _np.mean(_np.abs(jd_kcl))
    md_mean = _np.mean(_np.abs(md_kcl))
    check(f"K+ scored against MEASUREMENT (Kestin KCl): {jd_mean:.2f}% mean, "
          f"vs {md_mean:.1f}% if a NaCl-only leg is forced to call KCl NaCl - "
          f"this is what multi-salt is worth",
          jd_mean < 1.5 and md_mean > 10.0,
          f"JD={jd_mean:.3f}% MD-as-NaCl={md_mean:.3f}%")
    check(f"...and the K+ worst case is {_np.max(_np.abs(jd_kcl)):.1f}%, in the "
          f"cold concentrated corner - state it, do not hide it",
          _np.max(_np.abs(jd_kcl)) < 6.0,
          f"max={_np.max(_np.abs(jd_kcl)):.3f}%")
    # The MAX of the mis-assignment cost is pinned too. It was quoted as "32.0%"
    # across five documents for a day, unpinned and wrong on every grid that
    # gives the 15.9% mean it was paired with; the real figure is 51.2%. Only the
    # mean had a test. A number that appears in deliverables needs a check.
    md_max = _np.max(_np.abs(md_kcl))
    check(f"...and forcing a NaCl-only leg onto KCl costs {md_max:.1f}% at worst, "
          f"not the 32% once quoted - quote 15.9% mean / 51.2% max",
          50.0 < md_max < 53.0,
          f"MD-as-NaCl max={md_max:.3f}%")

    # --- the pressure factor: does grafting it actually improve both salts?
    for lab, ref, salt in (('NaCl', kv, 'NaCl'), ('KCl', kcl, 'KCl')):
        on, off = [], []
        for t in range(25, 155, 5):
            for p in (0.1, 10.0, 20.0, 35.0):
                for m in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
                    r = ref.salt_ratio(t, p, m)
                    on.append(abs(100.0 * (vr.salt_ratio(
                        t + 273.15, p, salts={salt: m}) / r - 1.0)))
                    off.append(abs(100.0 * (vr.salt_ratio(
                        t + 273.15, p, salts={salt: m},
                        pressure_term=False) / r - 1.0)))
        check(f"Kestin's pressure factor improves {lab}: {_np.mean(off):.3f}% -> "
              f"{_np.mean(on):.3f}% mean, {max(off):.2f}% -> {max(on):.2f}% max",
              _np.mean(on) < _np.mean(off) and max(on) < max(off),
              f"on={_np.mean(on):.4f}% off={_np.mean(off):.4f}%")

    # the factor is clamped outside Kestin's box, never extrapolated
    comp = {'Na+': 3.0, 'Cl-': 3.0}
    check("the pressure factor is CLAMPED outside Kestin's 35 MPa / 150 degC "
          "calibration box rather than extrapolated",
          vr.pressure_factor(423.15, 100.0, comp)
          == vr.pressure_factor(423.15, 35.0, comp)
          and vr.pressure_factor(473.15, 30.0, comp)
          == vr.pressure_factor(423.15, 30.0, comp),
          f"100 MPa={vr.pressure_factor(423.15, 100.0, comp):.6f} "
          f"35 MPa={vr.pressure_factor(423.15, 35.0, comp):.6f}")

    # --- a salinity with no species named is NaCl, on every input form
    ref = vr.brine_viscosity(373.15, 30.0, salts={'NaCl': 3.0})
    same = [vr.brine_viscosity(373.15, 30.0, m=3.0),
            vr.brine_viscosity(373.15, 30.0,
                               composition={'Na+': 3.0, 'Cl-': 3.0}),
            gas_saturated_viscosity(373.15, 30.0, m=3.0)]
    check("a salinity given with no species is NaCl - m=, salts={'NaCl':}, "
          "composition= and the full chain all agree exactly",
          all(abs(v / ref - 1.0) < 1e-12 for v in same),
          f"{ref:.8f} vs {same}")

    from brine_gas.brine_properties import salinity_from_molality
    by_S = vr.brine_viscosity(373.15, 30.0, S=salinity_from_molality(3.0))
    check(f"...and S=<weight fraction> round-trips to the same number "
          f"({by_S:.6f} vs {ref:.6f} mPa s)",
          abs(by_S / ref - 1.0) < 1e-9, f"{by_S:.8f} vs {ref:.8f}")

    # --- the superseded route refuses rather than pretending
    try:
        vr.brine_viscosity(373.15, 30.0, salts={'KCl': 4.0}, route='mao_duan')
        check("route='mao_duan' refuses a non-NaCl brine", False, "no raise")
    except ValueError:
        check("route='mao_duan' refuses a non-NaCl brine rather than silently "
              "treating it as NaCl", True)

    # --- the per-gas corrections still multiply the new base unchanged
    base = gas_saturated_viscosity(373.15, 30.0, m=3.0)
    with_co2 = gas_saturated_viscosity(373.15, 30.0, {'CO2': 0.02}, m=3.0)
    from brine_gas.garcia_mixing import viscosity_correction_single
    expect = viscosity_correction_single('CO2', 0.02, degf=212.0)
    check(f"the per-gas correction multiplies the new base unchanged "
          f"(CO2 at x=0.02, 100 degC: x{expect:.5f})",
          abs(with_co2 / base / expect - 1.0) < 1e-12,
          f"{with_co2 / base:.8f} vs {expect:.8f}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================


def test_mechanism_counterfactuals():
    """Mechanistic claims that SHIP IN PROSE, pinned as counterfactuals.

    Added 2026-08-08 after Mark caught a canonized mischaracterisation
    ("the shift absorbs the water-density error") that every consistency
    sweep had propagated: a mechanism that ships needs its own test, exactly
    as a number does. Each check here is a WHY-claim from the manuscript.
    """
    print("\n=== MECHANISM COUNTERFACTUALS (prose claims, tested) ===")
    import inspect
    from brine_gas.vphi_route import V_phi, salt_fraction
    import brine_gas.garcia_mixing as G

    # "Salinity enters the dissolved-gas volume exactly once, through g(m);
    #  the EOS's own salinity pathway is zeroed."  V(m) == V(0)*(1+g) EXACTLY.
    for gas in ('CO2', 'CH4', 'H2S', 'N2'):
        for (T, P, m) in ((298.15, 20.0, 1.0), (423.15, 35.0, 5.0)):
            lhs = V_phi(gas, T, P, m_nacl=m)
            rhs = V_phi(gas, T, P) * (1.0 + salt_fraction(m))
            check(f"{gas} {T:.0f}K/{P:.0f}MPa/m={m}: V(m) == V(0)*(1+g) exactly",
                  abs(lhs - rhs) < 1e-12, f"diff {lhs-rhs:.2e}")

    # "c_phi is the same in brine as in fresh water" - g(m) scales V_phi and
    # dV_phi/dp alike, so the log-derivative is m-independent.
    for gas, T in (('CO2', 323.15), ('CH4', 373.15)):
        dP = 0.05
        def cphi(m):
            v0 = V_phi(gas, T, 20.0, m_nacl=m)
            return -(V_phi(gas, T, 20.0 + dP, m_nacl=m)
                     - V_phi(gas, T, 20.0 - dP, m_nacl=m)) / (2 * dP) / v0
        check(f"{gas} {T:.0f}K: c_phi(m=3) == c_phi(0) (g cancels in the ratio)",
              abs(cphi(3.0) - cphi(0.0)) < 1e-12,
              f"diff {cphi(3.0)-cphi(0.0):.2e}")

    # "Substituting mole-fraction-weighted V_phi and M2 into the single-gas
    #  form reproduces the sum over gases identically."
    T, P = 352.59, 20.684
    S = 0.05
    xa, xb = 0.001, 0.008
    rho_mix = G.density_mixed_gas({'CH4': xa, 'CO2': xb}, T, P, S=S)
    w = xa / (xa + xb)
    v_eff = (w * V_phi('CH4', T, P, m_nacl=G.molality_from_salinity(S))
             + (1 - w) * V_phi('CO2', T, P, m_nacl=G.molality_from_salinity(S)))
    mw_eff = w * G.gas_mw('CH4') + (1 - w) * G.gas_mw('CO2')
    from brine_gas.brine_properties import rho_brine
    rho1 = rho_brine(T, P, S)
    rho_eq = G._garcia(xa + xb, mw_eff, v_eff, rho1 / 1000.0, S) * 1000.0
    check("mixed-gas balance == single-gas form with weighted (V_eff, M_eff)",
          abs(rho_mix - rho_eq) < 1e-9, f"diff {rho_mix-rho_eq:.2e}")

    # "The per-gas viscosity factors depend on temperature and dissolved mole
    #  fraction but not on pressure" - structural: no pressure argument exists.
    sig = inspect.signature(G.viscosity_correction_single)
    check("per-gas viscosity factor has no pressure argument (P-invariant)",
          not any(k.lower() in ('p', 'pres', 'p_mpa', 'pressure')
                  for k in sig.parameters))

    # "Undersaturated brine follows by reducing x2 alone" - V_phi carries no
    # composition argument, so the same volume serves any loading.
    sigv = inspect.signature(V_phi)
    check("V_phi carries no composition argument (x-independent volume)",
          not any(k in ('x', 'x2', 'loading') for k in sigv.parameters))

    # "Species contributions ADD, they do not compound" (Eq. exact_share,
    # 2026-08-16, Mark's question). The per-gas shares of Eq. exact_share sum
    # to the delivered density change exactly; the first-order terms of
    # Eq. first_order do not, over-counting by rho1*V/W_b. Multiplying the
    # single-gas results is a third answer, wrong in structure and closer to
    # neither: the cross term it introduces is 4e-6, a tenth of the gap the
    # denominator makes.
    m_nacl = G.molality_from_salinity(S)
    W_b = 1000.0 + m_nacl * 58.4428
    n_w = 1000.0 / 18.015268
    xt = xa + xb
    r1 = rho1 / 1000.0
    share, first_order, prod = 0.0, 0.0, 1.0
    V_tot = W_b / r1 + sum(n_w * x / (1 - xt)
                           * V_phi(g_, T, P, m_nacl=m_nacl)
                           for g_, x in (('CH4', xa), ('CO2', xb)))
    for g_, x in (('CH4', xa), ('CO2', xb)):
        m_i = n_w * x / (1 - xt)
        num = m_i * (G.gas_mw(g_) - r1 * V_phi(g_, T, P, m_nacl=m_nacl))
        share += num / (r1 * V_tot)
        first_order += num / W_b
        prod *= 1.0 + num / (W_b + r1 * m_i
                             * V_phi(g_, T, P, m_nacl=m_nacl))
    delivered = rho_mix / rho1 - 1.0
    check("Eq. exact_share: per-gas shares sum to the delivered density change",
          abs(share - delivered) < 1e-12, f"diff {share-delivered:.2e}")
    check("first-order terms do NOT sum to it, by exactly rho1*V/W_b",
          abs(first_order / (r1 * V_tot / W_b) - delivered) < 1e-12
          and abs(first_order - delivered) > 1e-5,
          f"first-order {100*first_order:.4f}% vs {100*delivered:.4f}%")
    check("multiplying single-gas results is a third answer, not the balance",
          abs(prod - 1.0 - delivered) > 1e-6,
          f"product {100*(prod-1):.4f}% vs {100*delivered:.4f}%")

    # The worked-example tables must be walkable: a reader recomputing each
    # row from the printed values must land on the printed value (Mark
    # 2026-08-16). worked_examples asserts this internally on every build.
    import examples.worked_examples as W
    P_ex = 3000.0 * W.PSIA_TO_MPA
    T_ex = (175.0 - 32.0) / 1.8 + 273.15
    ok = True
    try:
        W.salinity_columns([1.0, 1000.0 * 0.05 / (0.95 * 58.4428)])
        W.example1(T_ex, P_ex, 175.0)
        W.example2(T_ex, P_ex, 175.0)
    except AssertionError as exc:
        ok, why = False, str(exc)
    check("worked-example tables reproduce themselves from printed values",
          ok, '' if ok else why)


# ============================================================================
# TEST 17: pins from the 2026-09-05 independent review (every number the
# manuscript quotes from that round has its generator here)
# ============================================================================
def test_review_2026_09_05():
    print("\n=== TEST 17: 2026-09-05 review pins ===")
    from brine_gas.garcia_mixing import viscosity_correction_mixed
    from validation.calabrese_density_validation import x_saltfree_from_pseudo

    # 2.3  pseudo-component -> salt-free mole fraction, against a mole inventory
    nw0 = 1000.0 / MW_WATER
    for m, ng in ((0.77, 0.3), (2.5, 0.5), (5.0, 1.2)):
        x = ng / (ng + nw0 + m)
        check(f"x_sf round trip at m={m}, n_g={ng}",
              abs(x_saltfree_from_pseudo(x, m) - ng / (ng + nw0)) < 1e-12)

    # 2.4  input guards: unknown gas, x outside [0, 1], mixture sum above 1
    for gas, x in (('CO2', 1.2), ('CO2', -0.1), ('XYZ', 0.01), ('H2O', 0.01)):
        try:
            viscosity_correction_single(gas, x, degf=100.0)
            check(f"viscosity guard rejects ({gas}, x={x})", False, "no exception")
        except ValueError:
            check(f"viscosity guard rejects ({gas}, x={x})", True)
    try:
        viscosity_correction_mixed({'CO2': 0.6, 'CH4': 0.6}, degf=100.0)
        check("mixed guard rejects sum of x above 1", False, "no exception")
    except ValueError:
        check("mixed guard rejects sum of x above 1", True)
    for gas in ('N2', 'H2', 'C2H6', 'C3H8', 'NC4H10'):
        check(f"{gas} is a supported no-effect gas (factor 1)",
              viscosity_correction_single(gas, 0.01) == 1.0)

    # 1.5  Islam-Carlson over Calabrese at a STATED loading and exact T
    for T, exp in ((323.0, 1.52), (378.0, 4.02), (423.0, 8.85)):
        a = 65.560 * np.exp(-2.468 * (T / 142.0 - 1.0))
        r = 4.65 * 0.02 ** 1.0134 / np.expm1(a * 0.02)
        check(f"IC/Calabrese increment ratio at x=0.02, {T:.0f} K = {r:.2f}",
              abs(r - exp) < 0.01, f"got {r:.3f}")
    # 1.5  solvent-basis inflation: exact dilute limit 1/(1-S), and the finite
    #      loading ratio (W_b + q)/(W_w + q) at the Fig. basis state
    for m, exp in ((1.0, 5.844), (2.0, 11.689), (5.0, 29.221)):
        S = m * M_NACL / (1000.0 + m * M_NACL)
        check(f"dilute-limit inflation at {m:.0f} mol/kg = {100*(1/(1-S)-1):.3f}%",
              abs(100.0 * (1.0 / (1.0 - S) - 1.0) - exp) < 0.002)
    T, P, x2, m = 323.15, 20.0, 0.015, 5.0
    S = salinity_from_molality(m)
    rho1 = rho_brine(T, P, S)
    rho_sol = density_single_gas('CO2', x2, T, P, rho1=rho1, S=S)
    V = V_phi_route('CO2', T, P, 'auto', m)
    n2 = x2 * nw0 / (1.0 - x2)
    q = n2 * (rho1 / 1000.0) * V
    Wb = 1000.0 + m * M_NACL
    closed = (Wb + q) / (1000.0 + q)
    rho_old = ((MW_WATER * (1.0 - x2) + x2 * gas_mw('CO2'))
               / (x2 * V + MW_WATER * (1.0 - x2) / (rho1 / 1000.0)) * 1000.0)
    ratio = (rho_old / rho1 - 1.0) / (rho_sol / rho1 - 1.0)
    check(f"finite-loading inflation at the Fig. basis state = {100*(ratio-1):.1f}% "
          f"(closed form {100*(closed-1):.1f}%)",
          abs(ratio - closed) < 2e-4 and abs(ratio - 1.2832) < 0.001,
          f"ratio {ratio:.5f} closed {closed:.5f}")

    # 2.2  the ion-additive ratio's own pressure drift above the Kestin box
    from brine_gas.viscosity_route import salt_ratio
    r35 = salt_ratio(423.15, 35.0, m=5.0, pressure_term=False)
    r100 = salt_ratio(423.15, 100.0, m=5.0, pressure_term=False)
    drift = 100.0 * (r100 / r35 - 1.0)
    check(f"Jones-Dole ratio drifts {drift:+.3f}% from 35 to 100 MPa (423 K, 5 mol/kg)",
          abs(drift + 0.043) < 0.003, f"got {drift:.4f}")

    # 3.5  shifts are in closed form; equal-source weighting sensitivity
    from fits.fit_pr_vshift import calibration_set, HELD_OUT
    import brine_gas.pr_vphi_model as pr
    cs = calibration_set()
    for gas, exp_dv in (('CH4', -0.337), ('CO2', -0.451), ('H2S', -0.005), ('N2', -0.174)):
        pts = cs[gas]
        offs = np.array([pr.V2_inf_raw(gas, T_, P_) - v for T_, P_, v, _ in pts])
        b = pr.b_covolume(gas)
        check(f"{gas} shift equals the closed-form point-weighted mean offset / b",
              abs(offs.mean() / b - pr.VSHIFT[gas]) < 2e-6)
        srcs = sorted(set(p[3] for p in pts))
        by = [offs[[p[3] == s_ for p in pts]].mean() for s_ in srcs]
        dv = offs.mean() - np.mean(by)
        check(f"{gas}: equal-source weighting would move V by {dv:+.3f} cm3/mol",
              abs(dv - exp_dv) < 0.002, f"got {dv:.4f}")

    # 1.1  mixed-brine viscosity AGAINST MEASUREMENT (ambient pressure)
    import validation.mixed_brine_viscosity_validation as MB
    a = MB.score_arshad()
    h = MB.score_hoffert()
    check(f"Arshad 2020 KCl-CaCl2 viscosity: {a.dev_pct.abs().mean():.2f}% mean, "
          f"{a.dev_pct.abs().max():.2f}% max (n={len(a)})",
          abs(a.dev_pct.abs().mean() - 1.42) < 0.02 and abs(a.dev_pct.abs().max() - 5.61) < 0.02)
    check(f"Arshad 2020 KCl-CaCl2 density: {a.rho_dev_pct.abs().mean():.3f}% mean",
          abs(a.rho_dev_pct.abs().mean() - 0.149) < 0.003)
    hm = h[(h.kind == 'mixed') & (h.I <= 6.0)]
    hs = h[h.kind.isin(('NaCl', 'CaCl2')) & (h.I <= 6.0)]
    check(f"Hoffert 2025 mixed salt ratio: {hm.rdev_pct.abs().mean():.2f}% mean, "
          f"{hm.rdev_pct.abs().max():.2f}% max (n={len(hm)}); single-salt {hs.rdev_pct.abs().mean():.2f}%",
          abs(hm.rdev_pct.abs().mean() - 2.49) < 0.02 and abs(hm.rdev_pct.abs().max() - 10.22) < 0.02
          and len(hm) == 200 and 1.6 <= hs.rdev_pct.abs().mean() <= 1.8)
    hm3 = hm[hm.T_K <= 323]
    hs3 = hs[hs.T_K <= 323]
    check(f"Hoffert 293-323 K: mixed {hm3.rdev_pct.abs().mean():.2f}% vs single {hs3.rdev_pct.abs().mean():.2f}%",
          abs(hm3.rdev_pct.abs().mean() - 1.60) < 0.02 and abs(hs3.rdev_pct.abs().mean() - 1.19) < 0.02)
    check(f"Arshad ionic strength reaches {a.I.max():.1f} mol/kg",
          abs(a.I.max() - 12.5) < 0.01)

    # 3.2  the NaCl-equivalent pressure substitution tested on a DIVALENT salt
    #      (Abdulagatov & Azizov 2006 CaCl2, Paper 60; Mark supplied the PDF)
    import validation.cacl2_pressure_viscosity_validation as CA
    d, f = CA.load()
    box = d[d.in_box]
    check(f"Abdulagatov CaCl2 inside the Kestin box: {box.dev_pct.abs().mean():.2f}% mean, "
          f"{box.dev_pct.abs().max():.2f}% max, bias {box.dev_pct.mean():+.2f}% (n={len(box)})",
          abs(box.dev_pct.abs().mean() - 1.69) < 0.02 and abs(box.dev_pct.abs().max() - 7.59) < 0.02)
    m2 = box[box.m == 2.0]
    check(f"...bias at I = 6 (m = 2.00) is {m2.dev_pct.mean():+.2f}% (the Jones-Dole CaCl2 ratio, not pressure)",
          abs(m2.dev_pct.mean() + 3.16) < 0.02)
    f30 = f[(f.T_K <= 423.15) & (f.P_MPa == 30.0)]
    f60 = f[(f.T_K <= 423.15) & (f.P_MPa == 60.0)]
    check(f"CaCl2 salt-ratio pressure factor at 30 MPa: measured {f30.F_meas.mean():.4f}, chain {f30.F_chain.mean():.4f}; "
          f"chain dev {f30.dev_chain_pct.abs().mean():.2f}% mean vs none {f30.dev_none_pct.abs().mean():.2f}%",
          abs(f30.dev_chain_pct.abs().mean() - 0.64) < 0.02 and abs(f30.dev_none_pct.abs().mean() - 0.32) < 0.02
          and abs(f30.F_meas.mean() - 1.0018) < 0.0005)
    check(f"...at 60 MPa (clamped): chain {f60.dev_chain_pct.abs().mean():.2f}% vs none {f60.dev_none_pct.abs().mean():.2f}%, "
          f"chain bias {f60.dev_chain_pct.mean():+.2f}%",
          abs(f60.dev_chain_pct.abs().mean() - 0.77) < 0.02 and abs(f60.dev_chain_pct.mean() - 0.37) < 0.02)

    # Bignell 1987 H2 densities (Paper 61), reduced with the flash solubility:
    # HELD OUT, the route sits 5-16% above them (2026-09-06)
    import validation.bignell1987_h2_check as BG
    check("Bignell 1987 H2 reduction reproduces its pinned values and the route is +12.4% mean",
          BG.main(check=True))
    held = [p for p in HELD_OUT['H2'] if 'Bignell 1987' in p[3]]
    check(f"Bignell 1987 H2 points are HELD OUT ({len(held)} groups), not in the calibration set",
          len(held) == 12 and not any('Bignell 1987' in p[3] for p in cs['H2']))
    # Table A.1 is GENERATED (ezrokhi_fits --tex) since 2026-09-06 because the
    # H2S a0 sits near zero: the Barbero re-pin moved it 2.4% for 0.025% in V.
    import fits.ezrokhi_fits as EZ
    a_h2s = np.array([EZ.A_density('H2S', t_, EZ.P_REF, 0.0) for t_ in EZ.T_C])
    a0_h2s = EZ.quad_fit(EZ.T_C, a_h2s)[0]
    check(f"Ezrokhi H2S a0 = {a0_h2s:+.4e} (30 MPa density set; public README table)", abs(a0_h2s + 4.6214e-3) < 1e-6)
    for tc, exp in ((25.0, 1.42), (100.0, 2.06)):
        r = (0.0065950 - 0.000897 * tc) / EZ.A_density('H2S', tc, EZ.P_REF, 0.0)
        check(f"shipped H2S default / framework at {tc:.0f} C = {r:.2f} (manuscript 1.4, 2.1)", abs(r - exp) < 0.01)

    # McBride-Wright 2015 CO2-water densities (Paper 21), reduced at measured
    # loading against IAPWS water: HELD OUT, the route 2.5% below (2026-09-06)
    import validation.mcbridewright2015_density_check as MW
    check("McBride-Wright 2015 CO2 reduction reproduces its pinned isotherm means (route -0.2% with the source calibrated; slope -42)",
          MW.main(check=True))
    import validation.hebach2004_density_check as HB
    check("Hebach 2004 CO2-saturated water reduction reproduces its pinned isotherm means (route -3.1% isotherms, -4.8% coexistence; x from flash)",
          HB.main(check=True))
    check("Hebach 2004's 11 isotherm means are HELD OUT (x from flash), none in the CO2 calibration set",
          sum('Hebach' in p[3] for p in HELD_OUT['CO2']) == 11 and not any('Hebach' in p[3] for p in cs['CO2']))
    check("McBride-Wright 2015's 98 rows are IN the CO2 calibration set (n=109) since 2026-09-06",
          sum('McBride-Wright' in p[3] for p in cs['CO2']) == 98 and len(cs['CO2']) == 109
          and not any('McBride-Wright' in p[3] for p in HELD_OUT['CO2']))

    # "the size of the choice": shifts fitted to calibration + held-out points,
    # and the delivered-density consequence at saturation (manuscript 4.2)
    from brine_gas.brine_properties import rho_brine as _rb, salinity_from_molality as _sfm, M_NACL as _MN
    from pyrestoolbox import brine as _rtb
    for gas, s_all_exp, dens_exp in (('N2', -0.1377, (0.008, 0.022)), ('H2', -0.0540, (0.034, 0.150))):
        b = pr.b_covolume(gas)
        pts = cs[gas] + [p for p in HELD_OUT[gas] if 'finite' not in p[3]]
        s_all = np.mean([pr.V2_inf_raw(gas, T_, P_) - v for T_, P_, v, _ in pts]) / b
        check(f"{gas}: shift fitted to calibration + held-out = {s_all:+.4f}", abs(s_all - s_all_exp) < 0.0005)
        for (tc, P_), exp in zip(((30.0, 30.0), (177.0, 99.0)), dens_exp):
            kw = {'N2': dict(y_N2=1.0), 'H2': dict(y_H2=1.0)}[gas]
            x = float(_rtb.SoreideWhitson(pres=P_ * 10, temp=tc, ppm=_MN / (1000 + _MN) * 1e6,
                                          metric=True, framework='default', **kw).x[gas])
            T_ = tc + 273.15
            r1 = _rb(T_, P_, _sfm(1.0)) / 1000.0
            mm = x / (1 - x) * 55.5084
            W = 1000.0 + _MN
            dens = lambda V: (W + mm * gas_mw(gas)) / (W / r1 + mm * V)
            Vc = V_phi_route(gas, T_, P_, 'auto', 1.0)
            Va = Vc - (s_all - pr.VSHIFT[gas]) * b
            d = 100 * (dens(Va) / dens(Vc) - 1)
            check(f"{gas}: all-points shift would raise saturated density by {d:+.3f}% at {tc:.0f} C/{P_:.0f} MPa (manuscript {exp:.2f})",
                  abs(d - exp) < 0.004)


# ============================================================================
# TEST 18: pins from the 2026-09-06 ChatGPT revision round (Section 4.5 and
# Table 5 H2S transfer sensitivity; the equal-source vector of TEST 17 covers
# the -0.45 CO2 correction from the same round)
# ============================================================================
def test_revision_2026_09_06():
    print("\n=== TEST 18: 2026-09-06 revision pins ===")
    from brine_gas.vphi_route import V_phi as _V
    from brine_gas.brine_properties import rho_brine as _rb, salinity_from_molality as _sfm
    # The H2S sign question, quantified: the whole pressure-plus-salinity
    # transfer from the calibration state (fresh water, 0.1 MPa) to the
    # figure's brine state (1 mol/kg, 30 MPa) at 303.15 K, against the
    # sign-reversal margin there. 303 K in the manuscript means 30 degC
    # (303.15 K) everywhere in this project; evaluating at 303.0 K gives
    # 34.27/33.40 instead of 34.28/33.41.
    T_ = 303.15
    v0 = _V('H2S', T_, 0.1, m_nacl=0.0)
    v1 = _V('H2S', T_, 30.0, m_nacl=1.0)
    red = (v0 - v1) / v0
    v2 = v0 * (1 - 2 * red)
    rho1 = _rb(T_, 30.0, _sfm(1.0)) / 1000.0
    vn = gas_mw('H2S') / rho1
    margin = 100 * (1 - vn / v1)
    check(f"H2S V_phi fresh water 303.15 K/0.1 MPa = {v0:.2f} (manuscript 35.14)",
          abs(v0 - 35.14) < 0.005, f"got {v0:.4f}")
    check(f"H2S V_phi 1 mol/kg, 30 MPa = {v1:.2f} (manuscript 34.28)",
          abs(v1 - 34.28) < 0.005, f"got {v1:.4f}")
    check(f"combined pressure+salt transfer = {100*red:.1f}% (manuscript 2.5%)",
          abs(100 * red - 2.5) < 0.05, f"got {100*red:.3f}")
    check(f"doubled transfer gives {v2:.2f} (manuscript 33.41)",
          abs(v2 - 33.41) < 0.005, f"got {v2:.4f}")
    check(f"density-neutral volume M/rho1 = {vn:.2f} (manuscript 32.56)",
          abs(vn - 32.56) < 0.005, f"got {vn:.4f}")
    check(f"sign-reversal margin = {margin:.1f}% (manuscript 5%)",
          abs(margin - 5.0) < 0.05, f"got {margin:.3f}")
    check("lightening survives omitting or doubling the transfer",
          v0 > vn and v2 > vn)
    # Section 3's finite-loading solvent-basis ratio (W_b + q)/(W_w + q) at the
    # state the text names (formerly Fig. 1's state; the figure was cut
    # 2026-09-06 and the number had no pin)
    Tb, Pb, mb, xb = 323.15, 20.0, 5.0, 0.015
    Sb = _sfm(mb); r1 = _rb(Tb, Pb, Sb) / 1000.0
    m2b = xb / (1 - xb) * (1000.0 / MW_WATER)
    qb = m2b * r1 * _V('CO2', Tb, Pb, m_nacl=mb)
    ratio = 100 * ((1000.0 + mb * 58.443 + qb) / (1000.0 + qb) - 1)
    # Section 4.1's constant-vs-s(T) test with Murphy-Gaines held out. The
    # printed 1.3%/0.6% dated from the July calibration (mc3, no Barbero);
    # re-pinned 2026-09-06 under the delivered calibration. Same verdict.
    from scipy.optimize import minimize as _min, minimize_scalar as _mins
    from fits.fit_pr_vshift import calibration_set as _cal, _score as _sc
    import brine_gas.pr_vphi_model as _pr
    from fits.fit_pr_vshift import murphy_gaines_h2s as _mgh
    _mg = _mgh()
    _all = _cal()['H2S']
    _mgk = {(round(T_, 3), round(P_, 4)) for T_, P_, _ in _mg}
    _rest = [q for q in _all if (round(q[0], 3), round(q[1], 4)) not in _mgk]
    _b = _pr.b_covolume('H2S')
    _s1 = _mins(lambda s_: _sc(_rest, lambda T_, P_: _pr.V2_inf_raw('H2S', T_, P_) - s_ * _b)['rms'],
                bounds=(-1, 1), method='bounded', options={'xatol': 1e-9}).x
    _p2 = _min(lambda p_: _sc(_rest, lambda T_, P_: _pr.V2_inf_raw('H2S', T_, P_) - (p_[0] + p_[1] * 1000 / T_) * _b)['rms'],
               [_s1, 0.0], method='Nelder-Mead', options={'maxiter': 20000, 'fatol': 1e-12}).x
    _mgp = [(T_, P_, V_, 'MG') for T_, P_, V_ in _mg]
    _e1 = _sc(_mgp, lambda T_, P_: _pr.V2_inf_raw('H2S', T_, P_) - _s1 * _b)['map_']
    _e2 = _sc(_mgp, lambda T_, P_: _pr.V2_inf_raw('H2S', T_, P_) - (_p2[0] + _p2[1] * 1000 / T_) * _b)['map_']
    check(f"Murphy-Gaines held out (n={len(_rest)} remain): constant shift {_e1:.2f}% vs s(T) {_e2:.2f}% (manuscript 0.43 vs 0.63); constant wins",
          abs(_e1 - 0.43) < 0.005 and abs(_e2 - 0.63) < 0.005 and _e1 < _e2 and len(_rest) == 13, f"got {_e1:.4f} {_e2:.4f} n={len(_rest)}")
    # in-sample: the second parameter buys nothing for the temperature-resolved
    # gases and is undetermined for the 298 K clusters (Section 4.1)
    _exp = {'CH4': (1.44, 1.46), 'CO2': (1.06, 1.05), 'H2S': (0.48, 0.49)}
    for _g in ('CH4', 'CO2', 'H2S', 'H2', 'C2H6'):
        _pts = _cal()[_g]; _bg = _pr.b_covolume(_g)
        _sg = _mins(lambda s_: _sc(_pts, lambda T_, P_: _pr.V2_inf_raw(_g, T_, P_) - s_ * _bg)['rms'],
                    bounds=(-1, 1), method='bounded', options={'xatol': 1e-9}).x
        _pg = _min(lambda p_: _sc(_pts, lambda T_, P_: _pr.V2_inf_raw(_g, T_, P_) - (p_[0] + p_[1] * 1000 / T_) * _bg)['rms'],
                   [_sg, 0.0], method='Nelder-Mead', options={'maxiter': 20000, 'fatol': 1e-12}).x
        _m1 = _sc(_pts, lambda T_, P_: _pr.V2_inf_raw(_g, T_, P_) - _sg * _bg)['map_']
        _m2 = _sc(_pts, lambda T_, P_: _pr.V2_inf_raw(_g, T_, P_) - (_pg[0] + _pg[1] * 1000 / T_) * _bg)['map_']
        if _g in _exp:
            check(f"{_g} in-sample MAE constant {_m1:.2f}% vs s(T) {_m2:.2f}% (manuscript {_exp[_g][0]:.2f} / {_exp[_g][1]:.2f}; within 0.02 pp)",
                  abs(_m1 - _exp[_g][0]) < 0.005 and abs(_m2 - _exp[_g][1]) < 0.005 and abs(_m1 - _m2) <= 0.02)
        else:
            check(f"{_g}: 298 K cluster, temperature term undetermined (s1 = {_pg[1]:+.4f}, error unchanged {_m1:.2f}%)",
                  abs(_pg[1]) < 1e-3 and abs(_m1 - _m2) < 1e-3)
    check(f"solvent-basis over-count at 5 mol/kg, 323 K, 20 MPa, x = 0.015: {ratio:.1f}% (record value, cut from the prose 2026-09-06; dilute limit {100 * (1 / (1 - Sb) - 1):.1f})",
          abs(ratio - 28.3) < 0.05 and abs(100 * (1 / (1 - Sb) - 1) - 29.2) < 0.05, f"got {ratio:.3f}")
    # Section 5.3's H2S trend sentence, pinned at printed precision (2026-09-06:
    # the Plyasunov value read +0.02; the Part III correlation gives +0.01 at
    # the Murphy-Gaines group-mean states, and the old TEST only bounded it).
    import numpy as _np
    from fits.fit_pr_vshift import murphy_gaines_h2s
    from validation.akinfiev_diamond_h2s import v_inf_ad03
    _mg = murphy_gaines_h2s()
    _cold = [(T_, P_) for T_, P_, _ in _mg if T_ < 300]
    _warm = [(T_, P_) for T_, P_, _ in _mg if T_ > 305]
    Tc, Pc = _np.mean([t for t, _ in _cold]), _np.mean([p for _, p in _cold])
    Tw, Pw = _np.mean([t for t, _ in _warm]), _np.mean([p for _, p in _warm])
    meas = _np.mean([v for T_, _, v in _mg if T_ > 305]) - _np.mean([v for T_, _, v in _mg if T_ < 300])
    for name, fn, exp in (('this work', lambda t, p: _V('H2S', t, p), 0.53),
                          ('Plyasunov Part III', lambda t, p: float(V2_inf('H2S', t, p)), 0.01),
                          ('Akinfiev-Diamond 2003', lambda t, p: v_inf_ad03('H2S', t, p), -0.31)):
        d = fn(Tw, Pw) - fn(Tc, Pc)
        check(f"H2S cold->warm group trend, {name}: {d:+.2f} cm3/mol (manuscript {exp:+.2f}; measured {meas:+.2f})",
              abs(d - exp) < 0.005 and abs(meas - 0.69) < 0.005, f"got {d:.4f}, measured {meas:.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("BRINE PROPS - COMPREHENSIVE VALIDATION")
    print("=" * 60)

    test_V2inf_298K()
    test_V2inf_vs_hnedkovsky()
    test_V2inf_vs_garcia_polynomial()
    test_CO2_density()
    test_physical_reasonableness()
    test_N2_vs_osullivan()
    test_garcia_solvent_basis()
    test_vphi_route_dispatch()
    test_viscosity_corrections()
    test_independent_evidence()
    test_iapws2008_viscosity()
    test_pitzer_brine_leg()
    test_appelo_salt_leg()
    test_salt_viscosity()
    test_viscosity_route()
    test_mechanism_counterfactuals()
    test_review_2026_09_05()
    test_revision_2026_09_06()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
