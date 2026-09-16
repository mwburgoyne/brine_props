"""Ezrokhi coefficients for dissolved-gas brine density and viscosity.

PURPOSE. Reservoir simulators (Eclipse E300 and workalikes) model brine density
and viscosity with dissolved components through the Ezrokhi form

    log10(rho) = log10(rho_0(P,T)) + sum_i A_i(T) * w_i          Eq. 6.69
    log10(mu)  = log10(mu_0(P,T))  + sum_i B_i(T) * w_i          Eq. 6.71

verified against the ECLIPSE Technical Description, Enhanced oil recovery,
p. 472 (PDF page 486), keywords DENAQA and VISCAQA. Note two things that are
easy to get wrong and that the manual is explicit about:

  * rho_0 and mu_0 are PURE WATER, not brine.
  * the sum runs over every non-water component, so DISSOLVED SALT IS ITSELF
    AN EZROKHI COMPONENT with its own A and B, alongside each gas.

w_i is the weight fraction of non-water component i in the aqueous phase, and

    A_i(T) = a0 + a1*T + a2*T^2       T in DEGREES CELSIUS, always, even when
    B_i(T) = b0 + b1*T + b2*T^2       the simulation runs in field units.

This script regenerates those coefficients from the current framework, replacing
an earlier hand fit built on Garcia's CO2 volume correlation and Islam-Carlson
viscosity ('Solubility & Density Workbook.xlsm'). Two conventions changed and
both matter: that workbook fitted against MOLE fraction and used T in degrees
FAHRENHEIT, so its coefficients cannot be pasted into a deck expecting the
Eclipse convention. Both forms are reported below.

WHAT THIS MEANS FOR THE GAS COEFFICIENTS. Because the Ezrokhi sum is additive
in log10(rho), the salt and gas contributions separate exactly: the salt
coefficient must reproduce log10(rho_brine/rho_water) and the gas coefficient
must reproduce log10(rho_brine+gas/rho_brine). So the gas coefficients below are
fitted to the incremental effect of the gas at fixed salinity, which is the
correct quantity, and salt coefficients are reported alongside so the pair is
self-consistent.

WHERE THE DENSITY COEFFICIENT COMES FROM. It is not a numerical fit at all. The
density relation is a mass balance, so for small dissolved amounts

    log10(rho/rho_1) ~ w * (M2 - rho_1 * V_phi) / (M2 * ln 10)

which is exact to first order in w and gives A(T,P) analytically. Everything
gas-specific enters through V_phi and the molar mass, and the sign of A is the
sign of (M2 - rho_1 V_phi) - the same quantity that decides whether a gas
densifies or lightens brine.

THE STRUCTURAL LIMITATION, stated up front. A depends on PRESSURE through both
rho_1 and V_phi, and the Ezrokhi form has no pressure term. The fit is therefore
quoted at a reference pressure with the spread over the pressure range reported
alongside, so the user can see what the form cannot carry.
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

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from brine_gas.brine_properties import rho_brine, salinity_from_molality, molality_from_salinity
from brine_gas.plyasunov_model import gas_mw
from brine_gas.vphi_route import V_phi
from brine_gas.water_properties import MW_WATER

LN10 = np.log(10.0)
P_REF = 30.0                      # MPa, where the coefficients are quoted
P_RANGE = (10.0, 70.0)            # MPa, over which the spread is reported
T_C = np.arange(20.0, 155.0, 5.0)  # degC, the fitting window


def A_density(gas, T_C_, P_MPa, m_nacl):
    """Ezrokhi density coefficient at one state, per unit MASS fraction."""
    T = T_C_ + 273.15
    S = salinity_from_molality(m_nacl)
    rho1 = rho_brine(T, P_MPa, S) / 1000.0 if m_nacl > 0 else None
    if rho1 is None:
        from brine_gas.water_properties import rho_w
        rho1 = rho_w(T, P_MPa) / 1000.0
    M2 = gas_mw(gas)
    V = V_phi(gas, T, P_MPa, 'auto', m_nacl)
    return (M2 - rho1 * V) / (M2 * LN10)


def A_density_molefrac(gas, T_C_, P_MPa, m_nacl):
    """Same, but per unit MOLE fraction - the workbook's convention."""
    S = salinity_from_molality(m_nacl)
    W = MW_WATER / (1.0 - S)          # solvent mass carried by 1 mol of water
    T = T_C_ + 273.15
    from brine_gas.water_properties import rho_w
    rho1 = (rho_brine(T, P_MPa, S) if m_nacl > 0 else rho_w(T, P_MPa)) / 1000.0
    return (gas_mw(gas) - rho1 * V_phi(gas, T, P_MPa, 'auto', m_nacl)) / (W * LN10)


def B_viscosity(gas, T_C_, m_nacl, x_ref):
    """Ezrokhi viscosity coefficient per unit MASS fraction, at a reference x.

    The viscosity corrections are not all linear in composition, so B is not a
    constant of the gas alone: CH4 saturates, so its secant coefficient depends
    on how much gas is dissolved. That is reported rather than hidden - the
    coefficient is evaluated at a stated x_ref.
    """
    from brine_gas.garcia_mixing import viscosity_correction_single
    degf = T_C_ * 1.8 + 32.0
    try:
        f = viscosity_correction_single(gas, x_ref, degf=degf)
    except TypeError:
        f = viscosity_correction_single(gas, x_ref)
    if f == 1.0:
        return 0.0
    S = salinity_from_molality(m_nacl)
    W = MW_WATER / (1.0 - S)
    M2 = gas_mw(gas)
    # Weight fraction of dissolved gas: x_ref moles of gas accompany (1 - x_ref)
    # moles of water (salt-free basis), each mole of water carrying W grams of
    # brine. The pre-2026-08-08 form paired x_ref with a FULL mole of water
    # (w = x*M2/(W + x*M2)), overstating B by 1/(1 - x_ref) - the same solvent
    # basis error class the manuscript warns against, caught by the derivation
    # audit.
    w = x_ref * M2 / ((1.0 - x_ref) * W + x_ref * M2)
    return np.log10(f) / w


# --- NaCl as a VISCOSITY component (2026-09-08, Mark: a simulator has no salt
# ratio; it needs a B set, and ONE set, because a per-component constant is
# the contract of the Ezrokhi form). The baseline's salt ratio (ion-additive
# Jones-Dole x Kestin's pressure factor over IAPWS water) is not linear in
# weight fraction, so the single set is a least-squares fit of log10(mu_b/mu_w)
# on w THROUGH THE ORIGIN over 0.5-5 mol/kg (the form has no intercept), and
# its error across that range is reported rather than hidden. Quoted at P_REF;
# the pressure factor moves the ratio by up to ~3% of viscosity at 25 degC.
M_FIT = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)         # mol/kg, the fitting range
KESTIN_BANDS = ((0.5, 1.0, 1.5), (2.0, 3.0, 4.0), (4.0, 5.0, 6.0))
TNAV_NACL_B = (0.71800, 0.003590, 0.0)         # tNavigator manual Table 3.3


def _log_ratio(t, m, P):
    from brine_gas.viscosity_route import brine_viscosity
    from brine_gas.iapws_viscosity import mu_water_TP
    T = t + 273.15
    return np.log10(brine_viscosity(T, P, m=m) / (float(mu_water_TP(T, P)) * 1000.0))


def salt_B_secant(m, P=P_REF):
    """The secant log10(mu_b/mu_w)/S at one molality over T_C: shows how far
    the salt ratio departs from the linear-in-w form the single set assumes."""
    S = salinity_from_molality(m)
    return np.array([_log_ratio(t, m, P) / S for t in T_C])


def salt_B_fit(P=P_REF):
    """B_NaCl(T) over T_C: least squares of log10(mu_b/mu_w) on S through the
    origin over M_FIT, the one set the Ezrokhi form takes."""
    S = np.array([salinity_from_molality(m) for m in M_FIT])
    out = []
    for t in T_C:
        y = np.array([_log_ratio(t, m, P) for m in M_FIT])
        out.append(float((S * y).sum() / (S * S).sum()))
    return np.array(out)


def salt_viscosity_set(P=P_REF):
    """(b0, b1, b2): the quadratic-in-T fit of salt_B_fit."""
    return tuple(quad_fit(T_C, salt_B_fit(P))[:3])


def salt_set_vs_chain(bset, P=P_REF, temps=(25.0, 75.0, 125.0)):
    """Worst % error of 10^(B w) against the baseline it was fitted to, over
    M_FIT: what the linear-in-w form costs."""
    b0, b1, b2 = bset
    worst = 0.0
    for t in temps:
        for m in M_FIT:
            S = salinity_from_molality(m)
            worst = max(worst, abs(10 ** ((b0 + b1 * t + b2 * t * t) * S - _log_ratio(t, m, P)) - 1.0))
    return 100.0 * worst


def score_vs_kestin(bset, ms, temps=(20.0, 50.0, 80.0, 110.0, 150.0),
                    pressures=(0.1, 10.0, 20.0, 35.0)):
    """(mean %, max %) error of 10^(B(T) w) against Kestin's NaCl salt ratio
    over Kestin's own range (Paper 50) at the molalities ms."""
    from brine_gas.kestin_nacl_viscosity import salt_ratio
    b0, b1, b2 = bset
    errs = []
    for t in temps:
        for p in pressures:
            for m in ms:
                S = salinity_from_molality(m)
                errs.append(abs(10 ** ((b0 + b1 * t + b2 * t * t) * S) / salt_ratio(t, p, m) - 1.0))
    return 100.0 * float(np.mean(errs)), 100.0 * float(np.max(errs))


def salt_pressure_effect(m, t=25.0):
    """% change in the viscosity ratio between the 0.1 and 35 MPa secants at t."""
    S = salinity_from_molality(m)
    b_lo = np.interp(t, T_C, salt_B_secant(m, 0.1))
    b_hi = np.interp(t, T_C, salt_B_secant(m, 35.0))
    return 100.0 * (10 ** ((b_hi - b_lo) * S) - 1.0)


# --- ONE B per gas as well (Mark, 2026-09-08: "if it's not pressure dependent
# then x doesn't matter, does it?"): it matters for CH4 because its factor
# saturates in x, but a simulator applies one constant to whatever x a cell
# carries, so each gas set is a least-squares fit of log10 f on w through the
# origin over the loading the gas can reach in the envelope, with the cost of
# the linear form stated. CO2 and H2S are linear, so the fit costs nothing.
GAS_X_MAX = {'CO2': 0.03, 'CH4': 0.01, 'H2S': 0.04}


def _w_of(gas, x):
    M2 = gas_mw(gas)
    return x * M2 / ((1.0 - x) * MW_WATER + x * M2)


def _logf(gas, x, t):
    from brine_gas.garcia_mixing import viscosity_correction_single
    return np.log10(viscosity_correction_single(gas, x, degf=t * 1.8 + 32.0))


def gas_B_fit(gas):
    """B_gas(T) over T_C: least squares of log10 f on w through the origin over
    eight loadings up to GAS_X_MAX[gas]."""
    xmax = GAS_X_MAX[gas]
    xs = np.linspace(xmax / 8.0, xmax, 8)
    w = np.array([_w_of(gas, x) for x in xs])
    out = []
    for t in T_C:
        y = np.array([_logf(gas, x, t) for x in xs])
        out.append(float((w * y).sum() / (w * w).sum()))
    return np.array(out)


def gas_viscosity_set(gas):
    """(b0, b1, b2): the quadratic-in-T fit of gas_B_fit."""
    return tuple(quad_fit(T_C, gas_B_fit(gas))[:3])


def gas_set_vs_chain(gas, bset, temps=(25.0, 75.0, 125.0)):
    """Worst % error of 10^(B w) against the factor itself over the fitted
    loading range (plus a dilute point at xmax/20)."""
    b0, b1, b2 = bset
    xmax = GAS_X_MAX[gas]
    worst = 0.0
    for t in temps:
        for x in list(np.linspace(xmax / 8.0, xmax, 8)) + [xmax / 20.0]:
            worst = max(worst, abs(10 ** ((b0 + b1 * t + b2 * t * t) * _w_of(gas, x) - _logf(gas, x, t)) - 1.0))
    return 100.0 * worst


def gas_viscosity_report(check=False):
    print('\n' + '=' * 78)
    print('GAS B SETS, one per gas, fitted over the loading the gas reaches (through the origin)')
    print('=' * 78)
    sets = {}
    for gas in ('CO2', 'CH4', 'H2S'):
        bset = gas_viscosity_set(gas)
        worst = gas_set_vs_chain(gas, bset)
        sets[gas] = (bset, worst)
        print(f'  {gas:<4} fitted over x <= {GAS_X_MAX[gas]}: b0 = {bset[0]:+.5g}  b1 = {bset[1]:+.5g}  '
              f'b2 = {bset[2]:+.5g}; linear form within {worst:.2f}% of the factor')
    if check:
        assert abs(sets['CO2'][0][0] - 1.094) < 2e-3 and sets['CO2'][1] < 0.4, sets['CO2']
        assert abs(sets['CH4'][0][0] - 7.365) < 5e-3 and abs(sets['CH4'][1] - 3.60) < 0.05, sets['CH4']
        assert abs(sets['H2S'][0][0] - 0.39086) < 1e-4 and sets['H2S'][1] < 0.01, sets['H2S']
        print('  pins hold')
    return sets


def salt_viscosity_report(check=False):
    print('\n' + '=' * 78)
    print('NaCl AS AN EZROKHI VISCOSITY COMPONENT (one set, fitted over 0.5-5 mol/kg, baseline over IAPWS water)')
    print('=' * 78)
    bset = salt_viscosity_set()
    b0, b1, b2 = bset
    bt = salt_B_fit()
    print(f'  b0 = {b0:+.5f}  b1 = {b1:+.5e}  b2 = {b2:+.5e}; '
          f'B at 25/75/125 degC {np.interp(25, T_C, bt):.3f} {np.interp(75, T_C, bt):.3f} {np.interp(125, T_C, bt):.3f}')
    for m in (1.0, 3.0, 5.0):
        print(f'  secant at {m:g} mol/kg, 25 degC: {np.interp(25, T_C, salt_B_secant(m)):.3f}'
              f'  (pressure factor 0.1 -> 35 MPa: {salt_pressure_effect(m, 25):+.2f}% of viscosity)')
    worst = salt_set_vs_chain(bset)
    print(f'  linear-in-w form vs the baseline over 0.5-5 mol/kg, 25-125 degC: worst {worst:.2f}%')
    m5, x5 = score_vs_kestin(bset, M_FIT)
    m6, x6 = score_vs_kestin(bset, (0.5, 1.0, 2.0, 3.0, 4.0, 6.0))
    d6m, d6x = score_vs_kestin(TNAV_NACL_B, (0.5, 1.0, 2.0, 3.0, 4.0, 6.0))
    print(f'  vs Kestin: ours {m5:.2f}% mean / {x5:.2f}% max over 0.5-5 mol/kg, {m6:.2f} / {x6:.2f} over 0.5-6;'
          f' tNavigator default {d6m:.2f} / {d6x:.2f} over 0.5-6')
    for band in KESTIN_BANDS:
        print(f'    band {band}: ours {score_vs_kestin(bset, band)[0]:.2f}% mean, default {score_vs_kestin(TNAV_NACL_B, band)[0]:.2f}%')
    if check:
        # pinned 2026-09-08; the manuscript (App. A) and SIMULATOR.md quote these
        assert abs(b0 - 0.97967) < 5e-4, bset
        assert abs(worst - 4.12) < 0.05, worst
        assert abs(m5 - 1.84) < 0.05 and abs(x5 - 6.09) < 0.1, (m5, x5)
        assert abs(m6 - 2.48) < 0.05 and abs(x6 - 9.52) < 0.1, (m6, x6)
        assert abs(d6m - 3.69) < 0.05 and abs(d6x - 20.45) < 0.1, (d6m, d6x)
        assert abs(score_vs_kestin(bset, KESTIN_BANDS[0])[0] - 1.87) < 0.05
        assert abs(score_vs_kestin(TNAV_NACL_B, KESTIN_BANDS[0])[0] - 1.40) < 0.05
        assert abs(salt_pressure_effect(5.0, 25) - 2.91) < 0.05
        print('  pins hold')
    return bset


def quad_fit(T, y):
    c = np.polyfit(T, y, 2)
    fit = np.polyval(c, T)
    return c[2], c[1], c[0], float(np.max(np.abs(fit - y)))


def report(gas, m_nacl, x_ref):
    print(f'\n  {gas} in {m_nacl:g} mol/kg NaCl brine')
    print('  ' + '-' * 74)

    a = np.array([A_density(gas, t, P_REF, m_nacl) for t in T_C])
    a0, a1, a2, err = quad_fit(T_C, a)
    lo = np.array([A_density(gas, t, P_RANGE[0], m_nacl) for t in T_C])
    hi = np.array([A_density(gas, t, P_RANGE[1], m_nacl) for t in T_C])
    print(f'    DENSITY  a0 = {a0:+.6e}  a1 = {a1:+.6e}  a2 = {a2:+.6e}')
    print(f'             quadratic reproduces A(T) to {err:.2e} over {T_C[0]:.0f}-{T_C[-1]:.0f} degC')
    print(f'             A at 25/75/125 degC: {np.interp(25,T_C,a):+.5f} '
          f'{np.interp(75,T_C,a):+.5f} {np.interp(125,T_C,a):+.5f}')
    print(f'             pressure spread {P_RANGE[0]:.0f}-{P_RANGE[1]:.0f} MPa: '
          f'{np.max(np.abs(hi - lo)):.5f} in A '
          f'({100 * np.max(np.abs(hi - lo)) / max(np.max(np.abs(a)), 1e-12):.0f}% of its own size)')

    am = np.array([A_density_molefrac(gas, t, P_REF, m_nacl) for t in T_C])
    m0, m1, m2_, errm = quad_fit(T_C, am)
    print(f'    (mole-fraction convention, as in the workbook: '
          f'a0 = {m0:+.6e}  a1 = {m1:+.6e}  a2 = {m2_:+.6e})')

    b = np.array([B_viscosity(gas, t, m_nacl, x_ref) for t in T_C])
    if np.allclose(b, 0.0):
        print('    VISCOSITY  no correction for this gas (no measured effect)')
        return
    b0, b1, b2, errb = quad_fit(T_C, b)
    print(f'    VISCOSITY  b0 = {b0:+.6e}  b1 = {b1:+.6e}  b2 = {b2:+.6e}')
    print(f'               at x = {x_ref:g}; quadratic fits to {errb:.2e}')
    print(f'               B at 25/75/125 degC: {np.interp(25,T_C,b):+.4f} '
          f'{np.interp(75,T_C,b):+.4f} {np.interp(125,T_C,b):+.4f}')


def linearity_check(gas, m_nacl):
    """How much does B move with dissolved amount? Ezrokhi assumes not at all."""
    from brine_gas.garcia_mixing import viscosity_correction_single
    print(f'\n  {gas}: is the viscosity correction linear in composition?')
    print(f'    {"x":>10}{"factor":>10}{"B (per mass frac)":>20}')
    for x in (0.0005, 0.001, 0.002, 0.005, 0.01):
        b = B_viscosity(gas, 75.0, m_nacl, x)
        f = viscosity_correction_single(gas, x, degf=167.0)
        print(f'    {x:10.4f}{f:10.4f}{b:20.4f}')


def salt_coefficients(m_grid=(0.5, 1.0, 2.0, 3.0, 4.0, 5.0)):
    """Ezrokhi A for NaCl itself, from Spivey, so the pair is self-consistent."""
    from brine_gas.water_properties import rho_w
    print('\n' + '=' * 78)
    print('NaCl AS AN EZROKHI COMPONENT (from Spivey, relative to pure water)')
    print('=' * 78)
    A = []
    for t in T_C:
        T = t + 273.15
        num, den = [], []
        for m in m_grid:
            S = salinity_from_molality(m)
            r = rho_brine(T, P_REF, S) / 1000.0
            r0 = rho_w(T, P_REF) / 1000.0
            num.append(np.log10(r / r0)); den.append(S)
        A.append(np.polyfit(den, num, 1)[0])     # slope through the origin region
    A = np.array(A)
    a0, a1, a2, err = quad_fit(T_C, A)
    print(f'\n    a0 = {a0:+.6e}  a1 = {a1:+.6e}  a2 = {a2:+.6e}')
    print(f'    reproduces A_NaCl(T) to {err:.2e} over {T_C[0]:.0f}-{T_C[-1]:.0f} degC')
    print(f'    A at 25/75/125 degC: {np.interp(25,T_C,A):+.5f} '
          f'{np.interp(75,T_C,A):+.5f} {np.interp(125,T_C,A):+.5f}')


def emit_table(path):
    """Write the manuscript's Ezrokhi viscosity table (the B sets, freshwater
    reference) so it is \\input rather than retyped. The density sets are the
    pressure cubics of ezrokhi_pressure_fit (the 30 MPa density sets left the
    paper on 2026-09-06 as redundant with them; they still ship in the public
    README and are pinned in validation.py)."""
    def sci(v):
        if v == 0:
            return '0'
        m, e = f'{v:.4e}'.split('e')
        e = int(e)
        return f'${float(m):+.4f} \\times 10^{{{e}}}$' if abs(e) > 1 else f'${v:+.5g}$'
    rows = []
    label = {'CO2': 'CO$_2$', 'CH4': 'CH$_4$', 'H2S': 'H$_2$S', 'N2': 'N$_2$', 'H2': 'H$_2$'}
    for gas in ('CO2', 'CH4', 'H2S'):
        b0, b1, b2 = gas_viscosity_set(gas)
        worst = gas_set_vs_chain(gas, (b0, b1, b2))
        bcells = (f'${b0:+.5f}$ & 0 & 0' if gas == 'H2S'
                  else f'${b0:+.5g}$ & {sci(b1)} & {sci(b2)}')
        cost = 'exact' if worst < 0.05 else f'within {worst:.1f}\\%'
        rows.append(f'{label[gas]} & {bcells} & $x \\le {GAS_X_MAX[gas]:g}$, {cost} \\\\')
    rows.append(f'{label["N2"]} & 0 & 0 & 0 & no effect resolved at $x \\approx 10^{{-4}}$ \\\\')
    rows.append(f'{label["H2"]} & 0 & 0 & 0 & no measurement; no correction applied \\\\')
    rows.append('\\midrule')
    b0, b1, b2 = salt_viscosity_set()
    worst = salt_set_vs_chain((b0, b1, b2))
    rows.append(f'NaCl & ${b0:+.5g}$ & {sci(b1)} & {sci(b2)} & '
                f'0.5 to 5~mol~kg$^{{-1}}$ at 30~MPa, within {worst:.0f}\\% \\\\')
    with open(path, 'w') as f:
        f.write('% GENERATED by code/ezrokhi_fits.py --tex; do not edit\n')
        f.write('\\begin{tabular}{lrrrl}\n\\toprule\n')
        f.write('Solute & $b_0$ & $b_1$ & $b_2$ & fitted over (linear form) \\\\\n\\midrule\n')
        f.write('\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    print('wrote', path)


def main():
    if '--tex' in sys.argv:
        emit_table(os.path.join(HERE, '..', 'manuscript', 'tab_ezrokhi.tex'))
        return 0
    if '--check' in sys.argv:
        gas_viscosity_report(check=True)
        salt_viscosity_report(check=True)
        return 0
    print('=' * 78)
    print('EZROKHI COEFFICIENTS FROM THE CURRENT FRAMEWORK')
    print('=' * 78)
    print(f"""
  Convention (ECLIPSE Technical Description p.472, DENAQA / VISCAQA):
      log10(rho) = log10(rho_pure_water) + sum_i A_i(T)*w_i
      A_i(T) = a0 + a1*T + a2*T^2,  T in DEGREES CELSIUS, w = WEIGHT fraction.
  Salt is itself a component; its coefficients are given at the end so the set
  is self-consistent. The gas coefficients below are the INCREMENTAL effect at
  fixed salinity, which is what the additive form requires.
  Quoted at {P_REF:.0f} MPa; the pressure spread the form cannot carry is
  reported for each gas.""")
    for m in (0.0, 1.0):
        print('\n' + '=' * 78)
        print(f'SALINITY {m:g} mol/kg NaCl')
        print('=' * 78)
        for gas, xr in (('CO2', 0.010), ('CH4', 0.002), ('H2S', 0.010),
                        ('N2', 0.002), ('H2', 0.005)):
            report(gas, m, xr)
    salt_coefficients()
    gas_viscosity_report()
    salt_viscosity_report()
    linearity_check('CH4', 0.0)
    linearity_check('CO2', 0.0)
    print("""
  READ THE LINEARITY CHECK BEFORE USING THE VISCOSITY COEFFICIENTS. Ezrokhi is
  linear in composition by construction. CO2's correction is linear, so its B is
  a genuine constant. CH4's saturates, so its B falls as dissolved gas rises and
  a single value is only valid near the x it was evaluated at. Quote CH4's B
  with the x it belongs to, or accept the error at the dilute end.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
