"""V2inf of dissolved gases from the Soreide-Whitson modified PR EOS + VSHIFT.

An alternative to `plyasunov_model.V2_inf`, built 2026-07-25. It is NOT a
replacement: Plyasunov remains the reference and covers 8 gases to 573 K. This
route exists because it is far more parsimonious and because it repairs the one
V_phi defect the project has pinned (the H2S temperature trend).

THE EXACT RELATION. For any pressure-explicit EOS the partial molar volume is

    Vbar_2 = -(dP/dn2)_{T,V,n1} / (dP/dV)_{T,n}

and at x2 -> 0 that is V2inf. No flash and no root selection for the solute: the
volume root is the pure-water LIQUID root (x2 = 0 exactly), and the gas enters only through b2
and the cross term a12 = sqrt(a1 a2)(1 - kij). The solute never has a root of
its own, so there is nothing to "force" into a liquid state.

WHAT TO SAY WHEN CHALLENGED ON PR'S POOR WATER PROPERTIES. Two separate points,
and the first is a scope point rather than a numerical one:

  1. PR's water volumetrics never reach the delivered answer. This module
     supplies V_phi ONLY. The solvent density rho1 in the Garcia mixing rule
     comes from Spivey (`brine_properties.rho_brine`), never from this EOS. So
     PR's water error is confined to an intermediate quantity that is then
     calibrated against measured V_phi.

  2. A constant shift absorbs an OFFSET but cannot fix a SHAPE, so the question
     that matters is whether V_phi's T and P dependence survives. The answer is
     empirical: one constant s per gas reproduces Hnedkovsky densimetry to 0.5%
     mean absolute error for H2S, 0.9% for CO2 and 1.4% for CH4 across
     298-473 K. Lead with that measurement, not with a story about errors
     cancelling.

Do NOT justify the temperature cap by citing PR's 15-17% water density error:
the chain never uses the EOS water volume as a volume, and substituting the
true one is a translation of water, which leaves Vbar2 untouched (c1 never
appears in Vbar2). What the water DESCRIPTION does set is the evaluation root;
the constant shift's adequacy for that is the empirical point above. The cap
exists for two other reasons, given at `T_MAX`.

VOLUME SHIFT. Peneloux translation v = v_EOS - sum(n_i c_i) gives, exactly,

    V2inf(shifted) = V2inf(EOS) - c2,      c2 = s2 * b2

so a constant s2 is a pure offset with no temperature or pressure shape. The
shift is stored dimensionless (s = c/b, the standard VSHIFT convention) and
b2 is the gas's PR co-volume.

ONE PARAMETER PER GAS, NOT TWO. A held-out test (`fit_pr_vshift.py`) showed
s(T) = s0 + s1/T overfits: on the 21 Murphy & Gaines H2S points it scored 1.3%
mean absolute error against 0.6% for a constant s. Constant s is used.

CALIBRATION BASIS: DIRECT VOLUMETRIC MEASUREMENT ONLY. See `fit_pr_vshift.py`
for the set and the exclusions. Indirect determinations (solubility-versus-
pressure, electrochemical) and V_phi inverted from brine density are excluded
from the fit, the latter because they scatter five to seven times wider.

DEPENDENCY NOTE. The S&W alpha, the aqueous kij correlations and the critical
properties come from `pyrestoolbox.brine._lib_vle_engine`, which is the
canonical implementation - they are NOT re-copied here. That does mean this
module, unlike `plyasunov_model`, is not dependency-free, so it is not on the
Dart/Flutter port path unless the S&W engine goes with it (it would, since the
solubility side needs it anyway).
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

from pyrestoolbox.brine import _lib_vle_engine as _SW

from brine_gas.iapws_if97 import p_sat_if97

R = 8.31446261815324          # cm3.MPa/(mol.K), so volumes come out in cm3/mol

# PR constants (Peng-Robinson 1976)
_OMEGA_A = 0.45723553
_OMEGA_B = 0.07779607

# Validity envelope. Two reasons for the 473.15 K ceiling, neither of them the
# 15-17% water density error (the water volume never enters Vbar2; see above):
#   (a) the calibration data stops there. Hnedkovsky's top point is 473.15 K, so
#       anything above is unvalidated extrapolation of a fitted shift.
#   (b) V2inf diverges at the SOLVENT critical point, and a cubic EOS gets the
#       near-critical behaviour wrong in a way no constant offset can repair.
#       Measured on this implementation, the PR/IF97 compressibility ratio for
#       water is 0.27 at 298 K, 0.84 at 473 K, crosses 1.0 near 520 K and reaches
#       1.32 by 613 K - i.e. a growing SHAPE error, not an offset. It shows up
#       downstream as H2S drifting 13% from Plyasunov by 523 K and 37% by 573 K
#       (recomputed 2026-07-25 on the shipped shifts).
T_MIN, T_MAX = 273.15, 623.15
T_CALIBRATED_MAX = 473.15      # top of the calibration data (Hnedkovsky)
T_VOUCHED_MAX = 450.0          # ~350 degF; see below
P_MAX = 100.0

# RANGE WIDENED 2026-07-25 (Mark's call): the route no longer hands over to
# Plyasunov at 473 K. T_MAX is now the IAPWS-IF97 Region 1 ceiling, which the
# saturation-pressure guard needs and which is near where a liquid water root
# ceases to exist at all.
#
# THREE DIFFERENT CEILINGS, AND THEY ARE NOT THE SAME THING:
#   T_MAX (623 K)             - where the arithmetic stops working.
#   T_CALIBRATED_MAX (473 K)  - where the fitted shift stops being fitted.
#   T_VOUCHED_MAX (450 K)     - Mark, 2026-07-25: "realistically I'm not
#                               vouching for accuracy beyond 300-350 degF for
#                               any of this". 350 degF = 450 K. This is the
#                               number to quote in a deliverable; the other two
#                               are implementation limits, not accuracy claims.
#
# The practical envelope therefore sits INSIDE the calibration data, which is
# why removing the 473 K handover costs nothing in practice. Divergence from
# Plyasunov at 30 MPa, for reference only, since neither route is calibrated up
# there:
#
#   T/K     CH4    CO2    H2S     N2     H2   C2H6
#   473     0%     1%     5%      3%     9%    3%
#   523     1%     4%    12%      4%    10%    2%
#   573     4%    13%    35%      8%    13%    5%
#   623    11%    34%    96%     16%    19%   10%

SUPPORTED = ('CH4', 'CO2', 'H2S', 'N2', 'H2', 'C2H6', 'C3H8', 'NC4H10')

# This project spells butane 'NC4H10'; the S&W component tables spell it
# 'nC4H10'. That mismatch is why butane looked absent from the component set
# until 2026-07-25 - it was there all along, under the other spelling.
_SW_NAME = {'NC4H10': 'nC4H10'}

# WATER ALPHA INSENSITIVITY, measured 2026-07-25 and worth knowing before
# porting. The S&W framework uses `alpha_water_soreide` for the sw_original and
# dropin frameworks but Mathias-Copeman (`alpha_water_mc3`) otherwise, and the
# ResToolbox3 Dart path uses MC3. Swapping the water alpha moves V2inf_raw by
# only 0.00 to 0.05 cm3/mol over 298-473 K, i.e. delta-s of 0.0002 to 0.0018,
# under 2% of the shift values and negligible against the 0.5-1.4% fit error.
# So ONE VSHIFT set is valid for both water alphas and the ports do not need
# separate calibrations. (The alpha enters only a1, and its effect largely
# cancels in the ratio of derivatives.)

# Dimensionless volume shifts s = c2/b2, fitted to direct volumetric
# measurements only by `fit_pr_vshift.py`. Regenerate with that script if any
# calibration point changes; the values below are pinned in validation.py.
VSHIFT = {
    'CH4':  -0.111430,
    'CO2':  -0.070103,
    'H2S':  -0.079416,
    'N2':   -0.176768,
    'H2':   -0.178503,
    'C2H6': -0.073843,
    # C3H8 added 2026-07-25. Unlike the six above it is NOT fitted to a
    # densimetric data set, because none exists for propane in water. It is set
    # from the only two direct 298 K determinations on disk, Moore (1982) 70.7
    # and Zhou & Battino (2001) 75.0 cm3/mol, which imply s = -0.075126 and
    # -0.151527; the adopted value is their mean. Those two disagree by 6.1%,
    # which is the honest uncertainty on this gas and is far wider than for any
    # other. It replaces a fallback to a correlation that sat BELOW both of them
    # (66.99), so it is an improvement on a low bar rather than a calibration.
    'C3H8': -0.113326,
    # NC4H10 added 2026-07-25 from Moore (1982) Table I, 76.6 +/- 0.1 cm3/mol
    # (2 runs; his stated overall imprecision is +/-1.5 cm3/mol, so treat it as
    # +/-2%). Value read off a 700 dpi render of the page.
    #
    # TWO THINGS ABOUT THIS ONE ARE ODD AND SHOULD NOT BE SMOOTHED OVER:
    #   1. The shift is POSITIVE, alone among the eight. Every other gas needs
    #      the EOS volume increased; butane needs it reduced.
    #   2. It breaks Moore's own homologous series. His CH2 increments run
    #      +18.4 (C1->C2), +17.8 (C2->C3), then +5.9 (C3->C4). Either his
    #      butane is low or his propane is high; they cannot both sit on a line.
    # It is adopted because it is the only direct measurement of this quantity,
    # and this project's rule is to fit direct volumetric data. But a single
    # lab, two runs, and a broken series is thin, and C4 is outside the
    # five-gas validated scope.
    'NC4H10': +0.110920,
}

# Gases whose shift rests on 298 K data alone, so their temperature behaviour is
# the EOS's unaided prediction and is NOT calibrated. Reported, not hidden.
# C3H8 is uncalibrated in T for a stronger reason than the other three: it has
# no temperature-resolved data at all, only two mutually inconsistent 298 K
# points, so its T behaviour is entirely the EOS unaided.
UNCALIBRATED_IN_T = ('N2', 'H2', 'C2H6', 'C3H8', 'NC4H10')


def _ab(species, T, m_nacl=0.0):
    """PR a(T) and b for one species, cm6.MPa/mol2 and cm3/mol."""
    c = _SW.COMPONENTS[_SW_NAME.get(species, species)]
    Pc = c.Pc / 1e6
    Tr = T / c.Tc
    if species == 'H2O':
        alpha = _SW.alpha_water_soreide(Tr, m_nacl)
    else:
        alpha = _SW.alpha_standard_pr(Tr, c.omega)
    return (_OMEGA_A * R ** 2 * c.Tc ** 2 / Pc * alpha,
            _OMEGA_B * R * c.Tc / Pc)


def b_covolume(gas):
    """PR co-volume of the gas, cm3/mol - the normaliser for the VSHIFT."""
    return _ab(gas, 300.0)[1]


def _P_mix(T, V, n1, n2, gas, kij, m_nacl):
    """PR pressure of the binary water + gas mixture, MPa. V in cm3, n in mol."""
    a1, b1 = _ab('H2O', T, m_nacl)
    a2, b2 = _ab(gas, T)
    n = n1 + n2
    x1, x2 = n1 / n, n2 / n
    a12 = np.sqrt(a1 * a2) * (1.0 - kij)
    am = x1 * x1 * a1 + 2.0 * x1 * x2 * a12 + x2 * x2 * a2
    bm = x1 * b1 + x2 * b2
    return (n * R * T / (V - n * bm)
            - n * n * am / (V * V + 2.0 * n * bm * V - n * n * bm * bm))


def v_water_liquid(T, P, m_nacl=0.0):
    """Smallest real PR root for water/brine, cm3/mol.

    Above water's critical temperature the cubic has one real root; below it the
    smallest root is the liquid. Taking the minimum root is continuous across
    both cases, which is what the derivative below needs.
    """
    a, b = _ab('H2O', T, m_nacl)
    A = a * P / (R * T) ** 2
    B = b * P / (R * T)
    roots = np.roots([1.0, -(1.0 - B), A - 3.0 * B ** 2 - 2.0 * B,
                      -(A * B - B ** 2 - B ** 3)])
    real = roots[np.abs(roots.imag) < 1e-9].real
    real = real[real > B]
    if real.size == 0:
        raise ValueError(f'No admissible PR liquid root for water at {T} K, {P} MPa')
    return float(real.min()) * R * T / P


def _check(gas, T, P):
    if gas not in SUPPORTED:
        raise ValueError(
            f'{gas!r} has no fitted VSHIFT. Supported: {SUPPORTED}. Use '
            f'plyasunov_model.V2_inf for the other gases.')
    if not (T_MIN <= T <= T_MAX):
        raise ValueError(
            f'T = {T} K is outside {T_MIN}-{T_MAX} K. The upper bound is the '
            f'IAPWS-IF97 Region 1 ceiling, needed for the saturation-pressure '
            f'guard, and is near where a liquid water root ceases to exist; use '
            f'plyasunov_model.V2_inf above it.')
    if not (0.0 < P <= P_MAX):
        raise ValueError(f'P = {P} MPa is outside 0 to {P_MAX} MPa.')
    psat = float(p_sat_if97(T))
    if P < psat:
        raise ValueError(
            f'P = {P} MPa is below the water saturation pressure {psat:.4f} MPa '
            f'at {T} K, so the solvent is vapour and V2inf is undefined.')


def is_extrapolated(gas, T, P=None):
    """True where the volume shift is being used beyond its calibration data.

    The shift is fitted to direct volumetric measurement over 273-473 K. Above
    T_CALIBRATED_MAX the route still returns a value (that is deliberate, so the
    default does not hand over mid-range) but the shift is extrapolating and the
    near-critical shape error is growing. Callers that report uncertainty should
    check this rather than assume the whole range is equally earned.
    """
    return T > T_CALIBRATED_MAX or gas in UNCALIBRATED_IN_T


def V2_inf_raw(gas, T, P, m_nacl=0.0):
    """Unshifted V2inf from the exact EOS relation, cm3/mol. No range guard.

    The two derivatives of V2 = -(dP/dn2)/(dP/dV) are evaluated ANALYTICALLY
    at n1 = 1, n2 = 0, V = v_w (closed forms: manuscript Appendix B).
    Switched from finite differences 2026-07-31; the difference evaluation of
    the same derivatives agrees to 4e-5 cm3/mol, its truncation error.
    """
    v = v_water_liquid(T, P, m_nacl)
    # Pinned to the 'default' kij_AQ set: the refresh paper's published
    # recommendation (S&W alpha, refitted freshwater kij, embedded delta_kij),
    # adopted for this work 2026-09-01 (Mark's call) and the set the VSHIFT
    # values above are fitted against. The pin stays EXPLICIT even while it
    # matches the library default: the shift and the kij are one calibration,
    # and an unpinned call silently follows any future default move (which is
    # exactly what broke 3 validation checks in 2026-08 under the old 'mc3'
    # pairing).
    kij = _SW.get_kij_aq(_SW_NAME.get(gas, gas), T, m_nacl, framework='default')
    a1, b1 = _ab('H2O', T, m_nacl)
    a2, b2 = _ab(gas, T)
    a12 = np.sqrt(a1 * a2) * (1.0 - kij)
    D = v * v + 2.0 * b1 * v - b1 * b1
    dPdn2 = (R * T / (v - b1) + R * T * b2 / (v - b1) ** 2
             - (2.0 * a12 * D - 2.0 * a1 * b2 * (v - b1)) / D ** 2)
    dPdV = -R * T / (v - b1) ** 2 + 2.0 * a1 * (v + b1) / D ** 2
    return -dPdn2 / dPdV


def _derivatives(gas, T, P, m_nacl=0.0):
    """Every closed-form quantity behind V2inf and its pressure derivative at
    n1 = 1, n2 = 0, V = v_w (manuscript Appendix B): the two first derivatives
    of the PR mixture pressure, the two second derivatives the chain rule
    needs, and the assembled dV2inf/dp. Returned as a dict so the worked
    examples can print each step."""
    v = v_water_liquid(T, P, m_nacl)
    kij = _SW.get_kij_aq(_SW_NAME.get(gas, gas), T, m_nacl, framework='default')
    a1, b1 = _ab('H2O', T, m_nacl)
    a2, b2 = _ab(gas, T)
    a12 = np.sqrt(a1 * a2) * (1.0 - kij)
    D = v * v + 2.0 * b1 * v - b1 * b1
    Dp = 2.0 * (v + b1)                              # dD/dv
    g = 2.0 * a12 * D - 2.0 * a1 * b2 * (v - b1)
    gp = 2.0 * a12 * Dp - 2.0 * a1 * b2              # dg/dv
    N = R * T / (v - b1) + R * T * b2 / (v - b1) ** 2 - g / D ** 2     # dp/dn2
    Dv = -R * T / (v - b1) ** 2 + 2.0 * a1 * (v + b1) / D ** 2         # dp/dV
    Np = (-R * T / (v - b1) ** 2 - 2.0 * R * T * b2 / (v - b1) ** 3
          - (gp * D - 2.0 * g * Dp) / D ** 3)                          # d2p/dn2 dV
    Dvp = (2.0 * R * T / (v - b1) ** 3
           + 2.0 * a1 * (D - 2.0 * (v + b1) * Dp) / D ** 3)            # d2p/dV2
    V2 = -N / Dv
    dV2_dv = -(Np * Dv - N * Dvp) / Dv ** 2          # d(-N/Dv)/dv
    dv_dp = 1.0 / Dv                                 # pure water, fixed T
    return dict(v_w=v, kij=kij, a1=a1, b1=b1, a2=a2, b2=b2, a12=a12, D=D,
                dPdn2=N, dPdV=Dv, d2Pdn2dV=Np, d2PdV2=Dvp, V2_eos=V2,
                dV2_dv=dV2_dv, dvw_dp=dv_dp, dV2_dp=dV2_dv * dv_dp)


def dV2_inf_dp(gas, T, P, m_nacl=0.0):
    """Closed-form pressure derivative of V2inf, cm3/mol/MPa, at fixed T and
    infinite dilution: dV2/dp = -[(d2p/dn2dV)(dp/dV) - (dp/dn2)(d2p/dV2)] /
    (dp/dV)^3, the chain rule through the water root (dv_w/dp = 1/(dp/dV)).
    The constant volume shift has no pressure derivative, so this is also the
    derivative of the shifted V2inf. Implemented 2026-09-07 (Mark: the
    manuscript stated the chain rule; the code had used finite differences)."""
    _check(gas, T, P)
    return float(_derivatives(gas, T, P, m_nacl)['dV2_dp'])


def V2_inf(gas, T, P, m_nacl=0.0, s=None):
    """Apparent molar volume of the dissolved gas at infinite dilution, cm3/mol.

    Args:
        gas: one of SUPPORTED
        T: K, within T_MIN..T_MAX and above the water saturation pressure
        P: MPa, up to P_MAX
        m_nacl: NaCl molality. Enters the S&W water alpha and kij only; there is
            NO salinity term on the volume shift. The salt effect on V_phi
            is applied one level up (vphi_route, as a relative fraction since
            2026-07-30), never here, so there is no double counting. Leave at
            0 for the calibrated behaviour.
        s: override the dimensionless volume shift (for refitting).

    Returns:
        V2inf in cm3/mol.
    """
    _check(gas, T, P)
    shift = VSHIFT[gas] if s is None else s
    return V2_inf_raw(gas, T, P, m_nacl) - shift * b_covolume(gas)


# `plyasunov_model` exposes V_phi as an alias for V2_inf; mirror that so the two
# modules are drop-in comparable in scripts.
V_phi = V2_inf


if __name__ == '__main__':
    print('S&W PR + VSHIFT, V2inf in cm3/mol (Plyasunov in brackets)')
    from brine_gas.plyasunov_model import V2_inf as V_ply
    print(f"\n  {'gas':<6}{'s':>10}{'b':>8}  " +
          ''.join(f'{t:>18.0f} K' for t in (298, 348, 398, 448)))
    for g in SUPPORTED:
        cells = []
        for T in (298.15, 348.15, 398.15, 448.15):
            cells.append(f'{V2_inf(g, T, 20.0):7.2f} '
                         f'[{float(V_ply(g, T, 20.0)):6.2f}]')
        flag = ' *' if g in UNCALIBRATED_IN_T else ''
        print(f'  {g:<6}{VSHIFT[g]:10.4f}{b_covolume(g):8.2f}  '
              + ''.join(f'{c:>20}' for c in cells) + flag)
    print('\n  * temperature behaviour is the EOS unaided: these gases have no '
          'T-resolved\n    calibration data, only a 298 K cluster.')
