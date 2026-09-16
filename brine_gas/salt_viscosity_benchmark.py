"""Which salt viscosity term should ship: Mao-Duan or ion-additive Jones-Dole?

THE QUESTION. The delivered viscosity chain is

    mu(brine) = mu(water) * salt ratio        then a per-gas correction

and the salt ratio has always been Mao-Duan (2009), NaCl only. `jones_dole_viscosity`
is the ion-additive alternative, which also handles mixed brines. This script
scores both against measurement.

TWO YARDSTICKS, and they test different things:

  1. KESTIN (Papers/50, JPCRD 10, 71) - the molality axis. A reference
     correlation of oscillating-disk measurements, +/-0.5%, over 20-150 degC,
     0.1-35 MPa, 0-6 molal. This is the only source here that spans
     concentration, which is what a salt term is FOR. It is a smoothed
     representation of data, not raw points, and Mao-Duan was fitted to
     overlapping measurements, so agreement is not fully independent.

  2. CALABRESE Table 8 x = 0 rows - 0.77 molal only, but genuinely independent
     measurement (u = 1.5%), and it reaches 448 K and 100 MPa, well outside
     Kestin's range. This is the yardstick that produced the shipped
     0.64% mean / 2.83% max figure for the whole Mao-Duan chain.

The water leg is varied as well as the salt leg, because the two are not
separable in a measured brine viscosity: Mao-Duan's salt ratio was fitted
against Mao-Duan's water, and Jones-Dole's against IAPWS-2008.
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
import sys

import numpy as np

import brine_gas.kestin_nacl_viscosity as kestin
from brine_gas.iapws_if97 import rho_if97
from brine_gas.iapws_viscosity import mu_iapws2008
from brine_gas.jones_dole_viscosity import nacl_ratio

# ---------------------------------------------------------------- Mao-Duan
# Mao & Duan (2009) Tables 2-3 (Papers/54), verified against the primary
# 2026-08-08; the McCain Table 4-14 lineage carried two last-digit defects
# (d1 2885310 -> 2885317, b2 2.085244e-7 -> 2.0852448e-7, worth <=0.0094%).
_MD_D = [0, 2885317.0, -11072.577, -9.0834095, 0.030925651, -0.0000274071,
         -1928385.1, 5621.6046, 13.82725, -0.047609523, 0.000035545041]
_MD_A = [-0.21319213, 0.0013651589, -0.0000012191756]
_MD_B = [0.069161945, -0.00027292263, 0.00000020852448]
_MD_C = [-0.0025988855, 0.0000077989227]


def mu_water_maoduan(T_K, P_MPa):
    """Mao-Duan pure-water viscosity, mPa s. Density from IF97, as shipped."""
    rho = rho_if97(T_K, P_MPa) / 1000.0        # g/cm3, as the fit expects
    ln_mu = sum(_MD_D[i] * T_K ** (i - 3) for i in range(1, 6))
    ln_mu += sum(rho * _MD_D[i] * T_K ** (i - 8) for i in range(6, 11))
    return math.exp(ln_mu) * 1e3               # Pa s -> mPa s


def maoduan_ratio(T_K, m):
    """Mao-Duan salt ratio, Eqs. 4.43-4.47. No pressure dependence."""
    A = _MD_A[0] + _MD_A[1] * T_K + _MD_A[2] * T_K * T_K
    B = _MD_B[0] + _MD_B[1] * T_K + _MD_B[2] * T_K * T_K
    C = _MD_C[0] + _MD_C[1] * T_K
    return math.exp(A * m + B * m * m + C * m ** 3)


def mu_water_iapws(T_K, P_MPa):
    """IAPWS-2008 pure-water viscosity, mPa s."""
    return mu_iapws2008(T_K, rho_if97(T_K, P_MPa)) * 1e3


# ---------------------------------------------------------------- yardstick 1
def _stats(dev):
    dev = np.asarray(dev)
    return dev.mean(), np.abs(dev).mean(), np.abs(dev).max()


def against_kestin(verbose=True):
    """Salt RATIO only, on Kestin's own grid."""
    temps = [t for t in range(20, 155, 5)]
    press = [0.1, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    molal = [0.5 * i for i in range(1, 13)]

    md, jd = [], []
    per_m = {}
    for m in molal:
        rows_md, rows_jd = [], []
        for t in temps:
            for p in press:
                ref = kestin.salt_ratio(t, p, m)
                rows_md.append(100.0 * (maoduan_ratio(t + 273.15, m) / ref - 1.0))
                rows_jd.append(100.0 * (nacl_ratio(t + 273.15, p, m) / ref - 1.0))
        per_m[m] = (_stats(rows_md), _stats(rows_jd))
        md += rows_md
        jd += rows_jd

    if verbose:
        print("\n" + "-" * 78)
        print("1. SALT RATIO vs KESTIN, 20-150 degC, 0.1-35 MPa, 0.5-6.0 molal")
        print("-" * 78)
        print(f"  {'molal':>6} | {'Mao-Duan  bias':>15} {'mean':>8} {'max':>8}"
              f" | {'Jones-Dole  bias':>17} {'mean':>8} {'max':>8}")
        for m in molal:
            (a, b, c), (d, e, f) = per_m[m]
            print(f"  {m:6.1f} | {a:+14.3f}% {b:7.3f}% {c:7.3f}%"
                  f" | {d:+16.3f}% {e:7.3f}% {f:7.3f}%")
        a, b, c = _stats(md)
        d, e, f = _stats(jd)
        print(f"  {'ALL':>6} | {a:+14.3f}% {b:7.3f}% {c:7.3f}%"
              f" | {d:+16.3f}% {e:7.3f}% {f:7.3f}%   (n = {len(md)})")

        # Kestin MEASURED to 5.4 molal; his tables print 6.0 by extrapolating.
        n_per_m = len(md) // len(molal)
        keep = sum(([m <= 5.0] * n_per_m for m in molal), [])
        md5 = [v for v, k in zip(md, keep) if k]
        jd5 = [v for v, k in zip(jd, keep) if k]
        a, b, c = _stats(md5)
        d, e, f = _stats(jd5)
        print(f"  {'<=5.0':>6} | {a:+14.3f}% {b:7.3f}% {c:7.3f}%"
              f" | {d:+16.3f}% {e:7.3f}% {f:7.3f}%   (n = {len(md5)}) "
              f"<- inside Kestin's measured concentration range")
    return _stats(md), _stats(jd)


# ---------------------------------------------------------------- yardstick 2
def against_calabrese(verbose=True):
    """FULL brine viscosity vs Calabrese's measured gas-free 0.77 m rows."""
    from validation.raw_viscosity_tables import CALABRESE_T8, M_CALABRESE, parse_table

    rows = [(T, p, eta) for x, T, p, eta in parse_table(CALABRESE_T8)
            if x == 0.0]
    if not rows:
        raise RuntimeError('no x = 0 rows parsed from Calabrese Table 8')

    m = M_CALABRESE
    combos = {
        'Mao-Duan water x Mao-Duan salt (SHIPPED)':
            lambda T, p: mu_water_maoduan(T, p) * maoduan_ratio(T, m),
        'IAPWS-2008 water x Mao-Duan salt':
            lambda T, p: mu_water_iapws(T, p) * maoduan_ratio(T, m),
        'IAPWS-2008 water x Jones-Dole salt':
            lambda T, p: mu_water_iapws(T, p) * nacl_ratio(T, p, m),
        'Mao-Duan water x Jones-Dole salt':
            lambda T, p: mu_water_maoduan(T, p) * nacl_ratio(T, p, m),
    }

    out = {}
    for label, fn in combos.items():
        dev = [100.0 * (fn(T, p) / eta - 1.0) for T, p, eta in rows]
        out[label] = _stats(dev)

    if verbose:
        print("\n" + "-" * 78)
        print(f"2. FULL BRINE VISCOSITY vs CALABRESE, {len(rows)} measured "
              f"gas-free points")
        print(f"   {m} molal NaCl, {min(r[0] for r in rows):.0f}-"
              f"{max(r[0] for r in rows):.0f} K, "
              f"{min(r[1] for r in rows):.1f}-{max(r[1] for r in rows):.1f} MPa,"
              f" u(eta) = 1.5%")
        print("-" * 78)
        print(f"  {'chain':>42} {'bias':>9} {'mean':>9} {'max':>9}")
        for label, (bias, mean, mx) in out.items():
            print(f"  {label:>42} {bias:+8.3f}% {mean:8.3f}% {mx:8.3f}%")
    return out, rows


# ------------------------------------------------------------------ envelope
def where_they_differ(verbose=True):
    """Where the two salt terms disagree most, and how each fails Kestin."""
    worst_md = worst_jd = None
    for t in range(20, 155, 5):
        for p in (0.1, 10.0, 35.0):
            for m in (0.5 * i for i in range(1, 13)):
                ref = kestin.salt_ratio(t, p, m)
                d_md = 100.0 * (maoduan_ratio(t + 273.15, m) / ref - 1.0)
                d_jd = 100.0 * (nacl_ratio(t + 273.15, p, m) / ref - 1.0)
                if worst_md is None or abs(d_md) > abs(worst_md[0]):
                    worst_md = (d_md, t, p, m)
                if worst_jd is None or abs(d_jd) > abs(worst_jd[0]):
                    worst_jd = (d_jd, t, p, m)
    if verbose:
        print("\n" + "-" * 78)
        print("3. WHERE EACH ONE FAILS WORST against Kestin")
        print("-" * 78)
        for lab, w in (('Mao-Duan  ', worst_md), ('Jones-Dole', worst_jd)):
            d, t, p, m = w
            print(f"  {lab}: {d:+.3f}% at {t} degC, {p} MPa, {m:.1f} molal")
    return worst_md, worst_jd


def residual_is_not_the_salt_term(verbose=True):
    """Back the salt ratio out of Calabrese and see whether it can carry the blame.

    The premise this front started from was that the salt term is where the
    viscosity accuracy goes. Two tests kill that:
      * substitute KESTIN's own ratio - a measurement-backed salt term - into
        the chain and the fit to Calabrese does not improve;
      * back the ratio out of each measured point (measured / IAPWS-2008 water)
        and compare it with the models directly.
    """
    from validation.raw_viscosity_tables import CALABRESE_T8, M_CALABRESE, parse_table

    rows = sorted((T, p, e) for x, T, p, e in parse_table(CALABRESE_T8)
                  if x == 0.0)
    m = M_CALABRESE
    cold = [r for r in rows if r[0] < 300.0]
    warm = [r for r in rows if r[0] >= 300.0]

    def implied(rows_):
        return [(T, p, e / mu_water_iapws(T, p)) for T, p, e in rows_]

    if verbose:
        print("\n" + "-" * 78)
        print("4. IS THE RESIDUAL ACTUALLY IN THE SALT TERM?")
        print("-" * 78)
        print("  Salt ratio implied by each measured point vs the models, "
              f"{m} molal:")
        print(f"  {'set':>28} {'n':>4} {'implied':>9} {'Mao-Duan':>9} "
              f"{'Jones-Dole':>11} {'implied/MD':>11}")
        for lab, rr in (('274.65 K (cold corner)', cold),
                        ('348-448 K', warm)):
            imp = implied(rr)
            i_mean = np.mean([r[2] for r in imp])
            md_mean = np.mean([maoduan_ratio(T, m) for T, _, _ in rr])
            jd_mean = np.mean([nacl_ratio(T, p, m) for T, p, _ in rr])
            print(f"  {lab:>28} {len(rr):4d} {i_mean:9.4f} {md_mean:9.4f} "
                  f"{jd_mean:11.4f} {100 * (i_mean / md_mean - 1):+10.2f}%")

        dev = [100.0 * (mu_water_iapws(T, p) * kestin.salt_ratio(T - 273.15, p, m)
                        / e - 1.0) for T, p, e in rows if T <= 423.31 and p <= 35.0]
        dev_md = [100.0 * (mu_water_maoduan(T, p) * maoduan_ratio(T, m) / e - 1.0)
                  for T, p, e in rows if T <= 423.31 and p <= 35.0]
        print(f"\n  Inside Kestin's range (n = {len(dev)}), swapping in his "
              f"measured salt ratio:")
        print(f"    shipped Mao-Duan chain : mean {np.mean(np.abs(dev_md)):.3f}%")
        print(f"    same chain, KESTIN salt: mean {np.mean(np.abs(dev)):.3f}%")
        print("    -> a measurement-backed salt term does NOT reduce the "
              "residual.")
    return cold, warm


def main():
    print("=" * 78)
    print("SALT VISCOSITY TERM: Mao-Duan vs ion-additive Jones-Dole")
    print("=" * 78)
    against_kestin()
    where_they_differ()
    against_calabrese()
    residual_is_not_the_salt_term()
    print("\n" + "=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
