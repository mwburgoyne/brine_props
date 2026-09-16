"""Yan 2011 (Paper 16) validation: CO2-saturated NaCl brine density and solubility.

Yan, Huang & Stenby 2011, Int. J. Greenhouse Gas Control 5:1460-1477.
New experimental data: Tables 3 (solubility) and 4 (density), 323.2/373.2/413.2 K,
5-40 MPa, 0/1/5 mol/kg NaCl. Density meter +/-2e-4 g/ml same-load repeatability,
+/-1e-3 g/ml between loads; solubility repeatable within 2%.

Density leg (Mark's implied-V_phi directive): invert Garcia Eq. 18 per point for
the V_phi that reproduces the measured density, using the MEASURED x_CO2
(salt-free basis, as our implementation expects) and the Spivey gas-free baseline.
Compare against Plyasunov V2inf(T,P) - trends in T, P, m; the m=1 and m=5 rows
test the freshwater-V_phi-in-brine assumption to 5 mol/kg.

Solvent basis (corrected 2026-07-25): V_phi is defined against the whole gas-free
brine, so the inversion adds n_CO2*V_phi to the full brine volume, not to the
water-only part of it. Working per kg of water makes this unambiguous:
    rho = (Wb + m_CO2*M2) / (Wb/rho_brine + m_CO2*V_phi),  Wb = 1000 + m*M_NaCl
The previous inversion used water mass alone, which understated the solvent mass
and so overstated the implied V_phi (by 5.6% at 1 mol/kg, 29% at 5 mol/kg).
Identical at m=0.

Solubility leg: S&W frameworks (gamma_phi) and CO2_Brine_Mixture (Spycher-Pruess)
vs measured salt-free x_CO2.
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
import numpy as np, pandas as pd

# Analysis scripts must use the SHIPPED default route, not a hard-wired one.
# Before 2026-07-25 this imported V_phi from plyasunov_model directly, so after
# the default moved to PR+VSHIFT this script kept reporting the old route.
from brine_gas.plyasunov_model import gas_mw
from brine_gas.vphi_route import V_phi
from brine_gas.brine_properties import rho_brine, salinity_from_molality, M_NACL
from brine_gas.water_properties import rho_w
from pyrestoolbox import brine as rtb_brine

MW_WATER = 18.015268

# T_K, P_MPa, m_NaCl, m_CO2, x_CO2 (salt-free), rho_sat (g/ml)
DATA = [
    (323.2,  5.0, 0, 0.830, 0.01473, 0.99722), (323.2, 10.0, 0, 1.212, 0.02137, 1.00268),
    (323.2, 15.0, 0, 1.252, 0.02206, 1.00528), (323.2, 20.0, 0, 1.298, 0.02285, 1.00688),
    (323.2, 30.0, 0, 1.418, 0.02492, 1.01293), (323.2, 40.0, 0, 1.463, 0.02569, 1.01744),
    (373.2,  5.0, 0, 0.462, 0.00826, 0.96370), (373.2, 10.0, 0, 0.803, 0.01427, 0.96741),
    (373.2, 15.0, 0, 1.014, 0.01795, 0.97062), (373.2, 20.0, 0, 1.158, 0.02044, 0.97425),
    (373.2, 30.0, 0, 1.315, 0.02314, 0.97962), (373.2, 40.0, 0, 1.443, 0.02534, 0.98506),
    (413.2,  5.0, 0, 0.378, 0.00677, 0.92928), (413.2, 10.0, 0, 0.707, 0.01258, 0.93367),
    (413.2, 15.0, 0, 0.970, 0.01717, 0.93760), (413.2, 20.0, 0, 1.153, 0.02035, 0.94108),
    (413.2, 30.0, 0, 1.403, 0.02466, 0.94700), (413.2, 40.0, 0, 1.564, 0.02740, 0.95282),
    (323.2,  5.0, 1, 0.667, 0.01188, 1.03116), (323.2, 10.0, 1, 0.961, 0.01702, 1.03491),
    (323.2, 15.0, 1, 1.019, 0.01803, 1.03968), (323.2, 20.0, 1, 1.100, 0.01943, 1.04173),
    (323.2, 30.0, 1, 1.159, 0.02046, 1.04602), (323.2, 40.0, 1, 1.227, 0.02163, 1.05024),
    (373.2,  5.0, 1, 0.428, 0.00766, 1.00026), (373.2, 10.0, 1, 0.683, 0.01216, 1.00321),
    (373.2, 15.0, 1, 0.828, 0.01470, 1.00667), (373.2, 20.0, 1, 0.967, 0.01712, 1.00961),
    (373.2, 30.0, 1, 1.073, 0.01897, 1.01448), (373.2, 40.0, 1, 1.195, 0.02107, 1.01980),
    (413.2,  5.0, 1, 0.308, 0.00552, 0.96883), (413.2, 10.0, 1, 0.575, 0.01025, 0.97169),
    (413.2, 15.0, 1, 0.832, 0.01476, 0.97483), (413.2, 20.0, 1, 0.962, 0.01704, 0.97778),
    (413.2, 30.0, 1, 1.241, 0.02187, 0.98301), (413.2, 40.0, 1, 1.284, 0.02261, 0.98817),
    (323.2,  5.0, 5, 0.336, 0.00602, 1.15824), (323.2, 10.0, 5, 0.490, 0.00876, 1.16090),
    (323.2, 15.0, 5, 0.553, 0.00987, 1.16290), (323.2, 20.0, 5, 0.559, 0.00997, 1.16468),
    (323.2, 30.0, 5, 0.604, 0.01077, 1.16810), (323.2, 40.0, 5, 0.654, 0.01165, 1.17118),
    (373.2,  5.0, 5, 0.213, 0.00383, 1.12727), (373.2, 10.0, 5, 0.364, 0.00652, 1.12902),
    (373.2, 15.0, 5, 0.481, 0.00859, 1.13066), (373.2, 20.0, 5, 0.578, 0.01031, 1.13214),
    (373.2, 30.0, 5, 0.640, 0.01140, 1.13566), (373.2, 40.0, 5, 0.636, 0.01133, 1.13893),
    (413.2,  5.0, 5, 0.190, 0.00342, 1.09559), (413.2, 10.0, 5, 0.324, 0.00580, 1.10183),
    (413.2, 15.0, 5, 0.462, 0.00825, 1.10349), (413.2, 20.0, 5, 0.593, 0.01057, 1.10499),
    (413.2, 30.0, 5, 0.625, 0.01113, 1.10882), (413.2, 40.0, 5, 0.713, 0.01268, 1.11254),
]

rows = []
for T, P, m, mco2, x, rho_meas in DATA:
    S = salinity_from_molality(m) if m > 0 else 0.0
    rho1 = (rho_brine(T, P, S) if m > 0 else rho_w(T, P)) / 1000.0  # g/cm3
    # Invert Garcia Eq. 18 for implied V_phi, per kg of water so the gas volume
    # is added to the FULL gas-free brine volume (see module docstring).
    Wb = 1000.0 + m * M_NACL          # gas-free brine mass per kg water, g
    M2 = gas_mw('CO2')
    vphi_implied = ((Wb + mco2 * M2) / rho_meas - Wb / rho1) / mco2
    vphi_model = V_phi('CO2', T, P, 'auto', m)   # includes the salt shift
    # Density error if model V_phi used with measured x (isolates V_phi + baseline)
    rho_model = (Wb + mco2 * M2) / (Wb / rho1 + mco2 * vphi_model)
    # Fractional density increase framing
    dfrac_meas = rho_meas / rho1 - 1.0
    dfrac_model = rho_model / rho1 - 1.0
    rows.append(dict(T_K=T, P_MPa=P, m=m, x_exp=x, rho_meas=rho_meas,
                     rho1_spivey=rho1, vphi_implied=vphi_implied, vphi_model=vphi_model,
                     vphi_err_pct=(vphi_model/vphi_implied-1)*100,
                     rho_err_pct=(rho_model/rho_meas-1)*100,
                     dfrac_meas_pp=dfrac_meas*100, dfrac_model_pp=dfrac_model*100))
df = pd.DataFrame(rows)

print("=" * 78)
print("DENSITY LEG - implied V_phi (cm3/mol), Garcia Eq. 18 inverted, Spivey baseline")
print("=" * 78)
for m in (0, 1, 5):
    s = df[df.m == m]
    print(f"\nm = {m} mol/kg NaCl")
    print(s.pivot_table(index='P_MPa', columns='T_K', values='vphi_implied').round(1).to_string())
print("\nModel V_phi (delivered route via vphi_route, freshwater): "
      + ", ".join(f"{T}K: {V_phi('CO2',T,20):.1f}" for T in (323.2, 373.2, 413.2)) + " (at 20 MPa)")

print("\nV_phi model vs implied, % (positive = model high):")
piv = df.pivot_table(index='m', columns='T_K', values='vphi_err_pct', aggfunc='mean').round(1)
print(piv.to_string())
print(f"Overall: mean |err| {df.vphi_err_pct.abs().mean():.1f}%  bias {df.vphi_err_pct.mean():+.1f}%  max |err| {df.vphi_err_pct.abs().max():.1f}%")

print("\nDensity error using model V_phi + measured x (V_phi + baseline error only):")
piv = df.pivot_table(index='m', columns='T_K', values='rho_err_pct', aggfunc='mean').round(3)
print(piv.to_string())
print(f"Overall: mean |err| {df.rho_err_pct.abs().mean():.3f}%  max |err| {df.rho_err_pct.abs().max():.3f}%")

print("\nFractional density increase (rho_sat/rho_gasfree - 1), percentage points:")
for m in (0, 1, 5):
    s = df[df.m == m]
    print(f"  m={m}: measured {s.dfrac_meas_pp.min():.2f} to {s.dfrac_meas_pp.max():.2f} pp; "
          f"model-measured mean {(s.dfrac_model_pp-s.dfrac_meas_pp).mean():+.3f} pp, "
          f"max |diff| {abs(s.dfrac_model_pp-s.dfrac_meas_pp).max():.3f} pp")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SOLUBILITY LEG - x_CO2 (salt-free) vs S&W frameworks and Spycher-Pruess")
print("=" * 78)
MWSAL = 58.44
def ppm_of(m): return m*MWSAL/(1000+m*MWSAL)*1e6

# dropin's brine calibration uses embedded salinity (as in the CH4/CO2 benchmarks);
# proposed uses gamma_phi. Mismatching these misstates the frameworks badly.
FW_SAL = {'proposed': 'gamma_phi', 'dropin': 'embedded'}

def sw_x(row, framework):
    try:
        mix = rtb_brine.SoreideWhitson(pres=row.P_MPa*10, temp=row.T_K-273.15,
                                       ppm=ppm_of(row.m), y_CO2=1.0, sg=44.01/28.97,
                                       metric=True, framework=framework,
                                       salinity_method=FW_SAL[framework] if row.m > 0 else 'gamma_phi')
        return mix.x.get('CO2', np.nan)
    except Exception:
        return np.nan

def sp_x(row):
    try:
        mix = rtb_brine.CO2_Brine_Mixture(pres=row.P_MPa*10, temp=row.T_K-273.15,
                                          ppm=ppm_of(row.m), metric=True)
        return mix.x[0]
    except Exception:
        return np.nan

for name, fn in (('proposed', lambda r: sw_x(r, 'proposed')),
                 ('dropin', lambda r: sw_x(r, 'dropin')),
                 ('spycher_pruess', sp_x)):
    df[name] = df.apply(fn, axis=1)

# Note: measured x is salt-free basis; SW .x and CO2_Brine_Mixture x[0] conventions
# are true mole fractions incl. salt - convert measured to the same basis per model?
# Both pyrestoolbox models report x_CO2 with salt included in the aqueous phase for
# brine. Convert Yan to salt-inclusive: x_incl = mCO2/(mCO2 + 55.5084 + 2*m) using
# full dissociation, or + m undissociated. S&W 'explicit'/gamma_phi treats salt
# outside the flash: its x is on a salt-free water basis. Spycher-Pruess x[0] is
# mole fraction with fully dissociated ions in the denominator.
def x_incl_ions(row):  # 2 ions per NaCl
    return row_m(row) / (row_m(row) + 55.5084 + 2*row.m)
def row_m(row):
    return row.x_exp*55.5084/(1-row.x_exp)  # back to molality

print(f"\n{'model':16s} {'m':>3s} {'n':>3s} {'MARE%':>7s} {'bias%':>7s} {'max%':>6s}")
for name, basis in (('proposed', 'saltfree'), ('dropin', 'saltfree'), ('spycher_pruess', 'ions')):
    for m in (0, 1, 5):
        s = df[(df.m == m) & np.isfinite(df[name])]
        if basis == 'ions' and m > 0:
            x_exp = s.apply(lambda r: (r.x_exp*55.5084/(1-r.x_exp)) /
                            ((r.x_exp*55.5084/(1-r.x_exp)) + 55.5084 + 2*r.m), axis=1)
        else:
            x_exp = s.x_exp
        re = (s[name]-x_exp)/x_exp*100
        print(f"{name:16s} {m:3d} {len(s):3d} {re.abs().mean():7.1f} {re.mean():+7.1f} {re.abs().max():6.0f}")

df.to_csv(_os.path.join(_RESULTS, 'yan2011_results.csv'), index=False)
print("\nSaved code/yan2011_results.csv")
