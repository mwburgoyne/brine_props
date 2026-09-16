"""CaCl2 viscosity AT PRESSURE against measurement: Abdulagatov & Azizov (2006),
Fluid Phase Equilibria 240:204-219, doi 10.1016/j.fluid.2005.12.036, Table 3
(Paper 60): six molalities 0.10-2.00 mol/kg, 293-575 K, 0.1/10/30/60 MPa,
capillary flow, total viscosity uncertainty < 1.6%; their pure-water check
against IAPWS: AAD 0.5%, max 1.1%. Parsed from the text layer and checked
against the rendered page 7 (`abdulagatov2006_table3.csv`, 309 rows).

This is the test the 2026-09-05 review asked for (item 3.2): the chain's
pressure factor is Kestin's NaCl measurement evaluated at the brine's ionic
strength as an NaCl-equivalent molality, and NO measurement had tested that
substitution for a divalent salt. Two scorings, nothing fitted:

  A. absolute viscosity, chain as shipped (IAPWS water x Jones-Dole CaCl2
     ratio x Kestin factor at I = 3m), split by whether the state is inside
     Kestin's box (293-423 K, 0.1-35 MPa, I <= 6) or beyond it (60 MPa is
     clamped to the 35 MPa value; T > 423 K extrapolates the box and the
     Jones-Dole parameters).
  B. the SALT-RATIO PRESSURE FACTOR isolated: measured
     F_meas = [eta(P)/eta_w(P)] / [eta(P0)/eta_w(P0)] with P0 = 10 MPa (the
     lowest pressure tabulated at every temperature) against the chain's
     F_chain = f_p(P)/f_p(P0) at I = 3m, and against F = 1 (no salt pressure
     term). Water from IAPWS-2008 (the authors' own water agrees to 0.5%).

Appelo et al. (2014), the source of the Ca2+ Jones-Dole parameters, does not
cite this paper, so the comparison is independent of the ion-parameter
calibration as far as the citation trail shows.
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
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from brine_gas.viscosity_route import brine_viscosity, mu_water, pressure_factor   # noqa: E402
from brine_gas.salt_route import ions_from_salts                                   # noqa: E402

T_BOX = (293.15, 423.15)
P_BOX = 35.0
P0 = 10.0


def _stats(dev):
    dev = np.asarray(dev, float)
    return dict(n=len(dev), mean_abs=float(np.mean(np.abs(dev))),
                max_abs=float(np.max(np.abs(dev))), bias=float(np.mean(dev)))


def _row(label, s):
    return (f"  {label:<40} n={s['n']:>3}  mean|dev| {s['mean_abs']:5.2f}%  "
            f"max|dev| {s['max_abs']:5.2f}%  bias {s['bias']:+6.2f}%")


def load():
    d = pd.read_csv(os.path.join(_DATA, 'abdulagatov2006_table3.csv'))
    assert len(d) == 309, len(d)
    d['I'] = 3.0 * d.m
    d['in_box'] = (d.T_K <= T_BOX[1]) & (d.P_MPa <= P_BOX)
    d['mu_model'] = [brine_viscosity(r.T_K, r.P_MPa, salts={'CaCl2': r.m}) for r in d.itertuples()]
    d['mu_w'] = [mu_water(r.T_K, r.P_MPa) for r in d.itertuples()]
    d['dev_pct'] = 100.0 * (d.mu_model / d.eta - 1.0)
    # B. salt-ratio pressure factor, measured vs chain, referenced to P0
    ref = d[d.P_MPa == P0].set_index(['m', 'T_K'])
    rows = []
    for r in d.itertuples():
        if r.P_MPa == P0 or (r.m, r.T_K) not in ref.index:
            continue
        r0 = ref.loc[(r.m, r.T_K)]
        F_meas = (r.eta / r.mu_w) / (r0.eta / r0.mu_w)
        comp = ions_from_salts({'CaCl2': r.m})
        F_chain = pressure_factor(r.T_K, r.P_MPa, comp) / pressure_factor(r.T_K, P0, comp)
        rows.append(dict(m=r.m, T_K=r.T_K, P_MPa=r.P_MPa, I=r.I, in_box=r.in_box,
                         F_meas=F_meas, F_chain=F_chain,
                         dev_chain_pct=100.0 * (F_chain / F_meas - 1.0),
                         dev_none_pct=100.0 * (1.0 / F_meas - 1.0)))
    return d, pd.DataFrame(rows)


def main():
    d, f = load()
    print('CaCl2 VISCOSITY AT PRESSURE (Abdulagatov & Azizov 2006), chain as shipped')
    print('=' * 78)
    print('A. absolute viscosity')
    print(_row('inside Kestin box (T<=423 K, P<=30 MPa)', _stats(d[d.in_box].dev_pct)))
    print(_row('  ... at 0.1 MPa', _stats(d[d.in_box & (d.P_MPa == 0.1)].dev_pct)))
    print(_row('  ... at 30 MPa', _stats(d[d.in_box & (d.P_MPa == 30.0)].dev_pct)))
    print(_row('60 MPa, T<=423 K (factor clamped at 35)', _stats(d[(d.P_MPa == 60.0) & (d.T_K <= T_BOX[1])].dev_pct)))
    print(_row('T > 423 K, all P (extrapolated)', _stats(d[d.T_K > T_BOX[1]].dev_pct)))
    print('  inside the box, by molality:')
    for m, s in d[d.in_box].groupby('m'):
        print(_row(f'    m = {m:.2f} (I = {3*m:.2f})', _stats(s.dev_pct)))
    print('\nB. salt-ratio PRESSURE FACTOR relative to 10 MPa, T<=423 K')
    fb = f[f.T_K <= T_BOX[1]]
    for P in (0.1, 30.0, 60.0):
        s = fb[fb.P_MPa == P]
        print(f'  P = {P:5.1f} MPa: measured factor mean {s.F_meas.mean():.4f} '
              f'(range {s.F_meas.min():.4f}-{s.F_meas.max():.4f}); chain {s.F_chain.mean():.4f}')
        print(_row(f'    chain (Kestin at I = 3m) vs measured', _stats(s.dev_chain_pct)))
        print(_row(f'    no salt pressure term vs measured', _stats(s.dev_none_pct)))
    print('  30 MPa by molality (chain vs measured / none vs measured):')
    for m, s in fb[fb.P_MPa == 30.0].groupby('m'):
        print(f'    m = {m:.2f}: measured {s.F_meas.mean():.4f}  chain {s.F_chain.mean():.4f}  '
              f'dev chain {s.dev_chain_pct.mean():+.2f}%  none {s.dev_none_pct.mean():+.2f}%')
    out = os.path.join(_RESULTS, 'cacl2_pressure_viscosity_results.csv')
    d.to_csv(out, index=False)
    print(f'\nwrote {out}')
    print('\nSELF-CHECKS')
    ok = True
    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}: {name} {detail}")
    check('309 rows, six molalities', len(d) == 309 and d.m.nunique() == 6)
    check('every 10 MPa reference exists for the factor test', len(f) > 0)
    return ok, d, f


if __name__ == '__main__':
    ok, _, _ = main()
    sys.exit(0 if ok else 1)
