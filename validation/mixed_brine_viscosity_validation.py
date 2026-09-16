"""Mixed-brine viscosity (and density) against MEASUREMENT: the frozen chain
scored on two composition-resolved mixed-salt datasets found by the
2026-09-05 independent review, which showed the manuscript's "no measured
mixed-brine viscosity was located" was an incomplete search, not a data gap.

Datasets (both ambient pressure, so the Kestin pressure factor is exactly 1
and only the ion-additive Jones-Dole salt ratio is tested):

  Hoffert, Bloecher, Kranz, Milsch & Sass (2025), Geothermal Energy 13:15,
    doi 10.1186/s40517-025-00339-4, Appendix 1 Table 2: 520 rows, 293-353 K,
    NaCl to 6.0 mol/kg, CaCl2 to 5.3 mol/kg, seven Na:Ca mixing series,
    rolling-ball viscometer, stated precision 0.5%; the authors report their
    NaCl data run up to 5% ABOVE Laliberte (2007). Parsed from the article
    PDF text layer and checked against the rendered page (`hoffert2025_table2.csv`).
  Arshad, Qiblawey et al. (2020), Sci. Rep. 10:16312,
    doi 10.1038/s41598-020-73484-4, Tables 3-9: 210 rows of KCl (m1) +
    CaCl2 (m2) + water, 293.15-323.15 K, Uc(eta) = 0.003 mPa s,
    Uc(rho) = 5e-5 g/cm3 (`arshad2020_tables3to9.csv`). Their own check
    against Zhang & Chen at 298 K: density AAD 0.18%, viscosity AAD 2.2%.

NOTHING IS FITTED HERE. The chain is scored as shipped; single-salt rows are
kept separate from mixed rows so a dataset-level bias (Hoffert's +5%) is not
read as a mixing error. Row exclusions are listed in EXCLUDED with the reason.

Run: python3 mixed_brine_viscosity_validation.py   (self-checking)
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
from brine_gas.viscosity_route import brine_viscosity, pressure_factor       # noqa: E402
from brine_gas.salt_route import brine_density                                # noqa: E402
from brine_gas.appelo_volumes import ionic_strength                           # noqa: E402

P_ATM = 0.1   # the Kestin factor is normalised at 0.1 MPa, so it is exactly 1 here

# Printed-value defects, excluded rather than silently corrected.
EXCLUDED = [
    ('arshad', dict(T_K=293.15, m_KCl=0.1, m_CaCl2=3.5),
     'm1 printed as 0.1 inside a block whose other rows step 0.5, 1.0; the '
     'density 1.2888 exceeds the 0.5 row, so the label is a misprint for 1.0'),
]


def _stats(dev):
    dev = np.asarray(dev, float)
    return dict(n=len(dev), mean_abs=np.mean(np.abs(dev)), max_abs=np.max(np.abs(dev)),
                bias=np.mean(dev))


def _row(label, s):
    return (f"  {label:<34} n={s['n']:>3}  mean|dev| {s['mean_abs']:5.2f}%  "
            f"max|dev| {s['max_abs']:5.2f}%  bias {s['bias']:+6.2f}%")


def score_hoffert():
    d = pd.read_csv(os.path.join(_DATA, 'hoffert2025_table2.csv'))
    assert len(d) == 520, len(d)
    assert (d.Cl - (d.Na + 2 * d.Ca)).abs().max() < 1e-3
    d['kind'] = np.where((d.Na > 0) & (d.Ca > 0), 'mixed',
                np.where(d.Na > 0, 'NaCl', np.where(d.Ca > 0, 'CaCl2', 'water')))
    d['I'] = 0.5 * (d.Na + 4 * d.Ca + d.Cl)
    d['ca_frac'] = np.where(d.Na + d.Ca > 0, d.Ca / (d.Na + d.Ca), 0.0)
    mu = []
    for _, r in d.iterrows():
        comp = {'Na+': r.Na, 'Ca+2': r.Ca, 'Cl-': r.Cl}
        comp = {k: v for k, v in comp.items() if v > 0}
        assert abs(pressure_factor(r.T_K, P_ATM, comp) - 1.0) < 1e-12
        mu.append(brine_viscosity(r.T_K, P_ATM, composition=comp or None)
                  if comp else brine_viscosity(r.T_K, P_ATM, m=0.0))
    d['mu_model'] = mu
    d['dev_pct'] = 100.0 * (d.mu_model / d.eta_meas - 1.0)
    # Hoffert's main-table (sample 0) water rows run ABOVE IAPWS with temperature,
    # 13% at 353 K, while the paper's separate Appendix 2 water check matches NIST
    # to 0.3% - a source inconsistency, not instrument drift. The salt-ratio test
    # against the same table's water assumes a shared multiplicative error; the
    # 293-323 K numbers are the ones relied on. The clean test of the SALT RATIO is
    # each row against the same instrument's water at the same temperature.
    w_meas = d[d.kind == 'water'].set_index('T_K').eta_meas
    w_model = d[d.kind == 'water'].set_index('T_K').mu_model
    d['r_meas'] = d.eta_meas / d.T_K.map(w_meas)
    d['r_model'] = d.mu_model / d.T_K.map(w_model)
    d['rdev_pct'] = 100.0 * (d.r_model / d.r_meas - 1.0)
    return d


def score_arshad():
    d = pd.read_csv(os.path.join(_DATA, 'arshad2020_tables3to9.csv'))
    assert len(d) == 210, len(d)
    for src, key, why in EXCLUDED:
        if src != 'arshad':
            continue
        mask = np.ones(len(d), bool)
        for k, v in key.items():
            mask &= np.isclose(d[k], v)
        assert mask.sum() == 1, (key, mask.sum())
        d = d[~mask].copy()
    d['I'] = d.m_KCl + 3.0 * d.m_CaCl2
    d['ca_frac'] = d.m_CaCl2 / (d.m_KCl + d.m_CaCl2)
    mu, rho = [], []
    for _, r in d.iterrows():
        salts = {'KCl': r.m_KCl, 'CaCl2': r.m_CaCl2}
        mu.append(brine_viscosity(r.T_K, P_ATM, salts=salts))
        rho.append(brine_density(r.T_K, P_ATM, salts=salts) / 1000.0)
    d['mu_model'] = mu
    d['rho_model'] = rho
    d['dev_pct'] = 100.0 * (d.mu_model / d.eta_mPas - 1.0)
    d['rho_dev_pct'] = 100.0 * (d.rho_model / d.rho_g_cm3 - 1.0)
    return d


def main():
    print('MIXED-BRINE VISCOSITY AGAINST MEASUREMENT (chain as shipped, nothing fitted)')
    print('=' * 78)

    h = score_hoffert()
    print('\nHoffert et al. 2025, NaCl-CaCl2-H2O, 293-353 K, 0.1 MPa')
    print('  ABSOLUTE viscosity (model over measured):')
    print(_row('water (8 T) vs IAPWS', _stats(h[h.kind == 'water'].dev_pct)))
    for T, s in h[h.kind == 'water'].groupby('T_K'):
        print(f'      {T:3.0f} K water: measured {s.eta_meas.iloc[0]:.4f}, IAPWS {s.mu_model.iloc[0]:.4f}, {s.dev_pct.iloc[0]:+.2f}%')
    for kind in ('NaCl', 'CaCl2', 'mixed'):
        s = h[h.kind == kind]
        print(_row(f'{kind} all', _stats(s.dev_pct)))
    print('  SALT RATIO against the same instrument\'s water at the same T, I <= 6:')
    for kind in ('NaCl', 'CaCl2', 'mixed'):
        s = h[(h.kind == kind) & (h.I <= 6.0)]
        print(_row(f'{kind} ratio', _stats(s.rdev_pct)))
    print('  mixed-row salt ratio by series (Na:Ca), I <= 6:')
    for ser, s in h[(h.kind == 'mixed') & (h.I <= 6.0)].groupby('series'):
        print(_row(f'    series {ser}, Ca/(Na+Ca) = {s.ca_frac.mean():.2f}', _stats(s.rdev_pct)))
    print('  salt-ratio bias by temperature, I <= 6, single-salt vs mixed:')
    for T, s in h[h.I <= 6.0].groupby('T_K'):
        ns = s[s.kind == 'NaCl'].rdev_pct.mean()
        cs = s[s.kind == 'CaCl2'].rdev_pct.mean()
        ms = s[s.kind == 'mixed'].rdev_pct.mean()
        print(f'    {T:3.0f} K  NaCl {ns:+5.2f}%  CaCl2 {cs:+5.2f}%  mixed {ms:+5.2f}%')

    a = score_arshad()
    print('\nArshad et al. 2020, KCl-CaCl2-H2O, 293.15-323.15 K, 0.1 MPa '
          f'({len(a)} rows; 1 excluded, see EXCLUDED)')
    print(_row('viscosity, all rows', _stats(a.dev_pct)))
    print(_row('viscosity, I <= 6', _stats(a[a.I <= 6.0].dev_pct)))
    print(_row('density, all rows', _stats(a.rho_dev_pct)))
    print('  viscosity by CaCl2 molality:')
    for m2, s in a.groupby('m_CaCl2'):
        print(_row(f'    m_CaCl2 = {m2:.1f}', _stats(s.dev_pct)))
    print('  viscosity by temperature:')
    for T, s in a.groupby('T_K'):
        print(_row(f'    {T:.2f} K', _stats(s.dev_pct)))

    out = os.path.join(_RESULTS, 'mixed_brine_viscosity_results.csv')
    pd.concat([h.assign(source='Hoffert2025'), a.assign(source='Arshad2020')],
              ignore_index=True).to_csv(out, index=False)
    print(f'\nwrote {out}')

    # ---- self-checks: numbers quoted in the deliverables are pinned here ----
    print('\nSELF-CHECKS')
    ok = True
    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}: {name} {detail}")
    check('Hoffert row count 520', len(h) == 520)
    check('Arshad row count 209 after exclusion', len(a) == 209)
    hm = h[(h.kind == 'mixed') & (h.I <= 6.0)]
    hs = h[(h.kind != 'mixed') & (h.kind != 'water') & (h.I <= 6.0)]
    check('Hoffert main-table water rows vs IAPWS: within 1% at 293 K, beyond 5% at 353 K (source inconsistency, see docstring)',
          abs(h[(h.kind == 'water') & (h.T_K == 293)].dev_pct.iloc[0]) < 1.0
          and abs(h[(h.kind == 'water') & (h.T_K == 353)].dev_pct.iloc[0]) > 5.0)
    check('Hoffert mixed salt ratio: mean|dev| < 3%, max < 11% (single-salt rows 1.6-1.8%)',
          hm.rdev_pct.abs().mean() < 3.0 and hm.rdev_pct.abs().max() < 11.0,
          f"mixed {hm.rdev_pct.abs().mean():.2f}% / {hm.rdev_pct.abs().max():.2f}% "
          f"vs single {hs.rdev_pct.abs().mean():.2f}%")
    hm3 = hm[hm.T_K <= 323]; hs3 = hs[hs.T_K <= 323]
    check('Hoffert, 293-323 K only: mixed and single-salt salt-ratio errors within 0.5 pp',
          abs(hm3.rdev_pct.abs().mean() - hs3.rdev_pct.abs().mean()) < 0.5,
          f"mixed {hm3.rdev_pct.abs().mean():.2f}% (max {hm3.rdev_pct.abs().max():.2f}%) "
          f"vs single {hs3.rdev_pct.abs().mean():.2f}% (max {hs3.rdev_pct.abs().max():.2f}%)")
    check('Arshad viscosity mean|dev| < 1.6%, max < 6%',
          a.dev_pct.abs().mean() < 1.6 and a.dev_pct.abs().max() < 6.0,
          f"mean {a.dev_pct.abs().mean():.2f}% max {a.dev_pct.abs().max():.2f}%")
    check('Arshad density mean|dev| < 0.2%',
          a.rho_dev_pct.abs().mean() < 0.2, f"mean {a.rho_dev_pct.abs().mean():.3f}%")
    return ok, h, a


if __name__ == '__main__':
    ok, _, _ = main()
    sys.exit(0 if ok else 1)
