# brine_props

This package calculates how dissolved gases change brine density and viscosity, for CH4, CO2, H2S, N2 and H2. Its central purpose is the gas-induced correction, which applies on top of whatever gas-free brine density and viscosity models you already use; gas-free models (Spivey density, an IAPWS-based viscosity baseline) and an equilibrium-solubility workflow are also provided for a complete calculation. The repository holds the implementation, the measured data behind every fitted constant, and the scripts that reproduce the results of **Dissolved-Gas Corrections to Brine Density and Viscosity: A Single Method for CH4, CO2, H2S, N2 and H2** (Burgoyne, 2026, *Fluid Phase Equilibria*, submitted; not yet public), referred to below as the paper.

What it establishes, in one paragraph: the density change is a mass and volume balance (an identity) around one apparent molar volume per gas, which comes from the Peng-Robinson equation of state of the [Soreide-Whitson framework refresh](https://github.com/mwburgoyne/SW_Framework_Refresh) with one fitted volume shift per gas. CO2-loaded NaCl brine densities at measured dissolved amounts are reproduced within 0.5% of density with no parameter fitted to them. For the other four gases no gas-loaded brine density measurement exists; their predictions rest on molar volumes measured in water and a salinity factor fitted separately, and are combined here without a direct test. Viscosity uses one measured multiplicative factor per gas for CO2, CH4 and H2S; N2 and H2 are assigned no change. See [Validation coverage and limitations](#validation-coverage-and-limitations) before relying on a number.

The repository is `brine_props`; the importable package is `brine_gas`.

## Installation

Python 3.10 or later. There is no PyPI release of `brine_gas`; install from a source checkout:

```bash
git clone https://github.com/mwburgoyne/brine_props.git
cd brine_props
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e .
```

This installs `numpy`, `scipy`, `pandas`, `matplotlib` and [pyResToolbox](https://github.com/mwburgoyne/pyResToolbox) (`pyrestoolbox>=3.7.7`, which supplies the equation-of-state parameters and the flash). Tested environments, in which `python reproduce.py` passed and the examples below printed the numbers shown: Python 3.12.3 with numpy 2.5.2, scipy 1.18.1 and pyrestoolbox 3.7.7 from PyPI in a fresh virtual environment (Ubuntu on WSL2, 6 September 2026); an earlier build of this repository also reproduced under Python 3.11.16 with numpy 2.4.6, scipy 1.17.1 and pyrestoolbox 3.7.7. Not yet run on Windows or macOS. Minimum version bounds in `pyproject.toml` are not a pinned environment; if a result differs at the last printed digit, check those versions first.

## First calculation: known dissolved amounts

Temperature in K, pressure in MPa, NaCl as mass fraction `S` (use `salinity_from_molality` for mol/kg), dissolved gas as its salt-free mole fraction `x` (moles of gas over moles of gas plus water; the salt is not counted). Density returns kg/m3, viscosity mPa s.

```python
from brine_gas.brine_properties import rho_brine, salinity_from_molality
from brine_gas.garcia_mixing import density_single_gas, gas_saturated_viscosity
from brine_gas.viscosity_route import brine_viscosity

T, P = 350, 20                 # K, MPa
S = salinity_from_molality(1)      # 1 mol/kg NaCl -> mass fraction 0.0552
x = 0.01                            # dissolved CO2, salt-free mole fraction

rho1 = rho_brine(T, P, S)                       # gas-free density, Spivey
rho = density_single_gas('CO2', x, T, P, S=S)   # with dissolved CO2
mu1 = brine_viscosity(T, P, S=S)                # gas-free viscosity baseline
mu = gas_saturated_viscosity(T, P, gas_dict={'CO2': x}, S=S)
print(f'density  {rho1:.2f} -> {rho:.2f} kg/m3  ({100 * (rho / rho1 - 1):+.3f} %)')
print(f'viscosity {mu1:.4f} -> {mu:.4f} mPa s  (factor {mu / mu1:.4f})')
```

Expected output:

```
density  1019.43 -> 1023.04 kg/m3  (+0.354 %)
viscosity 0.4200 -> 0.4275 mPa s  (factor 1.0178)
```

`density_mixed_gas({'CH4': x1, 'CO2': x2}, T, P, S=S)` and `gas_saturated_viscosity(T, P, gas_dict={...}, S=S)` take several gases at once; each `x` is that gas's own salt-free mole fraction and the density terms add.

## Further examples

**Your own gas-free models.** The corrections are the product; the bundled gas-free models are a convenience. Pass your gas-free density as `rho1` (kg/m3); `S` is still needed because the balance uses the brine mass per kilogram of water. For viscosity, take the gas multiplier and apply it to your own baseline (the low-level correction functions take temperature in degF):

```python
from brine_gas.garcia_mixing import density_single_gas, viscosity_correction_mixed

rho = density_single_gas('CO2', x, T, P, rho1=my_rho1, S=S)
factor = viscosity_correction_mixed({'CO2': x}, degf=(T - 273.15) * 9 / 5 + 32)
mu = my_mu1 * factor
```

With `my_rho1 = 1010.0` and `my_mu1 = 0.40` at the state above, `rho` is 1013.76 and `factor` is 1.0178. `gas_saturated_viscosity` does not accept a custom baseline; this is the route for one.

**Dissolved amounts from an equilibrium flash.** Any solubility model can supply `x`. The paper's Worked Example 2 uses the Soreide-Whitson flash from pyResToolbox, whose inputs are in degC, bar and ppm NaCl:

```python
from pyrestoolbox import brine as rtb
from brine_gas.brine_properties import rho_brine
from brine_gas.garcia_mixing import density_mixed_gas, gas_saturated_viscosity
from brine_gas.viscosity_route import brine_viscosity

T, P, S = 352.59, 20.684, 0.05            # 175 degF, 3000 psia, 5 wt% NaCl
sw = rtb.SoreideWhitson(pres=P * 10, temp=T - 273.15, ppm=S * 1e6, metric=True,
                        framework='default', y_CO2=0.5, sg=0.554)   # equimolar CH4/CO2 free gas
x = {g: float(sw.x[g]) for g in ('CH4', 'CO2')}   # {'CH4': 0.000992, 'CO2': 0.008364}
rho1 = rho_brine(T, P, S)                          # 1014.61
rho = density_mixed_gas(x, T, P, rho1=rho1, S=S)   # 1016.38 (+0.175 %)
mu1 = brine_viscosity(T, P, S=S)                   # 0.4026
mu = gas_saturated_viscosity(T, P, gas_dict=x, S=S) # 0.4176 (factor 1.0373)
```

`examples/worked_examples.py` prints every intermediate of both worked examples in the paper's field units.

**Mixed salts.** The viscosity interface also takes `m=` (NaCl molality), `salts={'NaCl': 1.0, 'CaCl2': 0.5}` or `composition={ion: molality}`; a salinity with no species named is NaCl. The density functions take `S` only (NaCl mass fraction); for a mixed brine supply your own `rho1`. Argument definitions and a units table per interface: [docs/USAGE.md](docs/USAGE.md).

**Simulator input.** Ezrokhi density and viscosity coefficients for the five gases and NaCl, with their limits: [docs/SIMULATOR.md](docs/SIMULATOR.md).

**The method.** Equations, the fitted volume shifts with their calibration data, the salinity factor and the per-gas viscosity factors: [docs/METHOD.md](docs/METHOD.md).

## Validation coverage and limitations

The evidence is uneven across gases and properties. Calibration means the constant was fitted to that data; independent means measured data the constant was never fitted to; assumption means no measurement constrains it.

| gas | molar volume (density correction) | gas-loaded brine density | viscosity factor |
|---|---|---|---|
| CO2 | calibrated 275-473 K (vibrating-tube volumes and CO2-water densities at measured loading); independent: Enns 1965, Hebach 2004 (model loading) | **independent**: Calabrese 2019 (275-449 K, to 100 MPa, 0.77 and 2.50 mol/kg) and Yan 2011 (323-413 K, to 40 MPa, 0-5 mol/kg), within 0.5% of density | calibrated 274-449 K to CO2-water and CO2-brine viscosities |
| CH4 | calibrated 298-473 K; independent: O'Sullivan 1970 at 324.7 K (+2.8%) | none found; prediction only | calibrated 311-394 K (Ostermann 1985); extrapolated above |
| H2S | calibrated 283-473 K; the Murphy-Gaines density reduction uses modelled dissolved amounts (H2S is near density-neutral, so the derived volume is weakly sensitive to them) | none found; prediction only | five ratios below 309 K; a constant, extrapolated above |
| N2 | calibrated 276-298 K only; the trend above 298 K is the equation of state's; independent solubility-derived values sit 4.5 to 12% below the model | none found; prediction only | unity: a null measurement at x of order 1e-4, resolution about 2.5% |
| H2 | calibrated at 298 K only (three determinations spanning 23.1 to 26.7 cm3/mol); independent: Bignell 1987 densities 5 to 16% below | none found; prediction only | unity: no measurement of either sign, so no correction is applied |

The salinity factor on the molar volume is fitted to KCl dilatometry at 25 degC only and its magnitude is known to about a factor of two; the CO2 brine tests above exercise it, the other gases' brine predictions do not. The bundled viscosity baseline is validated separately from the gas factors: 0.300% (NaCl) and 0.764% (KCl) mean against Kestin's tables to 35 MPa; mixed salts 1.4% (KCl-CaCl2, Arshad 2020) and 1.6% (NaCl-CaCl2 below 323 K, Hoffert 2025) mean at atmospheric pressure, the latter conditional on which of that source's two pure-water series is the reference. Above 35 MPa the salt-ratio pressure factor is held at its 35 MPa value; on CaCl2 to 60 MPa it overstates the measured effect by 0.2 to 0.5%. No composition-resolved mixed-gas viscosity test exists.

The method is offered to about 450 K, 100 MPa, 5 mol/kg NaCl and a total dissolved mole fraction of about 0.05. Those maxima come from different datasets and are not a jointly tested rectangle; the CO2 brine tests reach x of about 0.03 and the model carries no composition dependence. Of the five gases only CO2 densifies brine at reservoir conditions; H2S is the closest to density-neutral, and in cold fresh water at atmospheric pressure its sign is within the data scatter. The paper's Table 5 lists each component's calibration range, test range and the consequence of its uncertainty.

## Reproducing the results

```bash
python reproduce.py --quick    # the validation suite only (293 checks)
python reproduce.py            # fits -> validation -> scorers -> figures -> examples
```

`reproduce.py` re-runs, in order, the validation suite, the dataset scorers behind the paper's tables, every fit that produced a shipped constant (each asserts the packaged value is its own output), the Ezrokhi tables, the worked examples and the figures. It then compares the regenerated result tables in `validation/results/` and `examples/tables/` field by field (numeric fields to a relative 1e-8) with the copies that were present before the run, reports any file that changed, appeared or disappeared, and exits non-zero on a failed step or a changed result. In a clean checkout those pre-run copies are the committed ones. **The full run rewrites those files and every figure in `figures/out/`, so run it in a disposable checkout.** Figures are regenerated but not compared (matplotlib output is not byte-stable across versions); printed outputs other than the compared tables are not checked. No PDF or PHREEQC installation is needed.

## Repository guide

```
brine_gas/      the delivered chain: IAPWS-IF97 water, Spivey and Pitzer gas-free density, the
                PR volume route with its shifts and salinity factor, the mass-volume balance,
                IAPWS-2008 viscosity, the ion-additive salt ratio (pitzer.dat vendored),
                Kestin's pressure factor, the per-gas viscosity corrections
data/           measured datasets as CSV or JSON, with PROVENANCE.md (source, DOI, table,
                page, extraction method, exclusions) and references.bib; smaller published
                tables live as arrays in the module that uses them, and PROVENANCE.md says which
fits/           the fit behind every constant, run on the data in data/
validation/     the validation suite (validation.py) and the dataset scorers; results/ holds
                the result tables reproduce.py compares
figures/        figure generators; out/ holds the rendered figures (index: docs/FIGURES.md)
examples/       the two worked examples; tables/ holds their printed tables
docs/           METHOD.md, USAGE.md, SIMULATOR.md, FIGURES.md
```

Module docstrings cite sources as `Papers/NN`; the numbering is listed at the end of [data/PROVENANCE.md](data/PROVENANCE.md) (the PDFs themselves are not distributed).

## Licence and citation

Code: MIT ([LICENSE](LICENSE)). Data tables and figures: CC-BY-4.0 ([data/LICENSE](data/LICENSE)); cite each table's source with this repository. `brine_gas/pitzer.dat` is the USGS PHREEQC database, public domain. Citation details in [CITATION.cff](CITATION.cff); the paper is cited there as submitted and a link will be added when it is public.

## Author

Mark Burgoyne.
