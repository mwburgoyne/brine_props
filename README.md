# brine_props

This package calculates how dissolved gases change brine density and viscosity, for CH4, CO2, H2S, N2 and H2. The gas correction is the product; it sits on top of whatever gas-free brine models you already use, and gas-free models (Spivey density, an IAPWS-based viscosity baseline) come bundled so a complete calculation runs out of the box.

The repository holds the code, the measured data behind every fitted constant, and the scripts that reproduce **Dissolved-Gas Corrections to Brine Density and Viscosity: A Single Method for CH4, CO2, H2S, N2 and H2** (Burgoyne and Nielsen, 2026, *Fluid Phase Equilibria*, submitted; not yet public), called the paper below. The repository is `brine_props`; the importable package is `brine_gas`.

**If you only want the numbers, these calculations already ship in [pyResToolbox](https://github.com/mwburgoyne/pyResToolbox).** `pyrestoolbox.brine.SoreideWhitson` flashes a gas mixture against brine and returns the gas-saturated density and viscosity in field or metric units. This repository is the reference implementation behind it, with the data and the fits alongside.

Only the CO2 predictions have been tested against gas-loaded brine density measurements; for CH4, H2S, N2 and H2 no such measurement exists, and those predictions rest on molar volumes measured in water plus a salinity factor fitted separately. [What the numbers rest on](#what-the-numbers-rest-on) gives it gas by gas.

Where to start:

- **Reproduce the paper**: [Installation](#installation), then [Reproducing the paper](#reproducing-the-paper).
- **See what data was used**: [The data](#the-data), or go straight to [data/PROVENANCE.md](data/PROVENANCE.md).
- **Calculate something of your own**: [First calculation](#first-calculation) and [Further examples](#further-examples).
- **Check what is and is not tested**: [What the numbers rest on](#what-the-numbers-rest-on).

## Installation

Python 3.10 or later. There is no PyPI release of `brine_gas`; install from a source checkout:

```bash
git clone https://github.com/mwburgoyne/brine_props.git
cd brine_props
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e .
```

This installs `numpy`, `scipy`, `pandas`, `matplotlib` and [pyResToolbox](https://github.com/mwburgoyne/pyResToolbox) (`pyrestoolbox>=3.7.7`, which supplies the equation-of-state parameters and the flash). Everything below has been run on Linux (Ubuntu on WSL2) under Python 3.11 and 3.12, not yet on Windows or macOS. `pyproject.toml` sets minimum versions rather than a pinned environment, so if a result differs in its last digit, check your `numpy`, `scipy` and `pyrestoolbox` versions first.

## Reproducing the paper

```bash
python reproduce.py --quick    # the validation suite only (293 checks)
python reproduce.py            # fits -> validation -> scorers -> figures -> examples
```

The full run re-fits every shipped constant from the data in `data/`, re-scores each dataset, redraws the figures and reprints the worked examples, then compares the regenerated tables against the committed ones and exits non-zero if a step fails or a number moved. It overwrites those tables and the figures as it goes, **so run it in a checkout you can throw away.** Nothing beyond the Python dependencies is needed: no PDFs, no PHREEQC.

## The data

Every measured value used anywhere in the repository is listed in [data/PROVENANCE.md](data/PROVENANCE.md): what it is, the file or module holding it, the source and its DOI, the table and page transcribed, and whether it was fitted or held out. The larger datasets are CSV or JSON in `data/`; a few short published tables live as arrays in the module that uses them, and the provenance table says which. Full references are in `data/references.bib`. The PDFs are not distributed, so module docstrings cite them as `Papers/NN`, numbered at the end of the provenance table.

Calibration and test data are kept apart on purpose. The volume shifts are fitted to gas molar volumes measured in water, while the CO2-loaded brine densities that test them (Calabrese 2019, Yan 2011) have nothing fitted to them.

## First calculation

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

## What the numbers rest on

The density change is a mass and volume balance, an identity, around one apparent molar volume per gas. That volume comes from the Peng-Robinson equation of state of the [Soreide-Whitson framework refresh](https://github.com/mwburgoyne/SW_Framework_Refresh) with one fitted volume shift per gas, and salinity enters through a single gas-generic factor. Viscosity is one measured factor per gas on a baseline of your choosing. Equations, constants and their calibration data: [docs/METHOD.md](docs/METHOD.md).

The evidence behind them is uneven, so the table says, per gas, what the constants were fitted to, what was then tested against data nothing was fitted to, and where a value rests on assumption instead.

| gas | molar volume (the density correction) | gas-loaded brine density | viscosity factor |
|---|---|---|---|
| CO2 | fitted 275-473 K to measured volumes and CO2-water densities | **tested**: Calabrese 2019 and Yan 2011, within 0.5% of density, nothing fitted to them | fitted 274-449 K to CO2-water and CO2-brine viscosities |
| CH4 | fitted 298-473 K; O'Sullivan 1970 sits +2.8% at 324.7 K | untested prediction; no measurement found | fitted 311-394 K (Ostermann 1985), extrapolated above |
| H2S | fitted 283-473 K, the Murphy-Gaines part at modelled rather than measured loading | untested prediction; no measurement found | five ratios below 309 K, held constant above |
| N2 | fitted 276-298 K only; above that the trend is the equation of state's, and solubility-derived values sit 4.5 to 12% below it | untested prediction; no measurement found | none applied: a null measured at x of order 1e-4 |
| H2 | fitted at 298 K only, three determinations spanning 23.1 to 26.7 cm3/mol; Bignell 1987 sits 5 to 16% below | untested prediction; no measurement found | none applied: no measurement of either sign |

The salinity factor on the molar volume is fitted to KCl dilatometry at 25 degC alone and its magnitude is good to about a factor of two; only the CO2 brine tests exercise it. The viscosity baseline is tested on its own (0.300% mean against Kestin's NaCl tables, 0.764% on KCl, 1.4 to 1.6% on mixed salts at atmospheric pressure), but no mixed-brine viscosity at pressure and no composition-resolved mixed-gas viscosity has been measured to test against.

The method is offered to about 450 K, 100 MPa, 5 mol/kg NaCl and a total dissolved mole fraction of about 0.05. Those maxima come from different datasets, so they are not a tested rectangle: the brine tests reach x of about 0.03, and the correction carries no composition dependence. Of the five gases only CO2 densifies brine at reservoir conditions; H2S is the closest to neutral, and in cold fresh water at atmospheric pressure its sign sits inside the data scatter. The paper's Table 5 gives each component's calibration range, test range and the consequence of its uncertainty.

## Repository guide

```
brine_gas/      the chain itself: IAPWS-IF97 water, gas-free brine density, the volume route
                with its shifts and salinity factor, the mass-volume balance, IAPWS-2008
                viscosity, the salt ratio, Kestin's pressure factor, the per-gas corrections
data/           the measured datasets, with PROVENANCE.md and references.bib
fits/           the fit behind every constant, run on the data in data/
validation/     the validation suite and the per-dataset scorers; results/ holds the tables
                reproduce.py compares
figures/        figure generators; out/ holds the rendered figures (index: docs/FIGURES.md)
examples/       the two worked examples; tables/ holds their printed tables
docs/           METHOD.md, USAGE.md, SIMULATOR.md, FIGURES.md
```

## Licence and citation

Code: MIT ([LICENSE](LICENSE)). Data tables and figures: CC-BY-4.0 ([data/LICENSE](data/LICENSE)); cite each table's source with this repository. `brine_gas/pitzer.dat` is the USGS PHREEQC database, public domain. Citation details in [CITATION.cff](CITATION.cff); the paper is cited there as submitted and a link will be added when it is public.

## Author

Mark Burgoyne. The method paper is co-authored with Markus H. Nielsen.
