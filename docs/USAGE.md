# Usage and units

## Interfaces

| function | T | p | salinity | dissolved gas | returns |
|---|---|---|---|---|---|
| `brine_properties.rho_brine(T, P, S)` | K | MPa | `S` NaCl mass fraction | none | gas-free density, kg/m3 (Spivey) |
| `garcia_mixing.density_single_gas(gas, x2, T, P, rho1=None, S=0.0)` | K | MPa | `S` only | `x2` salt-free mole fraction | density, kg/m3 |
| `garcia_mixing.density_mixed_gas({gas: x}, T, P, rho1=None, S=0.0)` | K | MPa | `S` only | per-gas `x` | density, kg/m3 |
| `viscosity_route.brine_viscosity(T, P, S= / m= / salts= / composition=)` | K | MPa | any of the four forms | none | gas-free viscosity, mPa s |
| `garcia_mixing.gas_saturated_viscosity(T, P, gas_dict=, S= / m= / salts= / composition=)` | K | MPa | any of the four forms | `{gas: x}` | viscosity, mPa s, on the bundled baseline |
| `garcia_mixing.viscosity_correction_single(gas, x2, degf=)` | **degF** | none | none | `x2` | multiplier for one gas |
| `garcia_mixing.viscosity_correction_mixed({gas: x}, degf=)` | **degF** | none | none | per-gas `x` | product of the multipliers |
| `vphi_route.V_phi(gas, T, P, m_nacl=0.0)` | K | MPa | `m_nacl` molality | none | apparent molar volume, cm3/mol |
| `brine_properties.salinity_from_molality(m)` / `molality_from_salinity(S)` | | | conversion | | |
| `pyrestoolbox.brine.SoreideWhitson(pres=, temp=, ppm=, metric=True, framework='default', ...)` | **degC** | **bar** | ppm NaCl | free-gas composition (`y_CO2=`, `sg=`, ...) | flash; `.x[gas]` are the aqueous mole fractions |

Gas names: `'CH4'`, `'CO2'`, `'H2S'`, `'N2'`, `'H2'` (also `'C2H6'`, `'C3H8'`, `'NC4H10'` for the volume route only).

## Definitions

- **Dissolved-gas mole fraction** `x`: moles of that gas over moles of all dissolved gas plus water. The salt is not counted. For a mixture each gas has its own `x` on this common basis, and their sum is the total dissolved mole fraction. The Soreide-Whitson flash returns `x` on this basis.
- **Salinity for density**, `S`: the NaCl mass fraction of the gas-free brine (0.05 for 5 wt%). The density balance needs it even when you pass your own `rho1`, because it sets the brine mass per kilogram of water. The density functions take NaCl only; for a mixed brine supply `rho1` from your own model and give `S` as the NaCl-equivalent mass fraction, or the salt-free value if the brine is dilute.
- **Salinity for viscosity**: `S=` (NaCl mass fraction), `m=` (NaCl molality), `salts={'NaCl': 1.0, 'CaCl2': 0.5}` (molalities) or `composition={'Na': 1.0, 'Cl': 2.0, 'Ca': 0.5}` (ion molalities). A salinity with no species named is NaCl.
- **`rho1`**: gas-free brine density in kg/m3 at the same `T`, `p` and salinity. Omit it to use Spivey's NaCl correlation.

## The correction on your own baseline

```python
from brine_gas.garcia_mixing import density_single_gas, viscosity_correction_mixed
rho = density_single_gas('CO2', x, T, P, rho1=my_rho1, S=S)          # kg/m3
mu = my_mu1 * viscosity_correction_mixed({'CO2': x}, degf=(T - 273.15) * 9 / 5 + 32)
```

`gas_saturated_viscosity` always uses the bundled baseline; the multiplier route above is how a custom baseline is corrected.

## Ranges

The molar-volume route runs to 623 K (the IAPWS-IF97 Region 1 limit) and the code raises outside it; accuracy is claimed only to about 450 K, 100 MPa, 5 mol/kg NaCl and total dissolved mole fraction about 0.05. The README's coverage table says which of those limits each gas has evidence for.
