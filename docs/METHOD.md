# The method

Notation: $T$ in K, $p$ in MPa, $m$ the NaCl molality (mol per kg water), $S$ the NaCl mass fraction, $x$ the dissolved-gas mole fraction on the salt-free basis (moles of gas over moles of gas plus water), $V_\phi$ in cm3/mol.

## Density: the mass and volume balance

Per kilogram of water, with $m_2$ the dissolved-gas molality, $M_2$ its molar mass, $\rho_1$ the gas-free brine density (g/cm3) and $W_b = 1000 + m\,M_{\mathrm{NaCl}}$ the gas-free brine mass,

$$\rho = \frac{W_b + m_2 M_2}{W_b/\rho_1 + m_2 V_\phi}$$

where $V_\phi$ is the apparent molar volume of the dissolved gas, defined against the whole gas-free brine. The relation is an identity, gas-generic, and needs no saturation; a mixture enters as the sum of per-gas terms. Using the water mass instead of $W_b$ inflates the whole effect by $1/(1-S)$ in the dilute limit, 29% at 5 mol/kg NaCl. The sign of $M_2 - \rho_1 V_\phi$ decides whether a gas densifies or lightens brine: of the five gases, only CO2 densifies at reservoir conditions.

## The apparent molar volume

$V_\phi$ is the infinite-dilution partial molar volume from the Peng-Robinson equation of state of the solubility framework, evaluated in closed form from two pressure derivatives at the pure-water liquid root (no flash), then corrected by one Peneloux volume translation per gas:

$$V_\phi = V_2^{\infty}(\mathrm{EOS}) - s\,b_2$$

with $b_2$ the gas co-volume and $s$ the dimensionless shift below. The shifts are fitted to direct volumetry (vibrating-tube densimetry, dilatometry, magnetic-float densimetry) and to fresh-water densities reduced to apparent molar volumes: McBride-Wright's CO2-water densities at measured loading, and Murphy and Gaines' H2S-water densities, whose dissolved amounts are modelled (a retained exception; H2S is near density-neutral, so the derived volume is weakly sensitive to them). Finite-loading apparent volumes are used to approximate the infinite-dilution volume; no composition dependence is fitted. Brine-density inversions, solubility-derived volumes and density sets whose loading had to be modelled otherwise are held out (`data/PROVENANCE.md`). Mean absolute error is against the calibration data at each point's own $(T, p)$; the five gases in the paper's scope are the first five rows.

| Gas | $s$ | calibration data | n | mean abs. error |
|---|---|---|---|---|
| CH4 | -0.111430 | 298-473 K | 12 | 1.4% |
| CO2 | -0.070103 | 275-473 K | 109 | 1.1% |
| H2S | -0.079416 | 283-473 K | 34 | 0.5% |
| N2 | -0.176768 | 276-298 K | 5 | 2.3% |
| H2 | -0.178503 | 298 K cluster | 3 | 5.1% |
| C2H6 | -0.073843 | 298 K cluster | 3 | 3.0% |
| C3H8 | -0.113326 | mean of two 298 K determinations | --- | not scored |
| nC4H10 | +0.110920 | single 298 K determination | --- | not scored |

Salinity enters $V_\phi$ once, through a gas-generic relative factor fitted to the KCl dilatometry of Tiepel and Gubbins with no parameter from any brine-density data:

$$V_\phi^{\mathrm{brine}} = V_\phi\,\bigl(1 + g(m)\bigr), \qquad g(m) = -\frac{1.7061\,m}{1 + 0.12371\,m}\ \%$$

giving -1.52% at 1 mol/kg and -5.27% at 5. The magnitude is known to about a factor of two; halving or doubling it moves CO2-saturated density by under 0.1%.

## Viscosity: per-gas factors on a supplied baseline

$$\mu = \mu_b(T,p)\prod_i f_i(T, x_i), \qquad \mu_b = \mu_w(T,p)\; r_{\mathrm{salt}}(T, m_i)\; f_p(T,p,I)$$

The baseline $\mu_b$ is the user's choice; the one supplied is IAPWS-2008 water, the ion-additive Jones-Dole salt ratio of Appelo et al. (parameters from PHREEQC's `pitzer.dat`, any combination of 13 ions), and Kestin's measured NaCl pressure factor evaluated at the brine's ionic strength (held at its 35 MPa value above). The per-gas factors take the salt-free dissolved mole fraction $x$:

| Gas | factor $f_i$ | constants | calibration |
|---|---|---|---|
| CO2 | $\exp[e_1 e^{-e_2(T/T_0-1)} x]$ | $e_1$ = 65.560, $e_2$ = 2.468, $T_0$ = 142 K | Calabrese et al. 2019 (CO2 slope from McBride-Wright et al. 2015 CO2-water measurements, 274-449 K, to 100 MPa) |
| CH4 | $1 + A e^{B/T} x/(K + x)$ | $A$ = 1.7174e-03, $B$ = 1239.8 K, $K$ = 1.5286e-03 | 23 measurements of Ostermann et al. 1985, 311-394 K, 3.4-48 MPa; RMS 0.72 percentage points |
| H2S | $1 + a x$ | $a$ = 1.70 | five ratios of Murphy and Gaines 1974, all below 309 K (the weakest calibration; 1.41-1.70 range) |
| N2 | 1 | --- | null resolved at $x \approx 10^{-4}$ |
| H2 | 1 | --- | no measurement; no correction applied |

N2 carries no correction because a low-loading measurement resolved none; H2 carries none because no measurement bounds the effect in either direction. Viscosity is not scaled from density: every measured gas thickens water or leaves it unchanged, including the gases that lighten brine. The Islam-Carlson CO2 correction, the only published correction of its kind, gives a viscosity increment 4.0 times the measured temperature-dependent one at 378 K and 8.8 times at 423 K ($x$ = 0.02).

## How well it works

| test | result |
|---|---|
| CO2-loaded NaCl brine densities at measured loading (Calabrese et al. 2019, 275-449 K, to 100 MPa, 0.77 and 2.50 mol/kg; Yan et al. 2011, 323-413 K, to 40 MPa, 0-5 mol/kg), no parameter fitted | within 0.5% of density; densification reproduced to 0.1-0.4 percentage points |
| Measured H2S temperature trend (Murphy and Gaines) | reproduced (+0.53 cm3/mol vs +0.69 measured) where published correlations give +0.02 and -0.31 |
| Gas-free viscosity baseline vs Kestin's NaCl and KCl tables | 0.300% and 0.764% mean |
| Ion mixing in the salt ratio vs measured mixed brines at atmospheric pressure (Arshad 2020 KCl-CaCl2; Hoffert 2025 NaCl-CaCl2 below 323 K, conditional on the water reference) | 1.4% and 1.6% mean |
| CaCl2 to 60 MPa (Abdulagatov and Azizov 2006) | 1.7% mean inside the calibration box; the NaCl-equivalent pressure substitution overstates by 0.2-0.5% |
| Islam-Carlson CO2 viscosity increment vs the measured temperature dependence | 4.0x at 378 K and 8.8x at 423 K ($x$ = 0.02) |

These test the gas-free baseline and the CO2 chain. What they do not test is listed in the README under validation coverage.
