# Simulator input: Ezrokhi coefficients

Most commercial reservoir simulators support the Ezrokhi form, $\log_{10}\rho = \log_{10}\rho_{\mathrm{water}} + \sum_i A_i(T) w_i$ and $\log_{10}\mu = \log_{10}\mu_{\mathrm{water}} + \sum_i B_i(T) w_i$, with $w_i$ the weight fraction of each non-water component, $A_i = a_0 + a_1 T + a_2 T^2$, $B_i = b_0 + b_1 T + b_2 T^2$ and $T$ in degC. The density coefficient follows from the mass and volume balance without fitting, $A = (M_2 - \rho_1 V_\phi)/(M_2 \ln 10)$, so the sets below are the method in simulator form: freshwater reference, quoted at 30 MPa, quadratic fits over 20-150 degC. The reference is pure water, and dissolved NaCl is itself a component, hence its row.

| Solute | $a_0$ | $a_1$ | $a_2$ | $b_0$ | $b_1$ | $b_2$ | $B$ fitted over (linear form) |
|---|---|---|---|---|---|---|---|
| CO2 | +0.10018 | -0.00024498 | -9.4035e-07 | +1.094 | -0.013563 | +4.6846e-05 | $x \le$ 0.03, within 0.4% |
| CH4 | -0.53324 | -0.00077506 | -3.3897e-06 | +7.365 | -0.072509 | +0.00023925 | $x \le$ 0.01, within 3.6% |
| H2S | -0.0046214 | -0.00025296 | -1.0727e-06 | +0.39086 | 0 | 0 | $x \le$ 0.04, exact |
| N2 | -0.093531 | -0.00041148 | -2.2187e-06 | 0 | 0 | 0 | no effect resolved at $x \approx 10^{-4}$ |
| H2 | -4.8017 | -0.0051093 | -2.5102e-05 | 0 | 0 | 0 | no measurement; no correction applied |
| NaCl | +0.29811 | -0.00031195 | +2.7559e-06 | +0.97967 | +0.0013655 | -4.8481e-06 | 0.5-5 mol/kg at 30 MPa, within 4% |

Limits:

- The form has no pressure term. `fits/ezrokhi_pressure_fit.py` gives each density coefficient as a cubic in pressure over 5-100 MPa, $a_j(p) = c_0 + c_1 p + c_2 p^2 + c_3 p^3$ with $p$ in MPa (the paper's Table A.1), and the NaCl viscosity coefficients likewise (Table A.3); both tables follow.
- A saline study should re-evaluate $A$ at its own brine: the CO2 coefficient falls 20% from fresh water to 2.5 mol/kg at 25 degC. `fits/ezrokhi_fits.py` evaluates the coefficients at any pressure and NaCl molality.
- Each $B$ set is one constant per solute, as the form requires, fitted through the origin over the loading the solute reaches; the last column states that range and what the linear form costs. The CH4 factor saturates with dissolved amount (its secant falls from 14.4 at $x = 5 \\times 10^{-4}$ to 2.5 at $10^{-2}$), so its single set reproduces the factor to within 3.6% over $x \\le 0.01$; CO2 and H2S are linear.
- N2 and H2 carry no viscosity coefficient: N2 because a low-loading measurement resolved no effect, H2 by assumption.
- NaCl needs coefficients of its own because the reference is pure water: the simulator computes brine viscosity as its water viscosity times $10^{B_{\mathrm{NaCl}} w_{\mathrm{NaCl}}}$, and the gas factors multiply on top. The salt ratio is not linear in weight fraction, which the form assumes, so the NaCl $B$ set is a least-squares fit through the origin over 0.5-5 mol/kg; it reproduces the baseline to within 4% over that range. Against Kestin's NaCl measurements (20-150 degC, 0.1-35 MPa) it scores 1.8% mean and 6.1% worst over 0.5-5 mol/kg; the tNavigator default NaCl set (0.718, 0.00359, 0) scores 3.7% mean and 20% worst over 0.5-6 mol/kg, better than ours only below 1.5 mol/kg (`python fits/ezrokhi_fits.py --check`). The 30 MPa set is quoted in the table; the baseline's pressure factor moves the salt ratio by up to 3% of viscosity between 0.1 and 35 MPa at 25 degC, so `fits/ezrokhi_pressure_fit.py` also gives the NaCl $b_j$ as cubics in pressure in the same form as the density sets (the paper's Table A.3; `examples/tables/tab_ezrokhi_nacl_b.tex` after a full reproduce), within 0.3% of viscosity at 5 mol/kg. NaCl follows both polynomial forms well: at 5 mol/kg the quadratic in temperature holds $A$ to 0.05% of density and $B$ to 0.2% of viscosity at every pressure, and the cubic in pressure adds nothing measurable to $A$; the errors that matter are the 4% linear-in-$w$ cost and the 1.8% disagreement with Kestin.

`validation/ezrokhi_vs_default.py` scores the CO2, H2S, CH4, NaCl and CaCl2 default sets printed in the tNavigator technical manual against the same measurements (the paper's Appendix A).

## Pressure dependence: the coefficients as cubics in pressure

Density coefficients, $a_j(p) = c_0 + c_1 p + c_2 p^2 + c_3 p^3$, $p$ in MPa, valid 5-100 MPa (freshwater reference, per weight fraction, $T$ in degC):

| Component | | $c_0$ | $c_1$ | $c_2$ | $c_3$ |
|---|---|---|---|---|---|
| CO$_2$ | $a_0$ | +1.0158e-01 | -4.2411e-05 | -1.5838e-07 | +6.5256e-10 |
|  | $a_1$ | -2.5144e-04 | +4.8662e-08 | +6.3096e-09 | -2.3965e-11 |
|  | $a_2$ | -1.3426e-06 | +1.6178e-08 | -1.0147e-10 | +2.9079e-13 |
| CH$_4$ | $a_0$ | -5.3023e-01 | -8.4544e-05 | -5.9685e-07 | +2.2485e-09 |
|  | $a_1$ | -7.8752e-04 | -1.6270e-07 | +2.1850e-08 | -8.1535e-11 |
|  | $a_2$ | -4.7801e-06 | +5.5792e-08 | -3.4586e-10 | +9.8886e-13 |
| H$_2$S | $a_0$ | -2.5910e-03 | -6.3357e-05 | -1.6740e-07 | +7.3601e-10 |
|  | $a_1$ | -2.6209e-04 | +1.2090e-07 | +6.9527e-09 | -2.6812e-11 |
|  | $a_2$ | -1.5159e-06 | +1.7863e-08 | -1.1319e-10 | +3.2552e-13 |
| N$_2$ | $a_0$ | -9.2131e-02 | -3.7043e-05 | -3.6316e-07 | +1.3259e-09 |
|  | $a_1$ | -4.1294e-04 | -3.0256e-07 | +1.3224e-08 | -4.8251e-11 |
|  | $a_2$ | -3.0523e-06 | +3.3337e-08 | -2.0315e-10 | +5.7803e-13 |
| H$_2$ | $a_0$ | -4.7957e+00 | -8.4726e-05 | -4.4020e-06 | +1.5160e-08 |
|  | $a_1$ | -5.1144e-03 | -3.8004e-06 | +1.4955e-07 | -5.4430e-10 |
|  | $a_2$ | -3.4613e-05 | +3.8009e-07 | -2.3076e-09 | +6.5623e-12 |
| NaCl | $a_0$ | +3.0787e-01 | -3.4663e-04 | +7.4873e-07 | -1.2477e-09 |
|  | $a_1$ | -3.8101e-04 | +2.5404e-06 | -8.5186e-09 | +1.8314e-11 |
|  | $a_2$ | +3.2686e-06 | -1.9150e-08 | +7.3772e-11 | -1.6467e-13 |

NaCl viscosity coefficients, $b_j(p)$ in the same form, fitted over 0.5-5 mol/kg (within 0.3% of viscosity at 5 mol/kg):

| | $c_0$ | $c_1$ | $c_2$ | $c_3$ |
|---|---|---|---|---|
| $b_0$ | +8.8764e-01 | +4.7821e-03 | -6.8902e-05 | +3.1709e-07 |
| $b_1$ | +2.8105e-03 | -7.5023e-05 | +1.0789e-06 | -4.9602e-09 |
| $b_2$ | -9.2664e-06 | +2.2949e-07 | -3.2999e-09 | +1.5139e-11 |
