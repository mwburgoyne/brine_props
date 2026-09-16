# Data provenance

Every measured value the fits, validation and figures consume is either a CSV or JSON file in this directory or a table in the source of the module named below. Each entry gives the source, its DOI, the table and page transcribed, how it was extracted and checked, and any rows excluded. `references.bib` holds the full references. Numerical tables of published measurements are reproduced for reproducibility; cite the source with this repository.

| what | where in this repository | source | DOI | notes |
|---|---|---|---|---|
| Apparent molar volumes of CH4, CO2, H2S in water, 298-573 K, to 30 MPa (vibrating tube) | `fits/fit_pr_vshift.py` (`HNEDKOVSKY`) | Hnedkovsky, Wood and Majer 1996, J. Chem. Thermodyn. 28:125 | 10.1006/jcht.1996.0011 | calibration set for the volume shifts |
| Partial molar volumes of gases in water at 298 K (densimetry) | `fits/fit_pr_vshift.py` (`AT_298`) | Moore, Battino, Rettich, Handa and Wilhelm 1982, J. Chem. Eng. Data 27:22 | 10.1021/je00027a005 | CH4, N2, H2, C2H6, C3H8, nC4H10 |
| Partial molar volumes at 298 K (dilatometry) | `fits/fit_pr_vshift.py` (`AT_298`) | Zhou and Battino 2001, J. Chem. Eng. Data 46:331 | 10.1021/je000215o | N2, H2, C2H6, C3H8 |
| Apparent molar volumes of Ar, CH4, C2H6, H2, O2 in water and electrolytes, 25 degC (dilatometry) | `fits/salt_effect_vphi.py` (`TIEPEL`) | Tiepel and Gubbins 1972, J. Phys. Chem. 76:3044 | 10.1021/j100665a024 | Table I; KCl concentrations are molar and are converted to molality in `fits/relative_salt_shift.py` |
| N2 partial molar volume 3-21 degC (magnetic float) | `fits/fit_pr_vshift.py` (`bignell_n2`) | Bignell 1984, J. Phys. Chem. 88:5409 | 10.1021/j150666a060 | three points sampled from his correlation inside his window |
| H2S infinite-dilution volumes at 283, 298 and 313 K (vibrating tube) | `fits/fit_pr_vshift.py` (`BARBERO_H2S`) | Barbero, McCurdy and Tremaine 1982, Can. J. Chem. 60:1872, Table 1 | 10.1139/v82-260 | calibration set; raw densities are in a CISTI depository, not printed |
| H2S in water: densities, viscosities and solubility-derived volumes, 294-314 K | `fits/murphy_gaines_h2s_refit.py` (Tables I, IV) | Murphy and Gaines 1974, J. Chem. Eng. Data 19:359 | 10.1021/je60063a015 | Table I re-reduced to V_phi with dissolved amounts from the Soreide-Whitson flash (loading not measured; a retained calibration exception, H2S being near density-neutral); Table IV viscosity ratios |
| H2S solubility used for the Murphy-Gaines viscosity mole fractions | `fits/murphy_gaines_h2s_refit.py` | Burgess and Germann 1969, AIChE J. 15:272 | 10.1002/aic.690150226 | |
| Solubility-derived volumes of N2 and CH4 in water and NaCl, 324.7 K | `fits/fit_pr_vshift.py` (`HELD_OUT`), `validation/validation.py` | O'Sullivan and Smith 1970, J. Phys. Chem. 74:1460 | 10.1021/j100702a012 | held out, never fitted |
| Solubility-derived volumes of N2, CO2 and O2 (sea water) | `fits/fit_pr_vshift.py` (`HELD_OUT`), `fits/salt_effect_vphi.py` | Enns, Scholander and Bradstreet 1965, J. Phys. Chem. 69:389 | see references.bib | held out |
| CO2 finite-loading volumes at 273 K | `fits/fit_pr_vshift.py` (`HELD_OUT`) | Lauder 1959 | see references.bib | shown, not scored |
| N2 volume at pressure, 298 K (Van Slyke) | `fits/fit_pr_vshift.py` (`HELD_OUT`) | Kennan and Pollack 1990, J. Chem. Phys. 93:2724 | 10.1063/1.458911 | held out; the Comment by Alvarez and Fernandez-Prini (10.1063/1.461935) gives the KK refit |
| CO2-loaded NaCl brine densities, 275-449 K, to 100 MPa, 0.77 and 2.50 mol/kg | `data/calabrese2019_table6.csv` | Calabrese, McBride-Wright, Maitland and Trusler 2019, J. Chem. Eng. Data 64:3831, Table 6 | 10.1021/acs.jced.9b00248 | 303 rows, from the authors' spreadsheet; `calabrese_frac_validation.csv` is the archived first reduction used as a self-check |
| CO2-loaded NaCl brine viscosities, 0.77 mol/kg | `data/calabrese2019_table8_viscosity.csv` | same, Table 8 | 10.1021/acs.jced.9b00248 | coordinate-parsed from the PDF and checked against a 400 dpi render (`validation/raw_viscosity_tables.py`) |
| CO2-water densities at three measured loadings, 274.7-449 K, 15-100 MPa | `data/mcbridewright2015_table5_density.csv` | McBride-Wright, Maitland and Trusler 2015, J. Chem. Eng. Data 60:171, Table 5 | 10.1021/je5009125 | coordinate-parsed and page-checked; reduced to CO2 volumes in `validation/mcbridewright2015_density_check.py`; in the CO2 calibration set (n = 109) since the point-weighted refit |
| CO2-saturated water densities, 283-333 K, 1-31 MPa (203 rows, 39 on the coexistence line) | `data/hebach2004_table1_density.csv` | Hebach, Oberhof and Dahmen 2004, J. Chem. Eng. Data 49:950, Table 1 | 10.1021/je034260i | coordinate-parsed and page-checked row by row; loading from the flash, so held out; reduced in `validation/hebach2004_density_check.py` |
| CO2-water viscosities, 294-449 K, 15-96 MPa | `data/mcbridewright2015_table7_viscosity.csv` | McBride-Wright, Maitland and Trusler 2015, J. Chem. Eng. Data 60:171, Table 7 | 10.1021/je5009125 | as above |
| CO2-saturated NaCl brine density and solubility, 323-413 K, 5-40 MPa, 0-5 mol/kg | `validation/yan2011_validation.py` (Tables 3, 4) | Yan, Huang and Stenby 2011, Int. J. Greenh. Gas Control 5:1460 | 10.1016/j.ijggc.2011.08.004 | 54 densities |
| CH4-saturated water viscosity ratios, 311-394 K, 3.4-48 MPa | `fits/ostermann_ch4_refit.py` (`OSTERMANN`) | Ostermann, Bloori and Dehghani 1985, SPE 14211 | 10.2118/14211-MS | 23 points with solution gas-water ratio |
| C2H6 null and CH4 scatter | `fits/ostermann_ch4_refit.py` | Ostermann, Paranjpe, Godbole and Kamath 1986, SPE 15081 | 10.2118/15081-MS | |
| NaCl brine viscosity, 293-423 K, 0.1-35 MPa, to 6 mol/kg | `brine_gas/kestin_nacl_viscosity.py` | Kestin, Khalifa and Correia 1981, J. Phys. Chem. Ref. Data 10:71 | 10.1063/1.555641 | the pressure factor's source |
| KCl brine viscosity, same ranges | `brine_gas/kestin_kcl_viscosity.py` | Kestin, Khalifa and Correia 1981, J. Phys. Chem. Ref. Data 10:57 | 10.1063/1.555640 | second-ion check |
| Water viscosity verification points | `brine_gas/iapws_viscosity.py` | Huber et al. 2009, J. Phys. Chem. Ref. Data 38:101 (IAPWS 2008) | 10.1063/1.3088050 | |
| Ion-additive volume and viscosity parameters (13 ions) | `brine_gas/pitzer.dat`, `brine_gas/appelo_volumes.py`, `brine_gas/jones_dole_viscosity.py` | Appelo, Parkhurst and Post 2014, Geochim. Cosmochim. Acta 125:49; USGS PHREEQC 3 database | 10.1016/j.gca.2013.10.003 | `pitzer.dat` is public domain (USGS) |
| Dead Sea brine densities, 25 four-salt compositions | `validation/deadsea_validation.py` (Tables III, IV) | Krumgalz and Millero 1982, Mar. Chem. 11:477 | 10.1016/0304-4203(82)90012-3 | 75 densities |
| NaCl-CaCl2 mixed-brine viscosities, 293-353 K, 0.1 MPa | `data/hoffert2025_table2.csv` | Hoffert, Bloecher, Kranz, Milsch and Sass 2025, Geotherm. Energy 13:15, Appendix 1 Table 2 | 10.1186/s40517-025-00339-4 | 520 rows from the text layer, checked against the rendered page; open access. The source's two pure-water series disagree, so its salt-ratio scores are conditional on the water reference |
| KCl-CaCl2 mixed-brine viscosity and density, 293-323 K | `data/arshad2020_tables3to9.csv` | Arshad et al. 2020, Sci. Rep. 10:16312, Tables 3-9 | 10.1038/s41598-020-73484-4 | 210 rows; one misprinted row (m1 = 0.1 in the m2 = 3.5 block at 293.15 K) is excluded by the scorer, not corrected |
| CaCl2 viscosity, 293-575 K, 0.1-60 MPa, six molalities | `data/abdulagatov2006_table3.csv` | Abdulagatov and Azizov 2006, Fluid Phase Equilib. 240:204, Table 3 | 10.1016/j.fluid.2005.12.036 | 309 rows from the text layer, checked against the rendered page |
| H2-saturated water density differences, 3.9-25 degC, atmospheric pressure (magnetic float) | `data/bignell1987_density_data.json` (all 108 Table I rows: 50 H2, 37 Ar, 21 H2-Ar) | Bignell 1987, J. Phys. Chem. 91:1687, Table I | 10.1021/j100290a080 | reduced to H2 volumes with the flash solubility in `validation/bignell1987_h2_check.py`; held out, never fitted; Table II's printed mixture intercept 0.00438 is a probable misprint for 0.0438 (refit), both kept |
| Infinite-dilution volume correlation coefficients | `brine_gas/plyasunov_model.py` | Plyasunov 2020, 2021 (Parts I, IV) | see references.bib | used only as the fallback route for C3H8 and nC4H10 and as a yardstick |
| Akinfiev-Diamond infinite-dilution correlation, H2S and CO2 parameters | `validation/akinfiev_diamond_h2s.py` | Akinfiev and Diamond 2003, Geochim. Cosmochim. Acta 67:613, Table 1 (preferred H2S set) | 10.1016/S0016-7037(02)01141-9 | the second published correlation the H2S trend check compares against; parameters verified against the paper |
| NaCl brine density (Spivey) | `brine_gas/brine_properties.py` | McCain, Spivey and Lenn 2011, Petroleum Reservoir Fluid Property Correlations | book | default gas-free density |
| NaCl Pitzer volumetric parameters | `brine_gas/rogers_pitzer_nacl.py`, `brine_gas/bradley_pitzer_dielectric.py` | Rogers and Pitzer 1982, J. Phys. Chem. Ref. Data 11:15; Bradley and Pitzer 1979 | 10.1063/1.555660; see references.bib | printed-source defects corrected in code and documented in each module |
| Simulator Ezrokhi defaults | `validation/ezrokhi_vs_default.py` | tNavigator technical manual (Rock Flow Dynamics) | manual | the shipped CO2/H2S/CH4/NaCl/CaCl2 sets audited in the paper's Appendix A |

Every DOI above was resolved against the publisher's record on 5 September 2026.

## Source numbers used in module docstrings

Module docstrings cite sources as `Papers/NN`, the numbering of the source directory in the working repository. The PDFs are not distributed; the numbers identify the source below (author, year, subject).

| number | source |
|---|---|
| 1 | Garcia 2001 CO2 Vphi framework |
| 2 | Hnedkovsky 1996 CH4 CO2 H2S Vphi data |
| 3 | OSullivan Smith 1970 N2 CH4 Vphi |
| 4 | Zhou Battino 2001 anchor points 25C |
| 5 | Plyasunov 2019 PartI H2 N2 CH4 SUPERSEDED |
| 6 | Plyasunov 2020 PartII CO2 C2H6 C3H8 nC4 |
| 7 | Plyasunov 2021 PartIII H2S polar |
| 8 | Plyasunov 2021 PartIV H2 N2 CH4 reparametrized |
| 9 | Ostermann 1985 SPE14211 CH4 brine viscosity |
| 10 | MurphyGaines 1974 H2S water density viscosity 20atm |
| 11 | Ostermann 1986 SPE15081 geothermal brine viscosity C2H6 null |
| 12 | Alboudwarej 2005 SPE96013 formation water viscosity GoM |
| 13 | Calabrese 2019 CO2 brine visc dens to449K 100MPa |
| 14 | Qin 2008 CH4 CO2 water ternary VLE |
| 15 | AlGhafri 2014 CH4 CO2 water phase compositions |
| 16 | Yan 2011 CO2 brine solubility density 0-5molal |
| 17 | DuanMao 2006 CH4 brine solubility density model |
| 18 | MaoDuan 2006 N2 brine solubility density model |
| 19 | Neuburg 1977 AECL5702 H2S water GS process properties |
| 20 | Dehaghani 2023 H2S brine MD visc dens IFT |
| 21 | McBrideWright 2015 CO2 water visc dens 274-449K 100MPa |
| 22 | Moore 1982 Vphi 20 gases infinite dilution water 298K |
| 23 | Hebach 2004 CO2 water density 284-332K 1-30MPa |
| 24 | Bignell 1984 Vphi atmospheric gases water |
| 25 | Handa 1982 dilatometer Vphi 40 liquid gas systems |
| 26 | TiepelGubbins 1972 Vphi gases in electrolyte solutions |
| 27 | TiepelGubbins 1973 thermo gases in electrolyte solutions |
| 28 | HeuslerGaiser 1972 Vphi H2 in electrolyte solutions |
| 29 | Lauder 1959 Vphi N2 O2 CO2 in water 0C |
| 30 | WilhelmBattino 2014 review Vphi gases in liquids |
| 31 | Enns 1965 hydrostatic pressure gases dissolved in water Vphi |
| 32 | Masterton 1961 densities gas solutions Vphi N2 Ar |
| 33 | Wiebe 1933 N2 solubility in water 50-100C |
| 34 | BurgessGermann 1969 H2S water physical properties |
| 35 | GibsonLoeffler 1941 PVT NaCl NaBr solutions Tait |
| 36 | HerrickGaines 1973 H2S surface tension |
| 37 | Gildseth 1972 precision water density 5-80C |
| 38 | RogersPitzer 1982 NaCl volumetric Pitzer 0-300C |
| 39 | Archer 1992 NaCl H2O thermodynamic extended Pitzer |
| 40 | Krumgalz 1996 Vphi single electrolytes to saturation 25C |
| 41 | Huber 2009 IAPWS2008 viscosity H2O |
| 42 | Archer 2000 NaNO3 H2O thermodynamic |
| 43 | Krumgalz 2000 volumetric ion interaction params vs T |
| 44 | BradleyPitzer 1979 dielectric DebyeHuckel slopes |
| 45 | ArcherWang 1990 dielectric constant DebyeHuckel slopes |
| 46 | PitzerPeiperBusey 1984 NaCl thermo JPCRD13 |
| 47 | AppeloParkhurstPost 2014 hydrogeochem reactions high PT |
| 48 | Appelo 2015 saline water databases 0-200C 1-1000atm |
| 49 | KrumgalzMillero 1982 DeadSea density measurements |
| 50 | Kestin 1981 NaCl viscosity 20-150C 0.1-35MPa |
| 51 | Kestin 1981 KCl viscosity 25-150C 0.1-35MPa |
| 52 | AkinfievDiamond 2003 EOS aqueous nonelectrolytes Vinf |
| 53 | IslamCarlson 2012 GRC viscosity models dissolved CO2 |
| 54 | MaoDuan 2009 alkali chloride viscosity 273-623K |
| 55 | Janjua 2024 H2 brine IFT salinity UHS |
| 56 | KennanPollack 1990 N2 Ar Kr Xe solubility pressure Vphi water 298K |
| 57 | AlvarezFernandezPrini 1992 Comment on KennanPollack |
| 58 | Hoffert 2025 NaCl CaCl2 mixed viscosity 293-353K 0.1MPa |
| 59 | Arshad 2020 KCl CaCl2 mixed visc dens 293-323K |
| 60 | Abdulagatov Azizov 2006 CaCl2 viscosity 293-575K 60MPa |
| 61 | Bignell 1987 H2 Ar mixed gas water density 3-25C |
| 62 | Barbero 1982 H2S NaHS Vphi Cp 10-40C |
