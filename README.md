# Beyond Linear Bias Expansions for AbacusSummit Halos at z=8

Analysis code for the paper *Beyond Linear Bias Expansions for AbacusSummit Halos at z=8*.

We measure the power spectrum and bispectrum of AbacusSummit halos at z = 8 (with a z = 5 comparison) and fit them with a one-loop power spectrum + tree-level bispectrum bias expansion. The bias templates are measured from simulated fields built out of a known linear power spectrum.

This is research code. Paths to data are hardcoded for the original machine and will need to be changed to run elsewhere.

## Pipeline

1. **Halo catalogs:** `Densities.py` reads AbacusSummit halo catalogs, applies the mass cut, and writes halo positions for each of the 25 phases (`ph000`–`ph024`).
2. **Halo P(k) and B(k):** `Bispectrum/BSKIT_Scripts/`
   - `Gala_BS.sh` paints halos to a 256³ mesh (`De_Pos_To_Mesh.py`) and measures the bispectrum.
   - `Gala_PS.sh` measures the power spectrum from the same mesh.
3. **Template fields:** `Bispectrum/Sandbox/Matter.py` generates the matter field from the linear power spectrum (`Bispectrum/Two_Points_Data/CLASS_power_z1`) and builds the bias operator fields (δ, δ², K², O_td, ∇²δ, ...).
4. **Template P(k) and B(k):**
   - `Bispectrum/BSKIT_Scripts/Sim_Bigfiles.py` converts the fields to BigFile meshes.
   - `Bispectrum/Sim_BSKIT/PS_Sim.sh` and `BS_Sim.sh` measure their auto and cross spectra. `BS_Sim.sh` uses a modified copy of BSKIT (`bskit_copy.py`, `measure_bs_fast_copy.py`).
   - `Average_Realizations.ipynb` averages over realizations.
   - `Display_Functions.ipynb` plots the templates.
5. **Covariance:** `Bispectrum/Covariance_Matrix/` holds the analytic power spectrum, bispectrum, and cross covariances. `Bispectrum/Fast_Bias_Fitting/Analytic_Covariance.ipynb` builds the versions used in the fits.
6. **Fitting and figures:** `Bispectrum/Fast_Bias_Fitting/`
   - `Bias_Fitting.ipynb` and `Bias_Fitting_Analytic_Cov.ipynb` run the fits.
   - `Model_Comparison.ipynb`, `Residuals.ipynb`, `Noise_Comp.ipynb`, and `Contour_Plots.ipynb` handle analysis and plots.
   - Shared loaders live in `Bispectrum/Sim_BiasFitting/` (`Fits_Helpers.py`, `Helpers.py`) and `Bispectrum/Convert_Bispectrum/Visual_Helpers.py`.

## Dependencies

- **Measurement (steps 1–2, 4):** Python 3.7 with [nbodykit](https://github.com/bccp/nbodykit) 0.3.15, [abacusutils](https://github.com/abacusorg/abacusutils), and [BSKIT](https://github.com/sjforeman/bskit). The shell scripts expect BSKIT to be checked out four directories above `Bispectrum/BSKIT_Scripts/`.
- **Fitting and analysis (steps 3, 5–6):** Python 3.12 with numpy, scipy, pandas, h5py, numexpr, matplotlib, JAX (float64), and getdist.
