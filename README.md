# TURB_Exact_laws

Python code to evaluate **exact laws of plasma turbulence** (Kolmogorov-like relations for the energy cascade rate) from 3D data:

- on the **full 3D cube**, as a function of the 3D increment vector **ℓ**;
- along **virtual spacecraft trajectories** (1, 4 or 9 satellites) crossing the cube, to mimic in-situ measurements.

It covers incompressible MHD (Politano & Pouquet 1998), compressible laws with several pressure closures (isothermal, polytropic, CGL, gyrotropic), their incompressible variants, and Hall corrections.

This repository is a fork of [`LaboratoryOfPlasmaPhysics/TURB_Exact_laws`](https://github.com/LaboratoryOfPlasmaPhysics/TURB_Exact_laws), originally written by Pauline Simon.

---

## Installation

```bash
git clone https://github.com/EwanBat/TURB_Exact_laws.git
cd TURB_Exact_laws
pip install numpy scipy h5py numba sympy numexpr
pip install mpi4py        # optional, for MPI runs
pip install -e .
```

---

## How it works

The code is organised in three parts. Each one is driven by an `.ini` configuration file; the example files in the repository document the available options.

### 1. Preprocessing

Reads the 3D data and writes a standard `.h5` file containing the grid, the physical parameters and all the quantities (velocity, magnetic field, density, pressures, gradients, …) needed by the chosen laws. The data can be reduced (binning or truncation) at this stage.

```bash
python -m exact_laws.reformat_oca_files -f example_input_process.ini
```

### 2. Exact laws on the 3D cube

Computes the terms of the chosen laws for every increment ℓ of a grid of increments, then combines them into the laws. Two methods are available:

- **incremental**: direct evaluation of the increments over the cube;
- **fourier**: FFT-based evaluation.

The results are saved on the 3D increment grid and reduced to (ℓ⊥, ℓ∥). Runs can be parallelised with MPI and resumed from checkpoints.

```bash
python -m exact_laws.calc_exact_law -f example_input_calc.ini
mpirun -n 8 python -m exact_laws.calc_exact_law -f example_input_calc.ini   # MPI
```

### 3. Exact laws along trajectories

Samples the 3D data along straight spacecraft trajectories, for a single satellite or a formation of 4 or 9 satellites, then computes the quantities, terms and laws along these 1D signals. Gradients are estimated from the satellite formation, as in multi-spacecraft missions.

```bash
python -m trajectories.test_1satellite      # uses trajectories/input_satellite.ini
```

### Filtering

A low-pass filter can be applied to the data before computing the terms, both along trajectories and on the 3D cube, to remove the smallest scales.

---

## Available laws

| Family | Laws |
| --- | --- |
| Incompressible MHD | `PP98`, `PP98_source`, `BG17` |
| Compressible (isothermal, polytropic, CGL, gyrotropic closures) | `SS21*`, `SS22*`, `TotSS21*`, `TotSS22*` |
| Incompressible versions of SS22 | `ISS22Iso`, `ISS22Cgl`, `ISS22Gyr`, `ISS22Gyr_source` |
| Hall corrections | `Hallcor`, `IHallcor` |
| Correlation functions | `COR_Etot` |

List them with `python -m exact_laws.calc_exact_law -e` (laws) and `-t` (terms).

New laws, terms and quantities are added by dropping a file in the corresponding folder: they are registered automatically.

---

## Repository layout

| Folder | Role |
| --- | --- |
| `exact_laws/preprocessing/` | preprocessing and computation of the physical quantities |
| `exact_laws/el_calc_mod/` | exact-law engine: laws, terms, increment grids, incremental and Fourier methods |
| `exact_laws/mathematical_tools/` | finite differences and Fourier transforms |
| `exact_laws/running_tools/` | serial / MPI run configuration, logs, checkpoints |
| `trajectories/` | trajectory pipeline (sampling, quantities, terms, laws for 1, 4 or 9 satellites) |
| `tests/` | unit tests (`python -m pytest tests`) |
| `notebooks/` | exploration notebooks |
