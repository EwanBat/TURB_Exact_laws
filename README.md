# TURB_Exact_laws

Repository map: this README only describes the files and folders tracked by Git (see `.gitignore`).
Focus is on the `exact_laws/` and `trajectories/` packages. Use it to locate a file, a function, or a class.

## Top level

| Path | Content |
| --- | --- |
| `exact_laws/` | Main package (preprocessing, exact-law calculation) |
| `trajectories/` | Trajectory-based (single / multi satellite) computation pipeline |
| `tests/` | Unit tests |
| `.coveragerc` | Coverage configuration (branch coverage, omit rules) |
| `.gitignore` | Ignored files: `*.h5`, `*.log`, `*.png`, `notebooks/`, `visualisation/`, `__pycache__/`, `*.pyc` |
| `example_input_calc.ini` | Example config for `calc_exact_law.py` (INPUT/OUTPUT data, GRID and RUN params) |
| `example_input_process.ini` | Example config for `reformat_oca_files.py` (OCA data -> standard h5) |
| `example_input_process_localP.ini` | Example config for `reformat_oca_files.py` with local pressure / `reduction = 2` |

---

# `exact_laws/`

```
exact_laws/
├── __init__.py                      (empty)
├── calc_exact_law.py                CLI entry point
├── reformat_oca_files.py            CLI entry point
├── el_calc_mod/                     exact law computation engine
│   ├── __init__.py
│   ├── datasets/                    Dataset object + h5 read/write
│   ├── grids/                       Grid objects and incremental grids
│   ├── laws/                        exact laws (LAWS registry)
│   ├── terms/                       terms (TERMS registry)
│   ├── terms_exemples/              example / test term implementations
│   ├── fourier.py                   "fourier" computation method
│   └── incremental.py               "incremental" computation method
├── mathematical_tools/
│   ├── derivation.py                finite-difference operators
│   └── fourier_transform.py         fft / ifft wrappers
├── preprocessing/
│   ├── __init__.py                  (empty)
│   ├── copy_structure_folder_of_h5.py
│   ├── process_on_oca_files.py      OCA files -> standard h5
│   ├── process_on_standard_h5_file.py  h5 inspection / binning / truncation
│   └── quantities/                  physical quantities (QUANTITIES registry)
└── running_tools/
    ├── __init__.py                  (empty)
    ├── backup_wrap.py               checkpoint save/download (pickle)
    └── run_config_wrap.py           run configuration (MPI / serial)
```

## Entry points

- `calc_exact_law.py` — CLI to compute exact laws.
  Options: `-f/--config-file`, `-e/--list-exactlaws`, `-t/--list-terms`.
  Loads config, sets `RunConfig` and `Backup`, then calls `calc_exact_laws_from_config()` (`el_calc_mod/__init__.py`).
- `reformat_oca_files.py` — CLI to convert OCA simulation files to the standard h5 format.
  Options: `-f/--config-file`, `-e/--list-exactlaws`, `-t/--list-terms`, `-q/--list-quantities`.
  Calls `reformat_oca_files()` from `preprocessing/process_on_oca_files.py`.

## `el_calc_mod/`

### `__init__.py`
- `calc_exact_laws_from_config(config_file, run_config, backup)` — main orchestration: builds the input filename, loads the original dataset, builds the incremental grid, dispatches to the selected method module (`fourier` or `incremental`), reduces 3D -> 2D, checks the output file.
- `initialise_original_dataset(input_filename)` — loads dataset, laws and terms.
- `multifile_distrib(laws, terms)` — splits laws/terms into per-file groups (`_inc`, `_ss22f`, `_ss22s`, `_hall`, `_other`) for the `multifile = True` case.

### `datasets/`
- `dataset.py` — `Dataset` class (attributes: `quantities`, `grid`, `params`; methods `describ()`, `check()`).
- `__init__.py` — `load()` (Dataset constructor), `read_standard_file(filename)` (extract quantities / laws / terms / grid params / physical params from a standard h5 file), `load_from_standard_file()`, `record_incdataset_to_h5file()` (write a Dataset to h5).

### `grids/`
- `grid.py` — `Grid` class (attributes `N`, `L`, `c`, `axis`, `coords`; methods `describ()`, `check()`).
- `incgrid.py` — `IncGrid` class (incremental scale grid; attributes `spatial_grid`, `N`, `axis`, `coords`, `kind`; methods `describ()`, `check()`).
- `lincart.py` — linear Cartesian incremental grid: `load()`, `load_outputgrid()`, `div()`, `reorganise_quantities()`, `reformat_grid_compatible_to_h5()`.
- `logcyl.py` — log-regular cylindrical incremental grid: `load()`, `build_logregular_cylindrical_incremental_grid()`, `logregular_axis()`, `build_listcoords()`, `load_outputgrid()`, `coordinate_sec_in_primsec_grid()`, `div()`, `reorganise_quantities()`, `reformat_grid_compatible_to_h5()`.
- `__init__.py` — factory helpers: `load_grid()`, `load_grid_from_dict()`, `load_incgrid_from_grid()`, `load_outputgrid_from_incgrid()`, `div_on_incgrid()`, `reorganise_quantities()`, `reformat_grid_compatible_to_h5()`.

### `laws/`
- `abstract_law.py` — `AbstractLaw` base class: `terms_and_coeffs()`, `list_variables()`.
- `__init__.py` — auto-loads every `*.py` (except `__init__`, `abstract_law`) into the `LAWS` registry (key = file name); `load_law()`, `load_all()`.

Laws (each file exports `load()` returning a class instance with `terms_and_coeffs(physical_params)` and `variables()`):

| File | Terms used |
| --- | --- |
| `PP98.py` | `flux_dvdvdv`, `flux_dbdbdv`, `flux_dvdbdb` (Politano–Pouquet MHD) |
| `PP98_source.py` | `source_dvdvdv`, `source_dbdbdv`, `source_dvdbdb` |
| `BG17.py` | `bg17_vwv`, `bg17_jbv`, `bg17_vbj` |
| `Hallcor.py` | `flux_djbdrb`, `flux_drjbdb`, `source_bbdrdj`, `source_bjdrdb` |
| `IHallcor.py` | `flux_djdbdb`, `flux_dbdbdj` (Hall corrections, incremental variant) |
| `COR_Etot.py` | `cor_rvv`, `cor_ru`, `cor_rbb`, `source_dpan` (correlation energy) |
| `SS21Iso.py` / `SS21Pol.py` | SS21 isothermal / polytropic laws |
| `TotSS21Iso.py` / `TotSS21Pol.py` | SS21 total (flux + source) variants |
| `SS22Iso.py` / `SS22Iso_flux.py` / `SS22Iso_sources.py` | SS22 isothermal law and its flux / sources parts |
| `SS22Pol.py` / `TotSS22Pol.py` | SS22 polytropic law (+ total variant) |
| `SS22Cgl.py` | SS22 CGL law |
| `SS22Gyr.py` / `SS22Gyr_flux.py` / `SS22Gyr_sources.py` | SS22 gyrotropic law and its flux / sources parts |
| `ISS22Cgl.py` | Ion SS22 CGL law |
| `ISS22Gyr.py` / `ISS22Gyr_source.py` | Ion SS22 gyrotropic law (+ source variant) |
| `ISS22Iso.py` | Ion SS22 isothermal law |

### `terms/`
- `abstract_term.py` — `AbstractTerm` base class: `calc()`, `calc_fourier()`, `variables()`, `calc_incr_traj()`, `calc_filter()`; plus numba helpers `calc_source_with_numba`, `calc_flux_with_numba`, `calc_source_with_numba_traj`, `calc_flux_with_numba_traj`, `calc_source_with_numba_traj_filter`, `calc_flux_with_numba_traj_filter`, `calc_source_with_numba_traj_split`.
- `__init__.py` — auto-loads every `*.py` (except `__init__`, `abstract_term`) into the `TERMS` registry (key = file name); `load_term()`, `load_all()`.

Terms (each file exports `load()`; methods: `calc()`, `calc_fourier()`, `calc_incr_traj()`, `calc_filter()`, `variables()`). Naming convention: `flux_drXX` = flux with a displacement (`dr`), `dXX` = derivative; `source_*` = source terms; `term_div_*`/`div_*` = divergence.

| Prefix | Files |
| --- | --- |
| `bg17_*` | `bg17_jbv.py`, `bg17_vbj.py`, `bg17_vwv.py` |
| `cor_*` | `cor_bb.py`, `cor_rbb.py`, `cor_ru.py`, `cor_rvv.py`, `cor_vv.py` |
| `diss_*` | `diss_b.py`, `diss_v.py`, `diss_v2.py` |
| `flux_*` | `flux_dbdbdj.py`, `flux_dbdbdv.py`, `flux_djbdrb.py`, `flux_djdbdb.py`, `flux_drbdbdv.py`, `flux_drbdvdb.py`, `flux_drdpancgldv.py`, `flux_drdpandv.py`, `flux_drdpisodv.py`, `flux_drdpmdv.py`, `flux_drdpperpcgldv.py`, `flux_drdpperpdv.py`, `flux_drdppoldv.py`, `flux_drducgldv.py`, `flux_drdugyrdv.py`, `flux_drduisodv.py`, `flux_drdupoldv.py`, `flux_drjbdb.py`, `flux_drpisov.py`, `flux_drpmv.py`, `flux_drpmv2.py`, `flux_drppolv.py`, `flux_druisov.py`, `flux_drupolv.py`, `flux_drvdbdb.py`, `flux_drvdvdv.py`, `flux_dvdbdb.py`, `flux_dvdvdv.py` |
| `forc_*` | `forc_v.py`, `forc_vinc.py` (forcing) |
| `source_*` | `source_bbdrdj.py`, `source_bdrbdv.py`, `source_bdrvdb.py`, `source_bjdrdb.py`, `source_dbdbdv.py`, `source_dpan.py`, `source_dpancgl.py`, `source_dpantr.py`, `source_drbbdv.py`, `source_dvdbdb.py`, `source_dvdvdv.py`, `source_pancglvdrdr.py`, `source_panvdrdr.py`, `source_pisovdrdr.py`, `source_pmvdr.py`, `source_pmvdr2.py`, `source_pmvdrdr.py`, `source_pperpcglvdrdr.py`, `source_pperpvdrdr.py`, `source_ppolvdrdr.py`, `source_rbdbdv.py`, `source_rbdvdb.py`, `source_rdpancgldv.py`, `source_rdpandv.py`, `source_rdpisodv.py`, `source_rdpperpcgldv.py`, `source_rdpperpdv.py`, `source_rdppoldv.py`, `source_rducgldv.py`, `source_rdugyrdv.py`, `source_rduisodv.py`, `source_rdupoldv.py`, `source_rpisodv.py`, `source_rppoldv.py`, `source_rvbetadrho.py`, `source_rvbetadu.py`, `source_rvbetadupol.py`, `source_rvdbdb.py`, `source_rvdpancgldr.py`, `source_rvdpandr.py`, `source_rvdpisodr.py`, `source_rvdpmdr.py`, `source_rvdpperpcgldr.py`, `source_rvdpperpdr.py`, `source_rvdppoldr.py`, `source_rvdvdv.py` |

### `terms_exemples/`
Example / experimental term implementations (not part of the `TERMS` registry, kept for reference):
`flux_test*.py`, `flux_ss21*.py`, `flux_ss21hyb*.py`, `flux_ss22*.py`, `source_test*.py`, `source_ss21*.py`, `source_ss22*.py`.

### `fourier.py` / `incremental.py`
Both provide the same interface, selected via `method` in the config:
- `initialise_output_dataset()`, `list_terms_and_coeffs()`, `apply_method()`, `red3Dto2D()`.
- `fourier.py` also has `calc_term()` (calls `TERMS[...].calc_fourier()`), `reduction_output()`, `save_output_dataset_on_incgrid()`, `red3Dto2D_multifile()`, `reduction()`.
- `incremental.py` also has `init_ouput_quantities()`, `calc_terms()` (loop over `listprim`/`listsec`, calls `TERMS[...].calc()`), `reduction_output()`, `save_output_dataset_on_incgrid()`.

## `mathematical_tools/`

- `derivation.py` — finite-difference operators: `cdiff()`, `cdiff_2point()`, `cdiff_4point()`, `cdiff_point()`, `cdiff_prec4()`, `cdiff_prec2()`, `cdiff_array()`, `cdiff2_prec4()`, `div()`, `rot()`, `rot_gen()`, `grad()`, `grad_gen()`, `laplacien()`, `laplacien2()`.
- `fourier_transform.py` — `fft()` (rfftn), `ifft()` (irfftn); `traj=True` switches to 1D `rfft`/`irfft`.

## `preprocessing/`

- `process_on_oca_files.py` — converts raw OCA simulation files into the standard h5 format:
  `from_OCA_files_to_standard_h5_file()` (main), `reformat_oca_files(config_file)` (INI-driven wrapper), `extract_simu_param_from_OCA_file()`, `extract_quantities_from_OCA_file()`, `list_quantities()`.
- `process_on_standard_h5_file.py` — h5 file utilities:
  `check_file()`, `describ_file()`, `recursive_describ_of_h5file()`, `verif_file_existence()`, `copy_struct_h5file()`, `recursive_copy_of_file()`, `data_binning()`, `bin_an_array()`, `bin_arrays_in_h5()`, `data_reduction()`, `trunc_an_array()`, `trunc_arrays_in_h5()`, `extract_quantities_from_h5_file()`.
- `copy_structure_folder_of_h5.py` — `copy_struct_folder_of_h5file()`, `copy_struct_h5file()`, `recursive_copy_of_file()` (copy structure only, empty datasets).

### `preprocessing/quantities/`
Physical quantities, each file exports `load(incompressible=False)` and `create_datasets(g, dic_quant, dic_param)`; `__init__.py` builds the `QUANTITIES` registry (keys: `name` and `Iname` incompressible variant, e.g. `v` / `Iv`). `get_original_quantity()` appears in files where raw components must be pre-computed (e.g. `j`, `b`, `pcgl`).

| File | Quantity | File | Quantity |
| --- | --- | --- | --- |
| `v.py` | velocity | `b.py` | magnetic field |
| `v2.py` | v² | `bnorm.py` | \|b\| |
| `vnorm.py` | \|v\| | `pm.py` | magnetic pressure b²/2 |
| `w.py` | vorticity | `j.py` | current density |
| `rho.py` | density | `divb.py` | div b |
| `gradv.py` | velocity gradient | `divj.py` | div j |
| `gradv2.py` | gradient of v² | `gradb.py` | gradient of b |
| `gradrho.py` | gradient of rho | `divv.py` | div v |
| `f.py` | forcing | `hdk.py` / `hdk2.py` | Hall/kinetic terms |
| `pgyr.py` | gyrotropic pressure | `hdm.py` | Hall/magnetic term |
| `piso.py` | isotropic pressure | `ugyr.py` | gyrotropic velocity |
| `ppol.py` | polytropic pressure | `uiso.py` | isotropic velocity |
| `pcgl.py` | CGL pressure | `upol.py` | polytropic velocity |
| `graduiso.py` | grad u_iso | `ucgl.py` | CGL velocity |
| `gradupol.py` | grad u_pol | | |

## `running_tools/`

- `run_config_wrap.py` — `RunConfig` class (run/parallelism configuration: `NOP`, `MPI`, `OLD`): `barrier()`, `bcast()`, `reduce()`, `distrib()`, `counter()`, `set_nblayer()`, `set_bufnum()`, `configure_log()`; factory `load(config, numbap=False)`.
- `backup_wrap.py` — `Backup` class: `configure(config, time, rank)` (creates `backup_<timestamp>/` folder), `save(object, name, rank='', state='')`, `download(name, rank='')` (pickle).

---

# `trajectories/`

```
trajectories/
├── __init__.py                     (empty)
├── derivation_satellite.py         divergence / gradient / curl along trajectories
├── trajectory_experimental.py      experimental satellite data loader
├── preprocess_components/          trajectory extraction pipeline
├── quantity_components/            quantities along trajectories
├── terms_components/               terms along trajectories
├── laws_components/                law assembly along trajectories
├── input_satellite.ini             example config (OCA/simulation data)
├── input_experimental.ini          example config (experimental data)
├── test_1satellite.py              run script: OCA data, nbsatellite from INI
└── test_experimental.py            run script: experimental data
```

Data structure throughout the pipeline: `{sat_name: {var_name: array(n_trajectories, n_points)}}`.

## Top-level files

- `derivation_satellite.py` — spatial derivative operators on trajectories:
  `divergence_1satellite()`, `gradient_1satellite()`, `curl_1satellite()`,
  `gradient_4satellite()`, `divergence_4satellite()` (reciprocal vectors),
  `gradient_9satellite()`, `divergence_9satellite()` (star / cross formations).
- `trajectory_experimental.py` — `ExperimentalTrajectoryDataLoader` class for satellite measurements (1 or 4 satellites):
  `load_datas_dict()`, `compute_derived_quantities()` (tangents, trajectory length, relative positions), `load_config()`, `run()`, `_process_satellite_data()`, `_compute_stats()`; plus helpers `param_to_txt()`, `setup_logging()`.
- `input_satellite.ini` — config for simulation/OCA runs (`[INPUT_DATA]`, `[OUTPUT_DATA]`, `[PHYSICAL_PARAMS]`, `[RUN_PARAMS]`, `[TRAJECTORY_PARAMS]`: `nbsatellite` 1/4/9, `gap_satellite`, `trajectory_method`, `trajectory_kwargs`, `formation`, `step_traj`).
- `input_experimental.ini` — config for experimental runs (`[INPUT_DATA]`, `[OUTPUT_DATA]`, `[PHYSICAL_PARAMS]`, `[RUN_PARAMS]`).
- `test_1satellite.py` — end-to-end example: `TrajectoryPreprocessor` -> `TrajectoryQuantitiesComputer` -> `TrajectoryTermsComputer` -> `TrajectoryLawsComputer`, saves h5 outputs into `result_traj/{parameters,quantities,terms,laws}/`.
- `test_experimental.py` — same pipeline with `ExperimentalTrajectoryDataLoader` (random data placeholder for 4 satellites).

## `preprocess_components/`
- `preprocessor.py` — `TrajectoryPreprocessor` class: `load_config()`, `load_oca_data()` (3Dfields_*.h5), `run()`; helpers `param_to_txt()`, `setup_logging()`. Dispatches to trajectory generators via `TRAJECTORY_METHODS` / `GENERATE_ALL_FUNCTIONS`.
- `trajectories.py` — trajectory path definitions (in grid indices): `trajectory_linear_x()`, `trajectory_linear_minus_x()`, `trajectory_linear_y()`, `trajectory_linear_minus_y()`, `trajectory_linear_z()`, `trajectory_linear_minus_z()`, `trajectory_linear_xy()`; parameter generators `generate_all_trajectory_kwargs_linear_{x,y,z,xy}()`.
- `geometry.py` — satellite geometry and sampling: `_get_satellite_offsets()`, `_compute_tangent_vectors()`, `_compute_trajectory_coordinates()`, `interpolation_along_trajectory()`, `extract_quantities_along_trajectory()`, `combine_multiple_trajectories()`.

## `quantity_components/`
- `__init__.py` — `TrajectoryQuantitiesComputer(TrajectoryQuantitiesComputerBase, TrajectoryQuantitiesComputeMixin)`.
- `base.py` — `TrajectoryQuantitiesComputerBase`: `QUANTITY_DEPENDENCIES`, `GRADIENT_QUANTITIES`, `NINE_SATELLITE_TUPLES` dicts; `_compute_quantity_vectorized()`, `quantities_to_h5()`; `MockFile` (in-memory h5 stand-in).
- `compute.py` — `TrajectoryQuantitiesComputeMixin`: `extract_and_compute()` (public entry), `_list_required_quantities()`, `_compute_all_quantities()` (dispatch on nbsatellite 1/4/9), `_compute_all_single_pass()`, `_compute_non_gradient_quantities()`, `_merge_raw_data()`, `_compute_gradient_4satellite()`, `_compute_gradient_9satellite()`.

## `terms_components/`
- `__init__.py` — `TrajectoryTermsComputer(TrajectoryTermsComputerBase, TrajectoryTermsIncrementalMixin, TrajectoryTermsFourierMixin)`.
- `base.py` — `TrajectoryTermsComputerBase`: `VARIABLE_COMPONENTS` mapping, `FLUX_TERMS` / `SOURCE_TERMS` sets; `list_required_terms()`, `compute_all_terms_for_laws()` (dispatches on method + nbsatellite), `terms_to_h5()`, `_get_incremental_fs()`, `_prepare_dic_param_for_terms_and_coeffs()`, `_extract_sat_parameters()`.
- `incremental.py` — `TrajectoryTermsIncrementalMixin`: `_compute_terms_incremental_1sat()`, `_compute_terms_incremental_4sat()`, `_compute_terms_incremental_9sat()` (calls `TERMS[...].calc_incr_traj()` / `.calc_filter()`).
- `fourier.py` — `TrajectoryTermsFourierMixin`: `_compute_terms_fourier_1sat()`, `_compute_terms_fourier_multi()` (calls `TERMS[...].calc_fourier()` / `.calc_with_fourier_4sat()`).

## `laws_components/`
- `__init__.py` — `TrajectoryLawsComputer(TrajectoryLawsComputerBase, TrajectoryLawsCoefficientsMixin)`.
- `base.py` — `TrajectoryLawsComputerBase`: `_prepare_dic_param_for_terms_and_coeffs()`, `laws_to_h5()`.
- `coefficients.py` — `TrajectoryLawsCoefficientsMixin`: `compute_laws_terms()` (public entry, dispatches on nbsatellite), `_apply_law_coefficients_1satellite()`, `_apply_law_coefficients_4satellite()`, `_apply_law_coefficients_9satellite()` (uses `divergence_{1,4,9}satellite()` from `derivation_satellite.py`).

---

# `tests/`

- `tests_mathematical_tools/test_derivation.py` — tests for `mathematical_tools/derivation.py`.
- `tests_exact_laws_calc/tests_grids/test_grids.py` — tests for `el_calc_mod/grids/`.
- `tests_preprocessing/test_process_on_oca_files.py` — tests for `preprocessing/process_on_oca_files.py`.
- `tests_preprocessing/test_process_on_standard_h5_file.py` — tests for `preprocessing/process_on_standard_h5_file.py`.
- `tests_preprocessing/tests_quantities/` — one test file per quantity: `test_b.py`, `test_divb.py`, `test_divj.py`, `test_divv.py`, `test_gradrho.py`, `test_graduiso.py`, `test_gradv.py`, `test_j.py`, `test_pgyr.py`, `test_piso.py`, `test_pm.py`, `test_rho.py`, `test_ugyr.py`, `test_uiso.py`, `test_v.py`, `test_w.py`, plus `test_quantities.py`.