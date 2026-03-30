# %% import libraries
import numpy as np
import h5py
from exact_laws.preprocessing.process_on_oca_files import (
    extract_quantities_from_OCA_file,
    extract_simu_param_from_OCA_file
)
from trajectory_quantities import (
    load_config_from_ini,
    extract_trajectory_and_compute,
    display_results,
    list_required_quantities
)

# %% Configuration
input_folder = "data_oca"
cycle = "cycle_0"
sim_type = "CGL5"
config_file = "example_input_process.ini"

# %% Load from INI file
print("=== Loading configuration from INI ===")
laws, terms, quantities, physical_params = load_config_from_ini(config_file)

print(f"Laws: {laws}")
print(f"Terms: {terms}")
print(f"Quantities: {quantities}")
print(f"Physical params: {physical_params}\n")

# %% load raw data into dic_quant dictionary
dic_quant = {}
dic_param = {}

# Extract simulation parameters from velocity file
with h5py.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
    param_key = "3Dgrid" if sim_type.endswith(("CGL3", "CGL5")) else "Simulation_Parameters"
    dic_param = extract_simu_param_from_OCA_file(fv, dic_param, param_key)
    (dic_quant["vx"],
     dic_quant["vy"],
     dic_quant["vz"]) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], cycle)

print(f"✓ Velocity loaded: {dic_quant['vx'].shape}")

# Load density
with h5py.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
    dic_quant["rho"] = extract_quantities_from_OCA_file(frho, ["rho"], cycle)[0]

print(f"✓ Density loaded: {dic_quant['rho'].shape}")

# Load magnetic field
with h5py.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
    (dic_quant["bx"],
     dic_quant["by"],
     dic_quant["bz"]) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)

print(f"✓ Magnetic field loaded: {dic_quant['bx'].shape}")

# Load pressure components
with h5py.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
    (dic_quant["ppar"],
     dic_quant["pperp"]) = extract_quantities_from_OCA_file(fp, ["pparli", "pperpi"], cycle)
    dic_quant["ppar"] /= 2
    dic_quant["pperp"] /= 2

print(f"✓ Pressure loaded: {dic_quant['ppar'].shape}")

# Load force amplitude
try:
    with h5py.File(f"{input_folder}/3Dfields_forcl_ampl.h5", "r") as ff:
        (dic_quant["fp"],
         dic_quant["fm"]) = extract_quantities_from_OCA_file(ff, ["forcl_ampl_plus", "forcl_ampl_mins"], cycle)
    print(f"✓ Force amplitude loaded: {dic_quant['fp'].shape}")
except:
    print("⚠ Force amplitude not loaded")

# Print summary
print("\n=== Summary ===")
print(f"Grid dimensions (N): {dic_param['N']}")
print(f"Domain size (L): {dic_param['L']}")
print(f"Cell spacing (c): {dic_param['c']}")
print(f"Datas loaded: {list(dic_quant.keys())}\n")

# %% Determine required quantities
print("=== Determining required quantities ===")
required_qty = list_required_quantities(laws, terms, quantities)
print(f"Required quantities: {required_qty}\n")

# %% Compute quantities along trajectory
print("=== Computing trajectory quantities ===")
traj_quantities = extract_trajectory_and_compute(
    dic_quant,
    y_pos=100,
    z_pos=100,
    dic_param=dic_param,
    laws=laws,
    terms=terms,
    quantities=quantities,
    verbose=True
)
display_results(traj_quantities, "Trajectory Results")

# %% Calculate flux term using Fourier method
print("=== Calculating flux term using Fourier method ===")
from exact_laws.el_calc_mod.laws.PP98 import Pp98
from exact_laws.el_calc_mod.terms.flux_dvdvdv import FluxDvdvdv
from exact_laws.el_calc_mod.terms.flux_dbdbdv import FluxDbdbdv
from exact_laws.el_calc_mod.terms.flux_dvdbdb import FluxDvdbdb
import numpy.fft as ft

# Initialize law and terms
law = Pp98()
term_dvdvdv = FluxDvdvdv()
term_dbdbdv = FluxDbdbdv()
term_dvdbdb = FluxDvdbdb()

# Compute flux using Fourier method
flux_dvdvdv = term_dvdvdv.calc_fourier(dic_quant["vx"], dic_quant["vy"], dic_quant["vz"])
flux_dbdbdv = term_dbdbdv.calc_fourier(dic_quant["bx"], dic_quant["by"], dic_quant["bz"], dic_quant["vx"], dic_quant["vy"], dic_quant["vz"])
flux_dvdbdb = term_dvdbdb.calc_fourier(dic_quant["vx"], dic_quant["vy"], dic_quant["vz"], dic_quant["bx"], dic_quant["by"], dic_quant["bz"])

print("✓ Flux terms calculated using Fourier method")