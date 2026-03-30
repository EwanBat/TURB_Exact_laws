# %% import libraries
import numpy as np
import h5py
import logging
from datetime import datetime
from exact_laws.preprocessing.process_on_oca_files import (
    extract_quantities_from_OCA_file,
    extract_simu_param_from_OCA_file
)
from trajectory_quantities import (
    load_config_from_ini,
    list_required_quantities
)
from trajectory_laws import extract_trajectory_and_compute_law_terms, display_law_terms_results

# Configure logging with a better format
log_filename = f"test_1satellite_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# %% Configuration
input_folder = "data_oca"
cycle = "cycle_0"
sim_type = "CGL5"
config_file = "example_input_process.ini"

# %% Load from INI file
logging.info("\n" + "="*70)
logging.info("LOADING CONFIGURATION FROM INI FILE")
logging.info("="*70)
laws, terms, quantities, physical_params = load_config_from_ini(config_file)

logging.info(f"  Laws:            {laws}")
logging.info(f"  Terms:           {terms}")
logging.info(f"  Quantities:      {quantities}")
logging.info(f"  Physical params: {physical_params}")

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

logging.info(f"  [OK] Velocity loaded:         {dic_quant['vx'].shape}")

# Load density
with h5py.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
    dic_quant["rho"] = extract_quantities_from_OCA_file(frho, ["rho"], cycle)[0]

logging.info(f"  [OK] Density loaded:          {dic_quant['rho'].shape}")

# Load magnetic field
with h5py.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
    (dic_quant["bx"],
     dic_quant["by"],
     dic_quant["bz"]) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)

logging.info(f"  [OK] Magnetic field loaded:   {dic_quant['bx'].shape}")

# Load pressure components
with h5py.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
    (dic_quant["ppar"],
     dic_quant["pperp"]) = extract_quantities_from_OCA_file(fp, ["pparli", "pperpi"], cycle)
    dic_quant["ppar"] /= 2
    dic_quant["pperp"] /= 2

logging.info(f"  [OK] Pressure loaded:         {dic_quant['ppar'].shape}")

# Load force amplitude
try:
    with h5py.File(f"{input_folder}/3Dfields_forcl_ampl.h5", "r") as ff:
        (dic_quant["fp"],
         dic_quant["fm"]) = extract_quantities_from_OCA_file(ff, ["forcl_ampl_plus", "forcl_ampl_mins"], cycle)
    logging.info(f"  [OK] Force amplitude loaded:  {dic_quant['fp'].shape}")
except:
    logging.warning("  [SKIP] Force amplitude not loaded")

# Get grid real values
lx, ly, lz = np.linspace(0, dic_param['N'][0]*dic_param['c'][0], dic_param['N'][0], endpoint=False), \
             np.linspace(0, dic_param['N'][1]*dic_param['c'][1], dic_param['N'][1], endpoint=False), \
             np.linspace(0, dic_param['N'][2]*dic_param['c'][2], dic_param['N'][2], endpoint=False)
dic_param['lx'] = lx; dic_param['ly'] = ly; dic_param['lz'] = lz

# Print summary
logging.info("\n" + "-"*70)
logging.info("DATA LOADING SUMMARY")
logging.info("-"*70)
logging.info(f"  Grid dimensions (N):  {dic_param['N']}")
logging.info(f"  Domain size (L):      {dic_param['L']}")
logging.info(f"  Cell spacing (c):     {dic_param['c']}")
logging.info(f"  Data fields:          {len(dic_quant)} fields loaded")
for field in sorted(dic_quant.keys()):
    logging.info(f"    - {field}")

# %% Determine required quantities
logging.info("\n" + "-"*70)
logging.info("DETERMINING REQUIRED QUANTITIES")
logging.info("-"*70)
required_qty = list_required_quantities(laws, terms, quantities)
if required_qty:
    logging.info(f"  Required quantities: {required_qty}")
else:
    logging.info("  No additional quantities required")

# %% Compute quantities along trajectory
# Récupérer les deux dictionnaires
dic_terms, dic_law_terms, dic_law_coeff = extract_trajectory_and_compute_law_terms(
    dic_quant,
    y_pos=100,
    z_pos=100,
    dic_param=dic_param,
    laws=laws,
    verbose=True
)
logging.info("\n" + "-"*70)
logging.info("COMPUTATION RESULTS")
logging.info("-"*70)
logging.info("  Law terms computed:")
for term in sorted(dic_law_terms.keys()):
    logging.info(f"    - {term}")
logging.info("\n  Law coefficients:")
for coeff in sorted(dic_law_coeff.keys()):
    logging.info(f"    - {coeff}")

# Display results
logging.info("\n" + "="*70)
logging.info("FINAL RESULTS")
logging.info("="*70)
display_law_terms_results(dic_law_terms, "Trajectory Laws")
logging.info("="*70 + "\n")

# %% Plot the law
import matplotlib.pyplot as plt

def linear_op_from_list_term(coeffs,quantities,list_term):
    coeff = {k.split('_',1)[1]:coeffs[k] for k in coeffs if k in list_term}
    result = np.zeros(np.shape(quantities[list(coeff.keys())[0]]))
    for k in coeff.keys():
            result += coeff[k]*quantities[k]
    return result

dic_of_list_terms = {}
dic_of_list_terms['PP98'] = [k for k in dic_law_coeff.keys() if k.split('_',1)[0] == 'PP98']

results = {}
for law in dic_of_list_terms.keys():
    list_term = dic_of_list_terms[law]
    results[law] = np.transpose(linear_op_from_list_term(dic_law_coeff,dic_law_terms,list_term))

plt.figure(figsize=(8,6))
plt.plot(dic_param['lx'], results['PP98'], label='PP98')
plt.xlabel('X-axis [di]')
plt.ylabel('PP98 law terms')
plt.title('PP98 law terms along trajectory')
plt.legend()
plt.savefig("PP98_trajectory.png")