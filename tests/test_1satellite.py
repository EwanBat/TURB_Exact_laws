# %% import libraries
import numpy as np
import logging
from datetime import datetime
from trajectory_preprocess import preprocess_trajectory_from_ini, param_to_txt
from trajectory_quantities import extract_and_compute_trajectory_quantities
from trajectory_terms import compute_all_terms_for_laws, terms_to_h5
from trajectory_laws import compute_laws_terms_with_coefficients, laws_to_h5
import time

# Configure logging with a better format
log_filename = f"test_1satellite_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# %% Configuration
config_file = "tests/traj_satellite.ini"

# %% Preprocess trajectory
logging.info("\n" + "="*70)
logging.info("PREPROCESSING TRAJECTORY")

time_start = time.time()

config = preprocess_trajectory_from_ini(
    ini_file=config_file,
    verbose=True
)

# Extract results
dic_datas = config['dic_datas']  # 1D extracted data
grid_param = config['grid_param']
traj_param = config['traj_param']
physical_param = config['physical_param']
param_to_txt(grid_param, traj_param, physical_param, filename=config['name_output'] + '_' + config['trajectory_name'] + "_parameters.txt")

laws = config['laws']
terms = config['terms']
quantities = config['quantities']
method = config['method']
print(dic_datas.keys())

# see_trajectory_in_space(dic_param, trajectory, nbsatellite)

dic_quantities = extract_and_compute_trajectory_quantities(
    dic_datas, 
    grid_param=grid_param,
    traj_param=traj_param,
    physical_param=physical_param,
    laws=laws,
    terms=terms,
    quantities=quantities,
    verbose=True
)

print(dic_quantities.keys())

# %% Compute quantities along trajectory
if traj_param['nbsatellite'] == 1:

    dic_terms = compute_all_terms_for_laws(
        dic_quantities = dic_quantities, 
        laws = laws,
        physical_param = physical_param,
        traj_param = traj_param,
        method = method,
        verbose=True)

    terms_to_h5(dic_terms, filename=config['name_output'] + '_' + config['trajectory_name'] + "_terms.h5")
    
    dic_law_terms, dic_law_coeff = compute_laws_terms_with_coefficients(
        dic_terms=dic_terms,
        laws=laws,
        physical_param=physical_param,
        traj_param=traj_param,
        verbose=True)

    laws_to_h5(dic_law_terms, dic_law_coeff, filename=config['name_output'] + '_' + config['trajectory_name'] + "_laws.h5")

    time_end = time.time()
    logging.info(f"Time taken to compute laws terms: {time_end - time_start:.2f} seconds")

elif config['nbsatellite'] == 4:

    logging.warning("Computation for 4 satellites not implemented yet. Please set nbsatellite to 1 in the configuration.")
