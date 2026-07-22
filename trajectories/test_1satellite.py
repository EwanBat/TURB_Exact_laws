# %% import libraries
import logging
from datetime import datetime
from trajectories.trajectory_preprocess import preprocess_trajectory_from_ini, param_to_txt
import time
time_start = time.time()

# Configure logging with a better format
log_filename = f"test_1satellite_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# %% Configuration
config_file = "trajectories/input_satellite.ini"

# %% Preprocess trajectory
logging.info("\n" + "="*70)
logging.info("PREPROCESSING TRAJECTORY")

config = preprocess_trajectory_from_ini(
    ini_file=config_file,
    verbose=True
)

# Extract results
name_output = config['name_output']
grid_param = config['grid_param']
traj_param = config['traj_param']
physical_param = config['physical_param']
run_params = config['run_params']
trajectory_name = config['trajectory_name']

laws = config['laws']
terms = config['terms']
quantities = config['quantities']

method = run_params['method']
nbsatellite = traj_param['nbsatellite']

param_to_txt(grid_param, traj_param, physical_param, laws, filename='result_traj/parameters/'+name_output + '_' + trajectory_name + "_" + method + '_sat' + str(nbsatellite) + "_parameters.txt")


from trajectories.trajectory_quantities import extract_and_compute_trajectory_quantities
from trajectories.trajectory_terms import compute_all_terms_for_laws
from trajectories.trajectory_laws import compute_laws_terms_with_coefficients

dic_quantities = extract_and_compute_trajectory_quantities(
    config["dic_datas"], 
    grid_param=grid_param,
    traj_param=traj_param,
    physical_param=physical_param,
    laws=laws,
    terms=terms,
    quantities=quantities,
    method=run_params['method'],
    filename='result_traj/quantities/'+name_output + '_' + trajectory_name + "_" + method + '_sat' + str(nbsatellite) + "_quantities.h5",
    verbose=True,
)

del config

dic_terms = compute_all_terms_for_laws(
    dic_quantities = dic_quantities, 
    laws = laws,
    grid_param = grid_param,
    physical_param = physical_param,
    traj_param = traj_param,
    run_params = run_params,
    filename = 'result_traj/terms/'+name_output + '_' + trajectory_name + "_" + method + '_sat' + str(nbsatellite) + "_terms.h5",
    verbose=True
)

del dic_quantities

dic_law_terms, dic_law_coeff = compute_laws_terms_with_coefficients(
    dic_terms=dic_terms,
    laws=laws,
    physical_param=physical_param,
    traj_param=traj_param,
    method=method,
    filename = 'result_traj/laws/'+name_output + '_' + trajectory_name + "_" + method + '_sat' + str(nbsatellite) + "_laws.h5",
    verbose=True
)

del dic_terms
del dic_law_coeff
del dic_law_terms

time_end = time.time()
logging.info(f"Time taken to compute laws terms: {time_end - time_start:.2f} seconds")
    