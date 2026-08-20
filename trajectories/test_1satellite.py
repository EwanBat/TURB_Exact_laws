# %% import libraries
import logging
from datetime import datetime
from trajectories.preprocess_components import TrajectoryPreprocessor, param_to_txt
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

preprocessor = TrajectoryPreprocessor(verbose=True)
preprocessor.load_config(config_file)
preprocessor.load_oca_data()
config = preprocessor.run()
del preprocessor

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


from trajectories.quantity_components import TrajectoryQuantitiesComputer
from trajectories.terms_components import TrajectoryTermsComputer
from trajectories.laws_components import TrajectoryLawsComputer

quantities_computer = TrajectoryQuantitiesComputer(
    verbose=True,
    grid_param=grid_param,
    physical_param=physical_param,
    traj_param=traj_param,
)
dic_quantities = quantities_computer.extract_and_compute(
    config["dic_datas"],
    laws=laws,
    terms=terms,
    quantities=quantities,
    method=run_params['method'],
    filename='result_traj/quantities/'+name_output + '_' + trajectory_name + "_" + method + '_sat' + str(nbsatellite) + "_quantities.h5",
)
del quantities_computer

del config

terms_computer = TrajectoryTermsComputer(
    verbose=True,
    physical_param=physical_param,
    traj_param=traj_param,
    grid_param=grid_param,
    run_params=run_params,
)
dic_terms = terms_computer.compute_all_terms_for_laws(
    dic_quantities,
    laws,
    filename='result_traj/terms/'+name_output + '_' + trajectory_name + "_" + method + '_sat' + str(nbsatellite) + "_terms.h5",
)
del terms_computer

del dic_quantities

laws_computer = TrajectoryLawsComputer(
    verbose=True,
    physical_param=physical_param,
    traj_param=traj_param,
    grid_param=grid_param,
)
dic_law_terms, dic_law_coeff = laws_computer.compute_laws_terms(
    dic_terms,
    laws=laws,
    method=method,
    filename='result_traj/laws/'+name_output + '_' + trajectory_name + "_" + method + '_sat' + str(nbsatellite) + "_laws.h5",
)
del laws_computer

del dic_terms
del dic_law_coeff
del dic_law_terms

time_end = time.time()
logging.info(f"Time taken to compute laws terms: {time_end - time_start:.2f} seconds")