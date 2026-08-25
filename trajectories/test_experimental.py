# %% import libraries
import numpy as np
import logging
from datetime import datetime
from trajectories.trajectory_experimental import ExperimentalTrajectoryDataLoader, param_to_txt
from trajectories.quantity_components import TrajectoryQuantitiesComputer
from trajectories.terms_components import TrajectoryTermsComputer
from trajectories.laws_components import TrajectoryLawsComputer
import time

time_start = time.time()

# Configure logging with a better format
log_filename = f"log_traj_exp_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# %% Configuration
config_file = "trajectories/input_experimental.ini"

# %% Preprocess trajectory
logging.info("\n" + "="*70)
logging.info("PREPROCESSING TRAJECTORY")

x = [np.random.rand(10000) for _ in range(4)]  # Replace with actual data loading
y = [np.random.rand(10000) for _ in range(4)]
z = [np.random.rand(10000) for _ in range(4)]
bx = [np.random.rand(10000) for _ in range(4)]
by = [np.random.rand(10000) for _ in range(4)]
bz = [np.random.rand(10000) for _ in range(4)]
vx = [np.random.rand(10000) for _ in range(4)]
vy = [np.random.rand(10000) for _ in range(4)]
vz = [np.random.rand(10000) for _ in range(4)]
rho = [np.random.rand(10000) for _ in range(4)]
ppar = [np.random.rand(10000) for _ in range(4)]
pperp = [np.random.rand(10000) for _ in range(4)]

loader = ExperimentalTrajectoryDataLoader(verbose=True)
loader.load_config(config_file)
config = loader.run(
    x=x, y=y, z=z,
    bx=bx, by=by, bz=bz,
    vx=vx, vy=vy, vz=vz,
    rho=rho, ppar=ppar, pperp=pperp,
)
del loader

# Extract results
dic_datas = config['dic_datas']  # 1D extracted data
grid_param = config['grid_param']
traj_param = config['traj_param']
physical_param = config['physical_param']
max_workers = config['max_workers']
ltraj_list = traj_param['ltraj_list']

laws = config['laws']
terms = config['terms']
quantities = config['quantities']
method = config['method']
nbsatellite = traj_param['nbsatellite']

param_to_txt(grid_param, traj_param, physical_param, filename='result_traj/'+config['name_output'] + '_' + config['trajectory_name'] + "_" + method + '_sat' + str(nbsatellite) + "_parameters.txt")

quantities_computer = TrajectoryQuantitiesComputer(
    verbose=True,
    grid_param=grid_param,
    traj_param=traj_param,
    physical_param=physical_param,
)
dic_quantities = quantities_computer.extract_and_compute(
    dic_datas,
    laws=laws,
    terms=terms,
    quantities=quantities,
    method=method,
    filename='result_traj/'+config['name_output'] + '_' + config['trajectory_name'] + "_" + method + '_sat' + str(nbsatellite) + "_quantities.h5",
)
del quantities_computer

run_params = {'method': method, 'max_workers': max_workers}

# %% Compute quantities along trajectory
if traj_param['nbsatellite'] == 1:

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
        filename='result_traj/'+config['name_output'] + '_' + config['trajectory_name'] + "_" + method + '_sat' + str(nbsatellite) + "_terms.h5",
    )
    del terms_computer

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
        filename='result_traj/'+config['name_output'] + '_' + config['trajectory_name'] + "_" + method + '_sat' + str(nbsatellite) + "_laws.h5",
    )
    del laws_computer

    time_end = time.time()
    logging.info(f"Time taken to compute laws terms: {time_end - time_start:.2f} seconds")

elif traj_param['nbsatellite'] == 4:

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
        filename='result_traj/'+config['name_output'] + '_' + config['trajectory_name'] + "_" + method + '_sat' + str(nbsatellite) + "_terms.h5",
    )
    del terms_computer

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
        filename='result_traj/'+config['name_output'] + '_' + config['trajectory_name'] + "_" + method + '_sat' + str(nbsatellite) + "_laws.h5",
    )
    del laws_computer
    
    time_end = time.time()
    logging.info(f"Time taken to compute laws terms: {time_end - time_start:.2f} seconds")
    