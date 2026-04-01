# %% import libraries
import numpy as np
import logging
from datetime import datetime
from trajectory_preprocess import preprocess_trajectory_from_ini, trajectory_linear_x
from trajectory_quantities import extract_trajectory_and_compute
from trajectory_terms import compute_all_terms_for_laws
from trajectory_laws import compute_laws_terms_with_coefficients
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.gridspec import GridSpec
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
logging.info("="*70)

time_start = time.time()
results = preprocess_trajectory_from_ini(
    ini_file=config_file,
    trajectory_func=trajectory_linear_x,
    trajectory_kwargs={'y_pos': 100, 'z_pos': 100},
    verbose=True
)

# Extract results
config = results['config']
dic_datas = results['dic_datas']  # 1D extracted data
dic_param = results['dic_param']
trajectory = results['trajectory']

laws = config['laws']
terms = config['terms']
quantities = config['quantities']
physical_params = config['physical_params']
nbsatellite = config['nbsatellite']

dic_quantities = extract_trajectory_and_compute(
    results['dic_datas'], 
    dic_param=results['dic_param'],
    laws=laws,
    nbsatellite=results['config']['nbsatellite'],
    verbose=True
)


# %% Compute quantities along trajectory
if results['config']['nbsatellite'] == 1:

    dic_terms = compute_all_terms_for_laws(
        dic_quantities = dic_quantities, 
        dic_param=results['dic_param'], 
        laws=laws, 
        nbsatellite=results['config']['nbsatellite'],
        verbose=True)



    dic_law_terms, dic_law_coeff = compute_laws_terms_with_coefficients(
        dic_quantities=dic_quantities,
        dic_terms=dic_terms,
        laws=laws,
        dic_param=results['dic_param'],
        nbsatellite=results['config']['nbsatellite'],
        trajectory=trajectory,
        verbose=True
    ) 
    time_end = time.time()
    logging.info(f"Time taken to compute laws terms: {time_end - time_start:.2f} seconds")


    def linear_op_from_list_term(coeffs, quantities, list_term):
        coeff = {k.split('_', 1)[1]: coeffs[k] for k in coeffs if k in list_term}
        result = np.zeros(np.shape(quantities[list(coeff.keys())[0]]))
        for k in coeff.keys():
            result += coeff[k] * quantities[k]
        return result

    dic_of_list_terms = {}
    dic_of_list_terms['PP98'] = [k for k in dic_law_coeff.keys() if k.split('_', 1)[0] == 'PP98']

    results_plot = {}
    for law in dic_of_list_terms.keys():
        list_term = dic_of_list_terms[law]
        results_plot[law] = np.transpose(linear_op_from_list_term(dic_law_coeff, dic_law_terms, list_term))

    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e-10, base=10)
    plt.plot(dic_param['lx'], results_plot['PP98'], label='PP98')
    plt.xlabel('lx [di]')
    plt.ylabel(r'$epsilon_{PP98}$')
    plt.title('PP98 along trajectory')
    plt.title(f'Cut at y=100, z=100')
    plt.legend()
    plt.savefig("PP98_trajectory.png")

elif results['config']['nbsatellite'] == 4:

    dic_terms = compute_all_terms_for_laws(
        dic_quantities = dic_quantities, 
        dic_param=results['dic_param'], 
        laws=laws, 
        nbsatellite=results['config']['nbsatellite'],
        verbose=True)
    
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig)
    for sat in ['sat_0', 'sat_1', 'sat_2', 'sat_3']:
        ax = fig.add_subplot(gs[int(sat.split('_')[1])//2, int(sat.split('_')[1])%2])
        for term in dic_terms.keys():
            ax.plot(dic_param['lx'], dic_terms[term][sat], label=term)
        ax.set_xlabel('lx [di]')
        ax.set_ylabel(r'$epsilon_{PP98}$')
        ax.set_title(f"{sat} - Cut at y=100, z=100")
        ax.legend()
            
    plt.tight_layout()
    plt.savefig("terms_trajectory_4sat.png")