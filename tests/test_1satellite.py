# %% import libraries
import numpy as np
import logging
from datetime import datetime
from trajectory_preprocess import preprocess_trajectory_from_ini, trajectory_linear_x
from trajectory_quantities import extract_trajectory_and_compute
import matplotlib.pyplot as plt
import matplotlib.scale as mscale


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

results = preprocess_trajectory_from_ini(
    ini_file=config_file,
    trajectory_func=trajectory_linear_x,
    trajectory_kwargs={'y_pos': 100, 'z_pos': 100},
    verbose=True
)

# Extract results
config = results['config']
dic_datas = results['dic_datas']  # Données 1D extraites
dic_param = results['dic_param']
trajectory = results['trajectory']

laws = config['laws']
terms = config['terms']
quantities = config['quantities']
physical_params = config['physical_params']
nbsatellite = config['nbsatellite']

dic_quantities = extract_trajectory_and_compute(
    results['dic_datas'], 
    y_pos=100, z_pos=100,
    dic_param=results['dic_param'],
    laws=laws,
    nbsatellite=results['config']['nbsatellite'],
    verbose=True
)


# %% Compute quantities along trajectory
if results['config']['nbsatellite'] == 1:
    from trajectory_terms import compute_all_terms_for_laws

    dic_terms = compute_all_terms_for_laws(
        dic_quantities = dic_quantities, 
        dic_param=results['dic_param'], 
        laws=laws, 
        nbsatellite=results['config']['nbsatellite'],
        verbose=True)


    from trajectory_laws import compute_laws_terms_with_coefficients

    dic_law_terms, dic_law_coeff = compute_laws_terms_with_coefficients(
        dic_quantities=dic_quantities,
        dic_terms=dic_terms,
        laws=laws,
        dic_param=results['dic_param'],
        nbsatellite=results['config']['nbsatellite'],
        trajectory=trajectory,
        verbose=True
    )  

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
    print(dic_quantities.keys())
    print(dic_quantities['vx'].keys())
