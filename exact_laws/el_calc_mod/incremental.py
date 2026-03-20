import logging
import numpy as np

from .datasets import load, record_incdataset_to_h5file
from .grids import load_outputgrid_from_incgrid
from .terms import TERMS
from .laws import LAWS
from .grids import div_on_incgrid, reorganise_quantities, reformat_grid_compatible_to_h5

def initialise_output_dataset(incremental_grid, original_dataset, laws, terms):
    logging.info("INIT Initialisation output data ")
    params = {}
    params['laws'] = laws
    params['terms'], params['coeffs'] = list_terms_and_coeffs(laws, terms, original_dataset.params)
    params['state'] = {'index': 0, 'list': 'prim'}
    nb_sec_by_dirr = 0
    for term in params['terms']:
        if term.startswith('flux'):
            nb_sec_by_dirr = 1
    grid = load_outputgrid_from_incgrid(coord=incremental_grid.kind.split('_')[0], incgrid=incremental_grid, nb_sec_by_dirr=nb_sec_by_dirr)
    quantities = init_ouput_quantities(grid.coords['listprim'], grid.coords['listsec'], params['terms'])
    output_dataset = load(quantities=quantities, grid=grid, params=params)
    logging.info("END Initialisation output data ")
    return output_dataset

def list_terms_and_coeffs(laws, terms, physical_params):
    list_terms = terms.copy()
    coeffs = {}
    for law in laws:
        terms_for_law, coeffs[law] = LAWS[law].terms_and_coeffs(physical_params)
        list_terms += terms_for_law
    return list(set(list_terms)), coeffs

def init_ouput_quantities(listprim, listsec, terms):
    output_quantities = {}
    Nprim = len(listprim)
    Nsec = len(listsec)
    for term in terms:
        if term.startswith('flux'):
            output_quantities[term] = [np.zeros((Nprim, 3)), np.zeros((Nsec, 3))]
        else:
            output_quantities[term] = [np.zeros((Nprim))]
    return output_quantities

def calc_terms(dataset, output_dataset, run_config, save, backup_setp=5000):
    # dataset contient au moins les quantités nécessaires au calcul
    # grid_prim = [Np][3] et grid_sec = [Ns][3] contiennent des vecteurs à calculer sous le format [x,y,z]
    # terms contient les noms des termes à calculer 
    # sortie : {[term_scalaire]:[Np],array([term_flux]):[array([Np,3]),array([Ns,3])]}

    terms = output_dataset.params['terms']
    output_quantities = output_dataset.quantities
    saved_list = output_dataset.params['state']['list']

    logging.info("INIT Calculation of terms")
    Ndat = dataset.grid.N

    if saved_list == 'prim':
        state_index = output_dataset.params['state']['index']
        vectors = [(index, vector) for index, vector in enumerate(output_dataset.grid.coords['listprim']) if
                   (index >= state_index) and ((index % run_config.size) == run_config.rank)]
        backup_ind = state_index
        for index, vector in vectors:
            if backup_ind == 0: 
                logging.info(f'BEG {backup_ind+1}/{len(vectors)} of listprim attributed to proc {run_config.rank}')
            for term in terms:
                output_quantities[term][0][index] = TERMS[term].calc(vector, Ndat,
                                                                                   **dataset.quantities)
            if backup_ind % backup_setp == 0: 
                logging.info(f'... END {backup_ind+1}/{len(vectors)} of listprim attributed to proc {run_config.rank}')
                output_dataset.params['state']['index'] = index + 1
                save(output_dataset, 'data_output', rank=f"{run_config.rank}")
            backup_ind +=1
        output_dataset.params['state']['index'] = 0
        saved_list = 'sec'

    if saved_list == 'sec':
        terms_flux = list(filter(lambda e: e.startswith('flux'), terms))
        state_index = output_dataset.params['state']['index']
        vectors = [(index, vector) for index, vector in enumerate(output_dataset.grid.coords['listsec']) if
                   (index >= state_index) and ((index % run_config.size) == run_config.rank)]
        backup_ind = state_index
        for index, vector in vectors:
            if backup_ind == 0: 
                logging.info(f'BEG {backup_ind+1}/{len(vectors)} of listsec attributed to proc {run_config.rank}')
            for term in terms_flux:
                output_quantities[term][1][index] = TERMS[term].calc(vector, Ndat,
                                                                                   **dataset.quantities)
            if backup_ind % backup_setp == 0: 
                logging.info(f'... END {backup_ind+1}/{len(vectors)} of listsec attributed to proc {run_config.rank}')
                output_dataset.params['state']['index'] = index + 1
                output_dataset.params['state']['list'] = 'sec'
                save(output_dataset, 'data_output', rank=f"{run_config.rank}")
            backup_ind +=1
    run_config.barrier()
    del (output_dataset.params['state'])
    logging.info("END Calculation of terms")
    
def reduction_output(quantities, run_config):
    logging.info("INIT Reduction output data")
    output = {}
    for k in quantities.keys():
        output[k] = run_config.reduce(quantities[k])
    logging.info("END Reduction output data")
    return output
    
def save_output_dataset_on_incgrid(output_filename, output_dataset, incremental_grid, original_dataset, coord):
    logging.info(f"INIT Record final result in {output_filename} ")
    nb_sec_by_dirr = 0
    for term in output_dataset.quantities:
        if term.startswith('flux'):
            nb_sec_by_dirr = 1
    incgrid_quantities = reorganise_quantities(coord =coord,  
                                               incgrid = incremental_grid, 
                                               output_grid = output_dataset.grid, 
                                               output_quantities = output_dataset.quantities,
                                               nb_sec_by_dirr = nb_sec_by_dirr)
    incgrid_h5 = reformat_grid_compatible_to_h5(coord=coord, incgrid= incremental_grid)
    params = original_dataset.params
    for k in output_dataset.params:
        params[k] = output_dataset.params[k]
    incgrid_dataset = load(quantities=incgrid_quantities, grid=incgrid_h5, params=params)
    record_incdataset_to_h5file(output_filename, incgrid_dataset)
    logging.info("END Record final result ")
   
def apply_method(original_dataset, incremental_grid, coord, laws, terms, output_filename, run_config, backup): 
    # Init Output_dataset 
    if backup.already:
        output_dataset = backup.download('data_output', rank=f"{run_config.rank}")
    else:
        output_dataset = initialise_output_dataset(incremental_grid, original_dataset, laws, terms)
        backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
    output_dataset.check('output_dataset')
    
    # ## CALCUL LOI EXACTE
    if 'state' in output_dataset.params:
        calc_terms(
            dataset=original_dataset,
            output_dataset=output_dataset,
            run_config=run_config,
            save=backup.save
        )
    backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
    
    # ## DIVERGENCE
    div_quantities = div_on_incgrid(coord, incremental_grid, output_dataset)
    for k in div_quantities.keys():
        output_dataset.quantities[k] = [div_quantities[k]]
    backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")

    # ## RASSEMBLEMENT
    output_dataset.quantities = reduction_output(output_dataset.quantities, run_config)
    if run_config.rank == 0: 
        backup.save(output_dataset, 'data_output_final')

    # ## Enregistrement données finales sur la grille incrémentale
    if run_config.rank == 0:
        save_output_dataset_on_incgrid(output_filename, output_dataset, incremental_grid, original_dataset, coord)
        
