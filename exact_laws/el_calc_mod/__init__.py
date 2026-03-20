import logging
import configparser
import importlib
import os

from .datasets import load_from_standard_file
from .grids import load_incgrid_from_grid
from ..preprocessing import process_on_standard_h5_file


def initialise_original_dataset(input_filename):
    logging.info("INIT Initialisation original data")
    original_dataset, laws, terms = load_from_standard_file(input_filename)
    logging.info("END Initialisation original data ")
    return original_dataset, laws, terms


def calc_exact_laws_from_config(config_file, run_config, backup):
    '''
    config_file example:
        [INPUT_DATA]
        path = ../../data_process/TestEff_CGL2/
        name = OCA_CGL2_cycle0_TestEff_bin2

        [OUTPUT_DATA]
        path = ./
        name = EL_logcyl40_cls100

        [RUN_PARAMS]
        config = NOP
        numbap = O
        nblayer = 8
        nbbuf = 4
        backup = None

        [GRID_PARAMS]
        Nmax_scale = 40
        Nmax_list = 100
        kind = cls
        coord = logcyl
        
        [METHOD_PARAMS]
        method = fourier ou incremental
        multifile = False
    '''

    config = configparser.ConfigParser()
    config.read(config_file)

    # translate information useful for the computation
    input_filename = f"{config['INPUT_DATA']['path']}/{config['INPUT_DATA']['name']}.h5"
    output_filename = f"{config['OUTPUT_DATA']['path']}/{config['INPUT_DATA']['name']}_{config['OUTPUT_DATA']['name']}.h5"
    output_filename_beg = f"{config['OUTPUT_DATA']['path']}/{config['INPUT_DATA']['name']}"
    output_filename_end = f"_{config['OUTPUT_DATA']['name']}.h5"
    method = config['METHOD_PARAMS']['method']
    if config['METHOD_PARAMS']['multifile']:
        multifile = eval(config['METHOD_PARAMS']['multifile'])
    else : multifile = False
    mod = importlib.import_module(f"exact_laws.el_calc_mod.{method}", "*")
    
    message = (
        f"Begin calc exact law with config:"
        f"\n\t - input_file: {input_filename}"
        f"\n\t - output_file: {output_filename}"
        f"\n\t - method: {method} on {os.cpu_count()} cpu"
    )
    logging.info(message)
    
    # Init Original_dataset
    #if backup.already:
    #    original_dataset = backup.download('data_origin')
    #else:
    #    original_dataset, laws, terms = initialise_original_dataset(input_filename)
    #    if run_config.rank == 0:
    #        backup.save(original_dataset, 'data_origin')
    original_dataset, laws, terms = initialise_original_dataset(input_filename)
    original_dataset.check('original_dataset')
    
    # Init Incremental_grid
    input_grid = {}
    input_grid["coord"] = config['GRID_PARAMS']["coord"]
    input_grid["kind"] = config['GRID_PARAMS']["kind"]
    if method == 'incremental':
        input_grid["Nmax_scale"] = int(eval(config['GRID_PARAMS']["Nmax_scale"]))
        input_grid["Nmax_list"] = int(eval(config['GRID_PARAMS']["Nmax_list"]))

    if backup.already:
        incremental_grid = backup.download('inc_grid')
    else:
        incremental_grid = load_incgrid_from_grid(original_grid=original_dataset.grid, **input_grid)
        if run_config.rank == 0:
            backup.save(incremental_grid, 'inc_grid')
    incremental_grid.check('incremental_grid')

    # ## CALCUL LOI EXACTE
    if multifile:
        distrib = multifile_distrib(laws,terms)
        filenames = []
        for k in distrib:
            logging.info(f'INIT apply method for {k}')
            filename = output_filename_beg+k+output_filename_end
            tag = mod.apply_method(original_dataset, incremental_grid, input_grid["coord"], 
                             distrib[k]['laws'], distrib[k]['terms'], filename, 
                             run_config, backup)
            if tag :  filenames.append(filename)
            logging.info(f'END apply method for {k}')
        logging.info("INIT reduction 3D to 2D ")
        if run_config.rank == 0:
            mod.red3Dto2D_multifile(output_filename,filenames,incremental_grid)
        logging.info("END reduction 3D to 2D")
    else :
        mod.apply_method(original_dataset, incremental_grid, input_grid["coord"], 
                         laws, terms, output_filename, run_config, backup)
        logging.info("INIT reduction 3D to 2D ")
        if run_config.rank == 0:
            mod.red3Dto2D(output_filename,incremental_grid)
        logging.info("END reduction 3D to 2D")
    
    if run_config.rank == 0:
        if not multifile:
            process_on_standard_h5_file.check_file(output_filename)
        
def multifile_distrib(laws,terms):
    set_laws = set(laws)
    set_terms = set(terms)
    out = {}
    
    out['_inc'] = {}
    inc_laws = {'PP98','BG17','ISS22Cgl','ISS22Gyr','ISS22Iso'}
    out['_inc']['laws'] = list(set_laws.intersection(inc_laws))
    set_laws = set_laws - inc_laws
    inc_terms = {'source_dpantr','forc_vinc'} 
    out['_inc']['terms'] = list(set_terms.intersection(inc_terms))
    set_terms = set_terms - inc_terms
    
    comp_laws = set_laws.intersection({'SS22Gyr','SS22Iso','SS22Cgl','SS22Pol'})
    
    out['_ss22f'] = {}
    ss22f_laws = {'SS22Gyr_flux','SS22Iso_flux','SS22Cgl_flux','SS22Pol_flux'}
    out['_ss22f']['laws'] = list(set(list(set_laws.intersection(ss22f_laws)) + [f'{s}_flux' for s in comp_laws]))
    set_laws = set_laws - ss22f_laws
    out['_ss22f']['terms'] = []
    
    out['_ss22s'] = {}
    ss22s_laws = {'SS22Gyr_sources','SS22Iso_sources','SS22Cgl_sources','SS22Pol_sources'}
    out['_ss22s']['laws'] = list(set(list(set_laws.intersection(ss22s_laws)) + [f'{s}_sources' for s in comp_laws]))
    set_laws = set_laws - ss22s_laws
    out['_ss22s']['terms'] = []
    
    set_laws = set_laws - comp_laws
    
    out['_hall'] = {}
    hall_laws = {'IHallcor','Hallcor'}
    out['_hall']['laws'] = list(set_laws.intersection(hall_laws))
    set_laws = set_laws - hall_laws
    out['_hall']['terms'] = []
    
    out['_other'] = {}
    out['_other']['laws'] = list(set_laws)
    out['_other']['terms'] = list(set_terms)
    
    return out