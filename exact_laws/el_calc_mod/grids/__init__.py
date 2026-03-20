from typing import Dict
from warnings import warn
import importlib
import logging
import numpy as np

from .grid import Grid


def load_grid(N=[], L=[], c=[], axis=[], coords={}):
    if len(N) == 0 and len(L) == 0 and len(c) == 0 and len(coords.keys()) == 0: 
        raise Exception('Impossible to creat an empty grid, add at least the axis or the coordinates argument or two between N, L and c')
    if len(axis) == 0 and (len(c) != 0 or len(N) != 0 or len(L) != 0) and len(coords.keys()) == 0:
        warn("WARNING: The names of the axis of the grid are not defined, it will be by default ['axis_0',...]"
             "If it is not your intention, add axis in the arguments.")
        nb_axis = len(N) * (len(N) != 0) + (len(N) == 0) * (len(L) * (len(L) != 0) + len(c) * (len(L) == 0))    
        axis = [f"axis_{i}" for i in range(nb_axis)]
    if len(axis) == 0 and len(N) == 0 and len(coords.keys()) != 0 : 
        axis = [k for k in coords.keys() if len(coords[k]) != 0]
        N = [len(coords[k]) for k in coords.keys() if len(coords[k]) != 0]
    if len(c) == 0 and len(N) != 0 and len(L) != 0:
        warn('WARNING: The grid is only defined with N and L, c is by default computed as L/(N-1). If it is not your intention, add c in the arguments.')
        c = [L[i]/(N[i]-1) for i in range(len(N))]
    return Grid(N = N, L = L, c = c, axis = axis, coords = coords) 

def load_grid_from_dict(dict: Dict):
        return Grid(N=dict['N'].astype(int),L=dict['L'],c=dict['c'])   

def load_incgrid_from_grid(coord, **kargs):
    """
    Args:
        coord (str): nom du module (type de grille)
        if coord is logcyl, kargs must contain at least: 
            - original_grid (Grid object with attributes N, L and c)
            - Nmax_scale (int)
            - Nmax_list (int) 
            - kind (str) : cls or rdm
    Returns:
        IncGrid object
    """
    mod = importlib.import_module(f"exact_laws.el_calc_mod.grids.{coord}", "*")
    return mod.load(**kargs)

def load_outputgrid_from_incgrid(coord, incgrid, nb_sec_by_dirr=1):
    """
    Args:
        coord (str): nom du module (type de grille)
        incgrid (IncGrid object)
        nb_sec_by_dirr (int) : 0, 1, 2 
    Returns:
        Grid object that contains list of coordinates
    """
    mod = importlib.import_module(f"exact_laws.el_calc_mod.grids.{coord}", "*")
    return mod.load_outputgrid( incgrid,  nb_sec_by_dirr)
    
def div_on_incgrid(coord, incgrid, dataset_terms):
    logging.info("INIT Calculation of the divergence")
    mod = importlib.import_module(f"exact_laws.el_calc_mod.grids.{coord}", "*")
    output = mod.div(incgrid, dataset_terms)
    logging.info("END Calculation of the divergence")
    return output

def reorganise_quantities(coord, incgrid, output_grid, output_quantities, nb_sec_by_dirr=1):
    mod = importlib.import_module(f"exact_laws.el_calc_mod.grids.{coord}", "*")
    return mod.reorganise_quantities(output_quantities, incgrid, output_grid,  nb_sec_by_dirr)

def reformat_grid_compatible_to_h5(coord, incgrid):
    mod = importlib.import_module(f"exact_laws.el_calc_mod.grids.{coord}", "*")
    return mod.reformat_grid_compatible_to_h5(incgrid)
