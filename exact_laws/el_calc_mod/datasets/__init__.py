import h5py as h5
import numpy as np
from .dataset import Dataset
from ..grids.grid import Grid
from ..grids import load_grid_from_dict


def load(quantities={},grid=Grid(),params={}):  
    return Dataset(params,quantities,grid)      
        
def read_standard_file(filename):
    """extract contents of the standard file called filename"""
    with h5.File(filename, "r") as f:
        quantities = {k:np.ascontiguousarray(f[k]) for k in f.keys() if not "param" in k}
        laws = f['param/laws'][()]
        laws = [law.decode() for law in laws]
        terms = f['param/terms'][()]
        terms = [term.decode() for term in terms]
        grid_params = {k:np.ascontiguousarray(f[f'param/{k}']) for k in f["param"].keys() if k in ['axis','L', 'c', 'N']}
        params = {k:eval(str(f[f'param/{k}'][()])) for k in f["param"].keys() if k not in ['axis','L', 'c', 'N', 'laws', 'terms', 'quantities']}
    params["rho_mean"] = 1  # np.mean(np.sort(data['rho'].flatten()))
    return quantities, params, grid_params, laws, terms

def load_from_standard_file(filename):
    quantities, params, grid_params, laws, terms = read_standard_file(filename)
    dataset = load(quantities = quantities, params = params, grid = load_grid_from_dict(grid_params))
    return dataset, laws, terms

def record_incdataset_to_h5file(filename, dataset):
    with h5.File(filename, "w") as f:
        f.create_group("params")
        for key in dataset.params:
            if key != 'coeffs' and key != 'state':
                f["params"].create_dataset(key, data=dataset.params[key])
            elif key == 'coeffs': 
                f["params"].create_group("coeffs")
                for k in dataset.params['coeffs']:
                    ksuffix = k
                    if k.endswith('flux'): 
                        ksuffix = k[:-5]
                    if k.endswith('sources'): 
                        ksuffix = k[:-8]
                    for t in dataset.params['coeffs'][k]:
                        f["params"]['coeffs'].create_dataset(ksuffix+'_'+t, data=dataset.params['coeffs'][k][t])
            elif key == 'state': 
                f["params"].create_group("state")
                for k in dataset.params['state']:
                    f["params"]['state'].create_dataset(k, data=dataset.params['state'][k])
        for key in dataset.quantities:
            f.create_dataset(key, data=dataset.quantities[key])
        f.create_group("grid")
        for key in dataset.grid:
            if key != 'coords':
                f['grid'].create_dataset(key, data=dataset.grid[key])
            else: 
                f['grid'].create_group('coords')
                for k in dataset.grid['coords']:
                    f['grid']['coords'].create_dataset(k, data=dataset.grid['coords'][k])
        
        
