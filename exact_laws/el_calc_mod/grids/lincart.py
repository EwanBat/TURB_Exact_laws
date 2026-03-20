import numpy as np
from .incgrid import IncGrid
from ...mathematical_tools.derivation import div as mtdiv

def load(original_grid, kind, **kargs):
    N = [original_grid.N[0],original_grid.N[1],int(original_grid.N[2]/2)]
    grid = {} 
    grid['lx'] = np.arange(0,N[0])-int(N[0]/2)
    grid['ly'] = np.arange(0,N[1])-int(N[1]/2)
    grid['lz'] = np.arange(0,int(N[2]))
    return IncGrid(original_grid=original_grid, N=N, axis=['lx','ly','lz'], coords=grid, kind=kind, coord='lincart')
    
def load_outputgrid(incgrid, *args):
    """
    Args:
        incgrid (IncGrid object)
    Returns:
        Grid object that contains list of coordinates
    """   
    grid = incgrid.spatial_grid
    grid.axis = ['lx','ly','lz']
    return grid

def div(incgrid, dataset_terms):
    c = incgrid.spatial_grid.c
    output = {}
    for t in dataset_terms.quantities:
        if t.startswith("flux") :
            output["div_" + t] = mtdiv(dataset_terms.quantities[t],c)
    return output

def reorganise_quantities(output_quantities, incgrid, *args):
    output = {}
    
    N = incgrid.N

    list_flux = [k for k in output_quantities if k.startswith("flux")]
    for t in list_flux:
        output[t] = [np.roll(output_quantities[t][i],(int(N[0]/2),int(N[1]/2)),axis=(0,1))[:,:,:N[2]] for i in range(len(output_quantities[t]))]

    list_other = [k for k in output_quantities if not k.startswith("flux")]
    for t in list_other:
        output[t] = np.roll(output_quantities[t],(int(N[0]/2),int(N[1]/2)),axis=(0,1))[:,:,:N[2]]

    return output

def reformat_grid_compatible_to_h5(incgrid):
    output = {}
    output['inc_axis'] = incgrid.axis
    output['inc_N'] = incgrid.N
    output['kind'] = incgrid.kind
    output['axis'] = incgrid.spatial_grid.axis
    output['N'] = incgrid.spatial_grid.N
    output['L'] = incgrid.spatial_grid.L
    output['c'] = incgrid.spatial_grid.c
    output['coords'] = {k:incgrid.coords[k] for k in incgrid.coords }
    return output
