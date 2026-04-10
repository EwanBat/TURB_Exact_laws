# trajectory_terms.py
"""
Module to compute terms along a trajectory.
Analog to trajectory_quantities.py but for terms.
Uses calc_fourier() methods from terms for trajectories.
"""

import numpy as np
import logging
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS
logger = logging.getLogger(__name__)


# Mapping of abstract variables to their concrete components
VARIABLE_COMPONENTS = {
    # === Raw data ===
    'v': ['vx', 'vy', 'vz'],
    'Iv': ['Ivx', 'Ivy', 'Ivz'],
    'b': ['bx', 'by', 'bz'],
    'Ib': ['Ibx', 'Iby', 'Ibz'],
    'w': ['wx', 'wy', 'wz'],  # Vitesse de vent (compressible)
    'j': ['jx', 'jy', 'jz'],  # Courant
    'Ij': ['Ijx', 'Ijy', 'Ijz'],  # Courant incompressible
    'f': ['fx', 'fy', 'fz'],  # Force
    
    # === Scalaires directs ===
    'rho': ['rho'],
    'Irho': ['Irho'],
    'pm': ['pm'],  # Magnetic pressure
    'Ipm': ['Ipm'],  # Incompressible magnetic pressure
    'pgyr': ['pgyr'],  # Gyrotropic pressure
    'Ipgyr': ['Ipgyr'],  # Incompressible gyrotropic pressure
    'piso': ['piso'],  # Isotropic pressure
    'Ipiso': ['Ipiso'],  # Incompressible isotropic pressure
    'ppol': ['ppol'],  # Poloidal pressure
    'Ippol': ['Ippol'],  # Incompressible poloidal pressure
    'pcgl': ['pcgl'],  # CGL pressure
    'Ipcgl': ['Ipcgl'],  # Incompressible CGL pressure
    'ugyr': ['ugyr'],  # Gyrotropic velocity
    'Iugyr': ['Iugyr'],  # Incompressible gyrotropic velocity
    'uiso': ['uiso'],  # Isotropic velocity
    'Iuiso': ['Iuiso'],  # Incompressible isotropic velocity
    'upol': ['upol'],  # Poloidal velocity
    'Iupol': ['Iupol'],  # Incompressible poloidal velocity
    'ucgl': ['ucgl'],  # CGL velocity
    'Iucgl': ['Iucgl'],  # Incompressible CGL velocity
    
    # === Divergences ===
    'divv': ['divvx', 'divvy', 'divvz'],  # Velocity divergence
    'divb': ['divbx', 'divby', 'divbz'],  # Magnetic field divergence
    'divj': ['divjx', 'divjy', 'divjz'],  # Current divergence
    
    # === Gradients ===
    'gradrho': ['dxrho', 'dyrho', 'dzrho'],  # Density gradient
    'gradv': ['grad_v_x', 'grad_v_y', 'grad_v_z'],  # Velocity gradient
    'graduiso': ['grad_uiso_x', 'grad_uiso_y', 'grad_uiso_z'],  # Isotropic velocity gradient
    'gradupol': ['grad_upol_x', 'grad_upol_y', 'grad_upol_z'],  # Poloidal velocity gradient
    
    # === Hyperdissipation ===
    'hdk': ['hdkx', 'hdky', 'hdkz'],  # Kinetic hyperdissipation
    'hdm': ['hdmx', 'hdmy', 'hdmz'],  # Magnetic hyperdissipation
    'hdk2': ['hdk2x', 'hdk2y', 'hdk2z'],  # Kinetic hyperdissipation order 2
}


def _prepare_dic_param_for_terms_and_coeffs(dic_param, nbsatellite=1):
    """
    Prepare dic_param for terms_and_coeffs() by converting list-based parameters to scalars.
    
    Parameters:
    -----------
    dic_param : dict
        Dictionary with potentially list-based parameters (one value per trajectory)
    nbsatellite : int
        Number of satellites
    
    Returns:
    -------
    dict : Cleaned dictionary with scalar parameters
    """
    params_clean = {}
    
    for key, value in dic_param.items():
        if isinstance(value, list):
            # Extract first element from list (same parameter for all trajectories)
            params_clean[key] = value[0]
        elif isinstance(value, dict):
            # For nbsatellite=4, extract first satellite and first trajectory
            if 'sat_0' in value:
                first_sat_value = value['sat_0']
                if isinstance(first_sat_value, list):
                    params_clean[key] = first_sat_value[0]
                else:
                    params_clean[key] = first_sat_value
            else:
                params_clean[key] = value
        else:
            params_clean[key] = value
    
    return params_clean


def list_required_terms(laws=None, physical_param=None, nbsatellite=1):
    """
    Returns the list of required terms to compute the given laws.
    
    Parameters:
    -----------
    laws : list[str]
        List of law names
    dic_param : dict
        Simulation parameters
    nbsatellite : int
        Number of satellites (1 or 4)
    
    Returns:
    -------
    set : Set of required terms
    """
    if laws is None:
        laws = []
    
    terms = set()
    
    if not laws:
        return terms
    
    # Prepare parameters for terms_and_coeffs (convert lists to scalars)
    params_clean = _prepare_dic_param_for_terms_and_coeffs(physical_param, nbsatellite)
    
    # Add terms from laws
    for law_name in laws:
        if law_name in LAWS:
            law_obj = LAWS[law_name]
            # terms_and_coeffs() returns (terms_list, coeffs_dict)
            law_terms, _ = law_obj.terms_and_coeffs(params_clean)
            terms.update(law_terms)
    
    return terms

def get_concrete_variables_from_abstract(abstract_vars, dic_quant):
    """
    Convert abstract variables to concrete components.
    
    Parameters:
    -----------
    abstract_vars : list[str]
        List of abstract variables (ex: ['v', 'b'])
    dic_quant : dict
        Dictionary containing the data
    
    Returns:
    -------
    list : List of np.ndarray corresponding to concrete variables
    """
    concrete_data = []
    for var in abstract_vars:
        components = VARIABLE_COMPONENTS.get(var, [var])  # Default to var itself
        for comp in components:
            if comp not in dic_quant:
                raise ValueError(f"Component '{comp}' (from '{var}') not found")
            concrete_data.append(dic_quant[comp])
    return concrete_data


def compute_term_from_TERMS(term_name, dic_quant, physical_param, nbsatellite=1, verbose=False):
    """
    Compute a single term using the calc_fourier method from TERMS.
    
    Adapted for 1D trajectory data. Handles single satellite or 4-satellite 
    formations by extracting satellite-specific data before computation.
    
    Parameters:
    -----------
    term_name : str
        Name of the term to compute (e.g., "flux_dvdvdv", "bg17_vwv")
    dic_quant : dict
        Dictionary of 1D data (trajectory) with computed quantities
    physical_param : dict
        Physical parameters
    nbsatellite : int
        Number of satellites (1 or 4)
    verbose : bool
        Logging flag (normally False to avoid spam in loops)
    
    Returns:
    -------
    np.ndarray or dict
        Computed term value(s)
    """
    
    if term_name not in TERMS:
        raise ValueError(f"Term '{term_name}' not found in TERMS")
    
    term_obj = TERMS[term_name]
    
    try:
        # Get abstract variables required by the term
        abstract_vars = term_obj.variables()
        
        if nbsatellite == 1:
            # Convert to concrete components
            args = get_concrete_variables_from_abstract(abstract_vars, dic_quant)
            # Call calc_fourier for 1D data
            result = term_obj.calc_fourier(*args, dic_param=physical_param, traj=True)
            if type(result) != np.ndarray:
                result = np.asarray(result)
        elif nbsatellite == 4:
            result = {}
            for sat_name in ['sat_0', 'sat_1', 'sat_2', 'sat_3']:
                # Extract data for this satellite
                dic_quant_sat = {}
                dic_param_sat = {}
                for key, value in dic_quant.items():
                    dic_quant_sat[key] = value[sat_name] if isinstance(value, dict) and sat_name in value else value
                for key, value in physical_param.items():
                    dic_param_sat[key] = value[sat_name] if isinstance(value, dict) and sat_name in value else value
                
                # Convert abstract variables to concrete components for this satellite
                args_sat = get_concrete_variables_from_abstract(abstract_vars, dic_quant_sat)
                
                # Compute term for this satellite
                result_sat = term_obj.calc_fourier(*args_sat, dic_param=dic_param_sat, traj=True)
                if type(result_sat) != np.ndarray:
                    result_sat = np.array(result_sat)
                result[sat_name] = result_sat
        else:
            raise ValueError(f"Unsupported number of satellites: {nbsatellite}")
        
    except Exception as e:
        if verbose:
            logger.error(f"Failed to compute {term_name}: {e}")
        raise
    
    return result


def compute_all_terms_for_laws(dic_quantities = None, traj_param = None, physical_param = None, laws=None, verbose=False):
    """
    Compute all terms required for the given laws.
    
    Determines the set of terms needed from law specifications, then computes
    them from the provided quantities. Accumulated per-trajectory if verbose=False.
    
    Parameters:
    -----------
    dic_quantities : dict
        Dictionary of 1D or 3D computed quantities
    traj_param : dict
        Trajectory parameters (nbsatellite, n_trajectories, etc.)
    physical_param : dict
        Physical parameters
    laws : list[str]
        List of law names to compute terms for
    verbose : bool
        If True, logs detailed information (normally False to avoid spam)
    
    Returns:
    -------
    dict : Dictionary of computed terms with multi-trajectory structure
    """
    
    nbsatellite = traj_param.get('nbsatellite', 1)

    if laws is None:
        laws = []
    
    # Get required terms
    required_terms = list_required_terms(laws, physical_param=physical_param, nbsatellite=nbsatellite)
    
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("FLUX AND SOURCE TERMS COMPUTATION")
        logging.info(f"  Computing {len(required_terms)} terms for {len(laws)} law(s)")
    
    result = {}
    
    for term_name in required_terms:
        try:
            computed = compute_term_from_TERMS(term_name, 
                    dic_quantities, 
                    physical_param=physical_param, 
                    nbsatellite=nbsatellite, 
                    verbose=False)  # Disable per-term logs to avoid spam
            
            result[term_name] = computed
        except Exception as e:
            if verbose:
                logger.error(f"Failed to compute {term_name}: {str(e)}")
    
    if verbose:
        logging.info(f"  [OK] All {len(result)} terms computed successfully")

    return result

def terms_to_h5(result_terms, filename="terms_trajectory.h5"):
    """
    Save computed terms to an HDF5 file.
    
    Parameters:
    -----------
    result_terms : dict
        Dictionary of computed terms (potentially with satellite sub-dictionaries)
    filename : str
        Output filename for the HDF5 file
    """
    import h5py
    
    with h5py.File(filename, 'w') as f:
        for term_name, term_value in result_terms.items():
            if isinstance(term_value, dict):  # Multiple satellites
                group = f.create_group(term_name)
                for sat_name, sat_value in term_value.items():
                    group.create_dataset(sat_name, data=sat_value)
            else:  # Single satellite
                f.create_dataset(term_name, data=term_value)