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
from trajectory_quantities import list_computable_quantities

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


def list_required_terms(laws=None, dic_param=None):
    """
    Returns the list of required terms to compute the given laws.
    
    Parameters:
    -----------
    laws : list[str]
        List of law names
    
    Returns:
    -------
    set : Set of required terms
    """
    if laws is None:
        laws = []
    
    terms = set()
    
    # Add terms from laws
    if laws:
        for law_name in laws:
            if law_name in LAWS:
                law_obj = LAWS[law_name]
                # terms_and_coeffs() returns (terms_list, coeffs_dict)
                # Pass empty parameters for this step
                law_terms, _ = law_obj.terms_and_coeffs(dic_param)
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
        if var in VARIABLE_COMPONENTS:
            components = VARIABLE_COMPONENTS[var]
            for comp in components:
                if comp not in dic_quant:
                    raise ValueError(f"Component '{comp}' (from variable '{var}') not found in data")
                concrete_data.append(dic_quant[comp])
        else:
            # Variable not in mapping, search directly
            if var not in dic_quant:
                raise ValueError(f"Variable '{var}' not found in data and not in VARIABLE_COMPONENTS mapping")
            concrete_data.append(dic_quant[var])
    return concrete_data


def compute_term_from_TERMS(term_name, dic_quant, dic_param, nbsatellite=1, verbose=False):
    """
    Compute a term using the calc_fourier method from TERMS.
    Adapted for 1D trajectory data.
    
    Parameters:
    -----------
    term_name : str
        Name of the term (ex: "flux_dvdvdv", "bg17_vwv")
    dic_quant : dict
        Dictionary of 1D data (trajectory) with computed quantities
    dic_param : dict
        Simulation parameters
    verbose : bool
    
    Returns:
    -------
    np.ndarray : Value of the computed term (always an array)
    """
    
    if verbose:
        logger.info(f"Computing term {term_name}...")
    
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
            result = term_obj.calc_fourier(*args, dic_param=dic_param, traj=True)
            if type(result) != np.ndarray:
                result = np.array(result)
        elif nbsatellite == 4:
            result = {}
            for sat_name in ['sat_0', 'sat_1', 'sat_2', 'sat_3']:
                # Extract data for this satellite
                dic_quant_sat = {}
                dic_param_sat = {}
                for key, value in dic_quant.items():
                    dic_quant_sat[key] = value[sat_name] if isinstance(value, dict) and sat_name in value else value
                for key, value in dic_param.items():
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
    
    if verbose:
        logger.info(f"Term {term_name} computed")
    
    return result


def compute_all_terms_for_laws(dic_quantities = None, dic_param = None, laws=None, nbsatellite=1, verbose=False):
    """
    Compute all terms required for the given laws.
    Derived quantities are computed automatically before terms.
    
    Parameters:
    -----------
    dic_quant : dict
        Dictionary of 1D or 3D data
    dic_param : dict
        Simulation parameters
    laws : list[str]
        List of laws
    nbsatellite : int
        Number of satellites
    verbose : bool
    
    Returns:
    -------
    dict : Dictionary of computed terms
    """
    
    if laws is None:
        laws = []
    
    # Get required terms
    required_terms = list_required_terms(laws, dic_param=dic_param)
    
    if verbose:
        logger.info("\n" + "-"*70)
        logger.info("FLUX AND SOURCE TERMS COMPUTATION ALONG TRAJECTORY")
        logger.info(f"Computing {len(required_terms)} terms")
        logger.info(f"Required terms: {required_terms}")
    
    result = {}
    
    for term_name in required_terms:
        try:
            computed = compute_term_from_TERMS(term_name, 
                    dic_quantities, 
                    dic_param, 
                    nbsatellite=nbsatellite, 
                    verbose=verbose)
            
            result[term_name] = computed
        except Exception as e:
            if verbose:
                logger.error(f"Failed to compute {term_name}: {str(e)}")
    
    return result

def display_results(dic_terms, title="Results along trajectory"):
    """
    Display results of terms along a trajectory.
    
    Parameters:
    -----------
    dic_terms : dict
        Dictionary of terms
    title : str
        Display title
    """
    logger.info(f"\n{title}")
    logger.info("-" * 70)
    
    for key in sorted(dic_terms.keys()):
        value = dic_terms[key]
        if isinstance(value, np.ndarray) and value.ndim == 1:
            logger.info(f"  {key:30s}: min={value.min():12.6e} | max={value.max():12.6e} | mean={value.mean():12.6e}")
        elif isinstance(value, np.ndarray):
            logger.info(f"  {key:30s}: shape={value.shape} | mean={value.mean():12.6e}")
        elif isinstance(value, (int, float, np.number)):
            logger.info(f"  {key:30s}: {value:15.6e}")
        else:
            logger.info(f"  {key:30s}: {value}")