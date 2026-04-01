# trajectory_quantities.py - VERSION WITH GRADIENT SUPPORT
"""
Module to compute non-derivative quantities along a trajectory.
Analog to process_on_oca_files.py but adapted for trajectories and using QUANTITIES.
Quantities (v or Iv, etc.) are determined by the required variables from laws.
Support for gradient and divergence with 4-satellite formations.
"""

import numpy as np
import numexpr as ne
import configparser
import logging
from pathlib import Path
from exact_laws.preprocessing.quantities import QUANTITIES
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS

logger = logging.getLogger(__name__)

# ========== MAPPING OF QUANTITIES AND THEIR DEPENDENCIES ==========

QUANTITY_DEPENDENCIES = {
    # === Raw data ===
    "v": {"requires": ["vx", "vy", "vz"]},
    "Iv": {"requires": ["Ivx", "Ivy", "Ivz"]},
    "rho": {"requires": ["rho"]},
    "Irho": {"requires": ["Irho"]},
    "b": {"requires": ["bx", "by", "bz"]},
    "Ib": {"requires": ["bx", "by", "bz"]},
    
    # === Velocity ===
    "v2": {"requires": ["vx", "vy", "vz"]},
    "Iv2": {"requires": ["Ivx", "Ivy", "Ivz"]},
    "vnorm": {"requires": ["vx", "vy", "vz"]},
    "Ivnorm": {"requires": ["Ivx", "Ivy", "Ivz"]},
    
    # === Magnetic field ===
    "bnorm": {"requires": ["bx", "by", "bz"]},
    "Ibnorm": {"requires": ["bx", "by", "bz"]},
    "pm": {"requires": ["bx", "by", "bz"]},
    "Ipm": {"requires": ["bx", "by", "bz"]},
    
    # === Pressures ===
    "pgyr": {"requires": ["pperp", "rho"]},
    "Ipgyr": {"requires": ["pperp"]},
    "piso": {"requires": ["ppar", "pperp"]},
    "Ipiso": {"requires": ["ppar", "pperp"]},
    "ppol": {"requires": ["pperp"]},
    "Ippol": {"requires": ["pperp"]},
    "pcgl": {"requires": ["bx", "by", "bz", "rho", "ppar", "pperp"]},
    "Ipcgl": {"requires": ["bx", "by", "bz", "ppar", "pperp"]},
    
    # === Pressure-derived velocities ===
    "ugyr": {"requires": ["pperp", "rho"]},
    "Iugyr": {"requires": ["pperp"]},
    "uiso": {"requires": ["ppar", "pperp", "rho"]},
    "Iuiso": {"requires": ["ppar", "pperp"]},
    "ucgl": {"requires": ["bx", "by", "bz", "rho", "ppar", "pperp"]},
    "Iucgl": {"requires": ["bx", "by", "bz", "ppar", "pperp"]},
    "upol": {"requires": ["ppar", "pperp", "rho"]},
    "Iupol": {"requires": ["pperp"]},
}

# Quantities with gradients and divergences and their dependencies
GRADIENT_QUANTITIES = {
    # === Spatial gradients (require 4 satellites in trajectory) ===
    'gradv': {'requires': ['vx', 'vy', 'vz']},
    'gradv2': {'requires': ['v2']},  # v2 computed from vx, vy, vz
    'gradrho': {'requires': ['rho']},
    'graduiso': {'requires': ['uiso']},  # uiso computed from ppar, pperp, rho
    'gradupol': {'requires': ['upol']},  # upol computed from ppar, pperp, rho
    'gradugyr': {'requires': ['ugyr']},  # ugyr computed from pperp, rho
    'gradpcgl': {'requires': ['pcgl']},  # pcgl computed from bx, by, bz, rho, ppar, pperp
    
    # === Divergences and vector rotations (require grid) ===
    'divv': {'requires': ['vx', 'vy', 'vz']},
    'divb': {'requires': ['bx', 'by', 'bz']},
    'divj': {'requires': ['bx', 'by', 'bz']},  # j (current) computed from rot(b)
    
    # === Incompressible gradient versions ===
    'Igradv': {'requires': ['Ivx', 'Ivy', 'Ivz']},
    'Igradv2': {'requires': ['Iv2']},
    'Igradrho': {'requires': ['Irho']},
    'Igraduiso': {'requires': ['Iuiso']},
    'Igradupol': {'requires': ['Iupol']},
    'Idivj': {'requires': ['bx', 'by', 'bz']},
    
    # === Derived quantities (using derivation functions) ===
    # Current j = rot(b)
    'j': {'requires': ['bx', 'by', 'bz']},
    'Ij': {'requires': ['bx', 'by', 'bz']},
    
    # Vorticity w = rot(v)
    'w': {'requires': ['vx', 'vy', 'vz']},
    'Iw': {'requires': ['Ivx', 'Ivy', 'Ivz']},
    
    # Force f (derivative of Langevin force)
    'f': {'requires': ['fp', 'fm']},
    'If': {'requires': ['fp', 'fm']},
    
    # Kinetic hydrodynamics - 4th order (4 x Laplacian)
    'hdk': {'requires': ['vx', 'vy', 'vz']},
    'Ihdk': {'requires': ['Ivx', 'Ivy', 'Ivz']},
    
    # Kinetic hydrodynamics - 6th order (4 x Laplacian2)
    'hdk2': {'requires': ['vx', 'vy', 'vz']},
    'Ihdk2': {'requires': ['Ivx', 'Ivy', 'Ivz']},
    
    # Magnetic hydrodynamics (4 x Laplacian)
    'hdm': {'requires': ['bx', 'by', 'bz', 'rho']},
    'Ihdm': {'requires': ['bx', 'by', 'bz']},
}

def list_computable_quantities(dic_quant, laws=None, terms=None, quantities=None, nbsatellite=1):
    """
    List computable quantities from data and requirements.
    
    Combines extraction of requirements (laws/terms) with availability check 
    based on data and nbsatellite.
    
    Parameters:
    -----------
    dic_quant : dict
        Dictionary of available data
    laws, terms, quantities : list, optional
        Requirement specifications
    nbsatellite : int
        Number of satellites (filters gradients if < 4)
    
    Returns:
    -------
    list : Computable quantities
    """
    # === STEP 1: Extract requirements ===
    if quantities is None:
        quantities = []
    else:
        quantities = list(quantities)
    
    # Add requirements from terms
    if terms:
        for term_name in terms:
            if term_name in TERMS:
                term_variables = TERMS[term_name].variables()
                quantities.extend(term_variables)
    
    # Add requirements from laws
    if laws:
        for law_name in laws:
            if law_name in LAWS:
                law_variables = LAWS[law_name].variables()
                quantities.extend(law_variables)
    
    required_quantities = set(quantities)
    
    # === STEP 2: Filter gradients by nbsatellite ===
    if nbsatellite < 4:
        required_quantities = {q for q in required_quantities if q not in GRADIENT_QUANTITIES}
    
    # === STEP 3: Check availability ===
    # Combine normal and gradient dependencies
    all_dependencies = {**QUANTITY_DEPENDENCIES, **GRADIENT_QUANTITIES}
    
    available = []
    for quantity_name in required_quantities:
        if quantity_name not in all_dependencies:
            # Undocumented quantity - attempt anyway if it exists
            if quantity_name in dic_quant:
                available.append(quantity_name)
            continue
        
        # Check if it's a direct raw quantity
        if quantity_name in dic_quant:
            available.append(quantity_name)
        # Check if it's a derived quantity we can compute
        elif all(req in dic_quant or req in available for req in all_dependencies[quantity_name]["requires"]):
            available.append(quantity_name)
    
    return available


def compute_gradient_stub(quantity_name: str, dic_quant: dict, separation: float = 1.0) -> np.ndarray:
    """
    Compute an approximation of a quantity's gradient.
    
    Uses a stub implementation pending a proper version.
    Assumes dic_quant[quantity_name] is a dictionary {sat_0, sat_1, sat_2, sat_3}
    
    Parameters:
    -----------
    quantity_name : str
        Name of the quantity whose gradient is desired
    dic_quant : dict
        Dictionary containing data for 4 satellites
    separation : float
        Separation between satellites in grid units
    
    Returns:
    -------
    dict : {sat_name: gradient_vector (n_points, 3)} or (n_points, 3) depending on quantity
    """
    
    # Retrieve base data (without "grad")
    base_quantity = quantity_name.lstrip('I').replace('grad', '')
    
    if base_quantity not in dic_quant:
        raise ValueError(f"Base quantity '{base_quantity}' not found to compute {quantity_name}")
    
    data_per_sat = dic_quant[base_quantity]
    
    if not isinstance(data_per_sat, dict):
        raise ValueError(f"{base_quantity} does not contain per-satellite data")
    
    n_points = len(data_per_sat['sat_0'])
    
    # Initialize result
    result = np.zeros((n_points, 3))
    
    # Retrieve data for each satellite
    sat_0 = data_per_sat['sat_0']  # front, up
    sat_1 = data_per_sat['sat_1']  # front, down
    sat_2 = data_per_sat['sat_2']  # rear, up
    sat_3 = data_per_sat['sat_3']  # rear, down
    
    # Gradient approximation (stub implementation)
    # To be replaced with real spatial interpolation
    
    # ∂f/∂x ≈ (f_front - f_rear) / (2 * separation)
    # Average between up and down for front and rear
    f_front = (sat_0 + sat_1) / 2.0
    f_rear = (sat_2 + sat_3) / 2.0
    result[:, 0] = (f_front - f_rear) / (separation if separation > 0 else 1.0)
    
    # ∂f/∂y ≈ (f_up - f_down) / (2 * separation)
    # Average between front and rear for up and down
    f_up = (sat_0 + sat_2) / 2.0
    f_down = (sat_1 + sat_3) / 2.0
    result[:, 1] = (f_up - f_down) / (separation if separation > 0 else 1.0)
    
    # ∂f/∂z ≈ 0 (no z variation with 4 satellites in xy plane)
    result[:, 2] = 0.0
    
    logger.warning(f"STUB: Gradient of {quantity_name} computed with simple approximation")
    
    return result


def compute_divergence_stub(base_quantity: str, dic_quant: dict, separation: float = 1.0) -> np.ndarray:
    """
    Compute an approximation of a vector's divergence.
    
    Uses a stub implementation pending a proper version.
    
    Parameters:
    -----------
    base_quantity : str
        Name of the vector (ex: "v", "b", "j")
    dic_quant : dict
        Dictionary containing data for 4 satellites
    separation : float
        Separation between satellites
    
    Returns:
    -------
    np.ndarray : (n_points,) - divergence
    """
    
    if base_quantity not in dic_quant:
        raise ValueError(f"Quantity '{base_quantity}' not found to compute divergence")
    
    data_per_sat = dic_quant[base_quantity]
    
    if not isinstance(data_per_sat, dict):
        raise ValueError(f"{base_quantity} does not contain per-satellite data")
    
    n_points = len(data_per_sat['sat_0'])
    
    # Retrieve vector components
    components = f"{base_quantity}x", f"{base_quantity}y", f"{base_quantity}z"
    
    # Stub approximation: divergence ≈ 0
    # To be replaced with real implementation
    result = np.zeros(n_points)
    
    logger.warning(f"STUB: Divergence of {base_quantity} = 0 (stub implementation)")
    
    return result


def compute_quantity_from_QUANTITIES(quantity_name, dic_quant, dic_param, nbsatellite=1, 
                                     separation=1.0, verbose=False):
    """
    Compute a quantity using QUANTITIES objects.
    Adapted for 1D trajectory data with gradient support.
    """
    
    if verbose:
        logger.info(f"Computing {quantity_name} (nbsatellite={nbsatellite})...")
    
    # ===== CAS GRADIENT/DIVERGENCE =====
    if quantity_name in GRADIENT_QUANTITIES:
        if nbsatellite < 4:
            raise ValueError(f"{quantity_name} requires nbsatellite >= 4")
        
        base_quantity = quantity_name.replace('div', '').lstrip('I').replace('grad', '')
        
        if 'div' in quantity_name:
            result = compute_divergence_stub(base_quantity, dic_quant, separation)
        else:
            result = compute_gradient_stub(quantity_name, dic_quant, separation)
        
        return result
    
    # ===== CASE NORMAL QUANTITY =====
    if quantity_name not in QUANTITIES:
        raise ValueError(f"Quantity '{quantity_name}' not found in QUANTITIES")
    
    class MockFile: # Use to capture the created datasets without actual file I/O
        def __init__(self):
            self.data = {}
        
        def create_dataset(self, name, data=None, **kwargs):
            self.data[name] = data if data is not None else np.empty(0)
    
    # Case nbsatellite = 1
    if nbsatellite == 1:
        mock_file = MockFile()
        try:
            QUANTITIES[quantity_name].create_datasets(mock_file, dic_quant, dic_param)
        except Exception as e:
            if verbose:
                logger.error(f"Failed to compute {quantity_name}: {e}")
            raise
        
        result = list(mock_file.data.values())[0] if len(mock_file.data) == 1 else mock_file.data
    
    # Case nbsatellite > 1 : calculate separately for each satellite
    elif nbsatellite == 4:
        result = {}
        for sat_name in ['sat_0', 'sat_1', 'sat_2', 'sat_3']:
            # Extract satellite-specific data and parameters
            dic_quant_sat = {}
            dic_param_sat = {}
            for key, value in dic_quant.items():
                dic_quant_sat[key] = value[sat_name] if isinstance(value, dict) and sat_name in value else value
            for key, value in dic_param.items():
                dic_param_sat[key] = value[sat_name] if isinstance(value, dict) and sat_name in value else value

            # Calculate quantity for this satellite
            mock_file = MockFile()
            try:
                QUANTITIES[quantity_name].create_datasets(mock_file, dic_quant_sat, dic_param)
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name} for {sat_name}: {e}")
                raise
            
            result[sat_name] = list(mock_file.data.values())[0] if len(mock_file.data) == 1 else mock_file.data
        
        result_ordered = {}
        for quantities in result['sat_0'].keys():
            result_ordered[quantities] = {sat: result[sat][quantities] for sat in result}
        result = result_ordered

    if verbose:
        logger.info(f"Quantity {quantity_name} computed")
    
    return result

def compute_all_available_quantities(trajectory_data, dic_param, available_quantities=None, 
                                     nbsatellite=1, separation=1.0, verbose=False):
    """
    Compute ALL available quantities based on data and requirements.
    Handles separately the cases nbsatellite=1 and nbsatellite=4.
    """
    
    if available_quantities is None:
        available_quantities = set(QUANTITY_DEPENDENCIES.keys())
    else:
        available_quantities = set(available_quantities)
    
    result = trajectory_data.copy()
    
    if nbsatellite == 1:
        # ===== CASE nbsatellite=1: no gradients =====
        for quantity_name in available_quantities:
            if quantity_name in result:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, dic_param, nbsatellite=nbsatellite, verbose=verbose
                )
                if isinstance(computed, dict):
                    result.update(computed)
                else:
                    result[quantity_name] = computed
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")
    
    elif nbsatellite == 4:
        # ===== CASE nbsatellite=4: with gradients =====
        # Step 1: normal quantities
        for quantity_name in available_quantities:
            if quantity_name in result or quantity_name in GRADIENT_QUANTITIES:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, dic_param, nbsatellite=nbsatellite, separation=separation, 
                    verbose=verbose
                )                
                if isinstance(computed, dict):
                    result.update(computed)
                else:
                    result[quantity_name] = computed
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")
        
        # Step 2: gradients and divergences
        for quantity_name in available_quantities:
            if quantity_name in result or quantity_name not in GRADIENT_QUANTITIES:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, dic_param, nbsatellite=nbsatellite, separation=separation, 
                    verbose=verbose
                )
                if isinstance(computed, dict):
                    result.update(computed)
                else:
                    result[quantity_name] = computed
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")
    
    else:
        raise ValueError(f"nbsatellite must be 1 or 4, received: {nbsatellite}")
    
    return result


def extract_trajectory_and_compute(dic_quant, dic_param=None, 
                                   laws=None, terms=None, quantities=None,
                                   nbsatellite=1, separation=1.0, verbose=False):
    """
    Extract a 1D trajectory and compute required quantities for laws/terms.
    
    Parameters:
    -----------
    dic_quant : dict
        Dictionary of 1D data (simple or multi-satellite trajectory)
    y_pos, z_pos : int
        For compatibility (not used if nbsatellite=4)
    dic_param : dict
        Parameters
    laws, terms, quantities : list
        Configurations
    nbsatellite : int
        Number of satellites
    separation : float
        Separation between satellites for gradients
    verbose : bool
    """
    
    if dic_param is None:
        dic_param = {}
    
    # Data is already extracted (1D or multi-satellite)
    trajectory_data = dic_quant.copy()
    
    if verbose:
        logger.info(f"Nbsatellite: {nbsatellite}")
        logger.info(f"Separation: {separation}")
    
    # Determine required quantities
    available_quantities = list_computable_quantities(
        trajectory_data, laws, terms, quantities, nbsatellite=nbsatellite
)    
    if verbose:
        logger.info(f"Required quantities from laws/terms: {available_quantities}")
    
    # Compute all available quantities
    return compute_all_available_quantities(
        trajectory_data, dic_param, available_quantities, nbsatellite=nbsatellite,
        separation=separation, verbose=verbose
    )


def display_results(traj_quantities, title="Results along trajectory"):
    """
    Display results of a trajectory.
    """
    logger.info(f"\n{title}")
    logger.info("-" * 70)
    for key in sorted(traj_quantities.keys()):
        value = traj_quantities[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                logger.info(f"  {key:20s}: shape={value.shape} | min={value.min():12.6e} | max={value.max():12.6e}")
            elif value.ndim == 2:
                logger.info(f"  {key:20s}: shape={value.shape} | min={value.min():12.6e} | max={value.max():12.6e}")
        elif isinstance(value, (int, float, np.number)):
            logger.info(f"  {key:20s}: {value:15.6e}")
