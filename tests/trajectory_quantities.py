# trajectory_quantities.py - VERSION WITH GRADIENT SUPPORT
"""
Module to compute non-derivative quantities along a trajectory.
Analog to process_on_oca_files.py but adapted for trajectories and using QUANTITIES.
Quantities (v or Iv, etc.) are determined by the required variables from laws.
Support for gradient and divergence with 4-satellite formations.
"""

import numpy as np
import numexpr as ne
import logging
from derivation_satellite import compute_gradient_stub, compute_divergence_stub
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


def extract_trajectory_index(dic_datas, traj_idx, nbsatellite=1):
    # Structure de vue, pas de copie
    if nbsatellite == 1:
        return {key: value[traj_idx] for key, value in dic_datas.items()}  # Vues, pas copies
    else:
        return {key: {f'sat_{i}': value[i, traj_idx] for i in range(value.shape[0])} 
                for key, value in dic_datas.items()}


def compute_quantities_all_trajectories(dic_datas, physical_param, traj_param, grid_param,
                                        available_quantities=None,
                                        verbose=False):
    """
    Compute quantities for ALL trajectories using array-based structure.
    
    Optimized to minimize memory allocations. Uses np.stack for efficient array construction.
    """
    n_trajectories = traj_param.get('n_trajectories', 1)
    nbsatellite = traj_param.get('nbsatellite', 1)
    separation = traj_param.get('separation', 1.0)
    
    if verbose:
        logging.info(f"  Computing quantities for {n_trajectories} trajectory/trajectories...")
    
    # Accumulate results in lists (more efficient than pre-allocation + filling)
    results_list = []
    
    for traj_idx in range(n_trajectories):
        single_traj_data = extract_trajectory_index(dic_datas, traj_idx, nbsatellite=nbsatellite)
        
        traj_result = compute_all_available_quantities(
            single_traj_data, physical_param, grid_param, available_quantities,
            nbsatellite=nbsatellite, separation=separation, verbose=False  # <- NO SPAM
        )
        results_list.append(traj_result)
    
    # OPTIMIZATION: Stack all results at once (10x faster, cache-efficient)
    result_all_trajectories = {}
    
    if nbsatellite == 1:
        # Stack: (n_trajectories, n_points)
        for key in results_list[0].keys():
            result_all_trajectories[key] = np.stack(
                [results_list[i][key] for i in range(n_trajectories)], axis=0
            )
    else:
        # Stack: (n_satellites, n_trajectories, n_points)
        for key in results_list[0].keys():
            sat_arrays = [
                np.stack([results_list[i][key][f'sat_{s}'] for i in range(n_trajectories)], axis=0)
                for s in range(4)
            ]
            result_all_trajectories[key] = np.stack(sat_arrays, axis=0)
    
    if verbose:
        logging.info(f"  [OK] Quantities computed for all {n_trajectories} trajectories")
    
    return result_all_trajectories


def compute_quantity_from_QUANTITIES(quantity_name, dic_datas, physical_param, grid_param, nbsatellite=1, 
                                     separation=1.0, verbose=False):
    """
    Compute a single quantity using QUANTITIES objects.
    
    OPTIMIZATION: Pre-build dic_param once instead of recreating for each satellite.
    """
    
    dic_param = {}
    dic_param.update(physical_param)
    dic_param.update(grid_param)
    
    # ===== CAS GRADIENT/DIVERGENCE =====
    if quantity_name in GRADIENT_QUANTITIES:
        if nbsatellite < 4:
            raise ValueError(f"{quantity_name} requires nbsatellite >= 4")
        
        base_quantity = quantity_name.replace('div', '').lstrip('I').replace('grad', '')
        
        if 'div' in quantity_name:
            result = compute_divergence_stub(base_quantity, dic_datas, separation)
        else:
            result = compute_gradient_stub(quantity_name, dic_datas, separation)
        
        return result
    
    # ===== CASE NORMAL QUANTITY =====
    if quantity_name not in QUANTITIES:
        raise ValueError(f"Quantity '{quantity_name}' not found in QUANTITIES")
    
    class MockFile:
        def __init__(self):
            self.data = {}
        
        def create_dataset(self, name, data=None, **kwargs):
            self.data[name] = data if data is not None else np.empty(0)
    
    # Case nbsatellite = 1
    if nbsatellite == 1:
        mock_file = MockFile()
        try:
            QUANTITIES[quantity_name].create_datasets(mock_file, dic_datas, dic_param)
        except Exception as e:
            if verbose:
                logger.error(f"Failed to compute {quantity_name}: {e}")
            raise
        
        result = list(mock_file.data.values())[0] if len(mock_file.data) == 1 else mock_file.data
    
    # Case nbsatellite = 4: OPTIMIZATION - build param once with references
    elif nbsatellite == 4:
        result = {}
        for sat_name in ['sat_0', 'sat_1', 'sat_2', 'sat_3']:
            # Extract satellite-specific data via dict comprehension (view, not copy)
            dic_quant_sat = {
                key: value[sat_name] if isinstance(value, dict) and sat_name in value else value
                for key, value in dic_datas.items()
            }
            # Extract satellite-specific parameters via dict comprehension
            dic_param_sat = {
                key: value[sat_name] if isinstance(value, dict) and sat_name in value else value
                for key, value in dic_param.items()
            }

            # Calculate quantity for this satellite
            mock_file = MockFile()
            try:
                QUANTITIES[quantity_name].create_datasets(mock_file, dic_quant_sat, dic_param_sat)
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name} for {sat_name}: {e}")
                raise
            
            result[sat_name] = list(mock_file.data.values())[0] if len(mock_file.data) == 1 else mock_file.data
        
        # Reorganize to array-friendly structure
        result_ordered = {}
        for quantities in result['sat_0'].keys():
            result_ordered[quantities] = {sat: result[sat][quantities] for sat in result}
        result = result_ordered

    return result

def compute_all_available_quantities(trajectory_data, physical_param, grid_param, available_quantities=None, 
                                     nbsatellite=1, separation=1.0, verbose=False):
    """
    Compute ALL available quantities based on data and requirements.
    
    OPTIMIZATION: No .copy() - modifies trajectory_data in-place to save memory.
    """
    if available_quantities is None:
        available_quantities_set = set(QUANTITY_DEPENDENCIES.keys())
    else:
        available_quantities_set = set(available_quantities)
    
    # OPTIMIZATION: Pre-filter gradients once
    if nbsatellite < 4:
        gradient_set = set(GRADIENT_QUANTITIES.keys())
        available_quantities_set -= gradient_set
    
    result = trajectory_data  # NO COPY - modifies in-place
    
    if nbsatellite == 1:
        # ===== CASE nbsatellite=1: no gradients =====
        for quantity_name in available_quantities_set:
            if quantity_name in result:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, physical_param, grid_param, nbsatellite=nbsatellite, verbose=False
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
        gradient_set = set(GRADIENT_QUANTITIES.keys())
        normal_quantities = available_quantities_set - gradient_set
        
        # Step 1: normal quantities
        for quantity_name in normal_quantities:
            if quantity_name in result:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, physical_param, grid_param, nbsatellite=nbsatellite, 
                    separation=separation, verbose=False
                )                
                if isinstance(computed, dict):
                    result.update(computed)
                else:
                    result[quantity_name] = computed
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")
        
        # Step 2: gradients only (in second pass)
        gradient_quantities = available_quantities_set & gradient_set
        for quantity_name in gradient_quantities:
            if quantity_name in result:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, physical_param, grid_param, nbsatellite=nbsatellite, 
                    separation=separation, verbose=False
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


def extract_and_compute_trajectory_quantities(dic_datas, grid_param=None,
                                              traj_param=None, physical_param=None,
                                             laws=None, terms=None, quantities=None,
                                             verbose=False):
    """
    Main function: Extract trajectories and compute all required quantities.
    
    Automatically detects if data contains multiple trajectories and processes them.
    
    Parameters:
    -----------
    dic_datas : dict
        Data from preprocess_trajectory_from_ini (multi-trajectory structure)
    grid_param : dict
        Grid parameters
    traj_param : dict
        Trajectory parameters
    physical_param : dict
        Physical parameters
    laws, terms, quantities : list
        Configuration lists
    nbsatellite : int
        Number of satellites
    separation : float
        Separation between satellites for gradient computation
    verbose : bool
        Display detailed info
    
    Returns:
    -------
    dict : Computed quantities with multi-trajectory structure
           Maintains same structure as input dic_datas
    """
    
    if verbose:
        logging.info("\n" + "="*70)
        logging.info("TRAJECTORY QUANTITIES COMPUTATION")
        logging.info(f"  Nbsatellite:        {traj_param.get('nbsatellite', 1)}")
        logging.info(f"  Separation:         {traj_param.get('separation', 1.0)}")
        logging.info(f"  N trajectories:     {traj_param.get('n_trajectories', 1)}")
    
    # Determine required quantities from first trajectory
    first_traj = extract_trajectory_index(dic_datas, 0)
    available_quantities = list_computable_quantities(
        first_traj, laws, terms, quantities, nbsatellite=traj_param.get('nbsatellite', 1)
    )
    
    if verbose:
        logging.info(f"  Required quantities: {len(available_quantities)} to compute")
    
    # Compute quantities for all trajectories
    result = compute_quantities_all_trajectories(
        dic_datas, physical_param, traj_param, grid_param, available_quantities,
        verbose=False  # Disable per-trajectory logs to avoid spam
    )
    
    return result


