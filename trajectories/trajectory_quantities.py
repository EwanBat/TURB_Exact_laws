# trajectory_quantities.py
"""
Module to compute non-derivative quantities along trajectories.
Analog to trajectory_terms.py but for quantities.
Uses QUANTITIES objects for vectorized trajectory computation.
Quantities (v or Iv, etc.) are determined by the required variables from laws.
Support for gradient and divergence with 4-satellite formations.

Structure: dic_quant = {key: array(n_trajectories, n_points)} (nbsatellite=1)
           or {sat_0: {key: array(n_trajectories, n_points)}, ...} (nbsatellite=4)
"""

import numpy as np
import logging
from exact_laws.preprocessing.quantities import QUANTITIES
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS

logger = logging.getLogger(__name__)


# ========== MOCK FILE FOR QUANTITY COMPUTATION ==========

class MockFile:
    """Mock HDF5 file object for storing computed quantities."""
    def __init__(self):
        self.data = {}
    
    def create_dataset(self, name, data=None, **kwargs):
        self.data[name] = data if data is not None else np.empty(0)


# ========== TRAJECTORY QUANTITIES COMPUTER CLASS ==========

class TrajectoryQuantitiesComputer:
    """
    Compute quantities along trajectories in a fully vectorized manner.
    
    Handles both single satellite and 4-satellite formation configurations.
    Manages quantity dependencies, availability checks, and vectorized computations.
    
    Attributes are maintained across operations to avoid repeatedly passing parameters.
    """
    
    # ========== CLASS CONSTANTS ==========
    
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
    
    SATELLITE_NAMES = ['sat_0', 'sat_1', 'sat_2', 'sat_3']
    
    # ========== INITIALIZATION ==========
    
    def __init__(self, verbose: bool = False, grid_param: dict = None, physical_param: dict = None, traj_param: dict = None):
        """
        Initialize the trajectory quantities computer.
        
        Parameters:
        -----------
        verbose : bool
            Enable detailed logging
        grid_param : dict, optional
            Grid parameters (can be set later via configure_params)
        physical_param : dict, optional
            Physical parameters (can be set later via configure_params)
        traj_param : dict, optional
            Trajectory parameters (can be set later via configure_params)
        """
        self.verbose = verbose
        self.grid_param = grid_param or {}
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.dic_param = {**self.physical_param, **self.grid_param}
        self.QUANTITIES = QUANTITIES
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)
       
    # ========== PUBLIC METHODS ==========
    
    def extract_and_compute(self, dic_datas: dict, laws=None, terms=None, quantities=None):
        """
        Main entry point: Compute all required quantities for vectorized trajectories.
        
        Fully vectorized - no trajectory loops. Input and output maintain same structure.
        
        Parameters:
        -----------
        dic_datas : dict
            Vectorized data structure (uniform):
            {sat_name: {var_name: array(n_trajectories, n_points)}, ...}
            For nbsatellite=1: {sat_0: {...}}
            For nbsatellite=4: {sat_0: {...}, sat_1: {...}, sat_2: {...}, sat_3: {...}}
        laws, terms, quantities : list
            Configuration lists for computing requirements
        
        Returns:
        -------
        dict : Computed quantities (same vectorized structure as input)
        """
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("TRAJECTORY QUANTITIES COMPUTATION (VECTORIZED)")
            logger.info(f"  Nbsatellite:        {self.nbsatellite}")
            logger.info(f"  Separation:         {self.traj_param.get('separation', 1)}")
        
        # Get first satellite's data to analyze available quantities
        first_sat_data = dic_datas.get(self.SATELLITE_NAMES[0], {})
        available_quantities = self.list_computable_quantities(
            first_sat_data, laws, terms, quantities
        )
        
        if self.verbose:
            logger.info(f"  Quantities to compute: {len(available_quantities)}")
            logger.info(f"  {available_quantities}")
        
        # Compute all quantities maintaining satellite structure
        result = self._compute_all_quantities(dic_datas, available_quantities)
        
        if self.verbose:
            logger.info(f"  [OK] All quantities computed successfully")
            logger.info(result['sat_0'].keys())
        
        return result
    
    def list_computable_quantities(self, dic_quant: dict, laws=None, terms=None, 
                                   quantities=None):
        """
        List computable quantities from available data and requirements.
        
        Parameters:
        -----------
        dic_quant : dict
            Dictionary of available data (vectorized structure)
        laws, terms, quantities : list, optional
            Requirement specifications
        
        Returns:
        -------
        list : Computable quantities
        """
        # STEP 1: Extract requirements
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
        
        # STEP 2: Check availability
        all_dependencies = {**self.QUANTITY_DEPENDENCIES, **self.GRADIENT_QUANTITIES}

        available = []
        for quantity_name in required_quantities:
            if quantity_name not in all_dependencies:
                # Undocumented quantity - check if it exists
                if quantity_name in dic_quant:
                    available.append(quantity_name)
                continue
            
            # Check if it's a direct raw quantity or derivable
            if self._quantity_is_available(quantity_name, dic_quant, 
                                          all_dependencies):
                available.append(quantity_name)
        
        return available
    
    # ========== PRIVATE METHODS ==========
        
    def _compute_all_quantities(self, dic_datas: dict, available_quantities: list):
        """
        Compute all quantities using vectorized operations (no trajectory loops).
        
        Parameters:
        -----------
        dic_datas : dict
            Vectorized data: {sat_name: {var_name: array(n_trajectories, n_points)}, ...}
        available_quantities : list
            Quantities to compute
        
        Returns:
        -------
        dict : dic_datas with added quantities (modifies in-place)
        """
        
        for quantity_name in available_quantities:
            try:
                if quantity_name not in self.QUANTITIES:
                    raise ValueError(f"Quantity '{quantity_name}' not found in QUANTITIES")
                    
                # Store result maintaining {sat_name: {var_name: array}} structure
                for sat_name in dic_datas.keys():
                    result = self._compute_quantity_vectorized(
                        quantity_name, dic_datas[sat_name]
                    )
                    if isinstance(result, dict):
                        # If result is a dict of multiple datasets, merge into dic_datas
                        for key, value in result.items():
                            dic_datas[sat_name][key] = value
                    else:
                        dic_datas[sat_name][quantity_name] = result
            
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {str(e)}")
        
        return dic_datas
    
    def _quantity_is_available(self, quantity_name: str, dic_quant: dict, all_dependencies: dict):
        """
        Check if a quantity is available directly or can be computed.
        dic_quant contains data from first satellite only: {var_name: array(...)}
        """
        deps = all_dependencies.get(quantity_name, {}).get("requires", [])
        # Check if all dependencies exist
        return all(dep in dic_quant for dep in deps) or quantity_name in dic_quant
    
    def _compute_quantity_vectorized(self, quantity_name: str, dic_quant_sat: dict):
        """
        Compute a single quantity for vectorized trajectory array of one satellite.
        
        Parameters:
        -----------
        quantity_name : str
            Quantity name
        dic_quant_sat : dict
            Data dict for one satellite: {var_name: array(n_trajectories, n_points)}
        
        Returns:
        -------
        np.ndarray : Computed quantity array (n_trajectories, n_points)
        """
        mock_file = MockFile()
        try:
            # Use QUANTITIES to compute the quantity for vectorized data
            self.QUANTITIES[quantity_name].create_datasets(
                mock_file, dic_quant_sat, self.dic_param, 
                traj=True, traj_param=self.traj_param
            )
        except Exception as e:
            if self.verbose:
                logger.error(f"Failed to compute {quantity_name}: {e}")
            raise
        
        # Extract the computed dataset
        if len(mock_file.data) == 1:
            return list(mock_file.data.values())[0]
        else:
            return mock_file.data

# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def extract_and_compute_trajectory_quantities(dic_datas: dict, grid_param: dict = None,
                                              traj_param: dict = None, physical_param: dict = None,
                                              laws=None, terms=None, quantities=None,
                                              verbose: bool = False):
    """
    Backward compatibility wrapper for extract_and_compute.
    
    Deprecated: Use TrajectoryQuantitiesComputer.extract_and_compute instead.
    
    Usage:
        computer = TrajectoryQuantitiesComputer(verbose=True, 
                                               grid_param=grid_param, 
                                               physical_param=physical_param,
                                               traj_param=traj_param)
        result = computer.extract_and_compute(dic_datas, laws, terms, quantities)
    """
    computer = TrajectoryQuantitiesComputer(verbose=verbose, 
                                           grid_param=grid_param, 
                                           physical_param=physical_param, 
                                           traj_param=traj_param)
    return computer.extract_and_compute(dic_datas, laws, terms, quantities)