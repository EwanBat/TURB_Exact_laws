# trajectory_quantities.py
"""
Compute non-derivative quantities along trajectories using fully vectorized operations.
Analog to trajectory_terms.py but for quantities, using QUANTITIES objects.
Quantities (v, Iv, etc.) are determined by requirements from laws/terms.
Support for gradient and divergence with 4-satellite formations.

Data structure (uniform across all methods):
    - Single satellite (nbsatellite=1):
        {sat_0: {var_name: array(n_trajectories, n_points), ...}}
    - Four satellites (nbsatellite=4):
        {sat_0: {...}, sat_1: {...}, sat_2: {...}, sat_3: {...}}
        where sat_i contains data for each satellite in the formation.

Key design: All data arrays maintain (n_trajectories, n_points) shape for vectorized operations.
"""

import numpy as np
import h5py
import logging
import scipy
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
        "v": {"requires": ["vx", "vy", "vz"]},
        "Iv": {"requires": ["Ivx", "Ivy", "Ivz"]},
        "rho": {"requires": ["rho"]},
        "Irho": {"requires": ["Irho"]},
        "b": {"requires": ["bx", "by", "bz"]},
        "Ib": {"requires": ["bx", "by", "bz"]},
        "v2": {"requires": ["vx", "vy", "vz"]},
        "Iv2": {"requires": ["Ivx", "Ivy", "Ivz"]},
        "vnorm": {"requires": ["vx", "vy", "vz"]},
        "Ivnorm": {"requires": ["Ivx", "Ivy", "Ivz"]},
        "bnorm": {"requires": ["bx", "by", "bz"]},
        "Ibnorm": {"requires": ["bx", "by", "bz"]},
        "pm": {"requires": ["bx", "by", "bz"]},
        "Ipm": {"requires": ["bx", "by", "bz"]},
        "pgyr": {"requires": ["pperp", "rho"]},
        "Ipgyr": {"requires": ["pperp"]},
        "piso": {"requires": ["ppar", "pperp"]},
        "Ipiso": {"requires": ["ppar", "pperp"]},
        "ppol": {"requires": ["pperp"]},
        "Ippol": {"requires": ["pperp"]},
        "pcgl": {"requires": ["bx", "by", "bz", "rho", "ppar", "pperp"]},
        "Ipcgl": {"requires": ["bx", "by", "bz", "ppar", "pperp"]},
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
        'gradv': {'requires': ['vx', 'vy', 'vz']},
        'gradv2': {'requires': ['v2']},
        'gradb': {'requires': ['bx', 'by', 'bz']},
        'gradrho': {'requires': ['rho']},
        'graduiso': {'requires': ['uiso']},
        'gradupol': {'requires': ['upol']},
        'gradugyr': {'requires': ['ugyr']},
        'gradpcgl': {'requires': ['pcgl']},
        'divv': {'requires': ['vx', 'vy', 'vz']},
        'divb': {'requires': ['bx', 'by', 'bz']},
        'divj': {'requires': ['bx', 'by', 'bz']},
        'Igradv': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'Igradv2': {'requires': ['Iv2']},
        'Igradrho': {'requires': ['Irho']},
        'Igraduiso': {'requires': ['Iuiso']},
        'Igradupol': {'requires': ['Iupol']},
        'Igradb': {'requires': ['Ibx', 'Iby', 'Ibz']},
        'Idivj': {'requires': ['bx', 'by', 'bz']},
        'j': {'requires': ['bx', 'by', 'bz']},
        'Ij': {'requires': ['bx', 'by', 'bz']},
        'w': {'requires': ['vx', 'vy', 'vz']},
        'Iw': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'f': {'requires': ['fp', 'fm']},
        'If': {'requires': ['fp', 'fm']},
        'hdk': {'requires': ['vx', 'vy', 'vz']},
        'Ihdk': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'hdk2': {'requires': ['vx', 'vy', 'vz']},
        'Ihdk2': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'hdm': {'requires': ['bx', 'by', 'bz', 'rho']},
        'Ihdm': {'requires': ['bx', 'by', 'bz']},
    }
        
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
    
    def extract_and_compute(self, dic_datas: dict, 
                            laws=None, terms=None, quantities=None, method:str = None,
                            filename: str = None):
        """
        Compute all required quantities for vectorized trajectories.
        
        Fully vectorized with no trajectory loops. Input and output maintain same structure.
        All arrays preserve (n_trajectories, n_points) shape throughout.
        
        Parameters:
        -----------
        dic_datas : dict
            {sat_name: {var_name: array(n_trajectories, n_points)}}
            Examples:
              - Single satellite: {sat_0: {vx: array(...), vy: array(...), ...}}
              - Four satellites: {sat_0: {...}, sat_1: {...}, sat_2: {...}, sat_3: {...}}
        laws : list, optional
            Law names; extract required variables via LAWS[name].variables()
        terms : list, optional
            Term names; extract required variables via TERMS[name].variables()
        quantities : list, optional
            Explicit list of quantities to compute
        
        Returns:
        -------
        dict : Same structure as input, with computed quantities added to each satellite
        """
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("TRAJECTORY QUANTITIES COMPUTATION (VECTORIZED)")
            logger.info(f"  Nbsatellite:        {self.nbsatellite}")
            logger.info(f"  Separation:         {self.traj_param.get('separation', 1)}")
        
        first_sat_data = dic_datas['sat_0']
        available_quantities = self.list_computable_quantities(
            first_sat_data, laws, terms, quantities, method=method
        )
        
        if self.verbose:
            logger.info(f"  Quantities to compute: {len(available_quantities)}")
            logger.info(f"  {available_quantities}")
                
        # self.quantities_to_h5(result, filename)
        return self._compute_all_quantities(dic_datas, available_quantities, filename)
    
    def list_computable_quantities(self, dic_quant: dict, laws=None, terms=None, 
                                   quantities=None, method:str = None):
        """
        Extract all required quantities from laws/terms specifications.
        
        Parameters:
        -----------
        dic_quant : dict
            Available data (only used for compatibility, not actively used)
        laws : list, optional
            Law names for extracting variables
        terms : list, optional
            Term names for extracting variables
        quantities : list, optional
            Explicit quantities to include
        
        Returns:
        -------
        list : Unique list of quantities to compute
        """
        if quantities is None:
            quantities = []
        else:
            quantities = list(quantities)
        
        if terms: # Check terms dependencies first because they can require quantities that laws also need
            for term_name in terms:
                if term_name in TERMS:
                    term_variables = TERMS[term_name].variables(nbsatellite=self.nbsatellite, method=method)
                    quantities.extend(term_variables)
        
        if laws: # Check laws dependencies
            for law_name in laws:
                if law_name in LAWS:
                    law_variables = LAWS[law_name].variables(nbsatellite=self.nbsatellite, method=method)
                    quantities.extend(law_variables)
        
        return list(set(quantities))
    
    # ========== PRIVATE METHODS ==========
        
    def _compute_all_quantities(self, dic_datas: dict, available_quantities: list, filename: str):
        """
        Compute all quantities using vectorized operations.
        Structure is preserved: {sat_name: {var_name: array(n_traj, n_pts)}}
        """
        dic_quantities = {sat_name: {} for sat_name in dic_datas.keys()}

        if self.nbsatellite == 1:
            # Single satellite: compute directly for each quantity
            for quantity_name in available_quantities:
                try:
                    if quantity_name not in self.QUANTITY_DEPENDENCIES and quantity_name not in self.GRADIENT_QUANTITIES:
                        raise ValueError(f"Quantity '{quantity_name}' not found in QUANTITIES")
                        
                    for sat_name in dic_datas.keys():
                        result = self._compute_quantity_vectorized(
                            quantity_name, dic_datas[sat_name]
                        )
                        if isinstance(result, dict):
                            for key, value in result.items():
                                dic_quantities[sat_name][key] = value
                        else:
                            dic_quantities[sat_name][quantity_name] = result
                
                except Exception as e:
                    if self.verbose:
                        logger.error(f"Failed to compute {quantity_name}: {str(e)}")
        
        elif self.nbsatellite == 4:
            # Four satellites: two-step process
            # Step 1: Compute non-gradient quantities (v, rho, b, etc.) for each satellite independently
            for quantity_name in available_quantities:
                if quantity_name in self.GRADIENT_QUANTITIES:
                    continue
                
                try:
                    if quantity_name not in self.QUANTITIES: # Check non-gradient quantities first because grad quantities can be computed from them
                        raise ValueError(f"Quantity '{quantity_name}' not found in QUANTITIES")
                    
                    for sat_name in dic_datas.keys():
                        result = self._compute_quantity_vectorized(
                            quantity_name, dic_datas[sat_name]
                        )
                        if isinstance(result, dict):
                            for key, value in result.items():
                                dic_quantities[sat_name][key] = value
                        else:
                            dic_quantities[sat_name][quantity_name] = result
                
                except Exception as e:
                    if self.verbose:
                        logger.error(f"Failed to compute {quantity_name} for {sat_name}: {str(e)}")
            
            # Merge raw data into dic_quantities for gradient computation
            for sat_name in dic_datas.keys():
                for key in dic_datas[sat_name].keys():
                    if key not in dic_quantities[sat_name]:
                        dic_quantities[sat_name][key] = dic_datas[sat_name][key]

            # Step 2: Compute gradients/divergences using all satellites simultaneously
            # Pass entire dic_quantities (all 4 satellites) to gradient computation
            for quantity_name in available_quantities:
                if quantity_name in self.QUANTITY_DEPENDENCIES:
                    continue
                
                try:
                    if quantity_name not in self.GRADIENT_QUANTITIES:
                        raise ValueError(f"Quantity '{quantity_name}' not found in GRADIENT_QUANTITIES")

                    result = self._compute_quantity_vectorized(
                        quantity_name, dic_quantities
                    )

                    if isinstance(result, dict):
                        for key, value in result.items():
                            dic_quantities['sat_0'][key] = value
                    else:
                        dic_quantities['sat_0'][quantity_name] = result
                
                except Exception as e:
                    if self.verbose:
                        logger.error(f"Failed to compute {quantity_name}: {str(e)}")

        if self.verbose:
            logger.info(f"  [OK] All quantities computed successfully")
            logger.info(dic_quantities['sat_0'].keys())

        self.quantities_to_h5(dic_quantities, filename)

        return dic_quantities

    def _compute_quantity_vectorized(self, quantity_name: str, dic_quant_sat: dict):
        """
        Compute a single quantity for vectorized trajectory arrays.
        
        For single satellite: dic_quant_sat = {var_name: array(n_traj, n_pts)}
        For 4 satellites: dic_quant_sat = {sat_0: {...}, sat_1: {...}, ...}
        
        Parameters:
        -----------
        quantity_name : str
            Quantity to compute
        dic_quant_sat : dict
            Either single-satellite data or multi-satellite data structure
        
        Returns:
        -------
        np.ndarray or dict : Computed quantity (array(n_traj, n_pts) or multi-sat dict)
        """
        mock_file = MockFile()
        try:
            self.QUANTITIES[quantity_name].create_datasets(
                mock_file, dic_quant_sat, self.dic_param, 
                traj=True, traj_param=self.traj_param
            )
        except Exception as e:
            if self.verbose:
                logger.error(f"Failed to compute {quantity_name}: {e}")
            raise
        
        if len(mock_file.data) == 1:
            return list(mock_file.data.values())[0]
        else:
            return mock_file.data

    def quantities_to_h5(self, dic_quant: dict, filename: str):
        """
        Save computed quantities to HDF5 file preserving satellite structure.
        Structure: /sat_name/var_name with arrays(n_traj, n_pts)
        
        Parameters:
        -----------
        dic_quant : dict
            {sat_name: {var_name: array(n_traj, n_pts)}}
        filename : str
            Output file path
        """
        with h5py.File(filename, 'w') as f:
            for sat_name, sat_data in dic_quant.items():
                group = f.create_group(sat_name)
                for var_name, data_array in sat_data.items():
                    group.create_dataset(var_name, data=data_array, compression="gzip", compression_opts=9)

# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def extract_and_compute_trajectory_quantities(dic_datas: dict, grid_param: dict = None,
                                              traj_param: dict = None, physical_param: dict = None,
                                              laws=None, terms=None, quantities=None, method: str = None,
                                              verbose: bool = False, filename: str = "computed_quantities.h5"):
    """
    Backward compatibility wrapper. Use TrajectoryQuantitiesComputer.extract_and_compute instead.
    """
    computer = TrajectoryQuantitiesComputer(verbose=verbose, 
                                           grid_param=grid_param, 
                                           physical_param=physical_param, 
                                           traj_param=traj_param)
    return computer.extract_and_compute(dic_datas, laws=laws, terms=terms, 
                                       quantities=quantities, method=method, 
                                       filename=filename)