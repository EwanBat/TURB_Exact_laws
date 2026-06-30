# trajectory_terms.py
"""
Module to compute terms along a trajectory.
Analog to trajectory_quantities.py but for terms.
Uses calc_fourier() methods from terms for trajectories.
Encapsulated in TrajectoryTermsComputer class for better parameter management.
"""

import numpy as np
from numba import njit, prange, set_num_threads
import logging
import h5py
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS

logger = logging.getLogger(__name__)


# ========== TRAJECTORY TERMS COMPUTER CLASS ==========

class TrajectoryTermsComputer:
    """
    Compute physics terms along trajectories.
    
    Encapsulates term computation logic with parameter storage as instance attributes
    to reduce repeated parameter passing.
    Handles both single satellite and 4-satellite formation configurations.
    """
    
    # ========== CLASS CONSTANTS ==========
    
    # Mapping of abstract variables to their concrete components
    VARIABLE_COMPONENTS = {
        'v': ['vx', 'vy', 'vz'],
        'Iv': ['Ivx', 'Ivy', 'Ivz'],
        'b': ['bx', 'by', 'bz'],
        'Ib': ['Ibx', 'Iby', 'Ibz'],
        'w': ['wx', 'wy', 'wz'],
        'j': ['jx', 'jy', 'jz'],
        'Ij': ['Ijx', 'Ijy', 'Ijz'],
        'f': ['fx', 'fy', 'fz'],
        'rho': ['rho'],
        'Irho': ['Irho'],
        'v2': ['v2'],
        'Iv2': ['Iv2'],
        'vnorm': ['vnorm'],
        'Ivnorm': ['Ivnorm'],
        'bnorm': ['bnorm'],
        'Ibnorm': ['Ibnorm'],
        'pm': ['pm'],
        'Ipm': ['Ipm'],
        'pgyr': ['ppar', 'pperp'],
        'Ipgyr': ['Ippar', 'Ipperp'],
        'piso': ['piso'],
        'Ipiso': ['Ipiso'],
        'ppol': ['ppol'],
        'Ippol': ['Ippol'],
        'pcgl': ['pparcgl', 'pperpcgl'],
        'Ipcgl': ['Ipparcgl', 'Ipperpcgl'],
        'ugyr': ['ugyr'],
        'Iugyr': ['Iugyr'],
        'uiso': ['uiso'],
        'Iuiso': ['Iuiso'],
        'upol': ['upol'],
        'Iupol': ['Iupol'],
        'ucgl': ['ucgl'],
        'Iucgl': ['Iucgl'],
        'divv': ['divv'],
        'Idivv': ['Idivv'],
        'divb': ['divb'],
        'Idivb': ['Idivb'],
        'divj': ['divj'],
        'Idivj': ['Idivj'],
        'gradrho': ['gradrhox', 'gradrhoy', 'gradrhoz'],
        'Igradrho': ['Igradrhox', 'Igradrhoy', 'Igradrhoz'],
        'gradv': ['dxvx', 'dyvx', 'dzvx', 'dxvy', 'dyvy', 'dzvy', 'dxvz', 'dyvz', 'dzvz'],
        'Igradv': ['Idxvx', 'Idyvx', 'Idzvx', 'Idxvy', 'Idyvy', 'Idzvy', 'Idxvz', 'Idyvz', 'Idzvz'],
        'gradb': ['dxbx', 'dybx', 'dzbx', 'dxby', 'dyby', 'dzby', 'dxbz', 'dybz', 'dzbz'],
        'Igradb': ['Idxbx', 'Idybx', 'Idzbx', 'Idxby', 'Idyby', 'Idzby', 'Idxbz', 'Idybz', 'Idzbz'],
        'graduiso': ['graduisox', 'graduisoy', 'graduisoz'],
        'Igraduiso': ['Igraduisox', 'Igraduisoy', 'Igraduisoz'],
        'gradupol': ['gradupolx', 'gradupoly', 'gradupolz'],
        'Igradupol': ['Igradupolx', 'Igradupoly', 'Igradupolz'],
        'hdk': ['hdkx', 'hdky', 'hdkz'],
        'Ihdk': ['Ihdkx', 'Ihdky', 'Ihdkz'],
        'hdm': ['hdmx', 'hdmy', 'hdmz'],
        'Ihdm': ['Ihdmx', 'Ihdmy', 'Ihdmz'],
        'hdk2': ['hdk2x', 'hdk2y', 'hdk2z'],
        'Ihdk2': ['Ihdk2x', 'Ihdk2y', 'Ihdk2z'],
    }

    FLUX_TERMS = frozenset([
        "flux_dvdvdv",
        "flux_dbdbdv",
        "flux_dvdbdb",
    ])
    
    SOURCE_TERMS = frozenset([
        "source_dpan",
    ])
        
    # ========== INITIALIZATION ==========
    
    def __init__(self, verbose: bool = False, grid_param: dict = None, physical_param: dict = None, traj_param: dict = None, max_workers: int = np.nan):
        """
        Initialize the trajectory terms computer.
        
        Parameters:
        -----------
        verbose : bool
            Enable detailed logger
        grid_param : dict, optional
            Grid parameters
        physical_param : dict, optional
            Physical parameters
        traj_param : dict, optional
            Trajectory parameters including 'nbsatellite'
        max_workers : int
            Maximum number of threads for ThreadPoolExecutor (default: 2)
        """
        self.verbose = verbose
        self.grid_param = grid_param or {}
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.max_workers = max_workers
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)
    
        self._sat_names = [f'sat_{i}' for i in range(self.nbsatellite)]
        self._sat_param_cache = {sat_name: self._extract_sat_parameters(sat_name) for sat_name in self._sat_names}

    # ========== PUBLIC METHODS ==========
    
    def list_required_terms(self, laws: list = None):
        """
        Returns the list of required terms to compute the given laws.
        
        Parameters:
        -----------
        laws : list[str]
            List of law names
        
        Returns:
        -------
        set : Set of required term names
        """
        if laws is None:
            laws = []
        
        terms = set()
        
        if not laws:
            return terms
        
        # Convert parameters from trajectory arrays to scalars for terms_and_coeffs()
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        
        # Collect terms from all laws
        for law_name in laws:
            if law_name in LAWS:
                law_obj = LAWS[law_name]
                # terms_and_coeffs() returns (terms_list, coeffs_dict)
                law_terms, _ = law_obj.terms_and_coeffs(params_clean)
                terms.update(law_terms)
        
        return terms

    def compute_all_terms_for_laws(self, dic_quantities: dict = None, laws: list = None, method: str = None, filename: str = "terms_trajectory.h5"):
        """
        Compute all terms required for the given laws.
        
        Determines the set of terms needed from law specifications, then computes
        them from the provided quantities.
        
        Parameters:
        -----------
        dic_quantities : dict
            Data structure: {sat_name: {var_name: array(n_trajectories, n_points)}, ...}
        laws : list[str]
            List of law names to compute terms for
        method : str
            Computation method ("fourier" or "incremental")
        filename : str
            Output HDF5 filename
        
        Returns:
        -------
        dict : {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
        """
        
        if laws is None:
            laws = []
        
        # Get required terms from law specifications
        required_terms = self.list_required_terms(laws)
        
        if self.verbose:
            logger.info("\n" + "-"*70)
            logger.info("FLUX AND SOURCE TERMS COMPUTATION")
            logger.info(f"  Computing {len(required_terms)} terms for {len(laws)} law(s)")
            logger.info(f"  Data structure: {{sat_name: {{term_name: array(n_trajectories, n_points)}}}}")
        
        # Initialize result with satellite names
        result = {sat_name: {} for sat_name in self._sat_names}
        
        if method == "incremental":
            if self.traj_param['trajectory_method'] == "linear_x":
                fs = 1/self.grid_param['c'][0]
            elif self.traj_param['trajectory_method'] == "linear_y":
                fs = 1/self.grid_param['c'][1]
            elif self.traj_param['trajectory_method'] == "linear_z":
                fs = 1/self.grid_param['c'][2]

            if self.nbsatellite == 1:
                try:
                    merged_quantities = {}  # Initialize an array to hold merged quantities for each trajectory and point
                    for quantity in dic_quantities['sat_0'].keys():
                        merged_quantities.update({quantity: np.concatenate((
                            dic_quantities['sat_0'][quantity],
                            dic_quantities['sat_0'][quantity]
                            ), axis=1)}) 
                    
                    for term_name in required_terms:
                        term_obj = TERMS[term_name]
                        result['sat_0'][term_name] = term_obj.calc_incr_traj(self.traj_param["n_points"], self.traj_param["n_trajectories"], **merged_quantities)

                    logger.info(f"  [OK] Terms computed for satellite sat_0")
                except Exception as e:
                    logger.error(f"Method {method}, nbsatellite={self.nbsatellite} : {e}")
                    raise
            
            elif self.nbsatellite == 4:
                try:
                    sat1 = 'sat_0'  # Reference satellite for merging quantities
                    for sat2 in self._sat_names:
                        merged_quantities = {}  # Initialize an array to hold merged quantities for each trajectory and point
                        for quantity in dic_quantities[sat1].keys():
                            if quantity in dic_quantities[sat2]:
                                merged_quantities.update({quantity: np.concatenate((
                                    dic_quantities[sat1][quantity],
                                    dic_quantities[sat2][quantity]
                                    ), axis=1)}) # Merge along points axis (axis=1) to create arrays of shape (n_trajectories, 2*n_points)
                        set_num_threads(self.max_workers)  # Set numba to use the specified number of threads
                        for term_name in required_terms:
                            term_obj = TERMS[term_name]
                            if term_name in self.FLUX_TERMS:
                                result[sat2][term_name] = term_obj.calc_incr_traj(self.traj_param["n_points"], self.traj_param["n_trajectories"], **merged_quantities)

                            elif term_name in self.SOURCE_TERMS and sat2 == 'sat_0':  # Compute source terms only for reference satellite
                                result[sat2][term_name] = term_obj.calc_incr_traj(self.traj_param["n_points"], self.traj_param["n_trajectories"], **merged_quantities)

                        logger.info(f"  [OK] Terms computed for satellite {sat2}")
                except Exception as e:
                    logger.error(f"Method {method}, nbsatellite={self.nbsatellite} : {e}")
                    raise

        elif method == "fourier":
            if self.nbsatellite == 1:
                try:
                    for term_name in required_terms:
                        term_obj = TERMS[term_name]
                        result['sat_0'][term_name] = term_obj.calc_fourier(**dic_quantities['sat_0'], dic_param=self._sat_param_cache['sat_0'], traj=True)
                        if not isinstance(result['sat_0'][term_name], np.ndarray):
                            result['sat_0'][term_name] = np.asarray(result['sat_0'][term_name])
                    logger.info(f"  [OK] Terms computed for satellite sat_0")
                except Exception as e:
                    logger.error(f"Method {method}, nbsatellite={self.nbsatellite} : {e}")
                    raise
        
            elif self.nbsatellite == 4:
                try:
                    for term_name in required_terms:
                        term_obj = TERMS[term_name]
                        if term_name in self.FLUX_TERMS:
                            result['sat_0'][term_name] = term_obj.calc_with_fourier_4sat(**dic_quantities['sat_0'], dic_param=self._sat_param_cache['sat_0'], traj=True)
                        elif term_name in self.SOURCE_TERMS:
                            result['sat_0'][term_name] = term_obj.calc_fourier(**dic_quantities['sat_0'], dic_param=self._sat_param_cache['sat_0'], traj=True)
                        if not isinstance(result['sat_0'][term_name], np.ndarray):
                            result['sat_0'][term_name] = np.asarray(result['sat_0'][term_name])
                    logger.info(f"  [OK] Terms computed for satellite sat_0")
                except Exception as e:
                    logger.error(f"Method {method}, nbsatellite={self.nbsatellite} : {e}")
                    raise
        
        if self.verbose:
            logger.info(f"  [OK] All {len(required_terms)} terms computed successfully:")
            logger.info(required_terms)
        
        self.terms_to_h5(result_terms=result, filename=filename)

        return result
    
    @njit(parallel=True)
    def calc_incremental_trajectories(self, result:dict[np.ndarray], merged_quantities: dict, n_trajectories:int, n_points:int):

        for dl in prange(n_points):
            for t in prange(n_points):
                tp = t + (n_points + dl) - n_points * (t + n_points + dl >= 2 * n_points)
                for term_name, term_obj in TERMS.items():
                    if term_name in self.FLUX_TERMS:
                        result[term_name][[0, 1, 2], :, dl] += term_obj.calc_in_point_with_sympy_traj(t, tp, **merged_quantities)
                    elif term_name in self.SOURCE_TERMS:
                        result[term_name][:,dl] += term_obj.calc_in_point_with_sympy_traj(t, tp, **merged_quantities)

    def terms_to_h5(self, result_terms: dict, filename: str = "terms_trajectory.h5"):
        """
        Save computed terms to HDF5 file.
        
        Parameters:
        -----------
        result_terms : dict
            Data structure: {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
        filename : str
            Output HDF5 filename
        """
        
        with h5py.File(filename, 'w') as f:
            for sat_name, terms_dict in result_terms.items():
                # Create satellite group and store all terms
                sat_group = f.create_group(sat_name)
                for term_name, term_value in terms_dict.items():
                    sat_group.create_dataset(term_name, data=term_value, 
                        compression='gzip', compression_opts=4)
    
    # ========== PRIVATE METHODS ==========
    
    def _prepare_dic_param_for_terms_and_coeffs(self, dic_param: dict):
        """
        Prepare dic_param for terms_and_coeffs() by converting list-based parameters to scalars.
        
        This is necessary because terms_and_coeffs() expects scalar values rather than
        arrays, while the trajectory data uses arrays. Takes first value for uniform parameters.
        
        Parameters:
        -----------
        dic_param : dict
            Dictionary with potentially list-based parameters
        
        Returns:
        -------
        dict : Cleaned dictionary with scalar parameters
        """
        params_clean = {}
        
        for key, value in dic_param.items():
            if isinstance(value, list):
                # Extract first element (same parameter for all trajectories)
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
    
    def _extract_sat_parameters(self, sat_name: str):
        """
        Extract satellite-specific physical parameters from dic_param.
        
        Handles complex parameter structures: dict of satellites, lists of values,
        and scalars. Returns parameters applicable to the given satellite.
        
        Parameters:
        -----------
        sat_name : str
            Satellite name (e.g., 'sat_0')
        
        Returns:
        -------
        dict : Satellite-specific parameters
        """
        dic_param_sat = {}
        for key, value in self.physical_param.items():
            if isinstance(value, dict) and sat_name in value:
                dic_param_sat[key] = value[sat_name]
            elif isinstance(value, list):
                dic_param_sat[key] = value[0]
            else:
                dic_param_sat[key] = value
        
        return dic_param_sat
    
# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def compute_all_terms_for_laws(dic_quantities: dict = None, grid_param: dict = None, traj_param: dict = None, physical_param: dict = None, filename:str = "terms_trajectory.h5", laws: list = None, method: str = None, verbose: bool = False, max_workers: int = np.nan):
    """
    Backward compatibility wrapper for compute_all_terms_for_laws.
    
    Deprecated: Use TrajectoryTermsComputer class instead.
    """
    computer = TrajectoryTermsComputer(verbose=verbose, 
                                      physical_param=physical_param, 
                                      traj_param=traj_param,
                                      grid_param=grid_param,
                                      max_workers=max_workers)
    return computer.compute_all_terms_for_laws(dic_quantities, laws, method, filename)
