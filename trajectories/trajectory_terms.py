# trajectory_terms.py
"""
Module to compute terms along a trajectory.
Analog to trajectory_quantities.py but for terms.
Uses calc_fourier() methods from terms for trajectories.
Encapsulated in TrajectoryTermsComputer class for better parameter management.
"""

import numpy as np
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
    
    def __init__(self, verbose: bool = False, physical_param: dict = None, traj_param: dict = None):
        """
        Initialize the trajectory terms computer.
        
        Parameters:
        -----------
        verbose : bool
            Enable detailed logger
        physical_param : dict, optional
            Physical parameters
        traj_param : dict, optional
            Trajectory parameters including 'nbsatellite'
        """
        self.verbose = verbose
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
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
    
    def compute_term_from_TERMS(self, term_name: str, dic_quant: dict, method: str = None):
        """
        Compute a single term using the calc_fourier method from TERMS.
        
        Handles single satellite or 4-satellite formations by extracting 
        satellite-specific data before computation.
        
        Parameters:
        -----------
        term_name : str
            Name of the term to compute (e.g., "flux_dvdvdv", "bg17_vwv")
        dic_quant : dict
            Data structure: {sat_name: {var_name: array(n_trajectories, n_points)}, ...}
            Each array contains data for multiple trajectories at multiple time points
        method : str
            Computation method ("fourier" or "incremental")
        
        Returns:
        -------
        dict : {sat_name: array(n_trajectories, n_points)}
        
        Raises:
        -------
        ValueError : If term not found in TERMS or method is unsupported
        """
        
        if term_name not in TERMS:
            raise ValueError(f"Term '{term_name}' not found in TERMS")
        
        term_obj = TERMS[term_name]
        result = {}
        
        try:                        
            if self.nbsatellite == 1:
                # Single satellite: extract sat_0 data, compute once, replicate structure
                sat_name = 'sat_0'
                dic_param_sat = self._sat_param_cache[sat_name]

                if method == "fourier":
                    result[sat_name] = term_obj.calc_fourier(**dic_quant[sat_name], dic_param=dic_param_sat, traj=True)
                elif method == "incremental":
                    result[sat_name] = term_obj.calc_incremental_trajectories(dic_quant, self.traj_param, 'sat_0', sat_name)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                if not isinstance(result[sat_name], np.ndarray):
                    result[sat_name] = np.asarray(result[sat_name])
                        
            elif self.nbsatellite == 4:
                # 4-satellite formation: different handling for flux vs source terms
                if method == "fourier":
                    sat_name = 'sat_0'
                    dic_param_sat = self._sat_param_cache[sat_name]

                    if term_name in self.FLUX_TERMS:
                        result[sat_name] = term_obj.calc_with_fourier_4sat(**dic_quant[sat_name], dic_param=dic_param_sat, traj=True)
                    elif term_name in self.SOURCE_TERMS:
                        result[sat_name] = term_obj.calc_fourier(**dic_quant[sat_name], dic_param=dic_param_sat, traj=True)
                    
                    if not isinstance(result[sat_name], np.ndarray):
                        result[sat_name] = np.asarray(result[sat_name])

                elif method == "incremental":
                    # Flux terms computed for all 4 satellites; source terms for sat_0 only
                    if term_name in self.FLUX_TERMS:
                        for sat_name in self._sat_names:
                            result[sat_name] = term_obj.calc_incremental_trajectories(dic_quant, self.traj_param,'sat_0', sat_name)
                            if not isinstance(result[sat_name], np.ndarray):
                                result[sat_name] = np.asarray(result[sat_name])
                    elif term_name in self.SOURCE_TERMS:
                        sat_name = 'sat_0'
                        result[sat_name] = term_obj.calc_incremental_trajectories(dic_quant, self.traj_param,'sat_0', sat_name)
                        if not isinstance(result[sat_name], np.ndarray):
                            result[sat_name] = np.asarray(result[sat_name])
                else:
                    raise ValueError(f"Unsupported method: {method}")

        except Exception as e:
            if self.verbose:
                logger.error(f"Failed to compute {term_name}: {e} for satellite {sat_name}")
            raise
        
        return result

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
        satellite_names = list(dic_quantities.keys())
        result = {sat_name: {} for sat_name in satellite_names}
        
        for term_name in required_terms:
            try:
                computed = self.compute_term_from_TERMS(term_name, dic_quantities, method=method)
                
                # Store results maintaining satellite structure
                for sat_name in computed.keys():
                    result[sat_name][term_name] = computed[sat_name]
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {term_name}: {str(e)}")
        
        if self.verbose:
            logger.info(f"  [OK] All {len(required_terms)} terms computed successfully:")
            logger.info(required_terms)
        
        self.terms_to_h5(result_terms=result, filename=filename)

        return result
    
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

def compute_all_terms_for_laws(dic_quantities: dict = None, traj_param: dict = None, physical_param: dict = None, filename:str = "terms_trajectory.h5", laws: list = None, method: str = None, verbose: bool = False):
    """
    Backward compatibility wrapper for compute_all_terms_for_laws.
    
    Deprecated: Use TrajectoryTermsComputer class instead.
    """
    computer = TrajectoryTermsComputer(verbose=verbose, 
                                      physical_param=physical_param, 
                                      traj_param=traj_param)
    return computer.compute_all_terms_for_laws(dic_quantities, laws, method, filename)
