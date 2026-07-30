# trajectory_terms.py
"""
Module to compute terms along a trajectory.
Analog to trajectory_quantities.py but for terms.
Uses calc_fourier() methods from terms for trajectories.
Encapsulated in TrajectoryTermsComputer class for better parameter management.
"""

import numpy as np
from numba import set_num_threads
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
        "source_dvdvdv",
        "source_dbdbdv",
        "source_dvdbdb",
    ])
        
    # ========== INITIALIZATION ==========
    
    def __init__(self, verbose: bool = False, grid_param: dict = None, physical_param: dict = None, traj_param: dict = None, run_params: dict = None):
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
        self.run_params = run_params or {}
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)
    
        self._sat_names = [f'sat_{i}' for i in range(self.nbsatellite)]
        self._sat_param_cache = self._extract_sat_parameters('sat_0')

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

    def _get_incremental_fs(self):
        if self.traj_param['trajectory_method'] == "linear_x":
            return 1 / self.grid_param['c'][0]
        elif self.traj_param['trajectory_method'] == "linear_y":
            return 1 / self.grid_param['c'][1]
        elif self.traj_param['trajectory_method'] == "linear_z":
            return 1 / self.grid_param['c'][2]

    def _compute_terms_incremental_1sat(self, dic_quantities, required_terms):
        result = {'sat_0': {}}
        fs = self._get_incremental_fs()
        n_points = self.traj_param["n_points"]
        n_trajectories = self.traj_param["n_trajectories"]
        filter_enabled = self.run_params.get('filter_enabled', False)
        computed = []
        missing = []

        merged_quantities = {}
        for quantity in dic_quantities['sat_0'].keys():
            merged_quantities[quantity] = np.concatenate((
                dic_quantities['sat_0'][quantity],
                dic_quantities['sat_0'][quantity]
            ), axis=1)

        for term_name in required_terms:
            try:
                term_obj = TERMS[term_name]
                if filter_enabled:
                    result['sat_0'][term_name] = term_obj.calc_filter(
                        n_points, n_trajectories, fs, **merged_quantities)
                else:
                    result['sat_0'][term_name] = term_obj.calc_incr_traj(
                        n_points, n_trajectories, **merged_quantities)
                computed.append(term_name)
            except Exception as e:
                missing.append(term_name)
                if self.verbose:
                    logger.error(f"  [ERROR] Failed to compute term {term_name}: {e}")

        if self.verbose:
            for t in computed:
                logger.info(f"  [OK] Term {t} computed for sat_0")
            for t in missing:
                logger.warning(f"  [WARNING] Term {t} NOT computed for sat_0")

        return result

    def _compute_terms_incremental_4sat(self, dic_quantities, required_terms):
        result = {sat_name: {} for sat_name in self._sat_names}
        fs = self._get_incremental_fs()
        sat1 = 'sat_0'
        n_points = self.traj_param["n_points"]
        n_trajectories = self.traj_param["n_trajectories"]
        filter_enabled = self.run_params.get('filter_enabled', False)

        set_num_threads(self.run_params.get('max_workers', 1))

        flux_terms = [t for t in required_terms if t in self.FLUX_TERMS]
        source_terms = [t for t in required_terms if t in self.SOURCE_TERMS]

        for sat2 in self._sat_names:
            computed = []
            missing = []

            merged_quantities = {}
            for quantity in dic_quantities[sat1].keys():
                if quantity in dic_quantities[sat2]:
                    merged_quantities[quantity] = np.concatenate((
                        dic_quantities[sat1][quantity],
                        dic_quantities[sat2][quantity]
                    ), axis=1)

            for term_name in flux_terms:
                try:
                    term_obj = TERMS[term_name]
                    if filter_enabled:
                        result[sat2][term_name] = term_obj.calc_filter(
                            n_points, n_trajectories, fs, **merged_quantities)
                    else:
                        result[sat2][term_name] = term_obj.calc_incr_traj(
                            n_points, n_trajectories, **merged_quantities)
                    computed.append(term_name)
                except Exception as e:
                    missing.append(term_name)
                    if self.verbose:
                        logger.error(f"  [ERROR] Failed to compute term {term_name} for {sat2}: {e}")

            if sat2 == 'sat_0':
                for term_name in source_terms:
                    try:
                        term_obj = TERMS[term_name]
                        if filter_enabled:
                            result[sat2][term_name] = term_obj.calc_filter(
                                n_points, n_trajectories, fs, **merged_quantities)
                        else:
                            result[sat2][term_name] = term_obj.calc_incr_traj(
                                n_points, n_trajectories, **merged_quantities)
                        computed.append(term_name)
                    except Exception as e:
                        missing.append(term_name)
                        if self.verbose:
                            logger.error(f"  [ERROR] Failed to compute term {term_name} for {sat2}: {e}")

            if self.verbose:
                for t in computed:
                    logger.info(f"  [OK] Term {t} computed for {sat2}")
                for t in missing:
                    logger.warning(f"  [WARNING] Term {t} NOT computed for {sat2}")

        return result

    def _compute_terms_incremental_9sat(self, dic_quantities, required_terms):
        result = {sat_name: {} for sat_name in self._sat_names}
        fs = self._get_incremental_fs()
        sat1 = 'sat_0'
        n_points = self.traj_param["n_points"]
        n_trajectories = self.traj_param["n_trajectories"]
        filter_enabled = self.run_params.get('filter_enabled', False)

        set_num_threads(self.run_params.get('max_workers', 1))

        flux_terms = [t for t in required_terms if t in self.FLUX_TERMS]
        source_terms = [t for t in required_terms if t in self.SOURCE_TERMS]
        other_terms = [t for t in required_terms if t not in self.FLUX_TERMS and t not in self.SOURCE_TERMS]

        for sat2 in self._sat_names:
            computed = []
            missing = []

            merged_quantities = {}
            for quantity in dic_quantities[sat1].keys():
                if quantity in dic_quantities[sat2]:
                    merged_quantities[quantity] = np.concatenate((
                        dic_quantities[sat1][quantity],
                        dic_quantities[sat2][quantity]
                    ), axis=1)

            for term_name in flux_terms + other_terms:
                try:
                    term_obj = TERMS[term_name]
                    if filter_enabled:
                        result[sat2][term_name] = term_obj.calc_filter(
                            n_points, n_trajectories, fs, **merged_quantities)
                    else:
                        result[sat2][term_name] = term_obj.calc_incr_traj(
                            n_points, n_trajectories, **merged_quantities)
                    computed.append(term_name)
                except Exception as e:
                    missing.append(term_name)
                    if self.verbose:
                        logger.error(f"  [ERROR] Failed to compute term {term_name} for {sat2}: {e}")

            if self.verbose:
                for t in computed:
                    logger.info(f"  [OK] Term {t} computed for {sat2}")
                for t in missing:
                    logger.warning(f"  [WARNING] Term {t} NOT computed for {sat2}")

        if source_terms:
            merged_quantities = {}
            for quantity in dic_quantities['sat_0'].keys():
                merged_quantities[quantity] = np.concatenate((
                    dic_quantities['sat_0'][quantity],
                    dic_quantities['sat_0'][quantity]
                ), axis=1)

            for term_name in source_terms:
                try:
                    term_obj = TERMS[term_name]
                    if filter_enabled:
                        result['sat_0'][term_name] = term_obj.calc_filter(
                            n_points, n_trajectories, fs, **merged_quantities)
                    else:
                        result['sat_0'][term_name] = term_obj.calc_incr_traj(
                            n_points, n_trajectories, **merged_quantities)
                    if self.verbose:
                        logger.info(f"  [OK] Source term {term_name} computed from sat_0")
                except Exception as e:
                    if self.verbose:
                        logger.error(f"  [ERROR] Failed source term {term_name} for sat_0: {e}")

        return result

    def _compute_terms_fourier_1sat(self, dic_quantities, required_terms):
        result = {'sat_0': {}}
        computed = []
        missing = []

        for term_name in required_terms:
            try:
                term_obj = TERMS[term_name]
                result['sat_0'][term_name] = term_obj.calc_fourier(
                    **dic_quantities['sat_0'], dic_param=self._sat_param_cache, traj=True)
                if not isinstance(result['sat_0'][term_name], np.ndarray):
                    result['sat_0'][term_name] = np.asarray(result['sat_0'][term_name])
                computed.append(term_name)
            except Exception as e:
                missing.append(term_name)
                if self.verbose:
                    logger.error(f"  [ERROR] Failed to compute term {term_name}: {e}")

        if self.verbose:
            for t in computed:
                logger.info(f"  [OK] Term {t} computed for sat_0")
            for t in missing:
                logger.warning(f"  [WARNING] Term {t} NOT computed for sat_0")

        return result

    def _compute_terms_fourier_multi(self, dic_quantities, required_terms):
        result = {'sat_0': {}}
        computed = []
        missing = []
        for term_name in required_terms:
            try:
                term_obj = TERMS[term_name]
                if term_name in self.FLUX_TERMS:
                    result['sat_0'][term_name] = term_obj.calc_with_fourier_4sat(
                        **dic_quantities['sat_0'], dic_param=self._sat_param_cache, traj=True)
                elif term_name in self.SOURCE_TERMS:
                    result['sat_0'][term_name] = term_obj.calc_fourier(
                        **dic_quantities['sat_0'], dic_param=self._sat_param_cache, traj=True)
                else:
                    missing.append(term_name)
                    continue
                if not isinstance(result['sat_0'][term_name], np.ndarray):
                    result['sat_0'][term_name] = np.asarray(result['sat_0'][term_name])
                computed.append(term_name)
            except Exception as e:
                missing.append(term_name)
                if self.verbose:
                    logger.error(f"  [ERROR] Failed to compute term {term_name}: {e}")

        if self.verbose:
            for t in computed:
                logger.info(f"  [OK] Term {t} computed for sat_0")
            for t in missing:
                logger.warning(f"  [WARNING] Term {t} NOT computed for sat_0")

        return result

    def compute_all_terms_for_laws(self, dic_quantities: dict = None, laws: list = None, filename: str = "terms_trajectory.h5"):
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
        filename : str
            Output HDF5 filename
        
        Returns:
        -------
        dict : {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
        """

        if laws is None:
            laws = []

        required_terms = self.list_required_terms(laws)

        if self.verbose:
            logger.info("\n" + "-" * 70)
            logger.info("FLUX AND SOURCE TERMS COMPUTATION")
            logger.info(f"  Computing {len(required_terms)} terms for {len(laws)} law(s)")

        method = self.run_params.get('method')

        if method == "incremental":
            if self.nbsatellite == 1:
                result = self._compute_terms_incremental_1sat(dic_quantities, required_terms)
            elif self.nbsatellite == 4:
                result = self._compute_terms_incremental_4sat(dic_quantities, required_terms)
            elif self.nbsatellite == 9:
                result = self._compute_terms_incremental_9sat(dic_quantities, required_terms)
            else:
                raise ValueError(f"Unsupported nbsatellite={self.nbsatellite} for method=incremental")
        elif method == "fourier":
            if self.nbsatellite == 1:
                result = self._compute_terms_fourier_1sat(dic_quantities, required_terms)
            elif self.nbsatellite in (4, 9):
                result = self._compute_terms_fourier_multi(dic_quantities, required_terms)
            else:
                raise ValueError(f"Unsupported nbsatellite={self.nbsatellite} for method=fourier")
        else:
            raise ValueError(f"Unknown method: {method}")

        if filename:
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

def compute_all_terms_for_laws(dic_quantities: dict = None, grid_param: dict = None, traj_param: dict = None, physical_param: dict = None, run_params: dict = None, filename:str = "terms_trajectory.h5", laws: list = None, verbose: bool = False):
    """
    Backward compatibility wrapper for compute_all_terms_for_laws.
    
    Deprecated: Use TrajectoryTermsComputer class instead.
    """
    computer = TrajectoryTermsComputer(verbose=verbose, 
                                      physical_param=physical_param, 
                                      traj_param=traj_param,
                                      grid_param=grid_param,
                                      run_params=run_params)
    return computer.compute_all_terms_for_laws(dic_quantities, laws, filename)
