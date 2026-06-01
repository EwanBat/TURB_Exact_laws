# trajectory_laws.py
"""
Compute law terms with coefficients along trajectories using fully vectorized operations.
Applies law coefficients to computed terms and handles divergence calculations.

Data structure (uniform across all methods):
    - Single satellite (nbsatellite=1):
        {sat_0: {term_name: array(n_trajectories, n_points), ...}}
    - Four satellites (nbsatellite=4):
        {sat_0: {...}, sat_1: {...}, sat_2: {...}, sat_3: {...}}
        where sat_i contains computed terms for each satellite.

Key design: Terms are computed once per law; data shape (n_trajectories, n_points) preserved.
"""

import numpy as np
import logging
import h5py

from exact_laws.el_calc_mod.laws import LAWS
from trajectories.derivation_satellite import divergence_1satellite, divergence_4satellite

logger = logging.getLogger(__name__)


# ========== TRAJECTORY LAWS COMPUTER CLASS ==========

class TrajectoryLawsComputer:
    """
    Compute law terms with coefficients along trajectories.
    
    Applies law coefficients to computed terms, handles divergence calculations,
    and manages both single-satellite and 4-satellite configurations.
    All data maintains structure: {sat_name: {term_name: array(n_traj, n_pts)}}
    """
    
    # ========== INITIALIZATION ==========
    
    def __init__(self, verbose: bool = False, physical_param: dict = None, traj_param: dict = None):
        """
        Initialize the trajectory laws computer.
        
        Parameters:
        -----------
        verbose : bool
            Enable detailed logging
        physical_param : dict, optional
            Physical parameters (can be set later)
        traj_param : dict, optional
            Trajectory parameters (can be set later)
        """
        self.verbose = verbose
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)
    
    # ========== PUBLIC METHODS ==========
    
    def compute_laws_terms(self, dic_terms: dict, laws=None, filename="laws_terms.h5", method: str = None):
        """
        Compute law terms with coefficients for all satellites.
        
        Processes each law by extracting its term requirements and coefficients,
        then applying coefficients to available computed terms. Divergence terms
        are computed on-demand using satellite geometry.
        
        Parameters:
        -----------
        dic_terms : dict
            {sat_name: {term_name: array(n_trajectories, n_points)}}
            Example: {sat_0: {v: array(...), b: array(...), ...}}
        laws : list[str]
            Law names to process (e.g., ['PP98', 'BG17'])
        filename : str
            Output HDF5 file path
        method : str, optional
            Computation method ('incremental' or 'fourier')
        
        Returns:
        -------
        tuple : (dic_law_terms, dic_coefficients)
                - dic_law_terms: {sat_name: {term_coeff_key: array(n_traj, n_pts)}}
                - dic_coefficients: {law_term_key: coefficient_value}
        """
        
        if self.verbose:
            logging.info("\n" + "="*70)
            logging.info("COMPUTING LAW TERMS WITH COEFFICIENTS")
            logging.info(f"  Nbsatellite:  {self.nbsatellite}")
        
        if laws is None:
            laws = []
        
        dic_law_terms = {'sat_'+str(i): {} for i in range(self.nbsatellite)}
        dic_coefficients = {}
        
        for law_name in laws:
            if law_name not in LAWS:
                if self.verbose:
                    logger.warning(f"Law '{law_name}' not found")
                continue
            
            if self.verbose:
                logging.info(f"Processing law: {law_name}")
            
            try:
                law_obj = LAWS[law_name]
                
                if self.nbsatellite == 1:
                    law_terms, law_coeffs = self._apply_law_coefficients_1satellite(
                        dic_terms['sat_0'], law_obj
                    )
                elif self.nbsatellite == 4:
                    law_terms, law_coeffs = self._apply_law_coefficients_4satellite(
                        dic_terms, law_obj, method=method
                    )
                
                dic_law_terms['sat_0'].update(law_terms)
                
                for term_key, coeff_value in law_coeffs.items():
                    dic_coefficients[f"{law_name}_{term_key}"] = coeff_value
                
                if self.verbose:
                    logging.info(f"  [OK] Terms computed for {len(list(dic_terms.keys()))} satellite(s)")
                    logging.info(f"    Applied terms: {list(law_terms.keys())}")
            
            except Exception as e:
                logger.error(f"Failed to process {law_name}: {e}")
        
        self.laws_to_h5(dic_law_terms, dic_coefficients, filename=filename)
        return dic_law_terms, dic_coefficients
    
    # ========== PRIVATE METHODS ==========
    
    def _apply_law_coefficients_1satellite(self, dic_terms_sat: dict, law_obj):
        """
        Apply law coefficients to computed terms for a single satellite.
        
        Processes three types of term coefficients:
        1. Divergence terms (div_*): compute via divergence_1satellite()
        2. Source terms (source_*): use directly from dic_terms_sat
        3. Simple terms (other): use directly from dic_terms_sat
        
        Parameters:
        -----------
        dic_terms_sat : dict
            {term_name: array(n_trajectories, n_points)}
        law_obj : AbstractLaw
            Has terms_and_coeffs() method
        
        Returns:
        -------
        tuple : (result_dict, coefficients_dict)
        """
        
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs = law_obj.terms_and_coeffs(params_clean)
        result = {}
        
        # Partition coefficients by type: divergence, source, or simple terms
        # This categorization happens once; terms are then matched with data
        div_coeffs = {k: v for k, v in coeffs.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs.items() 
                         if not k.startswith(('div_', 'source_'))}
        
        incomputable_terms = []
        
        # Divergence terms: extract base term name and apply divergence operator
        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms_sat:
                term_value = dic_terms_sat[term_name]
                result[coeff_key] = divergence_1satellite(term_value, self.traj_param)
            else:
                incomputable_terms.append((coeff_key, f"term '{term_name}' not computed"))
        
        # Source terms: use directly if available
        for coeff_key, coeff_value in source_coeffs.items():
            if coeff_key in dic_terms_sat:
                result[coeff_key] = dic_terms_sat[coeff_key]
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))
        
        # Simple terms: use directly if available
        for coeff_key, coeff_value in simple_coeffs.items():
            if coeff_key in dic_terms_sat:
                term_value = dic_terms_sat[coeff_key]
                result[coeff_key] = term_value
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))
        
        if self.verbose and incomputable_terms:
            logging.info(f"    [WARNING] {len(incomputable_terms)} terms could not be computed")
        
        return result, coeffs
    
    def _apply_law_coefficients_4satellite(self, dic_terms: dict, law_obj, method: str = None):
        """
        Apply law coefficients to computed terms for four satellites.
        
        Similar to 1-satellite case but:
        - Divergence terms computed using all 4 satellites (satellite geometry)
        - Method parameter ('incremental' or 'fourier') controls divergence calculation
        - Results stored in sat_0 (gradient-based terms)
        
        Parameters:
        -----------
        dic_terms : dict
            {sat_0: {...}, sat_1: {...}, sat_2: {...}, sat_3: {...}}
        law_obj : AbstractLaw
            Has terms_and_coeffs() method
        method : str, optional
            'incremental' or 'fourier' for divergence calculation
        
        Returns:
        -------
        tuple : (result_dict, coefficients_dict)
        """
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs_sat_0 = law_obj.terms_and_coeffs(params_clean)
        result = {}
        
        # Partition coefficients by type: divergence, source, or simple terms
        div_coeffs = {k: v for k, v in coeffs_sat_0.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs_sat_0.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs_sat_0.items() 
                         if not k.startswith(('div_', 'source_'))}
        
        incomputable_terms = []

        # Divergence terms: compute using satellite geometry or pass through (fourier)
        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms['sat_0']:
                if method == "incremental":
                    result[coeff_key] = divergence_4satellite(dic_terms, term_name, self.traj_param)
                elif method == "fourier":
                    result[coeff_key] = dic_terms['sat_0'][term_name]
            else:
                incomputable_terms.append((coeff_key, f"term '{term_name}' not computed"))

        # Source terms: use directly if available
        for coeff_key, coeff_value in source_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))

        # Simple terms: use directly if available
        for coeff_key, coeff_value in simple_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))
    
        return result, coeffs_sat_0

    def _prepare_dic_param_for_terms_and_coeffs(self, dic_param: dict):
        """
        Extract scalar values from parameter dictionary for law.terms_and_coeffs().
        
        The law computation expects scalar parameters, but dic_param may contain
        arrays or dictionaries (one value per trajectory or per satellite).
        Extract the first value uniformly for all trajectories.
        
        Parameters:
        -----------
        dic_param : dict
            Physical parameters (potentially list or dict values)
        
        Returns:
        -------
        dict : Cleaned dictionary with scalar values
        """
        params_clean = {}
        
        for key, value in dic_param.items():
            if isinstance(value, list):
                params_clean[key] = value[0]
            elif isinstance(value, dict):
                # For nbsatellite=4: extract first satellite and first trajectory
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

    def laws_to_h5(self, dic_law_terms, dic_coefficients, filename:str="laws_terms.h5"):
        """
        Save law terms and coefficients to HDF5 file.
        
        Structure:
            /law_terms/sat_0/term_key -> array(n_traj, n_pts)
            /law_terms/sat_1/term_key -> array(n_traj, n_pts)
            /coefficients/law_term_key -> scalar value
        
        Parameters:
        -----------
        dic_law_terms : dict
            {sat_name: {term_key: array(n_traj, n_pts)}}
        dic_coefficients : dict
            {law_term_key: coefficient_value}
        filename : str
            Output HDF5 file path
        """
        with h5py.File(filename, 'w') as f:
            # Save law terms with satellite groups
            law_terms_group = f.create_group('law_terms')
            for sat_name, terms_dict in dic_law_terms.items():
                sat_group = law_terms_group.create_group(sat_name)
                for term_key, value in terms_dict.items():
                    sat_group.create_dataset(term_key, data=value, compression="gzip", compression_opts=9)
            
            # Save coefficients
            coeffs_group = f.create_group('coefficients')
            for coeff_key, coeff_value in dic_coefficients.items():
                coeffs_group.create_dataset(coeff_key, data=coeff_value)
        
        logging.info(f"  [OK] Saved law terms for {len(dic_law_terms)} satellite(s) to {filename}")

# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def compute_laws_terms_with_coefficients(dic_terms, physical_param=None, traj_param=None,
                                        filename="laws_terms.h5",
                                        laws=None, method:str =None,
                                        verbose:bool=False):
    """
    Backward compatibility wrapper. Use TrajectoryLawsComputer.compute_laws_terms instead.
    """
    computer = TrajectoryLawsComputer(verbose=verbose, 
                                     physical_param=physical_param,
                                     traj_param=traj_param)
    return computer.compute_laws_terms(dic_terms, laws=laws, filename=filename, method=method)