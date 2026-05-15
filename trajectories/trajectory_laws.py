# trajectory_laws.py
"""
Module to compute law terms along a trajectory.
Uses terms computed by trajectory_terms.py directly.
For divergences, uses a common factor per law instead of computing it explicitly.

Architecture: Class-based (TrajectoryLawsComputer) for parameter management
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
    
    Applies law coefficients to computed terms and handles divergence calculations.
    Manages both single satellite and multi-satellite configurations with
    uniform data structure.
    
    Attributes are maintained across operations to avoid repeatedly passing parameters.
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
    
    def compute_laws_terms(self, dic_terms: dict, laws=None, filename="laws_terms.h5"):
        """
        Main entry point: Compute law terms with coefficients for all satellites.
        
        Groups flux and source terms according to law specifications, applying
        coefficients to determine the final contribution of each term to the law.
        Processes data with uniform satellite structure.
        
        Parameters:
        -----------
        dic_terms : dict
            Dictionary of computed terms with uniform structure:
            {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
        laws : list[str]
            List of law names to process
        
        Returns:
        -------
        tuple : (dic_law_terms dict, dic_coefficients dict)
                dic_law_terms: {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
                dic_coefficients: {law_name_term: coeff_value, ...}
        """
        
        if self.verbose:
            logging.info("\n" + "="*70)
            logging.info("COMPUTING LAW TERMS WITH COEFFICIENTS")
            logging.info(f"  Nbsatellite:  {self.nbsatellite}")
        
        if laws is None:
            laws = []
        
        # Initialize result structure with satellites
        dic_law_terms = {'sat_'+str(i): {} for i in range(self.nbsatellite)}
        dic_coefficients = {}
        
        # Compute terms only once (they are identical for all laws)
        for law_name in laws:
            if law_name not in LAWS:
                if self.verbose:
                    logger.warning(f"Law '{law_name}' not found")
                continue
            
            if self.verbose:
                logging.info(f"Processing law: {law_name}")
            
            try:
                law_obj = LAWS[law_name]
                
                if self.traj_param['nbsatellite'] == 1:
                    # Single satellite case: compute once and replicate for uniform structure                                        
                    # Calculate law terms and coefficients for this satellite
                    law_terms, law_coeffs = self._apply_law_coefficients_1satellite(
                        dic_terms['sat_0'], 
                        law_obj, 
                    )

                elif self.traj_param['nbsatellite'] == 4:
                    # Multi-satellite case: compute for the main satellite
                    # Calculate law terms and coefficients for the main satellite
                    law_terms, law_coeffs = self._apply_law_coefficients_4satellite(
                        dic_terms,
                        law_obj,
                    )

                # # Apply roll shift to center trajectories on last axis (points only)
                # for term_key, term_array in law_terms.items():
                #     if isinstance(term_array, np.ndarray) and term_array.ndim >= 1:
                #         # Roll shift on last axis (points dimension)
                #         shift = term_array.shape[-1] // 2
                #         law_terms[term_key] = np.roll(term_array, shift=shift, axis=-1)
                
                # Store terms for this satellite
                dic_law_terms['sat_0'].update(law_terms)

                # For the first law, store the coefficients (they will be identical for others)
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
        
        Handles multi-trajectory array structure by preserving dimensions
        through all operations. Matches terms with coefficients and applies
        divergence factors as needed.
        
        Parameters:
        -----------
        dic_terms_sat : dict
            Dictionary of computed terms for a single satellite:
            {term_name: array(n_trajectories, n_points)}
        law_obj : AbstractLaw
            Law object containing terms_and_coeffs() method
        physical_param_sat : dict
            Physical parameters (satellite-specific if nbsatellite=4)
        
        Returns:
        -------
        tuple : (result dict, coefficients dict)
        """
        
        # Clean parameters for terms_and_coeffs
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs = law_obj.terms_and_coeffs(params_clean)
        result = {}
        
        # PRE-FILTER coefficients by type (ONE-TIME, not per-iteration)
        div_coeffs = {k: v for k, v in coeffs.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs.items() 
                         if not k.startswith(('div_', 'source_'))}
        
        incomputable_terms = []
        applied_terms = []
        
        # Process divergence terms
        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms_sat:
                term_value = dic_terms_sat[term_name]
                result[coeff_key] = divergence_1satellite(term_value, self.traj_param)
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, f"term '{term_name}' not computed"))
        
        # Process source terms
        for coeff_key, coeff_value in source_coeffs.items():
            if coeff_key in dic_terms_sat:
                result[coeff_key] = dic_terms_sat[coeff_key]
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))
        
        # Process simple terms (no div_ or source_ prefix)
        for coeff_key, coeff_value in simple_coeffs.items():
            if coeff_key in dic_terms_sat:
                term_value = dic_terms_sat[coeff_key]
                result[coeff_key] = term_value
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))
        
        if self.verbose and incomputable_terms:
            logging.info(f"    [WARNING] {len(incomputable_terms)} terms could not be computed")
        
        return result, coeffs
    
    def _apply_law_coefficients_4satellite(self, dic_terms: dict, law_obj):
        """
        Apply law coefficients to computed terms for four satellites.
        
        Handles multi-trajectory array structure by preserving dimensions
        through all operations. Matches terms with coefficients and applies
        divergence factors as needed, using satellite-specific physical parameters.
        
        Parameters:
        -----------
        dic_terms : dict
            Dictionary of computed terms for all satellites with uniform structure:
            {term_name: array(n_trajectories, n_points)}
        law_obj : AbstractLaw
            Law object containing terms_and_coeffs() method
        physical_param_sat : dict
            Satellite-specific physical parameters
        Returns:
        -------
        tuple : (result dict, coefficients dict)
        """
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs_sat_0 = law_obj.terms_and_coeffs(params_clean)
        result = {}

        # PRE-FILTER coefficients by type (ONE-TIME, not per-iteration)
        div_coeffs = {k: v for k, v in coeffs_sat_0.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs_sat_0.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs_sat_0.items() 
                         if not k.startswith(('div_', 'source_'))}
        
        incomputable_terms = []
        applied_terms = [] 

        # Process divergence terms
        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms['sat_0']:
                result[coeff_key] = divergence_4satellite(dic_terms, term_name, self.traj_param)
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, f"term '{term_name}' not computed"))

        # Process source terms
        for coeff_key, coeff_value in source_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))

        # Process simple terms (no div_ or source_ prefix)
        for coeff_key, coeff_value in simple_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not computed"))
    
        return result, coeffs_sat_0

    def _prepare_dic_param_for_terms_and_coeffs(self, dic_param: dict):
        """
        Prepare dic_param for terms_and_coeffs() by converting list-based parameters to scalars.
        
        Parameters:
        -----------
        dic_param : dict
            Dictionary with potentially list-based parameters (one value per trajectory)
        
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

    def laws_to_h5(self, dic_law_terms, dic_coefficients, filename:str="laws_terms.h5"):
        """
        Save law terms and coefficients to an HDF5 file.
        
        Stores computed law terms and their associated coefficients for later analysis.
        Preserves satellite and term structure.
        
        Parameters:
        -----------
        dic_law_terms : dict
            Dictionary of computed law terms with uniform structure:
            {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
        dic_coefficients : dict
            Dictionary of coefficients mapping term names to their numerical weights
        filename : str
            Output HDF5 filename (default: "laws_terms.h5")
        """
    
        with h5py.File(filename, 'w') as f:
            # Save law terms with satellite groups
            law_terms_group = f.create_group('law_terms')
            for sat_name, terms_dict in dic_law_terms.items():
                sat_group = law_terms_group.create_group(sat_name)
                for term_key, value in terms_dict.items():
                    sat_group.create_dataset(term_key, data=value)
            
            # Save coefficients
            coeffs_group = f.create_group('coefficients')
            for coeff_key, coeff_value in dic_coefficients.items():
                coeffs_group.create_dataset(coeff_key, data=coeff_value)
        
        logging.info(f"  [OK] Saved law terms for {len(dic_law_terms)} satellite(s) to {filename}")

# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def compute_laws_terms_with_coefficients(dic_terms, physical_param=None, traj_param=None,
                                        filename="laws_terms.h5",
                                        laws=None, 
                                        verbose=False):
    """
    Backward compatibility wrapper for compute_laws_terms.
    
    Deprecated: Use TrajectoryLawsComputer.compute_laws_terms instead.
    
    Usage:
        computer = TrajectoryLawsComputer(verbose=True, 
                                         physical_param=physical_param,
                                         traj_param=traj_param)
        dic_law_terms, dic_coeffs = computer.compute_laws_terms(dic_terms, laws)
    """
    computer = TrajectoryLawsComputer(verbose=verbose, 
                                     physical_param=physical_param,
                                     traj_param=traj_param)
    return computer.compute_laws_terms(dic_terms, laws, filename=filename)