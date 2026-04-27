# trajectory_laws.py
"""
Module to compute law terms along a trajectory.
Uses terms computed by trajectory_terms.py directly.
For divergences, uses a common factor per law instead of computing it explicitly.
"""

import numpy as np
import logging
import h5py

from exact_laws.el_calc_mod.laws import LAWS
from trajectories.derivation_satellite import divergence_1satellite

logger = logging.getLogger(__name__)


# Divergence replacement factors
# A unique factor per law that applies to ALL divergence terms of that law
# Format: {'law_name': factor}
# For now, all factors are set to 1
DIVERGENCE_REPLACEMENT_FACTORS = {
    'PP98': 1.0,
    'BG17': 1.0,
    'COR_Etot': 1.0,
    'Hallcor': 1.0,
    'IHallcor': 1.0,
    'ISS22Cgl': 1.0,
    'ISS22Gyr': 1.0,
    'ISS22Iso': 1.0,
    'SS21Iso': 1.0,
    'SS21Pol': 1.0,
    'SS22Cgl': 1.0,
    'SS22Gyr_flux': 1.0,
    'SS22Gyr_sources': 1.0,
    'SS22Gyr': 1.0,
    'SS22Iso_flux': 1.0,
    'SS22Iso_sources': 1.0,
    'SS22Iso': 1.0,
    'SS22Pol': 1.0,
    'TotSS21Iso': 1.0,
    'TotSS21Pol': 1.0,
    'TotSS22Iso': 1.0,
    'TotSS22Pol': 1.0,
}


def _prepare_dic_param_for_terms_and_coeffs(dic_param, nbsatellite=1):
    """
    Prepare dic_param for terms_and_coeffs() by converting list-based parameters to scalars.
    Same as in trajectory_terms.py
    
    Parameters:
    -----------
    dic_param : dict
        Dictionary with potentially list-based parameters (one value per trajectory)
    nbsatellite : int
        Number of satellites
    
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


def apply_law_coefficients_1satellite(dic_terms_sat, law_obj, 
                                      physical_param, 
                                      traj_param,
                                      verbose=False):
    """
    Apply law coefficients to computed terms for a single satellite.
    
    Handles multi-trajectory array structure by preserving dimensions
    through all operations. Matches terms with coefficients and applies
    convergence factors as needed.
    
    Parameters:
    -----------
    dic_terms_sat : dict
        Dictionary of computed terms for a single satellite:
        {term_name: array(n_trajectories, n_points)}
    law_obj : AbstractLaw
        Law object containing terms_and_coeffs() method
    physical_param : dict
        Physical parameters (satellite-specific if nbsatellite=4)
    traj_param : dict
        Trajectory parameters (nbsatellite, etc.)
    verbose : bool
        If True, log detailed information about applied terms
    
    Returns:
    -------
    tuple : (result dict, coefficients dict)
    """
    
    # Clean parameters for terms_and_coeffs
    params_clean = _prepare_dic_param_for_terms_and_coeffs(physical_param, traj_param.get('nbsatellite', 1))
    law_terms, coeffs = law_obj.terms_and_coeffs(params_clean)
    result = {}
    
    # Get the list of flux terms of the law (those that will have a divergence)
    law_flux_terms = set(law_obj.terms) if hasattr(law_obj, 'terms') else set()
    
    # PRE-FILTER coefficients by type (ONE-TIME, not per-iteration)
    div_coeffs = {k: v for k, v in coeffs.items() if k.startswith('div_')}
    source_coeffs = {k: v for k, v in coeffs.items() if k.startswith('source_')}
    simple_coeffs = {k: v for k, v in coeffs.items() 
                     if not k.startswith(('div_', 'source_'))}
    
    incomputable_terms = []
    applied_terms = []
    
    # Process divergence terms
    for coeff_key, coeff_value in div_coeffs.items():
        base_term = coeff_key.replace('div_', '')
        if base_term in dic_terms_sat:
            term_value = dic_terms_sat[base_term]
            result[coeff_key] = divergence_1satellite(term_value, traj_param)
            applied_terms.append(coeff_key)
        else:
            incomputable_terms.append((coeff_key, f"term '{base_term}' not computed"))
    
    # Process source terms
    for coeff_key, coeff_value in source_coeffs.items():
        if coeff_key in dic_terms_sat:
            term_value = dic_terms_sat[coeff_key]
            result[coeff_key] = term_value
            applied_terms.append(coeff_key)
        else:
            incomputable_terms.append((coeff_key, "term not computed"))
    
    # Process simple terms (no div_ or source_ prefix)
    for coeff_key, coeff_value in simple_coeffs.items():
        if coeff_key in dic_terms_sat:
            term_value = dic_terms_sat[coeff_key]
            result[coeff_key] = term_value
            applied_terms.append(coeff_key)
        else:
            incomputable_terms.append((coeff_key, "term not computed"))
    
    if verbose:
        logging.info(f"  [OK] Matched {len(applied_terms)} terms")
        if incomputable_terms:
            logging.info(f"    [WARNING] {len(incomputable_terms)} terms could not be computed")
    
    return result, coeffs


def compute_laws_terms_with_coefficients(dic_terms, physical_param=None, traj_param=None,
                                        laws=None, 
                                        verbose=False):
    """
    Compute law terms with coefficients applied for all satellites.
    
    Groups flux and source terms according to law specifications, applying
    coefficients to determine the final contribution of each term to the law.
    Processes data with uniform satellite structure.
    
    Parameters:
    -----------
    dic_terms : dict
        Dictionary of computed terms with uniform structure:
        {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
    physical_param : dict
        Physical parameters
    traj_param : dict
        Trajectory parameters (nbsatellite, etc.)
    laws : list[str]
        List of law names to process
    verbose : bool
        If True, log detailed information per law
    
    Returns:
    -------
    tuple : (dic_law_terms dict, dic_coefficients dict)
            dic_law_terms: {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
            dic_coefficients: {law_name_term: coeff_value, ...}
    """
    
    if verbose:
        logging.info("\n" + "="*70)
        logging.info(f"COMPUTING LAW TERMS WITH COEFFICIENTS")

    nbsatellite = traj_param.get('nbsatellite', 1)
    if laws is None:
        laws = []
    
    # Initialize result structure with satellites
    satellite_names = list(dic_terms.keys())
    dic_law_terms = {sat_name: {} for sat_name in satellite_names}
    dic_coefficients = {}
    
    # Compute terms only once (they are identical for all laws)
    computed = False
    for law_name in laws:
        if law_name not in LAWS:
            if verbose:
                logger.warning(f"Law '{law_name}' not found")
            continue
        
        if verbose:
            logging.info(f"Processing law: {law_name}")
        
        try:
            law_obj = LAWS[law_name]
            
            # Process each satellite
            for sat_name in satellite_names:
                dic_terms_sat = dic_terms[sat_name]
                
                # Extract satellite-specific physical parameters if nbsatellite=4
                physical_param_sat = physical_param.copy() if physical_param else {}
                if nbsatellite == 4 and isinstance(physical_param_sat, dict):
                    # Convert list parameters to satellite-specific values
                    for key, value in physical_param_sat.items():
                        if isinstance(value, list):
                            # Use appropriate satellite index
                            sat_idx = int(sat_name.split('_')[1])
                            if sat_idx < len(value):
                                physical_param_sat[key] = value[sat_idx]
                            else:
                                physical_param_sat[key] = value[0]
                
                # Apply coefficients and divergence factors
                law_terms, law_coeffs = apply_law_coefficients_1satellite(
                    dic_terms_sat, 
                    law_obj, 
                    physical_param_sat,
                    traj_param,
                    verbose=verbose
                )
                
                # Apply roll shift to center trajectories on last axis (points only)
                for term_key, term_array in law_terms.items():
                    if isinstance(term_array, np.ndarray) and term_array.ndim >= 1:
                        # Roll shift on last axis (points dimension)
                        shift = term_array.shape[-1] // 2
                        law_terms[term_key] = np.roll(term_array, shift=shift, axis=-1)
                
                # Store terms for this satellite
                dic_law_terms[sat_name].update(law_terms)
            
            # For the first law, store the coefficients (they will be identical for others)
            if not computed:
                for term_key, coeff_value in law_coeffs.items():
                    dic_coefficients[f"{law_name}_{term_key}"] = coeff_value
                computed = True
            
            if verbose:
                logging.info(f"  [OK] Terms computed for {len(satellite_names)} satellite(s)")
                logging.info(f"    Applied terms: {list(law_terms.keys())}")
        
        except Exception as e:
            logger.error(f"Failed to process {law_name}: {e}")
    
    return dic_law_terms, dic_coefficients


def laws_to_h5(dic_law_terms, dic_coefficients, filename="laws_terms.h5"):
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