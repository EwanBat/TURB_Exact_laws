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
from derivation_satellite import divergence_1satellite

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


def apply_law_coefficients_1satellite(dic_terms, law_obj, 
                                      physical_param, 
                                      traj_param,
                                      verbose=False):
    """
    Apply law coefficients to computed terms.
    
    Handles multi-trajectory array structure by preserving dimensions
    through all operations. Matches terms with coefficients and applies
    convergence factors as needed.
    
    Parameters:
    -----------
    dic_terms : dict
        Dictionary of computed terms
        - nbsatellite=1: dict[term] -> array(n_trajectories, n_points)
        - nbsatellite=4: dict[term] -> array(n_satellites, n_trajectories, n_points)
    law_obj : AbstractLaw
        Law object containing terms_and_coeffs() method
    physical_param : dict
        Physical parameters
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
    
    # Determine array shape (from first computed term)
    array_shape = None
    for term_value in dic_terms.values():
        if isinstance(term_value, np.ndarray):
            array_shape = term_value.shape
            break
    
    incomputable_terms = []
    applied_terms = []
    
    for coeff_key, coeff_value in coeffs.items():
        # Determine if it's a divergence term
        is_divergence_term = coeff_key.startswith('div_')
        is_source_term = coeff_key.startswith('source_')
        
        # Find the corresponding term
        if is_divergence_term:
            # For divergences: div_flux_X → flux_X
            base_term = coeff_key.replace('div_', '')
            
            # Check if it's a flux term (listed in law_obj.terms)
            is_flux_term = base_term in law_flux_terms
            
            if base_term in dic_terms:
                term_value = dic_terms[base_term]
                # Apply the divergence factor (term already contains divergence)
                result[coeff_key] = divergence_1satellite(term_value, traj_param)
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, f"term '{base_term}' not computed"))
        
        elif is_source_term:
            # Source terms
            if coeff_key in dic_terms:
                term_value = dic_terms[coeff_key]
                # Apply ONLY the divergence factor (vectorized)
                result[coeff_key] = term_value
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, "term not computed"))
        
        else:
            # Simple term (no div_ or source_)
            if coeff_key in dic_terms:
                term_value = dic_terms[coeff_key]
                # Apply ONLY the divergence factor (vectorized)
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
    Compute law terms with coefficients applied.
    
    Groups flux and source terms according to law specifications, applying
    coefficients to determine the final contribution of each term to the law.
    Handles multi-trajectory array structure consistently.
    
    Parameters:
    -----------
    dic_terms : dict
        Dictionary of computed terms with multi-trajectory structure
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
    """
    
    if verbose:
        logging.info("\n" + "="*70)
        logging.info(f"COMPUTING LAW TERMS WITH COEFFICIENTS")

    nbsatellite = traj_param.get('nbsatellite', 1)
    if laws is None:
        laws = []
    
    dic_law_terms = {}
    dic_coefficients = {}
    
    # Compute terms only once (they are identical for all laws)
    computed = False
    for law_name in laws:
        if law_name not in LAWS:
            if verbose:
                logger.warning(f"Law '{law_name}' not found")
            continue
        
        if verbose:
            logging.info(f"\n  Processing law: {law_name}")
        
        try:
            law_obj = LAWS[law_name]
            
            # Apply coefficients and divergence factors
            if nbsatellite == 1:
                law_terms, law_coeffs = apply_law_coefficients_1satellite(
                    dic_terms, 
                    law_obj, 
                    physical_param,
                    traj_param,
                    verbose=verbose
                )
                
                # Apply roll shift to center trajectories on last axis (points only)
                for term_key, term_array in law_terms.items():
                    if isinstance(term_array, np.ndarray) and term_array.ndim >= 1:
                        # Roll shift on last axis (points dimension)
                        shift = term_array.shape[-1] // 2
                        law_terms[term_key] = np.roll(term_array, shift=shift, axis=-1)
            
            elif nbsatellite == 4:
                law_terms, law_coeffs = apply_law_coefficients_1satellite(
                    dic_terms,
                    law_obj,
                    physical_param,
                    traj_param,
                    nbsatellite=nbsatellite,
                    verbose=verbose
                )
                
                # Apply roll shift to center on last axis for all satellites and trajectories
                for term_key, term_array in law_terms.items():
                    if isinstance(term_array, np.ndarray) and term_array.ndim >= 1:
                        shift = term_array.shape[-1] // 2
                        law_terms[term_key] = np.roll(term_array, shift=shift, axis=-1)

            # For the first law, store the terms (they will be identical for others)
            if not computed:
                dic_law_terms.update(law_terms)
                computed = True
            
            # Add coefficients with formatted keys
            for term_key, coeff_value in law_coeffs.items():
                dic_coefficients[f"{law_name}_{term_key}"] = coeff_value
            
            if verbose:
                logging.info(f"  [OK] {len(law_terms)} terms added")
        
        except Exception as e:
            logger.error(f"Failed to process {law_name}: {e}")
    
    return dic_law_terms, dic_coefficients


def laws_to_h5(dic_law_terms, dic_coefficients, traj_param, filename="laws_terms.h5"):
    """
    Save law terms and coefficients to an HDF5 file.
    
    Stores computed law terms and their associated coefficients for later analysis.
    Only serializable trajectory parameters are saved to avoid data loss.
    
    Parameters:
    -----------
    dic_law_terms : dict
        Flat dictionary of computed law terms (fluxes and sources)
    dic_coefficients : dict
        Dictionary of coefficients mapping term names to their numerical weights
    traj_param : dict
        Trajectory parameters (filtered for serializable types)
    filename : str
        Output HDF5 filename (default: "laws_terms.h5")
    """
    
    def is_serializable(value):
        """Check if value can be saved to HDF5"""
        if value is None:
            return False
        if isinstance(value, (int, float, str, np.integer, np.floating)):
            return True
        if isinstance(value, np.ndarray) and value.ndim <= 2 and value.dtype != object:
            return True
        if isinstance(value, list) and all(isinstance(v, (int, float, str, np.integer, np.floating)) for v in value):
            return True
        return False
    
    with h5py.File(filename, 'w') as f:
        # Save law terms
        law_terms_group = f.create_group('law_terms')
        for term_key, value in dic_law_terms.items():
            law_terms_group.create_dataset(term_key, data=value)
        
        # Save coefficients
        coeffs_group = f.create_group('coefficients')
        for coeff_key, coeff_value in dic_coefficients.items():
            coeffs_group.create_dataset(coeff_key, data=coeff_value)
        
        # Save only serializable trajectory parameters
        traj_param_group = f.create_group('traj_param')
        for param_key, param_value in traj_param.items():
            if is_serializable(param_value):
                try:
                    traj_param_group.create_dataset(param_key, data=param_value)
                except Exception as e:
                    logger.warning(f"Could not save {param_key}: {e}")
    
    logging.info(f"  [OK] Saved {len(dic_law_terms)} law terms to {filename}")
