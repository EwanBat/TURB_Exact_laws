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


def get_divergence_factor_for_law(law_name):
    """
    Returns the divergence replacement factor for a given law.
    This factor applies to ALL divergence terms of that law.
    
    Parameters:
    -----------
    law_name : str
        Name of the law
    
    Returns:
    -------
    float : Divergence factor (default 1.0)
    """
    return DIVERGENCE_REPLACEMENT_FACTORS.get(law_name, 1.0)


def apply_law_coefficients_1satellite(dic_terms, law_obj, law_name, dic_param, trajectory=None, verbose=False):
    """
    Apply ONLY the divergence factor to the terms.
    Law-specific coefficients are not applied here.
    
    Parameters:
    -----------
    dic_quantities : dict
        Dictionary of computed base quantities
    dic_terms : dict
        Dictionary of computed terms
    law_obj : AbstractLaw
        Law object
    law_name : str
        Name of the law
    dic_param : dict
        Simulation parameters
    trajectory : np.ndarray, optional
        Trajectory along which laws are computed
    verbose : bool
    
    Returns:
    -------
    dict : Dictionary of law terms with divergence factor applied
    """
    
    law_terms, coeffs = law_obj.terms_and_coeffs(dic_param)
    result = {}
    
    # Get the common divergence factor for this law
    div_factor = get_divergence_factor_for_law(law_name)
    
    # Get the list of flux terms of the law (those that will have a divergence)
    law_flux_terms = set(law_obj.terms) if hasattr(law_obj, 'terms') else set()
    
    # Determine array size (from first computed term)
    array_size = None
    for term_value in dic_terms.values():
        if isinstance(term_value, np.ndarray):
            array_size = term_value.shape
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
                # Apply the divergence factor (not the coefficient)
                result[coeff_key] = divergence_1satellite(term_value, dic_param)
                applied_terms.append(coeff_key)
                if verbose and is_flux_term:
                    logger.info(f"{coeff_key} (flux): div_factor={div_factor:.4f} applied")
            else:
                incomputable_terms.append((coeff_key, f"term '{base_term}' not computed"))
        
        elif is_source_term:
            # Source terms
            # For sources: check if it's a gradient or divergence
            has_gradient = any(keyword in coeff_key for keyword in ['drdr', 'dr2', 'dx', 'dy', 'dz'])
            
            if has_gradient:
                # Source term with gradient/divergence: replace with factor array
                if array_size is not None:
                    # Create an array of same size filled with the factor
                    result[coeff_key] = np.full(array_size, div_factor)
                else:
                    # Fallback: create a scalar
                    result[coeff_key] = div_factor
                applied_terms.append(coeff_key)
                if verbose:
                    logger.info(f"{coeff_key} (source+grad): div_factor={div_factor:.4f} applied")
            else:
                # Normal source without gradient
                if coeff_key in dic_terms:
                    term_value = dic_terms[coeff_key]
                    # Apply ONLY the divergence factor
                    result[coeff_key] = div_factor * term_value
                    applied_terms.append(coeff_key)
                    if verbose:
                        logger.info(f"{coeff_key} (source): div_factor={div_factor:.4f} applied")
                else:
                    incomputable_terms.append((coeff_key, "term not computed"))
        
        else:
            # Simple term (no div_ or source_)
            if coeff_key in dic_terms:
                term_value = dic_terms[coeff_key]
                # Apply ONLY the divergence factor
                result[coeff_key] = div_factor * term_value
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, "term not computed"))
    
    if verbose:
        if applied_terms:
            logger.info(f"Applied terms ({len(applied_terms)}):")
            for term in applied_terms:
                logger.info(f"  {term}")
        if incomputable_terms:
            logger.info(f"Incomputable terms ({len(incomputable_terms)}):")
            for term, reason in incomputable_terms:
                logger.info(f"  {term}: {reason}")
    
    return result, coeffs  # Also return original coefficients


def apply_law_coefficients_4satellites(dic_quantities, dic_terms, law_obj, law_name, dic_param, verbose=False):
    # Work in progress: similar to 1 satellite but with handling for 4 satellites
    return None

def compute_laws_terms_with_coefficients(dic_terms, dic_param, laws=None, nbsatellite=1, trajectory=None, verbose=False):
    """
    Compute law terms with coefficients applied.
    Returns a flat dictionary of law terms (div_flux_*, source_*, etc.).
    
    Note: Since terms are identical for all laws (the divergence factor
    is applied to all terms regardless of the law), terms are computed
    only once for the first valid law.
    
    Parameters:
    -----------
    dic_quantities : dict
        Dictionary of computed base quantities
    dic_terms : dict
        Dictionary of computed base terms
    dic_param : dict
        Simulation parameters
    laws : list[str]
        List of laws
    verbose : bool
    
    Returns:
    -------
    dict : Flat dictionary of law terms with coefficients {{term_key: value}}
    """
    
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
            logger.info(f"Processing law: {law_name}")
        
        try:
            law_obj = LAWS[law_name]
            
            # Apply coefficients and divergence factors
            if nbsatellite == 1:
                law_terms, law_coeffs = apply_law_coefficients_1satellite(
                    dic_terms, 
                    law_obj, 
                    law_name, 
                    dic_param, 
                    trajectory=trajectory,
                    verbose=verbose
                )
            
            elif nbsatellite == 4:
                law_terms, law_coeffs = apply_law_coefficients_4satellites(
                    dic_terms,
                    law_obj,
                    law_name,
                    dic_param,
                    verbose=verbose
                )

            # For the first law, store the terms (they will be identical for others)
            if not computed:
                dic_law_terms.update(law_terms)
                computed = True
            
            # Add coefficients with formatted keys
            for term_key, coeff_value in law_coeffs.items():
                dic_coefficients[f"{law_name}_{term_key}"] = coeff_value
            
            if verbose:
                logger.info(f"Added {len(law_terms)} terms for {law_name}")
        
        except Exception as e:
            logger.error(f"Failed to process {law_name}: {e}")
    
    return dic_law_terms, dic_coefficients

def laws_to_h5(dic_law_terms, dic_coefficients, filename="laws_terms.h5"):
    """
    Save law terms and coefficients to an HDF5 file.
    
    Parameters:
    -----------
    dic_law_terms : dict
        Flat dictionary of law terms {term_key: value}
    dic_coefficients : dict
        Dictionary of coefficients {law_term_key: coeff_value}
    filename : str
        Output HDF5 filename
    """
    
    with h5py.File(filename, 'w') as f:
        # Save law terms
        law_terms_group = f.create_group('law_terms')
        for term_key, value in dic_law_terms.items():
            law_terms_group.create_dataset(term_key, data=value)
        
        # Save coefficients
        coeffs_group = f.create_group('coefficients')
        for coeff_key, coeff_value in dic_coefficients.items():
            coeffs_group.create_dataset(coeff_key, data=coeff_value)
    
    logger.info(f"Saved law terms and coefficients to {filename}")

def display_law_terms_results(dic_law_terms, title="Law Terms Results"):
    """
    Display law terms results along a trajectory.
    
    Parameters:
    -----------
    dic_law_terms : dict
        Flat dictionary of law terms {term: array}
    title : str
        Display title
    """
    logger.info(f"\n{title}")
    logger.info("-" * 70)
    
    for term_key in sorted(dic_law_terms.keys()):
        value = dic_law_terms[term_key]
        if isinstance(value, np.ndarray) and value.ndim == 1:
            if value.size > 0:
                logger.info(f"  {term_key:40s}: min={value.min():12.6e} | max={value.max():12.6e} | mean={value.mean():12.6e}")
            else:
                logger.info(f"  {term_key:40s}: empty array")
        elif isinstance(value, np.ndarray):
            logger.info(f"  {term_key:40s}: shape={value.shape} | mean={value.mean():12.6e}")
        elif isinstance(value, (int, float, np.number)):
            logger.info(f"  {term_key:40s}: {value:15.6e}")
        else:
            logger.info(f"  {term_key:40s}: {value}")