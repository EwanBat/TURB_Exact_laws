import numpy as np

def divergence_1satellite(term_value, dic_param):
    """
    Compute the divergence of a term along a trajectory for 1 satellite.
    
    Parameters:
    -----------
    term_value : np.ndarray
        3D array of the term values along the trajectory
    trajectory : np.ndarray
        3D array of the trajectory parameter (e.g., time or arc length)
    dic_param : dict
        Dictionary of simulation parameters
    """
    
    tangents = dic_param.get('tangents', None)
    if tangents is None:
        raise ValueError("Tangent vectors not found in dic_param. Ensure they are computed and added during trajectory preprocessing.")

    term_value_para = np.einsum('ij,ji->j', term_value, tangents)  # Project term onto tangent direction
    divergence_para = np.gradient(term_value_para, dic_param['ltraj'])  # Compute gradient along trajectory parameter

    divergence_perp = 0 # Placeholder for perpendicular divergence if needed (not computed here)

    return divergence_para + divergence_perp