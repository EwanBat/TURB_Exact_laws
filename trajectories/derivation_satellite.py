import numpy as np

def divergence_1satellite(term_value, traj_param):
    """
    Compute the divergence of a term along a trajectory for 1 satellite.
    
    Parameters:
    -----------
    term_value : np.ndarray
        3D array of the term values along the trajectory
    trajectory : np.ndarray
        3D array of the trajectory parameter (e.g., time or arc length)
    traj_param : dict
        Dictionary of trajectory parameters
    """
    
    tangents_list = traj_param.get('tangents_list', None)
    ltraj_list = traj_param.get('ltraj_list', None)
    if tangents_list is None:
        raise ValueError("Tangent vectors not found in traj_param. Ensure they are computed and added during trajectory preprocessing.")

    term_value_para = np.einsum('ijk,jki->jk', term_value, tangents_list)  # Project term onto tangent direction
    divergence_para = np.zeros_like(term_value_para)
    for traj_idx in range(traj_param['n_trajectories']):
        divergence_para[traj_idx, :] = np.gradient(
            term_value_para[traj_idx, :],
            ltraj_list[traj_idx]  # Distance pour cette trajectoire
        )
    
    divergence_perp = 0 # Placeholder for perpendicular divergence if needed (not computed here)

    return divergence_para + divergence_perp

def gradient_1satellite(term_value, traj_param):
    """
    Compute the gradient of a term along a trajectory for 1 satellite.
    
    Parameters:
    -----------
    term_value : np.ndarray
        3D array of the term values along the trajectory
    trajectory : np.ndarray
        3D array of the trajectory parameter (e.g., time or arc length)
    traj_param : dict
        Dictionary of trajectory parameters
    """

    tangents_list = traj_param.get('tangents_list', None)
    ltraj_list = traj_param.get('ltraj_list', None)
    if tangents_list is None:
        raise ValueError("Tangent vectors not found in traj_param. Ensure they are computed and added during trajectory preprocessing.")
    
    term_value_para = np.einsum('ijk,jki->jk', term_value, tangents_list)  # Project term onto tangent direction
    gradient_para = np.zeros_like(term_value)
    for traj_idx in range(traj_param['n_trajectories']):
        gradient_para[:, traj_idx, :] = np.gradient(
            term_value_para[traj_idx, :],
            ltraj_list[traj_idx]  # Distance pour cette trajectoire
        )[np.newaxis, :] * tangents_list[traj_idx, :, :]  # Convert back to vector form

    gradient_perp = 0 # Placeholder for pecurl gradient in pythonrpendicular gradient if needed (not computed here)

    return gradient_para + gradient_perp

def curl_1satellite(term_value, traj_param):
    """
    Compute the curl of a term along a trajectory for 1 satellite.
    
    Parameters:
    -----------
    term_value : np.ndarray
        3D array of the term values along the trajectory
    trajectory : np.ndarray
        3D array of the trajectory parameter (e.g., time or arc length)
    traj_param : dict
        Dictionary of trajectory parameters
    """

    term_value_para = np.einsum('ijk,jki->jk', term_value, traj_param['tangents_list'])  # Project term onto tangent direction
    term_value_perp = term_value - np.einsum('jk,jki->ijk', term_value_para, traj_param['tangents_list'])  # Get perpendicular component

    if traj_param['trajectory_method'] == 'linear_z':
        a = np.array([0, 1, 0])
    else:
        a = np.array([0, 0, 1])  # Default to z-axis if not linear_z
    # Compute perpendicular vectors
    perp_vector1 = np.cross(traj_param['tangents_list'], a)
    perp_vector2 = np.cross(traj_param['tangents_list'], perp_vector1)
    curl_perp = np.zeros_like(term_value)
    for traj_idx in range(traj_param['n_trajectories']):
        curl_perp[:, traj_idx, :] = np.gradient(
            np.einsum('ijk,jki->jk', term_value_perp, perp_vector1[traj_idx, :, :]),
            traj_param['ltraj_list'][traj_idx]
        )[np.newaxis, :] * perp_vector2[traj_idx, :, :] - np.gradient(
            np.einsum('ijk,jki->jk', term_value_perp, perp_vector2[traj_idx, :, :]),
            traj_param['ltraj_list'][traj_idx]
        )[np.newaxis, :] * perp_vector1[traj_idx, :, :]
    
    curl_para = np.zeros_like(term_value)

    return curl_para + curl_perp
        
