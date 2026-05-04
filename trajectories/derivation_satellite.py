import numpy as np

def divergence_1satellite(term_value, traj_param):
    """
    Compute the divergence of a term along a trajectory for 1 satellite.
    
    Parameters:
    -----------
    term_value : np.ndarray
        3D array of the term values along the trajectory
    traj_param : dict
        Dictionary of trajectory parameters
    """
    
    tangents_list = traj_param.get('tangents_list', None)
    ltraj_list = traj_param.get('ltraj_list', None)
    if tangents_list is None:
        raise ValueError("Tangent vectors not found in traj_param. Ensure they are computed and added during trajectory preprocessing.")

    term_value_para = np.einsum('ijk,jki->jk', term_value, tangents_list) # i: vector component of term_value, j: trajectory index, k: point index
    divergence_para = np.zeros_like(term_value_para)

    n_trajectories = traj_param['n_trajectories']
    divergence_para = np.array([
        np.gradient(term_value_para[traj_idx, :], ltraj_list[traj_idx, :])
        for traj_idx in range(n_trajectories)
    ])
    
    return divergence_para

def gradient_1satellite(term_value, traj_param):
    """
    Compute the gradient of a term along a trajectory for 1 satellite.
    Vectorized version.
    
    Parameters:
    -----------
    term_value : np.ndarray
        3D array (n_dim, n_trajectories, n_points)
    traj_param : dict
        Dictionary with tangents_list, ltraj_list, n_trajectories
    
    Returns:
    --------
    gradient_para : np.ndarray
        3D array (n_dim, n_trajectories, n_points)
    """

    tangents_list = traj_param.get('tangents_list', None)
    ltraj_list = traj_param.get('ltraj_list', None)
    if tangents_list is None:
        raise ValueError("Tangent vectors not found in traj_param.")
    
    n_trajectories = traj_param['n_trajectories']
    
    # Compute all 1D gradients (cannot be fully vectorized due to variable spacing)
    # Result: (n_trajectories, n_points)
    grad_1d_array = np.array([
        np.gradient(term_value[0, traj_idx, :], ltraj_list[traj_idx, :])
        for traj_idx in range(n_trajectories)
    ])
    
    # Vectorized projection using einsum
    # grad_1d_array[j,k] * tangents_list[j,i,k] -> gradient_para[i,j,k]
    # j: trajectory index, i: vector component (3), k: point index
    gradient_para = np.einsum('jk,jki->ijk', grad_1d_array, tangents_list)
    
    return gradient_para

def curl_1satellite(term_value, traj_param):
    """
    Compute the curl of a term along a trajectory for 1 satellite.
    
    Parameters:
    -----------
    term_value : np.ndarray
        3D array (n_dim, n_trajectories, n_points)
    traj_param : dict
        Dictionary with tangents_list, ltraj_list, trajectory_method, n_trajectories
    
    Returns:
    --------
    curl_result : np.ndarray
        3D array (n_dim, n_trajectories, n_points)
    """

    # Project onto tangent direction
    term_value_para = np.einsum('ijk,jki->jk', term_value, traj_param['tangents_list'])
    
    # Get perpendicular component
    term_value_perp = term_value - np.einsum('jk,jki->ijk', term_value_para, traj_param['tangents_list'])

    # Choose reference axis for perpendicular directions
    if traj_param.get('trajectory_method') == 'linear_z':
        a = np.array([0, 1, 0])
    else:
        a = np.array([0, 0, 1])
    
    # Compute perpendicular vectors
    perp_vector1 = np.cross(traj_param['tangents_list'], a)  # (n_trajectories, n_points, 3)
    perp_vector2 = np.cross(traj_param['tangents_list'], perp_vector1)  # (n_trajectories, n_points, 3)
    
    curl_result = np.zeros_like(term_value)  # (n_dim, n_trajectories, n_points)
    
    for traj_idx in range(traj_param['n_trajectories']):
        # Compute components along perpendicular directions
        term_perp1 = np.einsum('ijk,jk->i', term_value_perp[:, traj_idx, :, np.newaxis], perp_vector1[traj_idx, :, np.newaxis].T)
        term_perp2 = np.einsum('ijk,jk->i', term_value_perp[:, traj_idx, :, np.newaxis], perp_vector2[traj_idx, :, np.newaxis].T)
        
        # Simpler approach: project component-wise
        comp_perp1 = np.dot(term_value_perp[:, traj_idx, :], perp_vector1[traj_idx, :, :])  # (n_points,)
        comp_perp2 = np.dot(term_value_perp[:, traj_idx, :], perp_vector2[traj_idx, :, :])  # (n_points,)
        
        # Compute gradients
        grad_perp1 = np.gradient(comp_perp1, traj_param['ltraj_list'][traj_idx])  # (n_points,)
        grad_perp2 = np.gradient(comp_perp2, traj_param['ltraj_list'][traj_idx])  # (n_points,)
        
        # Curl = d(comp_perp2)/ds * perp_vector1 - d(comp_perp1)/ds * perp_vector2
        # grad_perp1: (n_points,)
        # perp_vector2[traj_idx, :, :].T: (n_dim, n_points)
        # Result: (n_dim, n_points) ✓
        curl_result[:, traj_idx, :] = (grad_perp2 * perp_vector1[traj_idx, :, :].T - 
                                        grad_perp1 * perp_vector2[traj_idx, :, :].T)

    return curl_result

