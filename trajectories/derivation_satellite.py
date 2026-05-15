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

def gradient_4satellite(dic_quant, quantity_name, traj_param):
    """
    Compute the gradient of a term along a trajectory for 4 satellites using reciprocal vectors.
    
    Parameters:
    -----------
    dic_quant : dict
        Dictionary containing the term values for all 4 satellites
    quantity_name : str
        Name of the quantity for which to compute the gradient
    traj_param : dict
        Dictionary with tangents_list, ltraj_list, n_trajectories
    
    Returns:
    --------
    gradient_para : np.ndarray
        3D array (n_dim, n_trajectories, n_points)
    """
    dR1 = traj_param['dR1']  # Vector from satellite 0 to satellite 1
    dR2 = traj_param['dR2']  # Vector from satellite 0 to satellite 2
    dR3 = traj_param['dR3']  # Vector from satellite 0 to satellite 3

    alpha1 = dic_quant['sat_1'][quantity_name] - dic_quant['sat_0'][quantity_name]
    alpha2 = dic_quant['sat_2'][quantity_name] - dic_quant['sat_0'][quantity_name]
    alpha3 = dic_quant['sat_3'][quantity_name] - dic_quant['sat_0'][quantity_name]

    gradalpha = alpha1[np.newaxis, :, :] * np.cross(dR2[:, np.newaxis, np.newaxis], dR3[:, np.newaxis, np.newaxis], axis=0) + \
                alpha2[np.newaxis, :, :] * np.cross(dR3[:, np.newaxis, np.newaxis], dR1[:, np.newaxis, np.newaxis], axis=0) + \
                alpha3[np.newaxis, :, :] * np.cross(dR1[:, np.newaxis, np.newaxis], dR2[:, np.newaxis, np.newaxis], axis=0)

    reciprocal_volume = 1 / np.dot(dR1, np.cross(dR2, dR3))
    return gradalpha * reciprocal_volume

def divergence_4satellite(dic_quant, quantity_name, traj_param):
    """
    Compute the divergence of a term along a trajectory for 4 satellites using reciprocal vectors.
    
    Parameters:
    -----------
    dic_quant : dict
        Dictionary containing the term values for all 4 satellites
    quantity_name : str
        Name of the quantity for which to compute the divergence
    traj_param : dict
        Dictionary with tangents_list, ltraj_list, n_trajectories
    
    Returns:
    --------
    divergence_para : np.ndarray
        2D array (n_trajectories, n_points)
    """

    dR1 = traj_param['dR1']  # Vector from satellite 0 to satellite 1
    dR2 = traj_param['dR2']  # Vector from satellite 0 to satellite 2
    dR3 = traj_param['dR3']  # Vector from satellite 0 to satellite 3
    
    if traj_param['trajectory_method'] == 'linear_x' and quantity_name.split('_')[0] == 'flux':
        alpha1 = np.roll(dic_quant['sat_1'][quantity_name], -1, axis=-1) - dic_quant['sat_0'][quantity_name]
        alpha2 = dic_quant['sat_2'][quantity_name] - dic_quant['sat_0'][quantity_name]
        alpha3 = dic_quant['sat_3'][quantity_name] - dic_quant['sat_0'][quantity_name]
    elif traj_param['trajectory_method'] == 'linear_y' and quantity_name.split('_')[0] == 'flux':
        alpha1 = dic_quant['sat_1'][quantity_name] - dic_quant['sat_0'][quantity_name]
        alpha2 = np.roll(dic_quant['sat_2'][quantity_name], -1, axis=-1) - dic_quant['sat_0'][quantity_name]
        alpha3 = dic_quant['sat_3'][quantity_name] - dic_quant['sat_0'][quantity_name]
    elif traj_param['trajectory_method'] == 'linear_z' and quantity_name.split('_')[0] == 'flux':
        alpha1 = dic_quant['sat_1'][quantity_name] - dic_quant['sat_0'][quantity_name]
        alpha2 = dic_quant['sat_2'][quantity_name] - dic_quant['sat_0'][quantity_name]
        alpha3 = np.roll(dic_quant['sat_3'][quantity_name], -1, axis=-1) - dic_quant['sat_0'][quantity_name]
    else:
        alpha1 = dic_quant['sat_1'][quantity_name] - dic_quant['sat_0'][quantity_name]
        alpha2 = dic_quant['sat_2'][quantity_name] - dic_quant['sat_0'][quantity_name]
        alpha3 = dic_quant['sat_3'][quantity_name] - dic_quant['sat_0'][quantity_name]

    print(traj_param['trajectory_method'], quantity_name)
    print(np.einsum('i...,i...', alpha1, np.cross(dR2, dR3)[:, np.newaxis, np.newaxis])[0,:])
    divalpha = np.einsum('i...,i...', alpha1, np.cross(dR2, dR3)[:, np.newaxis, np.newaxis]) + \
                np.einsum('i...,i...', alpha2, np.cross(dR3, dR1)[:, np.newaxis, np.newaxis]) + \
                np.einsum('i...,i...', alpha3, np.cross(dR1, dR2)[:, np.newaxis, np.newaxis])
    
    reciprocal_volume = 1 / np.dot(dR1, np.cross(dR2, dR3))
    return divalpha * reciprocal_volume
