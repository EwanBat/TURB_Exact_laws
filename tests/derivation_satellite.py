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

######## Derivation in trajectory_quantities.py
def compute_gradient_stub(quantity_name: str, dic_quant: dict, separation: float = 1.0) -> np.ndarray:
    """
    Compute an approximation of a quantity's gradient.
    
    Uses a stub implementation pending a proper version.
    Assumes dic_quant[quantity_name] is a dictionary {sat_0, sat_1, sat_2, sat_3}
    
    Parameters:
    -----------
    quantity_name : str
        Name of the quantity whose gradient is desired
    dic_quant : dict
        Dictionary containing data for 4 satellites
    separation : float
        Separation between satellites in grid units
    
    Returns:
    -------
    dict : {sat_name: gradient_vector (n_points, 3)} or (n_points, 3) depending on quantity
    """
    
    # Retrieve base data (without "grad")
    base_quantity = quantity_name.lstrip('I').replace('grad', '')
    
    if base_quantity not in dic_quant:
        raise ValueError(f"Base quantity '{base_quantity}' not found to compute {quantity_name}")
    
    data_per_sat = dic_quant[base_quantity]
    
    if not isinstance(data_per_sat, dict):
        raise ValueError(f"{base_quantity} does not contain per-satellite data")
    
    n_points = len(data_per_sat['sat_0'])
    
    # Initialize result
    result = np.zeros((n_points, 3))
    
    # Retrieve data for each satellite
    sat_0 = data_per_sat['sat_0']  # front, up
    sat_1 = data_per_sat['sat_1']  # front, down
    sat_2 = data_per_sat['sat_2']  # rear, up
    sat_3 = data_per_sat['sat_3']  # rear, down
    
    # Gradient approximation (stub implementation)
    # To be replaced with real spatial interpolation
    
    # ∂f/∂x ≈ (f_front - f_rear) / (2 * separation)
    # Average between up and down for front and rear
    f_front = (sat_0 + sat_1) / 2.0
    f_rear = (sat_2 + sat_3) / 2.0
    result[:, 0] = (f_front - f_rear) / (separation if separation > 0 else 1.0)
    
    # ∂f/∂y ≈ (f_up - f_down) / (2 * separation)
    # Average between front and rear for up and down
    f_up = (sat_0 + sat_2) / 2.0
    f_down = (sat_1 + sat_3) / 2.0
    result[:, 1] = (f_up - f_down) / (separation if separation > 0 else 1.0)
    
    # ∂f/∂z ≈ 0 (no z variation with 4 satellites in xy plane)
    result[:, 2] = 0.0
    
    
    return result


def compute_divergence_stub(base_quantity: str, dic_quant: dict, separation: float = 1.0) -> np.ndarray:
    """
    Compute an approximation of a vector's divergence.
    
    Uses a stub implementation pending a proper version.
    
    Parameters:
    -----------
    base_quantity : str
        Name of the vector (ex: "v", "b", "j")
    dic_quant : dict
        Dictionary containing data for 4 satellites
    separation : float
        Separation between satellites
    
    Returns:
    -------
    np.ndarray : (n_points,) - divergence
    """
    
    if base_quantity not in dic_quant:
        raise ValueError(f"Quantity '{base_quantity}' not found to compute divergence")
    
    data_per_sat = dic_quant[base_quantity]
    
    if not isinstance(data_per_sat, dict):
        raise ValueError(f"{base_quantity} does not contain per-satellite data")
    
    n_points = len(data_per_sat['sat_0'])
    
    # Retrieve vector components
    components = f"{base_quantity}x", f"{base_quantity}y", f"{base_quantity}z"
    
    # Stub approximation: divergence ≈ 0
    # To be replaced with real implementation
    result = np.zeros(n_points)
    
    
    return result
