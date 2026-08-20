"""
Trajectory path definitions and parameter generators.

All paths are defined in index space (grid indices). The physical conversion
is handled by the geometry helpers in geometry.py.
"""
import numpy as np


# ====================== Type of trajectories ======================
# ========== TRAJECTORY DEFINITIONS ==========

def trajectory_linear_x(t: np.ndarray, y_pos: int, z_pos: int, 
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the x axis (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[0]-1)
    y_pos : int
        Fixed position on y (index)
    z_pos : int
        Fixed position on z (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = t
    y = np.full_like(t, y_pos, dtype=int)
    z = np.full_like(t, z_pos, dtype=int)
    
    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_minus_x(t: np.ndarray, y_pos: int, z_pos: int,
                             N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the -x axis (indices).

    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[0]-1)
    y_pos : int
        Fixed position on y (index)
    z_pos : int
        Fixed position on z (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])

    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = N[0] - 1 - t  # Reverse direction along x
    y = np.full_like(t, y_pos, dtype=int)
    z = np.full_like(t, z_pos, dtype=int)

    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_y(t: np.ndarray, x_pos: int, z_pos: int,
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the y axis (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[1]-1)
    x_pos : int
        Fixed position on x (index)
    z_pos : int
        Fixed position on z (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = np.full_like(t, x_pos, dtype=int)
    y = t
    z = np.full_like(t, z_pos, dtype=int)
    
    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_minus_y(t: np.ndarray, x_pos: int, z_pos: int,
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the -y axis (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[1]-1)
    x_pos : int
        Fixed position on x (index)
    z_pos : int
        Fixed position on z (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = np.full_like(t, x_pos, dtype=int)
    y = N[1] - 1 - t  # Reverse direction along y
    z = np.full_like(t, z_pos, dtype=int)
    
    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_z(t: np.ndarray, x_pos: int, y_pos: int,
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the z axis (indices).

    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[2]-1)
    x_pos : int
        Fixed position on x (index)
    y_pos : int
        Fixed position on y (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])

    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = np.full_like(t, x_pos, dtype=int)
    y = np.full_like(t, y_pos, dtype=int)
    z = t

    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_minus_z(t: np.ndarray, x_pos: int, y_pos: int,
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the -z axis (indices).

    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[2]-1)
    x_pos : int
        Fixed position on x (index)
    y_pos : int
        Fixed position on y (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])

    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = np.full_like(t, x_pos, dtype=int)
    y = np.full_like(t, y_pos, dtype=int)
    z = N[2] - 1 - t  # Reverse direction along z

    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_xy(t: np.ndarray, x_pos: int, y_pos: int, z_pos: int, N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the diagonal in the xy-plane (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to min(N[0], N[1])-1)
    x_pos : int
        Fixed position on x (index)
    y_pos : int
        Fixed position on y (index)
    z_pos : int
        Fixed position on z (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = x_pos + t
    y = y_pos - t
    z = np.full_like(t, z_pos, dtype=int)
    
    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int) % N[0]
    y = np.clip(y, 0, N[1]-1).astype(int) % N[1]
    z = np.clip(z, 0, N[2]-1).astype(int) % N[2]
    
    return np.array([x, y, z]).T


# ====================== Trajectory parameter generators ======================

def generate_all_trajectory_kwargs_linear_x(N: np.ndarray, step: int) -> list:
    """
    Generate all possible trajectory_kwargs combinations for linear_x trajectory.
    
    Creates a trajectory for each (y_pos, z_pos) combination in the grid, covering
    all possible positions perpendicular to the x-axis propagation.
    
    Parameters:
    -----------
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    step : int
        Step size for trajectory generation
    
    Returns:
    -------
    list : List of dictionaries with all (y_pos, z_pos) combinations
    """
    trajectory_kwargs_list = []
    
    for y_pos in range(0, N[1], step):
        for z_pos in range(0, N[2], step):
            trajectory_kwargs_list.append({
                'y_pos': int(y_pos),
                'z_pos': int(z_pos)
            })
    
    return trajectory_kwargs_list

def generate_all_trajectory_kwargs_linear_y(N: np.ndarray, step: int) -> list:
    """
    Generate all possible trajectory_kwargs combinations for linear_y trajectory.
    
    Creates a trajectory for each (x_pos, z_pos) combination in the grid, covering
    all possible positions perpendicular to the y-axis propagation.
    
    Parameters:
    -----------
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    step : int
        Step size for trajectory generation
    
    Returns:
    -------
    list : List of dictionaries with all (x_pos, z_pos) combinations
    """
    trajectory_kwargs_list = []
    
    for x_pos in range(0, N[0], step):
        for z_pos in range(0, N[2], step):
            trajectory_kwargs_list.append({
                'x_pos': int(x_pos),
                'z_pos': int(z_pos)
            })
    
    return trajectory_kwargs_list

def generate_all_trajectory_kwargs_linear_z(N: np.ndarray, step: int) -> list:
    """
    Generate all possible trajectory_kwargs combinations for linear_z trajectory.
    
    Creates a trajectory for each (x_pos, y_pos) combination in the grid, covering
    all possible positions perpendicular to the z-axis propagation.
    
    Parameters:
    -----------
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    step : int
        Step size for trajectory generation
    
    Returns:
    -------
    list : List of dictionaries with all (x_pos, y_pos) combinations
    """
    trajectory_kwargs_list = []
    
    for x_pos in range(0, N[0], step):
        for y_pos in range(0, N[1], step):
            trajectory_kwargs_list.append({
                'x_pos': int(x_pos),
                'y_pos': int(y_pos)
            })
    
    return trajectory_kwargs_list

def generate_all_trajectory_kwargs_linear_xy(N: np.ndarray, step: int) -> list:
    """
    Generate all possible trajectory_kwargs combinations for linear_xy trajectory.

    Creates a trajectory for each z position in the grid, covering all possible
    fixed planes perpendicular to the xy diagonal trajectory.

    Parameters:
    -----------
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    step : int
        Step size for trajectory generation

    Returns:
    -------
    list : List of dictionaries with all z_pos values
    """
    trajectory_kwargs_list = []

    for z_pos in range(0, N[2], step):
        for x_pos, y_pos in zip(range(0, N[0], step), range(N[1]-1, -1, -step)):
            trajectory_kwargs_list.append({
                'x_pos': int(x_pos),
                'y_pos': int(y_pos),
                'z_pos': int(z_pos)
            })

    return trajectory_kwargs_list