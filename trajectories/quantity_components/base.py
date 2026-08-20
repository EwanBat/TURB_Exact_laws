"""
Base class for trajectory quantities computation.
Contains MockFile, constants, initialization, core helper, and I/O.
"""
import numpy as np
import h5py
import logging

from exact_laws.preprocessing.quantities import QUANTITIES

logger = logging.getLogger('trajectories.quantity_components')


class MockFile:
    """Mock HDF5 file object for storing computed quantities."""
    def __init__(self):
        self.data = {}

    def create_dataset(self, name, data=None, **kwargs):
        self.data[name] = data if data is not None else np.empty(0)


class TrajectoryQuantitiesComputerBase:
    """
    Compute quantities along trajectories in a fully vectorized manner.

    Handles single-satellite, 4-satellite formation, and 9-satellite cube
    configurations. Manages quantity dependencies and vectorized computations.
    """

    QUANTITY_DEPENDENCIES = {
        "v": {"requires": ["vx", "vy", "vz"]},
        "Iv": {"requires": ["Ivx", "Ivy", "Ivz"]},
        "rho": {"requires": ["rho"]},
        "Irho": {"requires": ["Irho"]},
        "b": {"requires": ["bx", "by", "bz"]},
        "Ib": {"requires": ["bx", "by", "bz"]},
        "v2": {"requires": ["vx", "vy", "vz"]},
        "Iv2": {"requires": ["Ivx", "Ivy", "Ivz"]},
        "vnorm": {"requires": ["vx", "vy", "vz"]},
        "Ivnorm": {"requires": ["Ivx", "Ivy", "Ivz"]},
        "bnorm": {"requires": ["bx", "by", "bz"]},
        "Ibnorm": {"requires": ["bx", "by", "bz"]},
        "pm": {"requires": ["bx", "by", "bz"]},
        "Ipm": {"requires": ["bx", "by", "bz"]},
        "pgyr": {"requires": ["pperp", "rho"]},
        "Ipgyr": {"requires": ["pperp"]},
        "piso": {"requires": ["ppar", "pperp"]},
        "Ipiso": {"requires": ["ppar", "pperp"]},
        "ppol": {"requires": ["pperp"]},
        "Ippol": {"requires": ["pperp"]},
        "pcgl": {"requires": ["bx", "by", "bz", "rho", "ppar", "pperp"]},
        "Ipcgl": {"requires": ["bx", "by", "bz", "ppar", "pperp"]},
        "ugyr": {"requires": ["pperp", "rho"]},
        "Iugyr": {"requires": ["pperp"]},
        "uiso": {"requires": ["ppar", "pperp", "rho"]},
        "Iuiso": {"requires": ["ppar", "pperp"]},
        "ucgl": {"requires": ["bx", "by", "bz", "rho", "ppar", "pperp"]},
        "Iucgl": {"requires": ["bx", "by", "bz", "ppar", "pperp"]},
        "upol": {"requires": ["ppar", "pperp", "rho"]},
        "Iupol": {"requires": ["pperp"]},
    }

    GRADIENT_QUANTITIES = {
        'gradv': {'requires': ['vx', 'vy', 'vz']},
        'gradv2': {'requires': ['v2']},
        'gradb': {'requires': ['bx', 'by', 'bz']},
        'gradrho': {'requires': ['rho']},
        'graduiso': {'requires': ['uiso']},
        'gradupol': {'requires': ['upol']},
        'gradugyr': {'requires': ['ugyr']},
        'gradpcgl': {'requires': ['pcgl']},
        'divv': {'requires': ['vx', 'vy', 'vz']},
        'divb': {'requires': ['bx', 'by', 'bz']},
        'divj': {'requires': ['bx', 'by', 'bz']},
        'Igradv': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'Igradv2': {'requires': ['Iv2']},
        'Igradrho': {'requires': ['Irho']},
        'Igraduiso': {'requires': ['Iuiso']},
        'Igradupol': {'requires': ['Iupol']},
        'Igradb': {'requires': ['Ibx', 'Iby', 'Ibz']},
        'Idivj': {'requires': ['bx', 'by', 'bz']},
        'j': {'requires': ['bx', 'by', 'bz']},
        'Ij': {'requires': ['bx', 'by', 'bz']},
        'w': {'requires': ['vx', 'vy', 'vz']},
        'Iw': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'f': {'requires': ['fp', 'fm']},
        'If': {'requires': ['fp', 'fm']},
        'hdk': {'requires': ['vx', 'vy', 'vz']},
        'Ihdk': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'hdk2': {'requires': ['vx', 'vy', 'vz']},
        'Ihdk2': {'requires': ['Ivx', 'Ivy', 'Ivz']},
        'hdm': {'requires': ['bx', 'by', 'bz', 'rho']},
        'Ihdm': {'requires': ['bx', 'by', 'bz']},
    }

    NINE_SATELLITE_TUPLES = (
        (1, 2, 4, 5),
        (2, 1, 3, 6),
        (3, 2, 4, 7),
        (4, 3, 1, 8),
        (5, 1, 6, 8),
        (6, 5, 7, 2),
        (7, 6, 8, 3),
        (8, 7, 5, 4),
    )

    def __init__(self, verbose: bool = False, grid_param: dict = None,
                 physical_param: dict = None, traj_param: dict = None):
        """
        Initialize the trajectory quantities computer.

        Parameters:
        -----------
        verbose : bool
            Enable detailed logger
        grid_param : dict, optional
            Grid parameters
        physical_param : dict, optional
            Physical parameters
        traj_param : dict, optional
            Trajectory parameters including 'nbsatellite', 'gap_satellite', 'satellite_offsets'
        """
        self.verbose = verbose
        self.grid_param = grid_param or {}
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.dic_param = {**self.physical_param, **self.grid_param}
        self.quantities_registry = QUANTITIES
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)

    def _compute_quantity_vectorized(self, quantity_name: str, dic_quant_sat: dict,
                                     traj_param_override: dict = None,
                                     grid_param_override: dict = None):
        """Compute a single quantity via the QUANTITIES registry.

        Parameters:
        -----------
        quantity_name : str
        dic_quant_sat : dict
            Single-satellite {var: array} or multi-satellite {sat: {var: array}}
        traj_param_override : dict, optional
            Temporary trajectory params (for 9-satellite subgroups)
        grid_param_override : dict, optional
            Temporary grid params (for 9-satellite subgroups)

        Returns:
        -------
        np.ndarray or dict
        """
        mock_file = MockFile()
        traj_param = traj_param_override or self.traj_param
        grid_param = grid_param_override or self.grid_param
        self.quantities_registry[quantity_name].create_datasets(
            mock_file, dic_quant_sat, self.dic_param,
            traj=True, traj_param=traj_param, grid_param=grid_param
        )
        if len(mock_file.data) == 1:
            return list(mock_file.data.values())[0]
        return mock_file.data

    def quantities_to_h5(self, dic_quant: dict, filename: str):
        """
        Save computed quantities to HDF5 file.

        Structure: /sat_name/var_name with gzip compression.

        Parameters:
        -----------
        dic_quant : dict
            {sat_name: {var_name: array(n_trajectories, n_points)}}
        filename : str
            Output HDF5 filename
        """
        with h5py.File(filename, 'w') as f:
            for sat_name, sat_data in dic_quant.items():
                group = f.create_group(sat_name)
                for var_name, data_array in sat_data.items():
                    group.create_dataset(var_name, data=data_array,
                                         compression="gzip", compression_opts=9)