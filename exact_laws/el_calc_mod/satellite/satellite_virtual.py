import numpy as np
import logging
from scipy.interpolate import RegularGridInterpolator
from typing import Dict
from ..laws import LAWS
from ..terms import TERMS


class SatelliteVirtual:
    def __init__(self, n_samples, original_dataset, trajectory):
        self.n_samples = n_samples
        self.grid = original_dataset.grid
        self._init_trajectory(trajectory)
        self._build_interpolator(original_dataset)
        self._get_values_at_trajectory()
    
    def _init_trajectory(self, trajectory_type='diagonal'):
        if trajectory_type == 'diagonal':
            self.trajectory = self.trajectory_diagonal()
        else:
            raise ValueError(f"Trajectory type {trajectory_type} not supported")

    def _build_interpolator(self, original_dataset):
        # Build an interpolator for each quantity of the original dataset
        self.interpolators = {}

        # Get the axis values from the grid, if they exist, otherwise create them as linspace
        if hasattr(self.grid, 'coords') and self.grid.coords:
            x_axis = self.grid.coords.get('x', np.arange(self.grid.N[0]))
            y_axis = self.grid.coords.get('y', np.arange(self.grid.N[1]))
            z_axis = self.grid.coords.get('z', np.arange(self.grid.N[2]))
        else:
            x_axis = np.linspace(0, 1, self.grid.N[0])
            y_axis = np.linspace(0, 1, self.grid.N[1])
            z_axis = np.linspace(0, 1, self.grid.N[2])
        
        self.axes = (x_axis, y_axis, z_axis)

        for name, values in original_dataset.data.items():
            self.interpolators[name] = RegularGridInterpolator(self.axes, 
                                        values,
                                        bounds_error=False,
                                        fill_value=np.nan,
                                        method='linear'
                                        )
        # It doesn't contain the datas yet, only the interpolators, which will be used to get the values at the trajectory points later
    
    def _get_values_at_trajectory(self):
        # Get the values of each quantity at the trajectory points using the interpolators
        self.trajectory_values = {}
        for name, interpolator in self.interpolators.items():
            self.trajectory_values[name] = interpolator(self.trajectory)

    def calc_term_satellite(self, original_dataset, output_dataset):
        terms = output_dataset.params['terms']
        output_quantities = output_dataset.quantities
        ind_term = output_dataset.params['state']['nb_term_done']
        
        logging.info(f"INIT Calculation of term for satellite {ind_term} {terms[ind_term]}")
        output_quantities[terms[ind_term]] = TERMS[terms[ind_term]].calc_fourier(**self.trajectory_values)
        logging.info(f"END Calculation of term for satellite {ind_term} {terms[ind_term]}")



























    def trajectory_diagonal(self):
        # Create a diagonal trajectory in the grid
        x = np.linspace(self.grid[0][0], self.grid[0][-1], self.n_samples)
        y = np.linspace(self.grid[1][0], self.grid[1][-1], self.n_samples)
        z = np.linspace(self.grid[2][0], self.grid[2][-1], self.n_samples)
        return np.array([x, y, z]).T