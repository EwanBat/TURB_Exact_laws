"""
Compute quantities along trajectories using fully vectorized operations.
Analog to trajectory_terms.py but for quantities, using QUANTITIES objects.
Quantities (v, Iv, etc.) are determined by requirements from laws/terms.

Data structure (uniform across all satellite counts):
    {sat_0: {var_name: array(n_trajectories, n_points), ...}}

For nbsatellite=9, gradient quantities are computed on sub-tuples from the
8 corner satellites forming 4-satellite subgroups.

Key design: All data arrays maintain (n_trajectories, n_points) shape.
"""

import numpy as np
import h5py
import logging

from exact_laws.preprocessing.quantities import QUANTITIES
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS


logger = logging.getLogger(__name__)


class MockFile:
    """Mock HDF5 file object for storing computed quantities."""
    def __init__(self):
        self.data = {}

    def create_dataset(self, name, data=None, **kwargs):
        self.data[name] = data if data is not None else np.empty(0)


class TrajectoryQuantitiesComputer:
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
        self.verbose = verbose
        self.grid_param = grid_param or {}
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.dic_param = {**self.physical_param, **self.grid_param}
        self.quantities_registry = QUANTITIES
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)

    # ========== PUBLIC METHODS ==========

    def extract_and_compute(self, dic_datas: dict,
                            laws=None, terms=None, quantities=None, method: str = None,
                            filename: str = None):
        """Compute all required quantities from laws/terms specifications.

        Parameters:
        -----------
        dic_datas : dict
            {sat_name: {var_name: array(n_trajectories, n_points)}}
        laws : list, optional
            Law names; required variables extracted via LAWS[name].variables()
        terms : list, optional
            Term names; required variables extracted via TERMS[name].variables()
        quantities : list, optional
            Explicit list of quantities to compute
        filename : str, optional
            If set, save results to HDF5

        Returns:
        -------
        dict : Same structure as input, with computed quantities added
        """
        if self.verbose:
            logger.info("\n" + "=" * 70)
            logger.info("TRAJECTORY QUANTITIES COMPUTATION (VECTORIZED)")
            logger.info(f"  Nbsatellite:   {self.nbsatellite}")
            logger.info(f"  Gap satellite: {self.traj_param.get('gap_satellite', 1)}")

        available_quantities = self._list_required_quantities(
            laws, terms, quantities, method=method
        )

        if self.verbose:
            logger.info(f"  Quantities to compute: {len(available_quantities)}")
            logger.info(f"  {available_quantities}")

        return self._compute_all_quantities(dic_datas, available_quantities, filename)

    def _list_required_quantities(self, laws=None, terms=None,
                                  quantities=None, method: str = None):
        """Collect all quantities required by the given laws and terms."""
        quantities = list(quantities) if quantities else []

        if terms:
            for term_name in terms:
                if term_name in TERMS:
                    quantities.extend(
                        TERMS[term_name].variables(nbsatellite=self.nbsatellite, method=method)
                    )

        if laws:
            for law_name in laws:
                if law_name in LAWS:
                    quantities.extend(
                        LAWS[law_name].variables(nbsatellite=self.nbsatellite, method=method)
                    )

        return list(set(quantities))

    # ========== PRIVATE COMPUTATION METHODS ==========

    def _compute_all_quantities(self, dic_datas: dict, available_quantities: list, filename: str):
        """Dispatch to the correct computation path based on satellite count."""
        dic_quantities = {sat_name: {} for sat_name in dic_datas.keys()}
    
        all_quantity_names = list(set(
            [req for deps in self.QUANTITY_DEPENDENCIES.values() for req in deps["requires"] if deps in available_quantities]
            + [req for deps in self.GRADIENT_QUANTITIES.values() for req in deps["requires"] if deps in available_quantities]
        ))

        if self.nbsatellite == 1:
            self._compute_all_single_pass(dic_datas, dic_quantities, available_quantities)
            if self.verbose:
                missing = [q for q in all_quantity_names if q not in dic_quantities["sat_0"].keys()]
                if missing:

                    logger.error(f"  [ERROR] Missing quantities in sat_0 for nbsatellite=1: {missing}")

                else:
                    logger.info(f"  [OK] All quantities computed successfully for nbsatellite=1.")
                    logger.info(f"  dic_quantities['sat_0'].keys(): {list(dic_quantities['sat_0'].keys())}")
        elif self.nbsatellite == 4:
            self._compute_non_gradient_quantities(dic_datas, dic_quantities, available_quantities)
            self._merge_raw_data(dic_datas, dic_quantities)
            self._compute_gradient_4satellite(dic_quantities, available_quantities)
            if self.verbose:
                missing = [q for q in all_quantity_names if q not in dic_quantities["sat_0"].keys()]
                if missing:

                    logger.error(f"  [ERROR] Missing quantities in sat_0 for nbsatellite=4: {missing}")


                else:
                    logger.info(f"  [OK] All quantities computed successfully for nbsatellite=4.")
                    logger.info(f"  dic_quantities['sat_0'].keys(): {list(dic_quantities['sat_0'].keys())}")
        elif self.nbsatellite == 9:
            self._compute_non_gradient_quantities(dic_datas, dic_quantities, available_quantities)
            self._merge_raw_data(dic_datas, dic_quantities)
            self._compute_gradient_9satellite(dic_quantities, available_quantities)
            if self.verbose:
                missing = [q for q in all_quantity_names if q not in dic_quantities["sat_0"].keys()]
                if missing:

                    logger.error(f"  [ERROR] Missing quantities in sat_0 for nbsatellite=9: {missing}")

                else:
                    logger.info(f"  [OK] All quantities computed successfully for nbsatellite=9.")
                    logger.info(f"  dic_quantities['sat_1'].keys(): {list(dic_quantities['sat_1'].keys())}")

        if filename:
            self.quantities_to_h5(dic_quantities, filename)

        return dic_quantities

    def _compute_all_single_pass(self, dic_datas: dict, dic_quantities: dict, quantities: list):
        """Compute all quantities in one pass (nbsatellite=1)."""
        for quantity_name in quantities:
            try:
                for sat_name in dic_datas.keys():
                    result = self._compute_quantity_vectorized(quantity_name, dic_datas[sat_name])
                    if isinstance(result, dict):
                        dic_quantities[sat_name].update(result)
                    else:
                        dic_quantities[sat_name][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")

    def _compute_non_gradient_quantities(self, dic_datas: dict, dic_quantities: dict, quantities: list):
        """Compute non-gradient quantities independently for each satellite."""
        for quantity_name in quantities:
            if quantity_name in self.GRADIENT_QUANTITIES:
                continue
            try:
                for sat_name in dic_datas.keys():
                    result = self._compute_quantity_vectorized(quantity_name, dic_datas[sat_name])
                    if isinstance(result, dict):
                        dic_quantities[sat_name].update(result)
                    else:
                        dic_quantities[sat_name][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")

    def _merge_raw_data(self, dic_datas: dict, dic_quantities: dict):
        """Copy raw input fields into dic_quantities so gradients can reference them."""
        for sat_name in dic_datas.keys():
            for key, value in dic_datas[sat_name].items():
                if key not in dic_quantities[sat_name]:
                    dic_quantities[sat_name][key] = value

    def _compute_gradient_4satellite(self, dic_quantities: dict, quantities: list):
        """Compute gradient/divergence quantities using all 4 satellites."""
        for quantity_name in quantities:
            if quantity_name not in self.GRADIENT_QUANTITIES:
                continue
            try:
                result = self._compute_quantity_vectorized(quantity_name, dic_quantities)
                if isinstance(result, dict):
                    dic_quantities['sat_0'].update(result)
                else:
                    dic_quantities['sat_0'][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")

    def _get_9satellite_faces_with_sat0(self):
        """Generate 4 tetrahedral tuples per face of the cube with sat0 as reference.

        For each face of the cube (6 faces), forms 4 combinations of 3 vertices,
        defining tetrahedrons with sat_0 at the center.
        This mirrors the approach used for flux divergence in trajectory_law.py.
        """
        faces = [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
        valid_tuples = []
        for surface in faces:
            for i in range(4):
                t = tuple(surface[:i] + surface[i+1:])
                valid_tuples.append(t)
        return valid_tuples

    def _compute_gradient_9satellite(self, dic_quantities: dict, quantities: list):
        """Compute gradients using face-based tetrahedrons with sat0 as reference.

        For each face of the cube, computes 4 gradients using sat_0 and 3 of
        the 4 face vertices, then averages all 24 tetrahedron results and
        stores the final gradient at sat_0.
        Mirrors the divergence calculation in trajectory_law.py.
        """
        satellite_offsets = self.traj_param.get('satellite_offsets', {})
        if not satellite_offsets:
            raise ValueError("Missing satellite_offsets in traj_param for nbsatellite=9")

        tuples = self._get_9satellite_faces_with_sat0()
        ref_offset = np.asarray(satellite_offsets['sat_0'])

        tetra_setups = []
        for (i, j, k) in tuples:
            source_sat_names = ["sat_0", f"sat_{i}", f"sat_{j}", f"sat_{k}"]
            dR1 = np.asarray(satellite_offsets[f'sat_{i}']) - ref_offset
            dR2 = np.asarray(satellite_offsets[f'sat_{j}']) - ref_offset
            dR3 = np.asarray(satellite_offsets[f'sat_{k}']) - ref_offset
            tetra_setups.append((source_sat_names, dR1, dR2, dR3))

        base_traj_param = dict(self.traj_param)
        base_traj_param['nbsatellite'] = 4

        for quantity_name in quantities:
            if quantity_name not in self.GRADIENT_QUANTITIES:
                continue

            grad_results = {}

            for source_sat_names, dR1, dR2, dR3 in tetra_setups:
                tuple_dic_quant = {
                    f"sat_{local_idx}": dic_quantities[source_sat_names[local_idx]]
                    for local_idx in range(4)
                }

                tuple_traj_param = dict(base_traj_param)
                tuple_traj_param['dR1'] = dR1
                tuple_traj_param['dR2'] = dR2
                tuple_traj_param['dR3'] = dR3

                result = self._compute_quantity_vectorized(
                    quantity_name, tuple_dic_quant, traj_param_override=tuple_traj_param,
                )

                if isinstance(result, dict):
                    for i, (key, value) in enumerate(result.items()):
                        if key not in grad_results:
                            grad_results[key] = []
                        grad_results[key].append(value)
                else:
                    if quantity_name not in grad_results:
                        grad_results[quantity_name] = []
                    grad_results[quantity_name].append(result)

            for key, values in grad_results.items():
                if key not in dic_quantities['sat_0']:
                    dic_quantities['sat_0'][key] = np.mean(values, axis=0)
                else:
                    dic_quantities['sat_0'][key].update(np.mean(values, axis=0))

    def _compute_quantity_vectorized(self, quantity_name: str, dic_quant_sat: dict,
                                     traj_param_override: dict = None):
        """Compute a single quantity via the QUANTITIES registry.

        Parameters:
        -----------
        quantity_name : str
        dic_quant_sat : dict
            Single-satellite {var: array} or multi-satellite {sat: {var: array}}
        traj_param_override : dict, optional
            Temporary trajectory params (for 9-satellite subgroups)

        Returns:
        -------
        np.ndarray or dict
        """
        mock_file = MockFile()
        traj_param = traj_param_override or self.traj_param
        self.quantities_registry[quantity_name].create_datasets(
            mock_file, dic_quant_sat, self.dic_param,
            traj=True, traj_param=traj_param
        )
        if len(mock_file.data) == 1:
            return list(mock_file.data.values())[0]
        return mock_file.data

    def quantities_to_h5(self, dic_quant: dict, filename: str):
        """Save computed quantities to HDF5 (structure: /sat_name/var_name)."""
        with h5py.File(filename, 'w') as f:
            for sat_name, sat_data in dic_quant.items():
                group = f.create_group(sat_name)
                for var_name, data_array in sat_data.items():
                    group.create_dataset(var_name, data=data_array,
                                         compression="gzip", compression_opts=9)


# ========== BACKWARD COMPATIBILITY ==========

def extract_and_compute_trajectory_quantities(dic_datas: dict, grid_param: dict = None,
                                              traj_param: dict = None, physical_param: dict = None,
                                              laws=None, terms=None, quantities=None,
                                              method: str = None, verbose: bool = False,
                                              filename: str = "computed_quantities.h5"):
    """Backward compatibility wrapper. Use TrajectoryQuantitiesComputer instead."""
    computer = TrajectoryQuantitiesComputer(
        verbose=verbose, grid_param=grid_param,
        physical_param=physical_param, traj_param=traj_param
    )
    return computer.extract_and_compute(
        dic_datas, laws=laws, terms=terms,
        quantities=quantities, method=method, filename=filename
    )
