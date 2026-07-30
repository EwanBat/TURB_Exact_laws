import numpy as np
import logging

from exact_laws.preprocessing.quantities import QUANTITIES
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS


logger = logging.getLogger(__name__)


class MockFile:
    def __init__(self):
        self.data = {}

    def create_dataset(self, name, data=None, **kwargs):
        self.data[name] = data if data is not None else np.empty(0)


class TrajectoryQuantitiesBase:

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

    def __init__(self, verbose: bool = False, grid_param: dict = None,
                 physical_param: dict = None, traj_param: dict = None):
        self.verbose = verbose
        self.grid_param = grid_param or {}
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.dic_param = {**self.physical_param, **self.grid_param}
        self.quantities_registry = QUANTITIES
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)

    def extract_and_compute(self, dic_datas: dict,
                            laws=None, terms=None, quantities=None, method: str = None,
                            filename: str = None):
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

    def _compute_all_quantities(self, dic_datas: dict, available_quantities: list, filename: str):
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
                    logger.info(f"  dic_quantities['sat_0'].keys(): {list(dic_quantities['sat_0'].keys())}")

        if filename:
            self.quantities_to_h5(dic_quantities, filename)

        return dic_quantities

    def _compute_all_single_pass(self, dic_datas: dict, dic_quantities: dict, quantities: list):
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
        for sat_name in dic_datas.keys():
            for key, value in dic_datas[sat_name].items():
                if key not in dic_quantities[sat_name]:
                    dic_quantities[sat_name][key] = value

    def _compute_quantity_vectorized(self, quantity_name: str, dic_quant_sat: dict,
                                     traj_param_override: dict = None):
        mock_file = MockFile()
        traj_param = traj_param_override or self.traj_param
        self.quantities_registry[quantity_name].create_datasets(
            mock_file, dic_quant_sat, self.dic_param,
            traj=True, traj_param=traj_param
        )
        if len(mock_file.data) == 1:
            return list(mock_file.data.values())[0]
        return mock_file.data
