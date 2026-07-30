import logging
from exact_laws.el_calc_mod.laws import LAWS

logger = logging.getLogger(__name__)


class TrajectoryTermsBase:

    VARIABLE_COMPONENTS = {
        'v': ['vx', 'vy', 'vz'],
        'Iv': ['Ivx', 'Ivy', 'Ivz'],
        'b': ['bx', 'by', 'bz'],
        'Ib': ['Ibx', 'Iby', 'Ibz'],
        'w': ['wx', 'wy', 'wz'],
        'j': ['jx', 'jy', 'jz'],
        'Ij': ['Ijx', 'Ijy', 'Ijz'],
        'f': ['fx', 'fy', 'fz'],
        'rho': ['rho'],
        'Irho': ['Irho'],
        'v2': ['v2'],
        'Iv2': ['Iv2'],
        'vnorm': ['vnorm'],
        'Ivnorm': ['Ivnorm'],
        'bnorm': ['bnorm'],
        'Ibnorm': ['Ibnorm'],
        'pm': ['pm'],
        'Ipm': ['Ipm'],
        'pgyr': ['ppar', 'pperp'],
        'Ipgyr': ['Ippar', 'Ipperp'],
        'piso': ['piso'],
        'Ipiso': ['Ipiso'],
        'ppol': ['ppol'],
        'Ippol': ['Ippol'],
        'pcgl': ['pparcgl', 'pperpcgl'],
        'Ipcgl': ['Ipparcgl', 'Ipperpcgl'],
        'ugyr': ['ugyr'],
        'Iugyr': ['Iugyr'],
        'uiso': ['uiso'],
        'Iuiso': ['Iuiso'],
        'upol': ['upol'],
        'Iupol': ['Iupol'],
        'ucgl': ['ucgl'],
        'Iucgl': ['Iucgl'],
        'divv': ['divv'],
        'Idivv': ['Idivv'],
        'divb': ['divb'],
        'Idivb': ['Idivb'],
        'divj': ['divj'],
        'Idivj': ['Idivj'],
        'gradrho': ['gradrhox', 'gradrhoy', 'gradrhoz'],
        'Igradrho': ['Igradrhox', 'Igradrhoy', 'Igradrhoz'],
        'gradv': ['dxvx', 'dyvx', 'dzvx', 'dxvy', 'dyvy', 'dzvy', 'dxvz', 'dyvz', 'dzvz'],
        'Igradv': ['Idxvx', 'Idyvx', 'Idzvx', 'Idxvy', 'Idyvy', 'Idzvy', 'Idxvz', 'Idyvz', 'Idzvz'],
        'gradb': ['dxbx', 'dybx', 'dzbx', 'dxby', 'dyby', 'dzby', 'dxbz', 'dybz', 'dzbz'],
        'Igradb': ['Idxbx', 'Idybx', 'Idzbx', 'Idxby', 'Idyby', 'Idzby', 'Idxbz', 'Idybz', 'Idzbz'],
        'graduiso': ['graduisox', 'graduisoy', 'graduisoz'],
        'Igraduiso': ['Igraduisox', 'Igraduisoy', 'Igraduisoz'],
        'gradupol': ['gradupolx', 'gradupoly', 'gradupolz'],
        'Igradupol': ['Igradupolx', 'Igradupoly', 'Igradupolz'],
        'hdk': ['hdkx', 'hdky', 'hdkz'],
        'Ihdk': ['Ihdkx', 'Ihdky', 'Ihdkz'],
        'hdm': ['hdmx', 'hdmy', 'hdmz'],
        'Ihdm': ['Ihdmx', 'Ihdmy', 'Ihdmz'],
        'hdk2': ['hdk2x', 'hdk2y', 'hdk2z'],
        'Ihdk2': ['Ihdk2x', 'Ihdk2y', 'Ihdk2z'],
    }

    FLUX_TERMS = frozenset([
        "flux_dvdvdv",
        "flux_dbdbdv",
        "flux_dvdbdb",
    ])

    SOURCE_TERMS = frozenset([
        "source_dpan",
        "source_dvdvdv",
        "source_dbdbdv",
        "source_dvdbdb",
    ])

    def __init__(self, verbose: bool = False, grid_param: dict = None,
                 physical_param: dict = None, traj_param: dict = None,
                 run_params: dict = None):
        self.verbose = verbose
        self.grid_param = grid_param or {}
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.run_params = run_params or {}
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)

        self._sat_names = [f'sat_{i}' for i in range(self.nbsatellite)]
        self._sat_param_cache = self._extract_sat_parameters('sat_0')

    def list_required_terms(self, laws: list = None):
        if laws is None:
            laws = []

        terms = set()

        if not laws:
            return terms

        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)

        for law_name in laws:
            if law_name in LAWS:
                law_obj = LAWS[law_name]
                law_terms, _ = law_obj.terms_and_coeffs(params_clean)
                terms.update(law_terms)

        return terms

    def _get_incremental_fs(self):
        if self.traj_param['trajectory_method'] == "linear_x":
            return 1 / self.grid_param['c'][0]
        elif self.traj_param['trajectory_method'] == "linear_y":
            return 1 / self.grid_param['c'][1]
        elif self.traj_param['trajectory_method'] == "linear_z":
            return 1 / self.grid_param['c'][2]

    def compute_all_terms_for_laws(self, dic_quantities: dict = None, laws: list = None,
                                    filename: str = "terms_trajectory.h5"):
        if laws is None:
            laws = []

        required_terms = self.list_required_terms(laws)

        if self.verbose:
            logger.info("\n" + "-" * 70)
            logger.info("FLUX AND SOURCE TERMS COMPUTATION")
            logger.info(f"  Computing {len(required_terms)} terms for {len(laws)} law(s)")

        method = self.run_params.get('method')

        if method == "incremental":
            if self.nbsatellite == 1:
                result = self._compute_terms_incremental_1sat(dic_quantities, required_terms)
            elif self.nbsatellite == 4:
                result = self._compute_terms_incremental_4sat(dic_quantities, required_terms)
            elif self.nbsatellite == 9:
                result = self._compute_terms_incremental_9sat(dic_quantities, required_terms)
            else:
                raise ValueError(f"Unsupported nbsatellite={self.nbsatellite} for method=incremental")
        elif method == "fourier":
            if self.nbsatellite == 1:
                result = self._compute_terms_fourier_1sat(dic_quantities, required_terms)
            elif self.nbsatellite in (4, 9):
                result = self._compute_terms_fourier_multi(dic_quantities, required_terms)
            else:
                raise ValueError(f"Unsupported nbsatellite={self.nbsatellite} for method=fourier")
        else:
            raise ValueError(f"Unknown method: {method}")

        if filename:
            self.terms_to_h5(result_terms=result, filename=filename)

        return result

    def _prepare_dic_param_for_terms_and_coeffs(self, dic_param: dict):
        params_clean = {}

        for key, value in dic_param.items():
            if isinstance(value, list):
                params_clean[key] = value[0]
            elif isinstance(value, dict):
                if 'sat_0' in value:
                    first_sat_value = value['sat_0']
                    if isinstance(first_sat_value, list):
                        params_clean[key] = first_sat_value[0]
                    else:
                        params_clean[key] = first_sat_value
                else:
                    params_clean[key] = value
            else:
                params_clean[key] = value

        return params_clean

    def _extract_sat_parameters(self, sat_name: str):
        dic_param_sat = {}
        for key, value in self.physical_param.items():
            if isinstance(value, dict) and sat_name in value:
                dic_param_sat[key] = value[sat_name]
            elif isinstance(value, list):
                dic_param_sat[key] = value[0]
            else:
                dic_param_sat[key] = value

        return dic_param_sat
