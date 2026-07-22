from typing import List

from .abstract_law import AbstractLaw


class Ss22iGyrSource(AbstractLaw):
    def __init__(self):
        self.terms = ["source_dvdvdv", "source_dbdbdv", "source_dvdbdb", "source_dpan"]
        pass

    def terms_and_coeffs(self, physical_params):
        coeffs = {}
        coeffs["source_dvdvdv"] = - physical_params["rho_mean"] / 4
        coeffs["source_dbdbdv"] = - physical_params["rho_mean"] / 4
        coeffs["source_dvdbdb"] = physical_params["rho_mean"] / 2
        coeffs["source_dpan"] = physical_params["rho_mean"] / 4
        return self.terms, coeffs

    def variables(self, nbsatellite: int = 1, method: str = None) -> List[str]:
        return self.list_variables(self.terms, nbsatellite=nbsatellite, method=method)


def load():
    return Ss22iGyrSource()