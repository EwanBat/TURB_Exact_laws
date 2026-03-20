from typing import List

from .abstract_law import AbstractLaw


class Ss22IsoF(AbstractLaw):
    def __init__(self):
        self.terms = [
            "flux_drvdvdv",
            "flux_drbdbdv",
            "flux_drvdbdb",
            "flux_drbdvdb",
            "flux_drduisodv",
            "flux_drdpisodv",
            "flux_drdpmdv",
        ]
        pass

    def terms_and_coeffs(self, physical_params):
        coeffs = {}
        coeffs["div_flux_drvdvdv"] = - 1 / 4
        coeffs["div_flux_drbdbdv"] = - 1 / 4
        coeffs["div_flux_drvdbdb"] = 1 / 4
        coeffs["div_flux_drbdvdb"] = 1 / 4
        coeffs["div_flux_drduisodv"] = - 1 / 2
        coeffs["div_flux_drdpisodv"] = 1 / 4
        coeffs["div_flux_drdpmdv"] = 1 / 4 
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)


def load():
    return Ss22IsoF()
