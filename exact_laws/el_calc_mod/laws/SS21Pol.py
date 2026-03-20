from typing import List

from .abstract_law import AbstractLaw


class Ss21Pol(AbstractLaw):
    def __init__(self):
        self.terms = [
            "flux_drvdvdv",
            "flux_drbdbdv",
            "flux_drvdbdb",
            "flux_drbdvdb",
            "flux_drdupoldv",
            "flux_drupolv",
            "flux_drppolv",
            "flux_drpmv",
            "source_rvdvdv",
            "source_rbdbdv",
            "source_drbbdv",
            "source_rdupoldv",
            "source_rppoldv",
            "source_rvdbdb",
            "source_rbdvdb",
            "source_bdrvdb",
            "source_rvbetadupol"
        ]
        pass

    def terms_and_coeffs(self, physical_params):
        coeffs = {}
        coeffs["div_flux_drvdvdv"] = - 1 / 4
        coeffs["div_flux_drbdbdv"] = - 1 / 4
        coeffs["div_flux_drvdbdb"] = 1 / 4
        coeffs["div_flux_drbdvdb"] = 1 / 4
        coeffs["div_flux_drdupoldv"] = - 1 / 2
        coeffs["div_flux_drupolv"] = - 1 / 4
        coeffs["div_flux_drppolv"] = - 1 / 4
        coeffs["div_flux_drpmv"] = - 1 / 4
        coeffs["source_rvdvdv"] = - 1 / 4
        coeffs["source_rbdbdv"] = - 1 / 4
        coeffs["source_drbbdv"] = 1 / 8
        coeffs["source_rdupoldv"] = - 1 / 2
        coeffs["source_rppoldv"] = 1 / 2
        coeffs["source_rvdbdb"] = 1 / 2
        coeffs["source_rbdvdb"] = 1 / 4
        coeffs["source_bdrvdb"] = -1 / 4
        coeffs["source_rvbetadupol"] = 1 / 4       
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)


def load():
    return Ss21Pol()
