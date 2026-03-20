from typing import List

from .abstract_law import AbstractLaw


class Ss22Iso(AbstractLaw):
    def __init__(self):
        self.terms = [
            "flux_drvdvdv",
            "flux_drbdbdv",
            "flux_drvdbdb",
            "flux_drbdvdb",
            "flux_drduisodv",
            "flux_drdpisodv",
            "flux_drdpmdv",
            "source_rvdvdv",
            "source_rbdbdv",
            "source_bdrbdv",
            "source_rduisodv",
            "source_rdpisodv",
            "source_rvdbdb",
            "source_rbdvdb",
            "source_bdrvdb",
            "source_pmvdrdr",
            "source_pisovdrdr",
            "source_rvdpmdr",
            "source_rvdpisodr"
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
        coeffs["source_rvdvdv"] = - 1 / 4
        coeffs["source_rbdbdv"] = - 1 / 8
        coeffs["source_bdrbdv"] = 1 / 8
        coeffs["source_rduisodv"] = - 1 / 2
        coeffs["source_rdpisodv"] = 1 / 2
        coeffs["source_rvdbdb"] = 1 / 2
        coeffs["source_rbdvdb"] = 1 / 4
        coeffs["source_bdrvdb"] = -1 / 4   
        coeffs["source_pmvdrdr"] = -1 / 4   
        coeffs["source_pisovdrdr"] = -1 / 4   
        coeffs["source_rvdpmdr"] = 1 / 4   
        coeffs["source_rvdpisodr"] = 1 / 4    
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)


def load():
    return Ss22Iso()
