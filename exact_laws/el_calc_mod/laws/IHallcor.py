from typing import List

from .abstract_law import AbstractLaw

class IHallCor(AbstractLaw):
    def __init__(self):
        self.terms = ["flux_djdbdb","flux_dbdbdj"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["div_flux_dbdbdj"] = physical_params["di"] / 8
        coeffs["div_flux_djdbdb"] = - physical_params["di"] / 4
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return IHallCor()
