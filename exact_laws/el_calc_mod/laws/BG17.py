from typing import List

from .abstract_law import AbstractLaw

class Bg17(AbstractLaw):
    def __init__(self):
        self.terms = ["bg17_vwv","bg17_jbv","bg17_vbj"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["bg17_vwv"] = physical_params["rho_mean"] / 2
        coeffs["bg17_jbv"] = physical_params["rho_mean"] / 2
        coeffs["bg17_vbj"] = physical_params["rho_mean"] / 2
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return Bg17()
