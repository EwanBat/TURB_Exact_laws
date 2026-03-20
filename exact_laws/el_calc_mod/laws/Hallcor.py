from typing import List

from .abstract_law import AbstractLaw

class HallCor(AbstractLaw):
    def __init__(self):
        self.terms = ["flux_djbdrb","flux_drjbdb","source_bbdrdj","source_bjdrdb"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["div_flux_djbdrb"] = physical_params["di"] / 2
        coeffs["div_flux_drjbdb"] = - physical_params["di"] / 2
        coeffs["source_bbdrdj"] = - physical_params["di"] / 8
        coeffs["source_bjdrdb"] =  physical_params["di"] / 4 
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return HallCor()
