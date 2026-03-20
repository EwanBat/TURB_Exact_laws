from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxSs21Hyb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'", "rho'", "pm'", "p'", "u'",
                 "vx", "vy", "vz", "rho", "pm", "p", "u"
                )
        self.fctx = sp.lambdify(
            sp.symbols(quantities),
            self.exprx,
            "numpy",
        )
        self.fcty = sp.lambdify(
            sp.symbols(quantities),
            self.expry,
            "numpy",
        )
        self.fctz = sp.lambdify(
            sp.symbols(quantities),
            self.exprz,
            "numpy",
        )
        
    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        pmP, pmNP = sp.symbols(("pm'", "pm"))
        pP, pNP = sp.symbols(("p'", "p"))
        uP, uNP = sp.symbols(("u'", "u"))

        
        self.exprx = ((rhoP + rhoNP) * (pNP + pmNP) * vxP 
                      - (rhoP + rhoNP) * (pP + pmP) * vxNP  
                      + rhoP * uNP * vxP - rhoNP * uP * vxNP)
        self.expry = ((rhoP + rhoNP) * (pNP + pmNP) * vyP 
                      - (rhoP + rhoNP) * (pP + pmP) * vyNP  
                      + rhoP * uNP * vyP - rhoNP * uP * vyNP)
        self.exprz = ((rhoP + rhoNP) * (pNP + pmNP) * vzP 
                      - (rhoP + rhoNP) * (pP + pmP) * vzNP  
                      + rhoP * uNP * vzP - rhoNP * uP * vzNP)
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, rho, pm, p, u, **kwarg) -> List[float]:
        #return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, rho, pm, p, u)

    def variables(self) -> List[str]:
        return ['v', 'rho', 'pm', 'p', 'u']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxSs21Hyb()

def print_expr():
    return FluxSs21Hyb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, rho, pm, p, u, 
                             fx=njit(FluxSs21Hyb().fctx),
                             fy=njit(FluxSs21Hyb().fcty),
                             fz=njit(FluxSs21Hyb().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pmP, pmNP = pm[ip, jp, kp], pm[i, j, k]
    pP, pNP = p[ip, jp, kp], p[i, j, k]
    uP, uNP = u[ip, jp, kp], u[i, j, k]
    
    outx = fx(vxP, vyP, vzP, rhoP, pmP, pP, uP, vxNP, vyNP, vzNP, rhoNP, pmNP, pNP, uNP)
    outy = fy(vxP, vyP, vzP, rhoP, pmP, pP, uP, vxNP, vyNP, vzNP, rhoNP, pmNP, pNP, uNP)
    outz = fz(vxP, vyP, vzP, rhoP, pmP, pP, uP, vxNP, vyNP, vzNP, rhoNP, pmNP, pNP, uNP)
    
    return outx, outy, outz
