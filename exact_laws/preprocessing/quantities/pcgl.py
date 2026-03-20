import numpy as np
import numexpr as ne

def get_original_quantity(dic_quant, dic_param, delete=False):
    cstpar = np.mean(ne.evaluate(f"meanppar*(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param))
    cstperp = np.mean(ne.evaluate(f"meanpperp/sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param))
    bnorm = ne.evaluate(f"sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant)
    rho = dic_quant['rho']
            
    dic_quant['pparcgl'] = ne.evaluate("cstpar*(rho**3)/bnorm/bnorm")
    dic_quant['pperpcgl'] = ne.evaluate("cstperp*rho*bnorm")


class PCgl:
    def __init__(self, incompressible=False):
        self.name = "I" * incompressible + "pcgl"
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            cstpar = np.mean(ne.evaluate(f"meanppar*(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param))
            cstperp = np.mean(ne.evaluate(f"meanpperp/sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param))
            bnorm = ne.evaluate(f"sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant)
            rho = 1
        else:
            cstpar = np.mean(ne.evaluate(f"meanppar*(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param))
            cstperp = np.mean(ne.evaluate(f"meanpperp/sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param))
            bnorm = ne.evaluate(f"sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant)
            rho = dic_quant['rho']
            
        ds_name = f"{self.name[:-3]}parcgl"
        file.create_dataset(
            ds_name,
            data = ne.evaluate("cstpar*(rho**3)/bnorm/bnorm"),
            shape = dic_param["N"],
            dtype = np.float64,
        )
        ds_name = f"{self.name[:-3]}perpcgl"
        file.create_dataset(
            ds_name,
            data = ne.evaluate("cstperp*rho*bnorm"),
            shape = dic_param["N"],
            dtype = np.float64,
        )


def load(incompressible=False):
    pcgl = PCgl(incompressible=incompressible)
    return pcgl.name, pcgl
