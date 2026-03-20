import numpy as np
import numexpr as ne


class PPol:
    def __init__(self, incompressible=False):
        self.name = "I" * incompressible + "ppol"
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            ds_name = f"{self.name}"
            file.create_dataset(
                ds_name,
                data= np.mean(ne.evaluate(f"(ppar+pperp+pperp)/3", local_dict=dic_quant)),
                shape=dic_param["N"],
                dtype=np.float64,
            )
        else:
            if not "gamma" in dic_param.keys():
                dic_param['gamma'] = 5/3
                gamma = 5/3
            else: 
                gamma = dic_param['gamma']
            if "cst" in dic_param.keys():
                cst = dic_param['cst']
            else: 
                cst = (np.mean(ne.evaluate(f"(ppar+pperp+pperp)/3", local_dict=dic_quant))
                        / np.mean(ne.evaluate(f"rho**(gamma)", local_dict=dic_quant, global_dict=dic_param)))
            rho = dic_quant['rho']
            ds_name = f"{self.name}"
            file.create_dataset(
                ds_name,
                data = ne.evaluate("cst*rho**(gamma-1)"),
                shape = dic_param["N"],
                dtype = np.float64,
            )


def load(incompressible=False):
    ppol = PPol(incompressible=incompressible)
    return ppol.name, ppol
