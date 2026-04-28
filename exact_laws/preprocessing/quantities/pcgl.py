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

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        if traj:
            if self.incompressible:
                sat_key = list(dic_param['meanppar'].keys())[0]
                meanppar_array = np.array(dic_param['meanppar'][sat_key])
                meanpperp_array = np.array(dic_param['meanpperp'][sat_key])
                
                b2 = dic_quant['bx']**2 + dic_quant['by']**2 + dic_quant['bz']**2
                bnorm = np.sqrt(b2)
                
                cstpar = float(np.mean(meanppar_array[:, np.newaxis] * b2, axis=1).mean())
                cstperp = float(np.mean(meanpperp_array[:, np.newaxis] / bnorm, axis=1).mean())
                rho = 1
            else:
                cstpar = np.mean(ne.evaluate(f"meanppar*(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param), axis=1)
                cstperp = np.mean(ne.evaluate(f"meanpperp/sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant, global_dict=dic_param), axis=1)
                bnorm = ne.evaluate(f"sqrt(bx*bx+by*by+bz*bz)", local_dict=dic_quant)
                rho = dic_quant['rho']
            
            eval_dict_par = {'cstpar': cstpar, 'rho': rho, 'bnorm': bnorm}
            eval_dict_perp = {'cstperp': cstperp, 'rho': rho, 'bnorm': bnorm}
            
            file.create_dataset(
                f"{self.name[:-3]}parcgl",
                data=ne.evaluate("cstpar*(rho**3)/bnorm/bnorm", local_dict=eval_dict_par)
            )
            file.create_dataset(
                f"{self.name[:-3]}perpcgl",
                data=ne.evaluate("cstperp*rho*bnorm", local_dict=eval_dict_perp)
            )
        else:
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
            
            eval_dict_par = {'cstpar': cstpar, 'rho': rho, 'bnorm': bnorm}
            eval_dict_perp = {'cstperp': cstperp, 'rho': rho, 'bnorm': bnorm}
            
            ds_name = f"{self.name[:-3]}parcgl"
            file.create_dataset(
                ds_name,
                data=ne.evaluate("cstpar*(rho**3)/bnorm/bnorm", local_dict=eval_dict_par),
                shape=dic_param["N"],
                dtype=np.float64,
            )
            ds_name = f"{self.name[:-3]}perpcgl"
            file.create_dataset(
                ds_name,
                data=ne.evaluate("cstperp*rho*bnorm", local_dict=eval_dict_perp),
                shape=dic_param["N"],
                dtype=np.float64,
            )


def load(incompressible=False):
    pcgl = PCgl(incompressible=incompressible)
    return pcgl.name, pcgl
