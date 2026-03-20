import numpy as np

from ...mathematical_tools.derivation import laplacien

class HdM:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'hdm'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        for axisd in ('x', 'y', 'z'):
            if ("vax" in dic_quant.keys() or "vay" in dic_quant.keys() or "vaz" in dic_quant.keys()):
                hdm = dic_param["eta"]/np.sqrt(dic_quant["rho"]) * -laplacien(
                    -laplacien(
                        -laplacien(
                            -laplacien(np.sqrt(dic_quant["rho"])*dic_quant[f"va{axisd}"],
                                    [1,1,dic_param["an_hd"]],dic_param["c"])
                            ,[1,1,dic_param["an_hd"]],dic_param["c"])
                        ,[1,1,dic_param["an_hd"]],dic_param["c"])
                    ,[1,1,dic_param["an_hd"]],dic_param["c"])
            
            elif ("bx" in dic_quant.keys() or "by" in dic_quant.keys() or "bz" in dic_quant.keys()):
                if self.incompressible :
                    hdm = dic_param["eta"] * -laplacien(
                        -laplacien(
                            -laplacien(
                                -laplacien(dic_quant[f"b{axisd}"],
                                        [1,1,dic_param["an_hd"]],dic_param["c"])
                                ,[1,1,dic_param["an_hd"]],dic_param["c"])
                            ,[1,1,dic_param["an_hd"]],dic_param["c"])
                        ,[1,1,dic_param["an_hd"]],dic_param["c"])
                else :
                    hdm = dic_param["eta"]/np.sqrt(dic_quant["rho"]) * -laplacien(
                        -laplacien(
                            -laplacien(
                                -laplacien(dic_quant[f"b{axisd}"],
                                        [1,1,dic_param["an_hd"]],dic_param["c"])
                                ,[1,1,dic_param["an_hd"]],dic_param["c"])
                            ,[1,1,dic_param["an_hd"]],dic_param["c"])
                        ,[1,1,dic_param["an_hd"]],dic_param["c"])
        
            ds_name = f"{self.name}{axisd}"
            file.create_dataset(
                ds_name,
                data = hdm,
                shape = dic_param["N"],
                dtype = np.float64,
            )    
        
def load(incompressible=False):
    hdm = HdM(incompressible=incompressible)
    return hdm.name, hdm
