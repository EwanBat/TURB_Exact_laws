import numpy as np
import h5py as h5
import logging
import configparser

from ..el_calc_mod.laws import LAWS
from ..el_calc_mod.terms import TERMS
from .quantities import QUANTITIES
from . import process_on_standard_h5_file


def extract_simu_param_from_OCA_file(file, dic_param, param):
    dic_param['axis'] = ['x','y','z']
    dic_param["L"] = np.array([file[f"{param}/x"][-1], file[f"{param}/y"][-1], file[f"{param}/z"][-1]])
    dic_param["N"] = np.array(
        [
            np.shape(file[f"{param}/x"])[0],
            np.shape(file[f"{param}/y"])[0],
            np.shape(file[f"{param}/z"])[0],
        ]
    )
    dic_param["c"] = np.array([file[f"{param}/x"][1], file[f"{param}/y"][1], file[f"{param}/z"][1]])
    return dic_param


def extract_quantities_from_OCA_file(file, list_quant, cycle):
    list_data = []
    for quant in list_quant:
        list_data.append(np.transpose(np.ascontiguousarray(file[f"{cycle}/{quant}"], dtype=np.float64)))
    return list_data


def list_quantities(laws, terms, quantities):
    quantities = quantities.copy()
    for term in terms:
        quantities += TERMS[term].variables()
    for law in laws:
        quantities += LAWS[law].variables()
    return list(set(quantities))


def from_OCA_files_to_standard_h5_file(
    input_folder, output_folder, name, cycle, laws, terms, quantities, sim_type, physical_params, reduction
):
    """
    Input: inputdic (Dictionnaire créé par Data_process.inputfile_to_dict())
    Ce qui est fait:
        - Ouverture les fichiers .h5 contenant les données données par l'OCA
        - Calcul des quantités voulues et indiquées dans inputdic
        - Enregistrements des tableaux de données dans un nouveau fichier .h5
        - Print du contenu de ce nouveau fichier via la fonction Data_process.check()
    En cours d'exécution: Print d'informations à propos des étapes intermédiaires.
    Output: nom du nouveau fichier .h5
    """

    output_file = f"{output_folder}/{name}.h5"

    message = (
        f"Begin process from_OCA_files_to_standard_h5_file() with config:"
        f"\n\t - input_folder: {input_folder}"
        f"\n\t - sim_type: {sim_type}"
        f"\n\t - cycle: {input_folder}"
        f"\n\t - output_file: {output_file}"
        f"\n\t - laws: {laws}"
        f"\n\t - terms: {terms}"
        f"\n\t - quantities: {quantities}"
        f"\n\t - reduction: {reduction}"
    )
    logging.info(message)
    
    needed_quantities = list_quantities(laws, terms, quantities)
    
    output_file = f"{output_folder}/{name}.h5"
    if process_on_standard_h5_file.verif_file_existence(output_file, "Process impossible."): 
        logging.info("End process from_OCA_files_to_standard_h5_file()\n") 
        #TODO manage case: file already existing
        #with h5.File(output_file, "a") as g: 
        #    for oqp in g.keys():
        #        if oqp in needed_quantities: needed_quantities.remove(oqp)
        #        if oqp in ['vx','vy','vz']: needed_quantities.remove('v')
        #        if oqp in ['wx','wy','wz']: needed_quantities.remove('w')
        #        if oqp in ['dxvx','dxvy','dxvz','dyvx','dyvy','dyvz','dzvx','dzvy','dzvz']: 
        #            needed_quantities.remove('gradv')
        #        if oqp in ['dxrho','dyrho','dzrho']: needed_quantities.remove('gradrho')
        #        if oqp in ['Ippar','Ipperp']: needed_quantities.remove('Ipgyr')
        #        if oqp in ['Ippar','Ipperp']: needed_quantities.remove('Ipgyr')
        #        if oqp in ['dxuiso','dyuiso','dzuiso']: needed_quantities.remove('graduiso')
        #        if oqp in ['Ibx','Iby','Ibz']: needed_quantities.remove('Ib')
        #        if oqp in ['bx','by','bz']: needed_quantities.remove('b')
        #        if oqp in ['Ijx','Ijy','Ijz']: needed_quantities.remove('Ij')
        #        if oqp in ['jx','jy','jz']: needed_quantities.remove('j')    
        return output_file
    
    dic_param = physical_params
    dic_quant = {}
    # param source file (obtained in velocity source file)
    with h5.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
        if  sim_type.endswith("CGL3") or  sim_type.endswith("CGL5"):
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "3Dgrid")
        else:
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "Simulation_Parameters")
    logging.info("... End extracting param")

    # velocity source file
    output_file = f"{output_folder}/{name}_v.h5temp"
    if not process_on_standard_h5_file.verif_file_existence(output_file, ""):
        g = h5.File(output_file, "a")
        with h5.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
            (dic_quant["vx"],
                dic_quant["vy"],
                dic_quant["vz"],
            ) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], cycle)
        accessible_quantities = ["v", "w", "gradv", "divv", "gradv2", "v2","vnorm","hdk","hdk2"]
        for aq in accessible_quantities:
            if aq in needed_quantities:
                logging.info(f"... computing {aq} from _v.h5")
                QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
        del (dic_quant["vx"], dic_quant["vy"], dic_quant["vz"])
        g.close()
    logging.info("... End computing quantities from _v.h5")

    # Density source file
    with h5.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
        dic_quant["rho"] = extract_quantities_from_OCA_file(
            frho,
            [
                "rho",
            ],
            cycle,
        )[0]
        
    output_file = f"{output_folder}/{name}_rho.h5temp"
    if not process_on_standard_h5_file.verif_file_existence(output_file, ""):
        g = h5.File(output_file, "a")
        accessible_quantities = ["rho", "gradrho"]
        for aq in accessible_quantities:
            if aq in needed_quantities:
                logging.info(f"... computing {aq} from _rho.h5")
                QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
        g.close()
    logging.info("... End computing quantities from _rho.h5")

    # Pressure source file
    output_file = f"{output_folder}/{name}_p.h5temp"
    if not process_on_standard_h5_file.verif_file_existence(output_file, ""):
        g = h5.File(output_file, "a")
        with h5.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
            dic_quant["ppar"], dic_quant["pperp"] = extract_quantities_from_OCA_file(fp, ["pparli", "pperpi"], cycle)
        dic_quant["ppar"] /= 2 #Correction de l'adimensionnement
        dic_quant["pperp"] /= 2 #Correction de l'adimensionnement
        accessible_quantities = ["Ipgyr", "pgyr", "ugyr", "piso", "uiso", "graduiso", "ppol", "upol", "gradupol"]
        for aq in accessible_quantities:
            if aq in needed_quantities:
                logging.info(f"... computing {aq} from _pi.h5")
                QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
        dic_quant["meanppar"] = np.mean(dic_quant["ppar"])
        dic_quant["meanpperp"] = np.mean(dic_quant["pperp"])
        del (dic_quant["ppar"], dic_quant["pperp"])
        
        accessible_quantities = ["Ipcgl", "pcgl", "ucgl"]
        tag = False
        for k in accessible_quantities:
            if k in needed_quantities : tag = True
        if tag:
            with h5.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
                (
                    dic_quant["bx"],
                    dic_quant["by"],
                    dic_quant["bz"],
                ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
            for aq in accessible_quantities:
                if aq in needed_quantities:
                    logging.info(f"... computing {aq} from _b et _rho.h5")
                    QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
            del(dic_quant['bx'],dic_quant['by'],dic_quant['bz'])
            g.close()
    logging.info(f"... End computing quantities from _pi.h5")

    # Magnetic field source file
    output_file = f"{output_folder}/{name}_b.h5temp"
    if not process_on_standard_h5_file.verif_file_existence(output_file, ""):
        g = h5.File(output_file, "a")
        accessible_quantities = ["Ib", "Ipm",'Ibnorm','Ihdm']
        tag = False
        for k in accessible_quantities:
            if k in needed_quantities : tag = True
        if tag:
            with h5.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
                (
                    dic_quant["bx"],
                    dic_quant["by"],
                    dic_quant["bz"],
                ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
            for aq in accessible_quantities:
                if aq in needed_quantities:
                    logging.info(f"... computing {aq} from _b.h5")
                    QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
            del(dic_quant['bx'],dic_quant['by'],dic_quant['bz'])
        
        accessible_quantities = ["Ij"]
        tag = False
        for k in accessible_quantities:
            if k in needed_quantities : tag = True
        if tag:
            with h5.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
                (
                    dic_quant["bx"],
                    dic_quant["by"],
                    dic_quant["bz"],
                ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
            from .quantities.j import get_original_quantity
            get_original_quantity(dic_quant, dic_param, delete=True)
            for aq in accessible_quantities:
                if aq in needed_quantities:
                    logging.info(f"... computing {aq} from _b.h5")
                    QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
            del(dic_quant['jx'],dic_quant['jy'],dic_quant['jz'])
            
        accessible_quantities = ["j", "divj"]
        tag = False
        for k in accessible_quantities:
            if k in needed_quantities : tag = True
        if tag:
            with h5.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
                (
                    dic_quant["bx"],
                    dic_quant["by"],
                    dic_quant["bz"],
                ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
            from .quantities.j import get_original_quantity
            get_original_quantity(dic_quant, dic_param, delete=True, inc=False)
            for aq in accessible_quantities:
                if aq in needed_quantities:
                    logging.info(f"... computing {aq} from _b.h5")
                    QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
            del(dic_quant['jcx'],dic_quant['jcy'],dic_quant['jcz'])
        
        accessible_quantities = ["b", "divb", "pm", "bnorm","hdm"]
        tag = False
        for k in accessible_quantities:
            if k in needed_quantities : tag = True
        if tag:
            with h5.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
                (
                    dic_quant["bx"],
                    dic_quant["by"],
                    dic_quant["bz"],
                ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
            from .quantities.b import get_original_quantity
            get_original_quantity(dic_quant, dic_param, delete=True)
            for aq in accessible_quantities:
                if aq in needed_quantities:
                    logging.info(f"... computing {aq} from _b.h5")
                    QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
            del(dic_quant['vax'],dic_quant['vay'],dic_quant['vaz'])
        g.close()
    logging.info(f"... End computing quantities from _b.h5")
    
    # forcing source file
    import os
    if os.path.isfile(f"{input_folder}/3Dfields_forcl_ampl.h5"):
        output_file = f"{output_folder}/{name}_f.h5temp"
        if not process_on_standard_h5_file.verif_file_existence(output_file, ""):
            g = h5.File(output_file, "a")
            with h5.File(f"{input_folder}/3Dfields_forcl_ampl.h5", "r") as fv:
                (dic_quant["fp"],
                    dic_quant["fm"],
                ) = extract_quantities_from_OCA_file(fv, ["forcl_ampl_plus", "forcl_ampl_mins"], cycle)
            accessible_quantities = ["f",]
            for aq in accessible_quantities:
                if aq in needed_quantities:
                    logging.info(f"... computing {aq} from _forcl_ampl.h5")
                    QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
            del (dic_quant["fp"],dic_quant["fm"])
            g.close()
        logging.info("... End computing quantities from _forcl_ampl.h5")
    else :
        logging.error("No file 3Dfields_forcl_ampl.h5, impossible to compute f.")
    del dic_quant

    output_file = f"{output_folder}/{name}.h5"
    g = h5.File(output_file, "w")
    for s in ['v','rho','p','b','f']:
        if os.path.isfile(f"{output_folder}/{name}_{s}.h5temp"):
            with h5.File(f"{output_folder}/{name}_{s}.h5temp",'r') as f:
                for k in f.keys():
                    g.create_dataset(k,data=np.ascontiguousarray(f[k]))
            os.remove(f"{output_folder}/{name}_{s}.h5temp")
                
    # Param
    g.create_group("param")
    for key in dic_param.keys():
        g["param"].create_dataset(key, data=dic_param[key])

    g["param"].create_dataset("laws", data=laws)
    g["param"].create_dataset("quantities", data=quantities)
    g["param"].create_dataset("terms", data=terms)
    g["param"].create_dataset("cycle", data=cycle)
    g["param"].create_dataset("name", data=name)
    g["param"].create_dataset("sim_type", data=sim_type)
    g["param"].create_dataset("reduction", data=reduction)

    #for key in physical_params.keys():
    #    g["param"].create_dataset(key, data=physical_params[key])

    g.close()
    logging.info(f"End process from_OCA_files_to_standard_h5_file()\n")

    return output_file

def reformat_oca_files(config_file):
    """
    config_file example: 
        [INPUT_DATA]
        path = /home/jeandet/Documents/DATA/Pauline/
        cycle = cycle_0
        sim_type = OCA_CGL2

        [OUTPUT_DATA]
        path = ./
        name = OCA_CGL2_cycle0_completeInc
        reduction_type = "trunc" ou "bin"
        reduction = 2 ou [2,2,4] ou [[0,0,0],[127,296,127]]
        laws = ['SS22I', 'BG17']
        terms = ['flux_dvdvdv']
        quantities = ['Iv']

        [PHYSICAL_PARAMS]
        di = 1
        eta = 0
        nu = 0
        an_hd = 0
        a1_forc = 0.5
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    file_process = from_OCA_files_to_standard_h5_file(
        input_folder=config["INPUT_DATA"]["path"],
        output_folder=config["OUTPUT_DATA"]["path"],
        name=config["OUTPUT_DATA"]["name"],
        sim_type=config["INPUT_DATA"]["sim_type"],
        cycle=config["INPUT_DATA"]["cycle"],
        quantities=eval(config["OUTPUT_DATA"]["quantities"]),
        laws=eval(config["OUTPUT_DATA"]["laws"]),
        terms=eval(config["OUTPUT_DATA"]["terms"]),
        reduction=1,
        physical_params={k: float(eval((config["PHYSICAL_PARAMS"][k]))) for k in config["PHYSICAL_PARAMS"].keys()},
    )
    process_on_standard_h5_file.check_file(file_process)

    
    if config["OUTPUT_DATA"]["reduction"] != "1":
        if "reduction_type" in config["OUTPUT_DATA"].keys():
            if "reduction_name" in config["OUTPUT_DATA"].keys():
                file_process = process_on_standard_h5_file.data_reduction(file_process, eval(config["OUTPUT_DATA"]["reduction"]), config["OUTPUT_DATA"]["reduction_type"], config["OUTPUT_DATA"]["reduction_name"])
            else:
                file_process = process_on_standard_h5_file.data_reduction(file_process, eval(config["OUTPUT_DATA"]["reduction"]), config["OUTPUT_DATA"]["reduction_type"])
        else:
            file_process = process_on_standard_h5_file.data_binning(file_process, eval(config["OUTPUT_DATA"]["reduction"]))
        process_on_standard_h5_file.check_file(file_process)
