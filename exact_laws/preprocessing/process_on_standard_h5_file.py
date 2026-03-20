import os
import h5py as h5
import numpy as np
import logging

def verif_file_existence(filename, message):
    if os.path.isfile(filename):
        logging.error(f"The file already exists. {message}")
        return True

def check_file(filename):
    """
    Affichage du contenu du fichier .h5 nommé "file".
    Les informations délivrées à propos des cubes de données sont la moyenne et l'écart-type.
    """
    logging.info(f"Check file {filename} ...") 
    message = describ_file(filename)              
    logging.info(message+"\n") 

def recursive_describ_of_h5file(file, path='/'):
    message = ''
    for k in file[path].keys():
        if not 'param' in k :
            splitted_key = str(file[path + '/' + k]).split()
            if splitted_key[1] == 'group':
                message += f"\n\t - {k}:"
                message += recursive_describ_of_h5file(file, path + '/' + k).replace('\n\t', '\n\t\t')
            else:
                tab = np.array(file[path + '/' + k])
                size = len(np.shape(tab))
                if size <= 1:
                    message += f"\n\t - {k}: {tab}"
                else : 
                    message += f"\n\t - {k}: {np.nanmean(tab):.5} $\pm$ {np.nanstd(tab):.5}"
    return message
    
def describ_file(filename):
    message = f"... file {filename} contains:"
    with h5.File(filename, 'r') as f:
        k = ""
        for key in f.keys():
            if "param" in key : k = key
        
        message += f"\n\t - {k}:"
        message += recursive_describ_of_h5file(f,path='/'+k).replace('\n\t', '\n\t\t')
        message += recursive_describ_of_h5file(f)
    return message

def recursive_copy_of_file(file_to_copy, output_file, path='/', input_to_record={}):
    for k in file_to_copy[path].keys():
        splitted_key = str(file_to_copy[path + '/' + k]).split()
        if splitted_key[1] == 'group':
            output_file[path].create_group(k)
            recursive_copy_of_file(file_to_copy, output_file, path + '/' + k)
        else:
            output_file[path].create_dataset(k, data=input_to_record.get(k, h5.Empty('f')))


def copy_struct_h5file(name_file_to_copy, name_output_file, input_to_record={}):
    with h5.File(name_file_to_copy, 'r') as fc:
        with h5.File(name_output_file, 'w') as fo:
            recursive_copy_of_file(fc, fo, input_to_record=input_to_record)
    

def data_binning(file_process, bin):
    """
    Vérification si le fichier contenant les données réduite existe déjà.
    Si non, enclenche le processus de création.
    Puis affiche le contenu du nouveau fichier.
    """
    output_filename = f"{file_process[:-3]}_bin{str(bin)}.h5"
    message = (f"Begin process data_binning() with config:"
               f"\n\t - input_file: {file_process}"
               f"\n\t - output_file: {output_filename}"
               f"\n\t - reduction: {bin}"
               )
    logging.info(message)
    
    
    if verif_file_existence(output_filename, "Data binning impossible."):
        logging.info(f"End process data_binning()\n")
        return output_filename
    else:
        bin_arrays_in_h5(file_process, output_filename, bin)
        
    logging.info(f"End process data_binning()\n")
    
    return output_filename


def bin_an_array(tab, binning):
    """
    Input: tab(np.array à réduire), binning(int,facteur de réduction)
    Output: np.array réduit
    """
    return (
        tab.reshape(
            np.shape(tab)[0] // binning,
            binning,
            np.shape(tab)[1] // binning,
            binning,
            np.shape(tab)[2] // binning,
            binning,
        )
        .mean(-1)
        .mean(3)
        .mean(1)
    )


def bin_arrays_in_h5(filename, output_filename, binning):
    """
    Input: file_name(str, name+ext of the original file), binning(int, binning factor)
    Apply a binning on the data contained in the .h5 file named "file_name".
    Record the new dataset.
    Output: name of the new file.
    """
    with h5.File(filename, "r") as g:
        with h5.File(output_filename, "w") as f:
            for k in g.keys():
                if not "param" in k:
                    tab = np.ascontiguousarray(g[k], dtype=np.float64)
                    new_tab = bin_an_array(tab, binning)
                    f.create_dataset(k, data=new_tab)
                else:
                    f.create_group("param")
                    f["param"]["N"] = np.array(g["param"]["N"]) // binning
                    f["param"]["c"] = np.array(g["param"]["c"]) * binning
                    f["param"]["reduction"] = binning
                    keys = list(g["param"].keys())
                    keys.remove("N")
                    keys.remove("c")
                    keys.remove("reduction")
                    for kp in keys:
                        f["param"].create_dataset(kp, data=g["param"][kp])
    return 0

def data_reduction(file_process, reduction_param, reduction_type, reduction_name="red"):
    """
    Vérification si le fichier contenant les données réduite existe déjà.
    Si non, enclenche le processus de création.
    Puis affiche le contenu du nouveau fichier.
    """ 
    output_filename = f"{file_process[:-3]}_{reduction_name}.h5"
    message = (f"Begin process data_truncation() with config:"
               f"\n\t - input_file: {file_process}"
               f"\n\t - output_file: {output_filename}"
               f"\n\t - reduction: {reduction_type} {reduction_param}"
               )
    logging.info(message)
    
    if verif_file_existence(output_filename, "Data reduction impossible."):
        logging.info(f"End process data_reduction()\n")
        return output_filename
    else:
        if reduction_type == 'trunc':
            trunc_arrays_in_h5(file_process, output_filename, reduction_param)
        else: 
            bin_arrays_in_h5(file_process, output_filename, reduction_param)
    logging.info(f"End process data_reduction()\n")
    
    return output_filename

def trunc_an_array(tab, trunc):
    """
    Input: tab(np.array à réduire), trunc(int,facteur de réduction)
    Output: np.array réduit
    """
    return tab[trunc[0][0]:trunc[1][0],trunc[0][1]:trunc[1][1],trunc[0][2]:trunc[1][2]]


def trunc_arrays_in_h5(filename, output_filename, trunc):
    """
    Input: file_name(str, name+ext of the original file), trunc(int, binning factor)
    Apply a binning on the data contained in the .h5 file named "file_name".
    Record the new dataset.
    Output: name of the new file.
    """
    with h5.File(filename, "r") as g:
        with h5.File(output_filename, "w") as f:
            for k in g.keys():
                tronc = [[],[]]
                i = 0
                for n in np.array(g["param"]["N"]):
                    if type(trunc) == 'int':
                        tronc[0].append(0)
                        tronc[1].append(n//trunc)
                    elif len(np.shape(trunc)) == 1:
                        tronc[0].append(0)
                        tronc[1].append(n//trunc[i])
                    elif len(np.shape(trunc)) == 2:
                        tronc[0].append(trunc[0][i])
                        tronc[1].append(trunc[1][i])
                    i += 1
                if not "param" in k:
                    tab = np.ascontiguousarray(g[k], dtype=np.float64)
                    new_tab = trunc_an_array(tab, trunc)
                    f.create_dataset(k, data=new_tab)
                else:
                    f.create_group("param")
                    f["param"]["N"] = np.array([t[1]-t[0] for t in len(tronc[0])])
                    f["param"]["L"] = np.array([(t[1]-t[0])*g["param"]["c"][t] for t in len(tronc[0])])
                    f["param"]["reduction"] = trunc
                    keys = list(g["param"].keys())
                    keys.remove("N")
                    keys.remove("L")
                    keys.remove("reduction")
                    for kp in keys:
                        f["param"].create_dataset(kp, data=g["param"][kp])
    return 0

def extract_quantities_from_h5_file(file, list_quant):
    list_data = []
    with h5.File(file, "r") as f:
        for quant in list_quant:
            list_data.append(np.ascontiguousarray(f[f"{quant}"], dtype=np.float64))
    return list_data
