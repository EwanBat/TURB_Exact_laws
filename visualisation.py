# %% Librairies
import matplotlib.pyplot as plt
import numpy as np
from visualisation_tools import *

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 15
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12
plt.rcParams['legend.fontsize'] = 10

simu = "CGL5"
file_to_read = "results/OCA_CGL5_processed_bin2_PP98_BG17_red2D.h5"
quantities, grid, coeffs = readfile(file_to_read)
print("Quantities:", quantities.keys())
print("Coeffs:", coeffs.keys())

# %% Define the parameters for the exact laws
kdi = 0.045
tantheta = np.tan(75/180*np.pi)
forc_par = 1/kdi*tantheta
forc_perp = 1/kdi
dissi_perp = 1
dissi_par = 1

# %% Create a dictionary of list to sort the terms with the exact laws
dic_of_list_terms = {}
dic_of_list_terms['PP98'] = [k for k in coeffs if k.split('_',1)[0] == 'PP98']
print(dic_of_list_terms['PP98'])

dic_of_list_terms['BG17'] = [k for k in coeffs if k.split('_',1)[0] == 'BG17']
print(dic_of_list_terms['BG17'])

# %% Sort the datas in dictionaries
lpar = grid["lpar"]; lperp = grid["lperp"]
results = {}
for law in dic_of_list_terms.keys():
    list_term = dic_of_list_terms[law]
    results[law] = linear_op_from_list_term(coeffs,quantities,list_term)
print(results.keys())
# %% Plot the results for PP98
save = False
num = "PP98"
law = "PP98"
xlab = r"l$_{\parallel}$"; ylab = r"l$_{\perp}$"; zlab = "PP98"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
display_map(num, lpar, lperp, results[law], xlab, ylab, zlab, title, xdissi, ydissi, xforc, yforc)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)
# %% Plot the results for BG17
num = "BG17"
law = "BG17"
xlab = r"l$_{\parallel}$"; ylab = r"l$_{\perp}$"; zlab = "BG17"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
display_map(num, lpar, lperp, results[law], xlab, ylab, zlab, title, xdissi, ydissi, xforc, yforc)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)

# %%
