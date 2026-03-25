# %% Librairies
import matplotlib.pyplot as plt
import numpy as np
from visualisation_tools import *

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 15
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.labelsize']= 15
plt.rcParams['ytick.labelsize']= 15
plt.rcParams['legend.fontsize'] = 10

simu = "CGL5"
cycle = "cycle0"
file_to_read = "results/OCA_CGL5_processed_all_laws_reduc1_red2D.h5"
quantities, grid, coeffs = readfile(file_to_read)
print("Quantities:", quantities.keys())
print("Coeffs:", coeffs.keys())
print("Grid:", grid.keys())

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

dic_of_list_terms['BG17'] = [k for k in coeffs if k.split('_',1)[0] == 'BG17']

dic_of_list_terms['ISS22Gyr'] = [k for k in coeffs if k.split('_',1)[0] == 'ISS22Gyr']

dic_of_list_terms['ISS22Iso'] = [k for k in coeffs if k.split('_',1)[0] == 'ISS22Iso']

dic_of_list_terms['ISS22Cgl'] = [k for k in coeffs if k.split('_',1)[0] == 'ISS22Cgl']

dic_of_list_terms['IHallcor'] = [k for k in coeffs if k.split('_',1)[0] == 'IHallcor']

# %% Sort the datas in dictionaries
lpar = grid["lpar"]*grid['c'][2]; lperp = grid["lperp"]*grid['c'][0]
lx = grid["lx"]*grid['c'][0]; ly = grid["ly"]*grid['c'][1]; lz = grid["lz"]*grid['c'][2]
print("lpar:", lpar.shape, "lperp:", lperp.shape)
print("lx:", lx.shape, "ly:", ly.shape, "lz:", lz.shape)
results = {}
for law in dic_of_list_terms.keys():
    list_term = dic_of_list_terms[law]
    results[law] = np.transpose(linear_op_from_list_term(coeffs,quantities,list_term))

# %% Plot the results for PP98
save = False
num = "PP98"
law = "PP98"
decimal = int(np.max(np.log10(np.abs(results[law]))))
xlab = r"l$_{\perp}$"; ylab = r"l$_{\parallel}$"; zlab = f"$\\epsilon_{{{law}}}$"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
vmin = -np.max(np.abs(results[law])); vmax = -vmin
levels = np.array([-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,-1e-7,-1e-8,-1e-9,0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
ticks = np.array([-1e-2,-1e-4,-1e-6,-1e-8,0,1e-8,1e-6,1e-4,1e-2])
display_map_log(num, lperp, lpar, results[law], xlab, ylab, zlab, title, 
            xdissi, ydissi, xforc, yforc, vmin = vmin, vmax = vmax,
            levels = levels, ticks = ticks)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)

# %% Plot the results for BG17
num = "BG17"
law = "BG17"
decimal = int(np.max(np.log10(np.abs(results[law]))))
xlab = r"l$_{\perp}$"; ylab = r"l$_{\parallel}$"; zlab = f"$\\epsilon_{{{law}}}$"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
vmin = -np.max(np.abs(results[law])); vmax = -vmin
levels = np.array([-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,-1e-7,-1e-8,-1e-9,0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
ticks = np.array([-1e-2,-1e-4,-1e-6,-1e-8,0,1e-8,1e-6,1e-4,1e-2])
display_map_log(num, lperp, lpar, results[law], xlab, ylab, zlab, title, 
            xdissi, ydissi, xforc, yforc, vmin = vmin, vmax = vmax,
            levels = levels, ticks = ticks)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)

# %% Plot the results for ISS22Gyr
num = "ISS22Gyr"
law = "ISS22Gyr"
decimal = int(np.max(np.log10(np.abs(results[law]))))
xlab = r"l$_{\perp}$"; ylab = r"l$_{\parallel}$"; zlab = f"$\\epsilon_{{{law}}}$"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
vmin = -np.max(np.abs(results[law])); vmax = -vmin
levels = np.array([-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,-1e-7,-1e-8,-1e-9,0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
ticks = np.array([-1e-2,-1e-4,-1e-6,-1e-8,0,1e-8,1e-6,1e-4,1e-2])
display_map_log(num, lperp, lpar, results[law], xlab, ylab, zlab, title, 
            xdissi, ydissi, xforc, yforc, vmin = vmin, vmax = vmax,
            levels = levels, ticks = ticks)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)

# %% Plot the results for ISS22Iso
num = "ISS22Iso"
law = "ISS22Iso"
decimal = int(np.max(np.log10(np.abs(results[law]))))
xlab = r"l$_{\perp}$"; ylab = r"l$_{\parallel}$"; zlab = f"$\\epsilon_{{{law}}}$"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
vmin = -np.max(np.abs(results[law])); vmax = -vmin
levels = np.array([-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,-1e-7,-1e-8,-1e-9,0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
ticks = np.array([-1e-2,-1e-4,-1e-6,-1e-8,0,1e-8,1e-6,1e-4,1e-2])
display_map_log(num, lperp, lpar, results[law], xlab, ylab, zlab, title, 
            xdissi, ydissi, xforc, yforc, vmin = vmin, vmax = vmax,
            levels = levels, ticks = ticks)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)

# %% Plot the results for ISS22Cgl
num = "ISS22Cgl"
law = "ISS22Cgl"
decimal = int(np.max(np.log10(np.abs(results[law]))))
xlab = r"l$_{\perp}$"; ylab = r"l$_{\parallel}$"; zlab = f"$\\epsilon_{{{law}}}$"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
vmin = -np.max(np.abs(results[law])); vmax = -vmin
levels = np.array([-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,-1e-7,-1e-8,-1e-9,0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
ticks = np.array([-1e-2,-1e-4,-1e-6,-1e-8,0,1e-8,1e-6,1e-4,1e-2])
display_map_log(num, lperp, lpar, results[law], xlab, ylab, zlab, title, 
            xdissi, ydissi, xforc, yforc, vmin = vmin, vmax = vmax,
            levels = levels, ticks = ticks)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)

# %% Plot the results for IHallcor
num = "IHallcor"
law = "IHallcor"
decimal = int(np.max(np.log10(np.abs(results[law]))))
xlab = r"l$_{\perp}$"; ylab = r"l$_{\parallel}$"; zlab = f"$\\epsilon_{{{law}}}$"
title = f"Total {law} OCA {simu}"
xdissi = dissi_perp; ydissi = dissi_par; xforc = forc_perp; yforc = forc_par
vmin = -np.max(np.abs(results[law])); vmax = -vmin
levels = np.array([-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,-1e-7,-1e-8,-1e-9,0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
ticks = np.array([-1e-2,-1e-4,-1e-6,-1e-8,0,1e-8,1e-6,1e-4,1e-2])
display_map_log(num, lperp, lpar, results[law], xlab, ylab, zlab, title, 
            xdissi, ydissi, xforc, yforc, vmin = vmin, vmax = vmax,
            levels = levels, ticks = ticks)
plt.tight_layout()
if save:
    plt.savefig(f"images/visualisation_{num}_{simu}.png", dpi=300)

# %% Average along lparallel for every laws
plt.figure("Average along lparallel")
for law in dic_of_list_terms.keys():
    plt.plot(lperp, np.mean(results[law], axis=0), label=law)
plt.xscale("log"); plt.yscale("log")
plt.xlabel(r"l$_{\perp}$"); plt.ylabel(r"$\epsilon_{law}$")
plt.title(f"Comparison of incompressible laws OCA {simu} {cycle}")
plt.suptitle(r"$l_{\parallel}$ = ["+str(np.round(np.min(lpar),1))+","+str(np.round(np.max(lpar),1))+"]")
plt.legend()

# %%
