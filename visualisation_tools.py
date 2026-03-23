import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
import os

def read_grid_in_h5file(file):
    with h5.File(file,'r') as f:
        grid = {}
        for k in f['grid']['coords'].keys():
            grid[k] = np.array(f['grid']['coords'][k])
        for k in f['grid'].keys():
            if k != 'coords':
                grid[k] = np.array(f['grid'][k])
    return grid

def readfile(file):
    with h5.File(file,'r') as f:
        quantities = {}
        for k in f.keys():
            if not k in ['params', 'grid']:
                quantities[k] = np.array(f[k])
        coeffs = {}
        for k in f['params']['coeffs']:
            coeffs[k] = np.array(f['params']['coeffs'][k])
        grid = {}
        for k in f['grid']['coords'].keys():
            grid[k] = np.array(f['grid']['coords'][k])
        for k in f['grid'].keys():
            if k != 'coords':
                grid[k] = np.array(f['grid'][k])
    return quantities, grid, coeffs

def logcyl_reduce_quantity(quantity,grid):
    if len(np.shape(quantity)) == 3:
        tab = np.ones((np.shape(grid['lz'])[0],np.shape(grid['lperp'])[0]))
        for i in range(np.shape(grid['lz'])[0]):
            for j in range(np.shape(grid['lperp'])[0]):
                tab[i,j] = np.mean(np.sort(quantity[i,j,:int(grid['count'][j])].flatten()))
        return tab
    elif len(np.shape(quantity)) == 4: #cas d'un flux
        list_tab  = []
        for a in range(3):
            tab = np.ones((np.shape(grid['lz'])[0],np.shape(grid['lperp'])[0]))
            for i in range(np.shape(grid['lz'])[0]):
                for j in range(np.shape(grid['lperp'])[0]):
                    tab[i,j] = np.mean(np.sort(quantity[i,j,:int(grid['count'][j]),a].flatten()))
            list_tab.append(tab)
        return list_tab

def logcyl_reduce(law,grid):
    law_reduce = {}
    for k in law.keys():
        if len(np.shape(law[k])) == 3:
            tab = np.ones((np.shape(grid['lz'])[0],np.shape(grid['lperp'])[0]))
            for i in range(np.shape(grid['lz'])[0]):
                for j in range(np.shape(grid['lperp'])[0]):
                    tab[i,j] = np.mean(np.sort(law[k][i,j,:int(grid['count'][j])].flatten()))
            law_reduce[k] = tab
        elif len(np.shape(law[k])) == 4:
            axis = ['x','y','z']
            for a in range(3):
                tab = np.ones((np.shape(grid['lz'])[0],np.shape(grid['lperp'])[0]))
                for i in range(np.shape(grid['lz'])[0]):
                    for j in range(np.shape(grid['lperp'])[0]):
                        tab[i,j] = np.mean(np.sort(law[k][i,j,:int(grid['count'][j]),a].flatten()))
                law_reduce[k+f"_{axis[a]}"] = tab
    return law_reduce

def linear_op_from_list_term(coeffs,quantities,list_term):
    coeff = {k.split('_',1)[1]:coeffs[k] for k in coeffs if k in list_term}
    result = np.zeros(np.shape(quantities[list(coeff.keys())[0]]))
    for k in coeff.keys():
            result += coeff[k]*quantities[k]
    return result


def display_map_duo(num,x,y,z,xlab,ylab,zlab,title,xdissi=1,ydissi=1,xforc=1,yforc=1):
    if num != -1 : 
        plt.figure(num,figsize=(10,5))  
        plt.clf()
    plt.suptitle(title)
    
    plt.subplot(121)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    #plt.grid()
    plt.contourf(x,y,z,locator=ticker.SymmetricalLogLocator(linthresh=1e-9,base=10), norm=colors.SymLogNorm(linthresh=1e-9, linscale=1e-9,base=10),
                       cmap='RdBu_r')#,shading='auto')#cmap='viridis') 
    plt.axvline(xforc,color='silver',linestyle='-.')
    plt.axvline(xdissi,color='silver',linestyle=':')
    plt.axhline(yforc,color='silver',linestyle='-.')
    plt.axhline(ydissi,color='silver',linestyle=':')
    plt.xlim(xmin=x[1],xmax=x[-1])
    plt.ylim(ymin=y[1],ymax=y[-1])
    plt.colorbar(format = ticker.LogFormatterSciNotation(),label=zlab) #format= ticker.SymLogFormatter()
    
    plt.subplot(122)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    #plt.grid()
    plt.contourf(x,y,z,cmap='viridis') #,locator=ticker.LogLocator()
    plt.axvline(xforc,color='silver',linestyle='-.')
    plt.axvline(xdissi,color='silver',linestyle=':')
    plt.axhline(yforc,color='silver',linestyle='-.')
    plt.axhline(ydissi,color='silver',linestyle=':')
    plt.xlim(xmin=x[1],xmax=x[-1])
    plt.ylim(ymin=y[1],ymax=y[-1])
    plt.colorbar(label=zlab)
    
    if num != -1 : 
        plt.tight_layout()
        plt.show()
        
def display_map_log(num,x,y,z,xlab,ylab,zlab,title,xdissi=None,ydissi=None,xforc=None,yforc=None,vmin=-1,vmax = 1,ticks=None,levels=None):
    if num != -1 : 
        plt.figure(num,figsize=(10,5))  
        plt.clf()
    plt.suptitle(title)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    #plt.grid()
    plt.contourf(x,y,z,locator=ticker.SymmetricalLogLocator(linthresh=1e-9,base=10), norm=colors.SymLogNorm(linthresh=1e-9, linscale=1e-9,base=10),
                       cmap='RdBu_r',levels = levels)#,shading='auto')#cmap='viridis') 
    if not xforc == None: plt.axvline(xforc,color='silver',linestyle='-.')
    if not xdissi == None: plt.axvline(xdissi,color='silver',linestyle=':')
    if not yforc == None:plt.axhline(yforc,color='silver',linestyle='-.')
    if not ydissi == None:plt.axhline(ydissi,color='silver',linestyle=':')
    plt.xlim(xmin=x[1],xmax=x[-1])
    plt.ylim(ymin=y[1],ymax=y[-1])
    plt.colorbar(format = ticker.LogFormatterSciNotation(),ticks=ticks,label=zlab) #format= ticker.SymLogFormatter()
    
    if num != -1 : 
        plt.tight_layout()
        plt.show()
    
def display_map(num,x,y,z,xlab,ylab,zlab,title,xdissi=None,ydissi=None,xforc=None,yforc=None,vmin=-1,vmax = 1,ticks=None,levels=None,formats=None):
    if num != -1 : 
        plt.figure(num,figsize=(10,5))  
        plt.clf()
    plt.suptitle(title)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    #plt.grid()
    plt.contourf(x,y,z,cmap='RdBu_r',levels = levels) #,locator=ticker.LogLocator()
    if not xforc == None: plt.axvline(xforc,color='silver',linestyle='-.')
    if not xdissi == None: plt.axvline(xdissi,color='silver',linestyle=':')
    if not yforc == None:plt.axhline(yforc,color='silver',linestyle='-.')
    if not ydissi == None:plt.axhline(ydissi,color='silver',linestyle=':')
    plt.xlim(xmin=x[1],xmax=x[-1])
    plt.ylim(ymin=y[1],ymax=y[-1])
    plt.colorbar(ticks=ticks,label=zlab,format=formats)
    
    if num != -1 : 
        plt.tight_layout()
        plt.show()

def display_range_curve(num,x,y,dirr,rng,rng_grid,xlab,ylab,title):
    plt.figure(num)
    plt.clf()
    plt.title(title)
    if dirr == 0:
        for i in range(*rng): 
            p, = plt.plot(x,np.abs(y[i,:]),'--')
            plt.plot(x,np.ma.masked_where(y[i,:]<0,y[i,:]),color=p.get_color(),label=str(rng_grid[i]))
        mean = np.mean(y[rng[0]:rng[1],:],axis=dirr)
    elif dirr == 1: 
        for i in range(*rng): 
            p, = plt.plot(x,np.abs(y[:,i]),'--')
            plt.plot(x,np.ma.masked_where(y[:,i]<0,y[:,i]),color=p.get_color(),label=str(rng_grid[i]))
        mean = np.mean(y[:,rng[0]:rng[1]],axis=dirr)
    plt.plot(x,np.abs(mean),'k',label='mean')
    plt.plot(x,np.ma.masked_where(mean<0,mean),'--k')
    plt.legend(ncol=6,bbox_to_anchor = (0,-0.2),loc='upper left')
    plt.xscale('log')
    plt.xlabel(xlab)
    plt.yscale('log')
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.show()

def display_terms(num,x,y,key,coeff,dirr,index,xlab,title=" ",labels=[None],xdissi = 1, xforc=0, leg = False):
    if num != -1 : 
        plt.figure(num,figsize=(10,5))
        plt.clf()
        plt.suptitle(title+f"index: {index}")
    if labels[0] == None :
        labels = key
    i = 0
    for k in key:
        if dirr == 0: 
            if k in coeff.keys(): y_tab = y[k.split('_',1)[1]][index,:]*coeff[k]
            else : y_tab = y[k][index,:]
        elif dirr == 1: 
            if k in coeff.keys(): y_tab = y[k.split('_',1)[1]][:,index]*coeff[k]
            else : y_tab = y[k][:,index]
        p, = plt.plot(x,np.abs(y_tab),'--')
        plt.plot(x,np.ma.masked_where(y_tab<0,y_tab),color=p.get_color(),label=labels[i])
        i +=1
    if xforc < x[-1] and xforc > 0 : plt.axvline(xforc,color='silver',linestyle='-.')
    if (xdissi > x[0] and x[0] > 0) or xdissi > x[1]: plt.axvline(xdissi,color='silver',linestyle=':')
    #plt.legend(bbox_to_anchor = (1, 1))
    if leg: plt.legend()
    plt.xscale('log')
    plt.xlabel(xlab)
    plt.yscale('log')
    if num != -1 : 
        plt.tight_layout()
        plt.show()

def display_mean(num,x,y,key,coeff,dirr,xlab,title=" ",mini=0,maxi=-1,labels=[None],xdissi = 1, xforc=0, leg = False):
    if num != -1 : 
        plt.figure(num,figsize=(10,5))
        plt.clf()
        plt.suptitle(title)
    if labels[0] == None :
        labels = key
    i = 0
    for k in key:
        if dirr == 0: 
            if k in coeff.keys(): y_tab = np.mean(y[k.split('_',1)[1]][mini:maxi,:],axis=0)*coeff[k]
            else : y_tab = np.mean(y[k][mini:maxi,:],axis=0)
        elif dirr == 1: 
            if k in coeff.keys(): y_tab = np.mean(y[k.split('_',1)[1]][:,mini:maxi],axis=1)*coeff[k]
            else : y_tab = np.mean(y[k][:,mini:maxi],axis=1)
        p, = plt.plot(x,np.abs(y_tab),'--')
        plt.plot(x,np.ma.masked_where(y_tab<0,y_tab),color=p.get_color(),label=labels[i])
        i+=1
    if xforc < x[-1] and xforc > 0 : plt.axvline(xforc,color='silver',linestyle='-.')
    if (xdissi > x[0] and x[0] > 0) or xdissi > x[1]: plt.axvline(xdissi,color='silver',linestyle=':')
    #plt.legend(bbox_to_anchor = (1, 1))
    if leg: plt.legend()
    plt.xscale('log')
    plt.xlabel(xlab)
    plt.yscale('log')
    if num != -1 : 
        plt.tight_layout()
        plt.show()
     
def splotold(x,y,label='',color=None,linewidth=None,alpha = None):
    if alpha :
        if linewidth and color : #
            pplus, = plt.plot(x,np.abs(y),'--',color=color,linewidth=linewidth,alpha = alpha)
        elif linewidth :
            pplus, = plt.plot(x,np.abs(y),'--',linewidth=linewidth,alpha = alpha) 
        elif color :
            pplus, = plt.plot(x,np.abs(y),'--',color=color,alpha = alpha)    
        else : pplus, = plt.plot(x,np.abs(y),'--',alpha = alpha) 
    else : 
        if linewidth and color : 
            pplus, = plt.plot(x,np.abs(y),'--',color=color,linewidth=linewidth)
        elif linewidth :
            pplus, = plt.plot(x,np.abs(y),'--',linewidth=linewidth) 
        elif color :
            pplus, = plt.plot(x,np.abs(y),'--',color=color)    
        else : pplus, = plt.plot(x,np.abs(y),'--',)
    pmoins = plt.plot(x,np.ma.masked_where(y<0,y),color=pplus.get_color(),label=label,linewidth=pplus.get_linewidth(),alpha = pplus.get_alpha())
    return pplus, pmoins
        
def splot(x,y,label='',color=None,linewidth=None,alpha = None):
    if alpha :
        if linewidth and color : #
            pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',color=color,linewidth=linewidth,alpha = alpha)
        elif linewidth :
            pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',linewidth=linewidth,alpha = alpha) 
        elif color :
            pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',color=color,alpha = alpha)    
        else : pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',alpha = alpha) 
    else : 
        if linewidth and color : 
            pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',color=color,linewidth=linewidth)
        elif linewidth :
            pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',linewidth=linewidth) 
        elif color :
            pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',color=color)    
        else : pplus, = plt.plot(x,np.abs(np.where(y>0,0,y)),'--',)
    pmoins = plt.plot(x,np.where(y<0,0,y),color=pplus.get_color(),label=label,linewidth=pplus.get_linewidth(),alpha = pplus.get_alpha())
    return pplus, pmoins

def subsplot(ax,x,y,label='',color=None,linewidth=None, orientation = 'normal'):
    if linewidth and color : 
        if orientation == 'inverse' : pplus, = ax.plot(np.abs(y),x,'--',color=color,linewidth=linewidth)
        else : pplus, = ax.plot(x,np.abs(np.where(y>0,0,y)),'--',color=color,linewidth=linewidth)
    elif linewidth :
        if orientation == 'inverse' : pplus, = ax.plot(np.abs(y),x,'--',linewidth=linewidth) 
        else : pplus, = ax.plot(x,np.abs(np.where(y>0,0,y)),'--',linewidth=linewidth) 
    elif color :
        if orientation == 'inverse' : pplus, = ax.plot(np.abs(y),x,'--',color=color)    
        else : pplus, = ax.plot(x,np.abs(np.where(y>0,0,y)),'--',color=color) 
    else : 
        if orientation == 'inverse' : pplus, = ax.plot(np.abs(y),x,'--',)
        else : pplus, = ax.plot(x,np.abs(np.where(y>0,0,y)),'--',)
    if orientation == 'inverse' : pmoins = ax.plot(np.where(y<0,0,y),x,color=pplus.get_color(),label=label,linewidth=pplus.get_linewidth())
    else : pmoins = ax.plot(x,np.where(y<0,0,y),color=pplus.get_color(),label=label,linewidth=pplus.get_linewidth())
    return pplus, pmoins
   
def sdotplot(x,y,label=''):
    pplus, = plt.plot(x,np.ma.masked_where(y<0,y),'.',label=label)
    pmoins = plt.plot(x,np.ma.masked_where(y>0,np.abs(y)),'.', markerfacecolor='none', markeredgecolor=pplus.get_color())
    return pplus, pmoins

def display_graph(record_file='', displ=True, rec=False, funcfig = None, **kwargs):
    if os.path.exists(record_file) and displ : 
        display(Image(filename=record_file))
        print('Displayed from ', record_file)
    elif rec:
        fig = funcfig(**kwargs)
        fig.savefig(record_file)
        print('Recorded in ', record_file)
        plt.show()
    else: 
        fig = funcfig(**kwargs)
        plt.show()