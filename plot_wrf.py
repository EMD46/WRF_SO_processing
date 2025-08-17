# -*- coding: utf-8 -*-
"""

@author: Estefania Montoya Feb 2023

Warning: NCAR developed a useful tool for
post-processing WRFout files
The package uses netCDF4 Dataset
Disclaimer also for some cartopy features
mercator projection is a bit anoying to put the lon labels
"""

import pyproj
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import matplotlib.ticker as mticker
from matplotlib.ticker import LogFormatter
import matplotlib.colors as mc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# modify colorbars
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

class plot_wrf():

    def  __init__(self,dir_fig):
        self.dir_save = dir_fig #main directory to save figures


    def plot_imshow(self,array,figsize,name_list,vmin,vmax,units,color_map,\
                        var,save,norm=None,clab=False,line=False,sc_sn=False,\
                        border=False):
        '''
        uses cartopy for projection and xarray values to make subplots

        array     : xarray annual cycle
        name_list : title for plot
        vmin      : min value of the colorbar
        vmax      : max value of the colorbar
        nround    : round colorbar # decimals
        units     : units of the variable to plots
        color_map : colormap to use
        var       : name of the variable
        save      : name save fig
        norm      : set center of colorbar at specific #
        clab      : create xticks...predefined in function (it can change to more general)
        line      : plot line, type dict line,color
        sc_sn     : add location of dropsonde as df
        '''

        map_proj  = ccrs.PlateCarree()
        cart_proj = get_cartopy(array)
        array = array


        fig,ax = plt.subplots(1,1,figsize=figsize,
                   subplot_kw={'projection': map_proj})

        ax.set_aspect('auto')

        cs = array.plot(x='XLONG',y='XLAT',vmin=vmin,vmax=vmax,cmap=color_map,\
                        norm=norm,add_colorbar=False)

        if line:
            for tr in line.keys():
                l = ax.plot(line[tr]['track'][0],line[tr]['track'][1],color=line[tr]['color'])

        if sc_sn:
            for sc in range(len(sc_sn[0])):
                stt = ax.scatter(sc_sn[0]['lon'][sc],sc_sn[0]['lat'][sc],color='k',s=20)

        ax.set_title(name_list,fontsize=15)
        if border:
            bc = border
        else:
            bc = 'white'
        ax.coastlines(linewidth=0.8,color=bc)
        ax.add_feature(cfeature.BORDERS,linewidth=0.8,color=bc)

        gl = ax.gridlines(crs=map_proj, draw_labels=True,
                          linestyle='--',color=bc)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines       = False
        gl.ylines       = False
        gl.xlabel_style = {'size':14}
        gl.ylabel_style = {'size':14}

        if not clab:
            #cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax)
            cbar = plt.colorbar(cs,fraction=0.046,pad=0.05,orientation='horizontal',ax=ax)
        if clab==1:
            cbar = fig.colorbar(cs,aspect=15,orientation='vertical',ax=ax,ticks=range(0,6))
            cbar.set_ticks(range(0,11))
            cbar.set_ticklabels(['Clear','Liquid','SLW','Mixed','Ice','Unk'])
        cbar.set_label('%s %s'%(var,units),fontsize=15)
        cbar.ax.tick_params(labelsize=14)

        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')
