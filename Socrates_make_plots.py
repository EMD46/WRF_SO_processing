# -*- coding: utf-8 -*-
"""
Class for making plots from flights of
SOCRATES project. Including 2D plots
and line plots

@author: Estefania Montoya September 2020
"""

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

from to_make_lines import *

#https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    ''' toma un pedazo de la barra de colores y con eso
    hace la colorbar'''
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# colorbars
#cl_p = ListedColormap(['white', 'lime','blueviolet','crimson','royalblue','black'])
cl_p       = ListedColormap(['white','limegreen','mediumpurple','pink','cornflowerblue','gray'])
cl_ty      = ListedColormap(['white','lightsteelblue','royalblue','navy','moccasin','gold','darkgoldenrod','sandybrown','salmon','red','k'])
cl_ty2      = ListedColormap(['white','lightgray','lightsteelblue','royalblue','navy','moccasin','gold','darkgoldenrod','sandybrown','salmon','red','k'])

nightfall  = ListedColormap(['#125A56', '#00767B', '#238F9D', '#42A7C6', '#60BCE9', '#9DCCEF', '#C6DBED', '#DEE6E7', '#ECEADA', '#F0E6B2',\
                            '#F9D576', '#FFB954', '#FD9A44', '#F57634', '#E94C1F', '#D11807', '#A01813'])
rainbow    = truncate_colormap(plt.cm.gist_ncar, minval=0.1, maxval=0.8)
ncar2      = truncate_colormap(plt.cm.gist_ncar, minval=0.1, maxval=1.)
prpl_red   = truncate_colormap(plt.cm.hsv_r, minval=0.2, maxval=1.0)
brown_blue = LinearSegmentedColormap.from_list("MyCmapName",["brown",'sienna','yellow',"cyan",'lightskyblue','lightsteelblue'][::-1])
fucsia_red = LinearSegmentedColormap.from_list("MyCmapName",["fuchsia",'b',"cyan",'darkgreen','gray','yellow','coral','red','brown'])
plasma     = 'plasma'
temp_lin   = ListedColormap(['lightgray','purple','royalblue','yellow','salmon','red','k'])
#ListedColormap(['white','green','yellow','red','blue','purple','k'])
BuRd       = ListedColormap(['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B'])
conv       = ListedColormap(['#0077BB','#EE7733','#EE3377'])

# for rain
#https://unidata.github.io/python-gallery/examples/Precipitation_Map.html
cmap_data = [(1.0, 1.0, 1.0),
             (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
             (0.0, 1.0, 1.0),
             (0.0, 0.8784313797950745, 0.501960813999176),
             (0.0, 0.7529411911964417, 0.0),
             (0.501960813999176, 0.8784313797950745, 0.0),
             (1.0, 1.0, 0.0),
             (1.0, 0.6274510025978088, 0.0),
             (1.0, 0.0, 0.0),
             (1.0, 0.125490203499794, 0.501960813999176),
             (0.9411764740943909, 0.250980406999588, 1.0),
             (0.501960813999176, 0.125490203499794, 1.0),
             (0.250980406999588, 0.250980406999588, 1.0),
             (0.125490203499794, 0.125490203499794, 0.501960813999176),
             (0.125490203499794, 0.125490203499794, 0.125490203499794),
             (0.501960813999176, 0.501960813999176, 0.501960813999176),
             (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
             (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
             (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
             (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
             (0.4000000059604645, 0.20000000298023224, 0.0)]
p_colors = mc.ListedColormap(cmap_data, 'precipitation')
slw_line = mc.ListedColormap(['gray','green','red','darkviolet'], 'slw')
#HCA      = mc.ListedColormap(cmap_data[-17:-7], 'microphysics')
HCA      = ListedColormap(['white','yellow','lime','teal','navy',\
                           'purple','crimson','coral','maroon'])


cmap_data2 = [(1.0, 1.0, 1.0),
             (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
             #(0.0, 1.0, 1.0),
             #(0.0, 0.8784313797950745, 0.501960813999176),
             (0.0, 0.7529411911964417, 0.0),
             #(0.501960813999176, 0.8784313797950745, 0.0),
             (1.0, 1.0, 0.0),
             (1.0, 0.6274510025978088, 0.0),
             (1.0, 0.0, 0.0),
             #(1.0, 0.125490203499794, 0.501960813999176),
             (0.9411764740943909, 0.250980406999588, 1.0),
             (0.501960813999176, 0.125490203499794, 1.0),
             #(0.250980406999588, 0.250980406999588, 1.0),
             (0.125490203499794, 0.125490203499794, 0.501960813999176),
             (0.125490203499794, 0.125490203499794, 0.125490203499794),
             (0.501960813999176, 0.501960813999176, 0.501960813999176),
             #(0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
             #(0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
             (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
             (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
             (0.4000000059604645, 0.20000000298023224, 0.0)]
p_colors2 = mc.ListedColormap(cmap_data2, 'precipitation')


v1 = truncate_colormap(plt.cm.get_cmap('tab10'), minval=0.0, maxval=0.68)
v2 = truncate_colormap(plt.cm.get_cmap('tab10'), minval=0.8, maxval=1)
v3 = truncate_colormap(plt.cm.get_cmap('Dark2'), minval=0.4, maxval=0.4)
v4 = truncate_colormap(plt.cm.get_cmap('Dark2'), minval=1.0, maxval=1.0)
new = np.vstack((v1(np.linspace(0, 1, 81)),
                v2(np.linspace(0, 1, 23)),
                 v3(np.linspace(0, 1, 12)),
                 v4(np.linspace(0, 1, 12))))
slw_line2 = mc.ListedColormap(new, 'microphysics')

rainbow2    = truncate_colormap(plt.cm.gist_ncar_r, minval=0.1, maxval=0.8)
grays    = truncate_colormap(plt.cm.gray_r, minval=0.1, maxval=0.9)
new = np.vstack((rainbow2(np.linspace(0, 1, 48)),
                grays(np.linspace(0, 1, 80))))
tbb  = mc.ListedColormap(new,'tbb')

v1 = truncate_colormap(plt.cm.get_cmap('terrain'), minval=0.0, maxval=0.9)
#v2 = truncate_colormap(plt.cm.get_cmap('gist_st'), minval=0.3, maxval=0.9)
new = np.vstack(v1(np.linspace(0, 1, 81))
                )#v2(np.linspace(0, 1, 81))))
doppler_c = mc.ListedColormap(new, 'dopplerc')

class plot_socrates():

    def  __init__(self,dir_fig):
        self.dir_save = dir_fig #main directory to save figures

    def plot_lines(self,xs,ys,minmax_tuple,color_list,ylab,save,twin=False,\
                   twin_double=False,leg_list=False,shade=False):
        '''
        plot one or more lines
        twin         : If True twinx
        twin_double  : If True more than one line on either side, type tuple
        shade        : False or tuple with dates to shade
        xs           : abscissa as array
        ys           : ordinate, list or tuple (multiple lines either side as list of list)
        color_list   : string or list or tuple (multiple lines either side as list of list)
        leg_list     : string, list or tuple (multiple lines either side as list of list)
        ylab         : ordinate label (ylabel) string or list
        save         : name to save fig
        minmax_tuple : tuple or list with min an max values of plot
        '''
        plt.figure(figsize=(11,3))
        ax=plt.gca()

        if not twin:
            # multiple lines same variability
            if (len(ys)>1) and (leg_list):
                for ind,i in enumerate(ys):
                    ax.plot(xs,i,color=color_list[ind],label=leg_list[ind])
                plt.legend(loc=0,fontsize=14,ncol=5)
                #plt.yscale('symlog', linthresh=500)
            if (len(ys)>1) and not (leg_list):
                for ind,i in enumerate(ys):
                    ax.plot(xs,i,color=color_list[ind])
                #plt.yscale('symlog', linthresh=500)
            # one line
            if len(ys)==1:
                ax.plot(xs,ys[0],color=color_list)

            ax.set_ylim(minmax_tuple[0],minmax_tuple[1])
            ax.set_ylabel(ylab,fontsize=14)
        else:
            if not twin_double:
                for snd,s in enumerate(ys):
                    if snd==0:
                        ss = ax
                    elif snd==1:
                        ss = ax.twinx()
                    else:
                        ss = ax.twinx()
                        pos = [0,0,1.13,1.26,1.39,1.52]
                        ss.spines['right'].set_position(('axes', pos[snd]))
                    ss.plot(xs[snd],s,color=color_list[snd])
                    ss.set_ylim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                    ss.set_ylabel(ylab[snd],color=color_list[snd],fontsize=14)
                    ss.tick_params('y',colors=color_list[snd],labelsize=13)

            if twin_double:
                #if twin_double[0]==0:
                #    r = ax
                #    l = ax.twinx()
                #elif twin_double[0] == 1:
                #    r = ax.twinx()
                #    l = ax

                for ind,i in enumerate(ys[0]):
                    ax.plot(xs[0],ys[0][ind],color_list[0][ind],label=leg_list[0][ind])
                ax.set_ylim(minmax_tuple[0][0],minmax_tuple[0][1])
                ax.set_ylabel(ylab[0],fontsize=14)
                ax.legend(loc=2,fontsize=14)

                l = ax.twinx()
                for jnd, j in enumerate(ys[1]):
                    l.plot(xs[1],ys[1][jnd],color_list[1][jnd],label=leg_list[1][jnd],ls='--')
                l.set_ylim(minmax_tuple[1][0],minmax_tuple[1][1])
                l.set_ylabel(ylab[1],fontsize=14)
                l.legend(loc=1,fontsize=14)
                l.tick_params('both',labelsize=13)


        if shade:
            if type(shade)!=dict:
                for sh in range(len(shade)):
                    a = xs.loc[shade[sh][0]]
                    b = xs.loc[shade[sh][1]]
                    ax.axvspan(a,b,color='lightgray',zorder=-10)
            if type(shade)==dict:
                colors_sh = ['lightcyan','palegoldenrod','gainsboro']
                for land,lab in enumerate(shade.keys()):
                    for sh in range(len(shade[lab])):
                        a = xs[shade[lab][sh][0]]
                        b = xs[shade[lab][sh][1]]
                        ax.axvspan(a,b,color=colors_sh[land],zorder=-10,label='_'*sh+lab)
                ax.legend(loc=0,fontsize=14)
        # regardles # lines this is the same
        #plt.title(title.capitalize(),fontsize=16)
        ax.set_xlabel('Time [hours]',fontsize=14)
        ax.set_xlim(xs[0][0],xs[0][-1])
        locator = md.AutoDateLocator(minticks=5, maxticks=10)
        formatter = md.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.tick_params('both',labelsize=13)
        ax.xaxis.get_offset_text().set_size(13)

        plt.savefig(self.dir_save+'%s.pdf'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')

    def plot_lines2(self,xs,ys,minmax_tuple,color_list,ylab,save,twin=False,\
                   twin_double=False,leg_list=False,shade=False):
        '''
        plot one or more lines
        twin         : If True twinx
        twin_double  : If True more than one line on either side, type tuple
        shade        : False or tuple with dates to shade
        xs           : abscissa as array
        ys           : ordinate, list or tuple (multiple lines either side as list of list)
        color_list   : string or list or tuple (multiple lines either side as list of list)
        leg_list     : string, list or tuple (multiple lines either side as list of list)
        ylab         : ordinate label (ylabel) string or list
        save         : name to save fig
        minmax_tuple : tuple or list with min an max values of plot
        '''
        plt.figure(figsize=(11,3))
        ax=plt.gca()

        if not twin:
            # multiple lines same variability
            if (len(ys)>1) and (leg_list):
                for ind,i in enumerate(ys):
                    if ind ==0:
                        ax.plot(xs[1],i, color=color_list[ind],label=leg_list[ind])
                    else:
                        ax.plot(xs[0],i,color=color_list[ind],label=leg_list[ind])
                plt.legend(loc=0,fontsize=14,ncol=5)
                #plt.yscale('symlog', linthresh=500)
            if (len(ys)>1) and not (leg_list):
                for ind,i in enumerate(ys):
                    if ind ==0:
                        ax.plot(xs[1],i, color=color_list[ind])
                    else:
                        ax.plot(xs[0],i,color=color_list[ind])
                #plt.yscale('symlog', linthresh=500)
            # one line
            if len(ys)==1:
                ax.plot(xs,ys[0],color=color_list)

            ax.set_ylim(minmax_tuple[0],minmax_tuple[1])
            ax.set_ylabel(ylab,fontsize=14)
        else:
            if not twin_double:
                for snd,s in enumerate(ys):
                    if snd==0:
                        ss = ax
                    elif snd==1:
                        ss = ax.twinx()
                    else:
                        ss = ax.twinx()
                        pos = [0,0,1.13,1.26,1.39,1.52]
                        ss.spines['right'].set_position(('axes', pos[snd]))
                    ss.plot(xs[snd],s,color=color_list[snd])
                    ss.set_ylim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                    ss.set_ylabel(ylab[snd],color=color_list[snd],fontsize=14)
                    ss.tick_params('y',colors=color_list[snd],labelsize=13)

            if twin_double:
                #if twin_double[0]==0:
                #    r = ax
                #    l = ax.twinx()
                #elif twin_double[0] == 1:
                #    r = ax.twinx()
                #    l = ax

                for ind,i in enumerate(ys[0]):
                    ax.plot(xs[0],ys[0][ind],color_list[0][ind],label=leg_list[0][ind])
                ax.set_ylim(minmax_tuple[0][0],minmax_tuple[0][1])
                ax.set_ylabel(ylab[0],fontsize=14)
                ax.legend(loc=2,fontsize=14)

                l = ax.twinx()
                for jnd, j in enumerate(ys[1]):
                    l.plot(xs[1],ys[1][jnd],color_list[1][jnd],label=leg_list[1][jnd],ls='--')
                l.set_ylim(minmax_tuple[1][0],minmax_tuple[1][1])
                l.set_ylabel(ylab[1],fontsize=14)
                l.legend(loc=1,fontsize=14)
                l.tick_params('both',labelsize=13)


        if shade:
            if type(shade)!=dict:
                for sh in range(len(shade)):
                    a = xs.loc[shade[sh][0]]
                    b = xs.loc[shade[sh][1]]
                    ax.axvspan(a,b,color='lightgray',zorder=-10)
            if type(shade)==dict:
                colors_sh = ['lightcyan','palegoldenrod','gainsboro']
                for land,lab in enumerate(shade.keys()):
                    for sh in range(len(shade[lab])):
                        a = xs[shade[lab][sh][0]]
                        b = xs[shade[lab][sh][1]]
                        ax.axvspan(a,b,color=colors_sh[land],zorder=-10,label='_'*sh+lab)
                ax.legend(loc=0,fontsize=14)
        # regardles # lines this is the same
        #plt.title(title.capitalize(),fontsize=16)
        ax.set_xlabel('Time [hours]',fontsize=14)
        ax.set_xlim(xs[0][0],xs[0][-1])
        locator = md.AutoDateLocator(minticks=5, maxticks=10)
        formatter = md.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.tick_params('both',labelsize=13)
        ax.xaxis.get_offset_text().set_size(13)

        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')


    def plot_lines3(self,xs,ys,minmax_tuple,color_list,ylab,save,twin=False,\
                   twin_double=False,leg_list=False,shade=False):
        '''
        QUICK FIX FOR ONE PLOT
        plot one or more lines
        twin         : If True twinx
        twin_double  : If True more than one line on either side, type tuple
        shade        : False or tuple with dates to shade
        xs           : abscissa as array
        ys           : ordinate, list or tuple (multiple lines either side as list of list)
        color_list   : string or list or tuple (multiple lines either side as list of list)
        leg_list     : string, list or tuple (multiple lines either side as list of list)
        ylab         : ordinate label (ylabel) string or list
        save         : name to save fig
        minmax_tuple : tuple or list with min an max values of plot
        '''
        plt.figure(figsize=(11,3))
        ax=plt.gca()

        if not twin:
            # multiple lines same variability
            if (len(ys)>1) and (leg_list):
                for ind,i in enumerate(ys):
                    if ind ==0:
                        ax.plot(xs[1],i, color=color_list[ind],label=leg_list[ind])
                    else:
                        ax.plot(xs[0],i,color=color_list[ind],label=leg_list[ind])
                plt.legend(loc=0,fontsize=14,ncol=5)
                #plt.yscale('symlog', linthresh=500)
            if (len(ys)>1) and not (leg_list):
                for ind,i in enumerate(ys):
                    if ind ==0:
                        ax.plot(xs[1],i, color=color_list[ind])
                    else:
                        ax.plot(xs[0],i,color=color_list[ind])
                #plt.yscale('symlog', linthresh=500)
            # one line
            if len(ys)==1:
                ax.plot(xs,ys[0],color=color_list)

            ax.set_ylim(minmax_tuple[0],minmax_tuple[1])
            ax.set_ylabel(ylab,fontsize=14)
        else:
            if not twin_double:
                for snd, s_group in enumerate(ys):
                    if snd == 0:
                        ss = ax
                    elif snd == 1:
                        ss = ax.twinx()
                    else:
                        ss = ax.twinx()
                        pos = [0, 0, 1.13, 1.26, 1.39, 1.52]
                        ss.spines['right'].set_position(('axes', pos[snd]))

                    # loop over each line in the group
                    if isinstance(s_group, list):
                        for idx, single_y in enumerate(s_group):
                            col = color_list[snd][idx] if isinstance(color_list[snd], (list, tuple)) else color_list[
                                snd]
                            lab = leg_list[snd][idx] if leg_list and isinstance(leg_list[snd], (list, tuple)) else None
                            ss.plot(xs[snd], single_y, color=col, label=lab)
                    else:
                        ss.plot(xs[snd], s_group, color=color_list[snd], label=leg_list[snd] if leg_list else None)

                    ss.set_ylim(minmax_tuple[snd][0], minmax_tuple[snd][1])
                    ss.set_ylabel(ylab[snd], color=color_list[snd][0] if isinstance(color_list[snd], (list, tuple)) else
                    color_list[snd], fontsize=14)
                    ss.tick_params('y', colors=color_list[snd][0] if isinstance(color_list[snd], (list, tuple)) else
                    color_list[snd], labelsize=13)

                    if leg_list:
                        ss.legend(loc=2 if snd == 0 else 1, fontsize=14)

            if twin_double:
                #if twin_double[0]==0:
                #    r = ax
                #    l = ax.twinx()
                #elif twin_double[0] == 1:
                #    r = ax.twinx()
                #    l = ax

                for ind,i in enumerate(ys[0]):
                    ax.plot(xs[0],ys[0][ind],color_list[0][ind],label=leg_list[0][ind])
                ax.set_ylim(minmax_tuple[0][0],minmax_tuple[0][1])
                ax.set_ylabel(ylab[0],fontsize=14)
                ax.legend(loc=2,fontsize=14)

                l = ax.twinx()
                for jnd, j in enumerate(ys[1]):
                    l.plot(xs[1],ys[1][jnd],color_list[1][jnd],label=leg_list[1][jnd],ls='--')
                l.set_ylim(minmax_tuple[1][0],minmax_tuple[1][1])
                l.set_ylabel(ylab[1],fontsize=14)
                l.legend(loc=1,fontsize=14)
                l.tick_params('both',labelsize=13)


        if shade:
            if type(shade)!=dict:
                for sh in range(len(shade)):
                    a = xs.loc[shade[sh][0]]
                    b = xs.loc[shade[sh][1]]
                    ax.axvspan(a,b,color='lightgray',zorder=-10)
            if type(shade)==dict:
                colors_sh = ['lightcyan','palegoldenrod','gainsboro']
                for land,lab in enumerate(shade.keys()):
                    for sh in range(len(shade[lab])):
                        a = xs[shade[lab][sh][0]]
                        b = xs[shade[lab][sh][1]]
                        ax.axvspan(a,b,color=colors_sh[land],zorder=-10,label='_'*sh+lab)
                ax.legend(loc=0,fontsize=14)
        # regardles # lines this is the same
        #plt.title(title.capitalize(),fontsize=16)
        ax.set_xlabel('Time [hours]',fontsize=14)
        ax.set_xlim(xs[0][0],xs[0][-1])
        locator = md.AutoDateLocator(minticks=5, maxticks=10)
        formatter = md.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.tick_params('both',labelsize=13)
        ax.xaxis.get_offset_text().set_size(13)

        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')




    def plot_scatter(self,xs,ys,minmax_tuple,color_list,xlab,save,twin=False,\
                   twin_double=False,leg_list=False,log=False,zorder=False,marker=False,\
                   sh2='o',ps=10):
        '''
        plot one or more scatter
        twin         : If True twiny
        twin_double  : If True more than one line on either side, type tuple
        xs           : abscissa list or tuple (multiple lines either side as list of list)
        ys           : ordinate, as array
        color_list   : string or list or tuple (multiple lines either side as list of list)
        leg_list     : string, list or tuple (multiple lines either side as list of list)
        xlab         : ordinate label (xlabel) string or list
        save         : name to save fig
        minmax_tuple : tuple or list with min an max values of plot
        log          : log scale False or index as list (0 for ax, 1 for ax.twin)
                       if only one x is log e.g [0,False]
        zorder       : change order of different layers
        marker       : facecolor filled (give color name) or unfilled (None)
        sh2          : change marker type
        ps           : change point size, default = 10, if more than one "line"
                       ps is a list
        '''

        plt.figure(figsize=(11,3))
        ax=plt.gca()
        if marker:
            sh=marker


        if not twin:
            # multiple lines same variability
            if len(ys)>1:
                for ind,i in enumerate(ys):
                    ax.scatter(xs,i,color=color_list[ind],label=leg_list[ind],\
                               zorder=zorder[ind],s=ps[ind],facecolors=sh[ind])
                if not marker:
                    plt.legend(loc=0,fontsize=14)
            # one line
            elif len(xs)==1:
                ax.scatter(xs,ys[0],color=color_list,s=ps,marker=sh2)

            if log:
                ax.set_yscale('log')
            ax.set_ylim(minmax_tuple[0],minmax_tuple[1])
            ax.set_ylabel(xlab,fontsize=14)
        else:
            if not twin_double:
                for snd,s in enumerate([ax,ax.twinx()]):
                    s.scatter(xs,ys[snd],color=color_list[snd],s=ps[snd])
                    s.zorder = zorder[snd]
                    if log:
                        if log[snd]==snd:
                            s.set_yscale('log')
                    s.set_ylim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                    s.set_ylabel(xlab[snd],color=color_list[snd],fontsize=14)
                    s.tick_params('y',colors=color_list[snd],labelsize=13)
                    s.set_facecolor("none")
            else:
                for ind,i in enumerate(ys[0]):
                    ax.scatter(xs,ys[0][ind],color=color_list[0][ind],\
                               label=leg_list[0][ind],s=ps[0][ind],facecolors=sh[0][ind])
                    ax.set_zorder(zorder[0])
                if log[0]==0:
                    ax.set_yscale('log')
                ax.set_ylim(minmax_tuple[0][0],minmax_tuple[0][1])
                ax.set_ylabel(xlab[0],color=color_list[0][0],fontsize=14)
                if not marker:
                    ax.legend(loc=4,fontsize=14)
                ax.tick_params('y',colors=color_list[0][0],labelsize=13)
                ax.set_facecolor("none")

                l = ax.twinx()
                for jnd, j in enumerate(ys[1]):
                    l.scatter(xs,ys[1][jnd],color=color_list[1][jnd],\
                              label=leg_list[1][jnd],facecolors=sh[1][jnd],s=ps[1][jnd])
                    l.set_zorder(zorder[1])
                if log[1]==1:
                    l.set_yscale('log')
                l.set_ylim(minmax_tuple[1][0],minmax_tuple[1][1])
                l.set_ylabel(xlab[1],color=color_list[1][0],fontsize=14)
                if not marker:
                    l.legend(loc=1,fontsize=14)
                l.tick_params('both',labelsize=13)
                l.tick_params('y',colors=color_list[1][0],labelsize=13)
                l.set_facecolor("none")

        ax.set_xlabel('Time [hours]',fontsize=14)
        ax.set_xlim(xs[0],xs[-1])
        locator = md.AutoDateLocator(minticks=5, maxticks=10)
        formatter = md.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.tick_params('both',labelsize=13)
        ax.xaxis.get_offset_text().set_size(13)

        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')


    def plot_lines_vertical(self,ys,xs,minmax_tuple,color_list,xlab,save,twin=False,\
                   twin_double=False,leg_list=False,log=False,zorder=False):
        '''
        plot one or more lines
        twin         : If True twiny
        twin_double  : If True more than one line on either side, type tuple
        xs           : abscissa list or tuple (multiple lines either side as list of list)
        ys           : ordinate, as array
        color_list   : string or list or tuple (multiple lines either side as list of list)
        leg_list     : string, list or tuple (multiple lines either side as list of list)
        xlab         : ordinate label (xlabel) string or list
        save         : name to save fig
        minmax_tuple : tuple or list with min an max values of plot
        log          : log scale False or index as list (0 for ax, 1 for ax.twin)
                       if only one x is log e.g [0,False]
        '''

        plt.figure(figsize=(3,7))
        ax=plt.gca()

        if not twin:
            # multiple lines same variability
            if len(xs)>1:
                for ind,i in enumerate(xs):
                    if leg_list:
                        ax.plot(i,ys,color=color_list[ind],label=leg_list[ind],zorder=zorder[ind])
                    else:
                        ax.plot(i,ys,color=color_list[ind],zorder=zorder[ind])
                if leg_list:
                    plt.legend(loc=0,fontsize=14)
            # one line
            elif len(xs)==1:
                ax.plot(xs[0],ys,color=color_list)

            if log:
                ax.set_xscale('log')
            ax.set_xlim(minmax_tuple[0],minmax_tuple[1])
            ax.set_xlabel(xlab,fontsize=14)
        else:
            if not twin_double:
                for snd,s in enumerate(xs):
                    if snd==0:
                        ss = ax
                    elif snd==1:
                        ss = ax.twiny()
                    else:
                        ss = ax.twiny()
                        pos = [0,0,1.13,1.26,1.39,1.52]
                        ss.spines['top'].set_position(('axes', pos[snd]))

                    ss.plot(s,ys,color=color_list[snd])
                    #if log:
                    if log[snd]==snd:
                        ss.set_xscale('log')

                    ss.set_xlim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                    ss.set_xlabel(xlab[snd],color=color_list[snd],fontsize=14)
                    ss.tick_params('x',colors=color_list[snd],labelsize=13)
                    ss.set_facecolor("none")
            else:
                for snd,s in enumerate(xs):
                    if snd==0:
                        ss = ax
                    elif snd==1:
                        ss = ax.twiny()
                    else:
                        ss = ax.twiny()
                        pos = [0,0,1.13,1.26,1.39,1.52]
                        ss.spines['top'].set_position(('axes', pos[snd]))

                    if len(leg_list[snd])>1:
                        for ind,i in enumerate(xs[snd]):
                            ss.plot(xs[snd][ind],ys,color_list[snd][ind],label=leg_list[snd][ind])
                            ss.set_zorder(zorder[snd])
                            if log[snd]==0:
                                ss.set_xscale('log')

                            ss.set_xlim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                            ss.set_xlabel(xlab[snd],color=color_list[snd][1],fontsize=14)
                            ss.legend(loc=2,fontsize=14)
                            ss.tick_params('x',colors=color_list[snd][1],labelsize=13)
                            ss.set_facecolor("none")
                    else:
                        ss.plot(xs[snd],ys,color_list[snd][0],label=leg_list[snd][0])
                        ss.set_zorder(zorder[snd])
                        if log[snd]==0:
                            ss.set_xscale('log')

                        ss.set_xlim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                        ss.set_xlabel(xlab[snd],color=color_list[snd][0],fontsize=14)
                        ss.tick_params('x',colors=color_list[snd][0],labelsize=13)
                        ss.set_facecolor("none")

        ax.set_ylabel('Altitude [km]',fontsize=14)
        ax.set_ylim(0,12)
        ax.tick_params('both',labelsize=13)

        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')

    def plot_lines_vertical_error(self,ys,xs,minmax_tuple,color_list,xlab,save,twin=False,\
                   twin_double=False,leg_list=False,log=False,zorder=False):
        '''
        plot one or more lines
        twin         : If True twiny
        twin_double  : If True more than one line on either side, type tuple
        xs           : abscissa list or tuple (multiple lines either side as list of list)
        ys           : ordinate, as array
        color_list   : string or list or tuple (multiple lines either side as list of list)
        leg_list     : string, list or tuple (multiple lines either side as list of list)
        xlab         : ordinate label (xlabel) string or list
        save         : name to save fig
        minmax_tuple : tuple or list with min an max values of plot
        log          : log scale False or index as list (0 for ax, 1 for ax.twin)
                       if only one x is log e.g [0,False]
        '''

        plt.figure(figsize=(3,7))
        ax=plt.gca()

        if not twin:
            # multiple lines same variability
            if len(xs)>1:
                for ind,i in enumerate(xs):
                    iin    = np.nanpercentile(i,50,axis=0)
                    error = [np.nanpercentile(i,50,axis=0)-np.nanpercentile(i,10,axis=0),
                             np.nanpercentile(i,90,axis=0)-np.nanpercentile(i,50,axis=0)]
                    if leg_list:
                        ax.errorbar(iin,ys,xerr=error,
                        color=color_list[ind],label=leg_list[ind],zorder=zorder[ind])
                    else:
                        ax.errorbar(iin,ys,xerr=error,
                        color=color_list[ind],zorder=zorder[ind])
                if leg_list:
                    plt.legend(loc=0,fontsize=14)
            # one line
            elif len(xs)==1:
                iin    = np.nanpercentile(xs[0],50,axis=0)
                error = [np.nanpercentile(xs[0],50,axis=0)-np.nanpercentile(xs[0],10,axis=0),
                         np.nanpercentile(xs[0],90,axis=0)-np.nanpercentile(xs[0],50,axis=0)]
                ax.errorbar(iin,ys,xerr=error,color=color_list)

            if log:
                ax.set_xscale('log')
            ax.set_xlim(minmax_tuple[0],minmax_tuple[1])
            ax.set_xlabel(xlab,fontsize=14)
        else:
            if not twin_double:
                for snd,s in enumerate(xs):
                    iin    = np.nanpercentile(s,50,axis=0)
                    error = [np.nanpercentile(s,50,axis=0)-np.nanpercentile(s,10,axis=0),
                             np.nanpercentile(s,90,axis=0)-np.nanpercentile(s,50,axis=0)]
                    if snd==0:
                        ss = ax
                    elif snd==1:
                        ss = ax.twiny()
                    else:
                        ss = ax.twiny()
                        pos = [0,0,1.13,1.26,1.39,1.52]
                        ss.spines['top'].set_position(('axes', pos[snd]))

                    ss.errorbar(iin,ys,xerr=error,color=color_list[snd])
                    if log:
                        if log[snd]==snd:
                            ss.set_xscale('log')
                    ss.set_xlim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                    ss.set_xlabel(xlab[snd],color=color_list[snd],fontsize=14)
                    ss.tick_params('x',colors=color_list[snd],labelsize=13)
                    ss.set_facecolor("none")
            else:
                for snd,s in enumerate(xs):
                    if snd==0:
                        ss = ax
                    elif snd==1:
                        ss = ax.twiny()
                    else:
                        ss = ax.twiny()
                        pos = [0,0,1.13,1.26,1.39,1.52]
                        ss.spines['top'].set_position(('axes', pos[snd]))

                    if len(leg_list[snd])>1:
                        for ind,i in enumerate(xs[snd]):
                            iin    = np.nanpercentile(s[ind],50,axis=0)
                            error = [np.nanpercentile(s[ind],50,axis=0)-np.nanpercentile(s[ind],10,axis=0),
                                     np.nanpercentile(s[ind],90,axis=0)-np.nanpercentile(s[ind],50,axis=0)]

                            ss.errorbar(iin,ys,xerr=error,color=color_list[snd][ind],label=leg_list[snd][ind])
                            ss.set_zorder(zorder[snd])
                            if log[snd]==0:
                                ss.set_xscale('log')
                            ss.set_xlim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                            ss.set_xlabel(xlab[snd],color=color_list[snd][1],fontsize=14)
                            ss.legend(loc=2,fontsize=14)
                            ss.tick_params('x',colors=color_list[snd][1],labelsize=13)
                            ss.set_facecolor("none")
                    else:
                        iin    = np.nanpercentile(s,50,axis=0)
                        error = [np.nanpercentile(s,50,axis=0)-np.nanpercentile(s,10,axis=0),
                                 np.nanpercentile(s,90,axis=0)-np.nanpercentile(s,50,axis=0)]

                        ss.errorbar(iin,ys,xerr=error,color=color_list[snd][0],label=leg_list[snd][0])
                        ss.set_zorder(zorder[snd])
                        if log[snd]==0:
                            ss.set_xscale('log')
                        ss.set_xlim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                        ss.set_xlabel(xlab[snd],color=color_list[snd][0],fontsize=14)
                        ss.tick_params('x',colors=color_list[snd][0],labelsize=13)
                        ss.set_facecolor("none")

        ax.set_ylabel('Altitude [km]',fontsize=14)
        ax.set_ylim(0,5)
        ax.tick_params('both',labelsize=13)

        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')



    def plot_scatter_vertical(self,ys,xs,minmax_tuple,color_list,xlab,save,twin=False,\
                   twin_double=False,leg_list=False,log=False,zorder=False,marker=False,\
                   sh2='o',ps=10):
        '''
        plot one or more scatter
        twin         : If True twiny
        twin_double  : If True more than one line on either side, type tuple
        xs           : abscissa list or tuple (multiple lines either side as list of list)
        ys           : ordinate, as array
        color_list   : string or list or tuple (multiple lines either side as list of list)
        leg_list     : string, list or tuple (multiple lines either side as list of list)
        xlab         : ordinate label (xlabel) string or list
        save         : name to save fig
        minmax_tuple : tuple or list with min an max values of plot
        log          : log scale False or index as list (0 for ax, 1 for ax.twin)
                       if only one x is log e.g [0,False]
        zorder       : change order of different layers
        marker       : facecolor filled (give color name) or unfilled (None)
        sh2          : change marker type
        ps           : change point size, default = 10, if more than one "line"
                       ps is a list
        '''

        plt.figure(figsize=(3,7))
        ax=plt.gca()
        if marker:
            sh=marker


        if not twin:
            # multiple lines same variability
            if len(xs)>1:
                for ind,i in enumerate(xs):
                    ax.scatter(i,ys,color=color_list[ind],label=leg_list[ind],\
                               zorder=zorder[ind],s=ps[ind],facecolors=sh[ind])
                if not marker:
                    plt.legend(loc=0,fontsize=14)
            # one line
            elif len(xs)==1:
                ax.scatter(xs[0],ys,color=color_list,s=ps,marker=sh2)

            if log:
                ax.set_xscale('log')
            ax.set_xlim(minmax_tuple[0],minmax_tuple[1])
            ax.set_xlabel(xlab,fontsize=14)
        else:
            if not twin_double:
                for snd,s in enumerate([ax,ax.twiny()]):
                    s.scatter(xs[snd],ys,color=color_list[snd],s=ps[snd])
                    s.zorder = zorder[snd]
                    if log:
                        if log[snd]==snd:
                            s.set_xscale('log')
                    s.set_xlim(minmax_tuple[snd][0],minmax_tuple[snd][1])
                    s.set_xlabel(xlab[snd],color=color_list[snd],fontsize=14)
                    s.tick_params('x',colors=color_list[snd],labelsize=13)
                    s.set_facecolor("none")
            else:
                for ind,i in enumerate(xs[0]):
                    ax.scatter(xs[0][ind],ys,color=color_list[0][ind],\
                               label=leg_list[0][ind],s=ps[0][ind],facecolors=sh[0][ind])
                    ax.set_zorder(zorder[0])
                if log[0]==0:
                    ax.set_xscale('log')
                ax.set_xlim(minmax_tuple[0][0],minmax_tuple[0][1])
                ax.set_xlabel(xlab[0],color=color_list[0][0],fontsize=14)
                if not marker:
                    ax.legend(loc=4,fontsize=14)
                ax.tick_params('x',colors=color_list[0][0],labelsize=13)
                ax.set_facecolor("none")

                l = ax.twiny()
                for jnd, j in enumerate(xs[1]):
                    l.scatter(xs[1][jnd],ys,color=color_list[1][jnd],\
                              label=leg_list[1][jnd],facecolors=sh[1][jnd],s=ps[1][jnd])
                    l.set_zorder(zorder[1])
                if log[1]==1:
                    l.set_xscale('log')
                l.set_xlim(minmax_tuple[1][0],minmax_tuple[1][1])
                l.set_xlabel(xlab[1],color=color_list[1][0],fontsize=14)
                if not marker:
                    l.legend(loc=1,fontsize=14)
                l.tick_params('both',labelsize=13)
                l.tick_params('x',colors=color_list[1][0],labelsize=13)
                l.set_facecolor("none")

        ax.set_ylabel('Altitude [km]',fontsize=14)
        ax.set_ylim(0,6)
        ax.tick_params('both',labelsize=13)

        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')



    def one_plot_imshow(self,array,lat,lon,figsize,name_list,vmin,vmax,units,color_map,\
                        var,save,norm=None,clab=False,line=False,sc_sn=False,border=False):
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

        map_proj = ccrs.PlateCarree()

        fig,ax = plt.subplots(1,1,figsize=figsize,
                   subplot_kw={'projection': map_proj})

        ax.set_aspect('auto')

        cs = ax.imshow(array,extent=[lon.min(), lon.max(), lat.min(), lat.max()]\
                    ,vmin=vmin,vmax=vmax,cmap=color_map,norm=norm)
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
            cbar.set_ticklabels(['Clear','Liquid','SLW','Mixed','Ice','Unkown'])
        if clab==2:
            cbar = fig.colorbar(cs,aspect=15,orientation='vertical',ax=ax,ticks=range(0,11))
            cbar.set_ticks(range(0,11))
            cbar.set_ticklabels(['Clear','Ci','Cs','DC','Ac','As','Ns','Cu','Sc','St','Unkown'])
        cbar.set_label('%s %s'%(var,units),fontsize=15)
        cbar.ax.tick_params(labelsize=14)

        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')


    def one_plot_imshow_capri_vs_socra(self,array,lat,lon,figsize,name_list,vmin,vmax,units,color_map,\
                        var,norm=None,clab=False,line=False,sc_sn=False):
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
        norm      : set center of colorbar at specific #
        clab      : create xticks...predefined in function (it can change to more general)
        line      : plot line, type dict line,color
        sc_sn     : add location of dropsonde as df
        '''

        map_proj = ccrs.PlateCarree()

        fig,ax = plt.subplots(1,1,figsize=figsize,
                   subplot_kw={'projection': map_proj})

        ax.set_aspect('auto',adjustable=None)

        cs = ax.imshow(array,extent=[lon.min(), lon.max(), lat.min(), lat.max()]\
                    ,vmin=vmin,vmax=vmax,cmap=color_map,norm=norm)

        ax.set_title(name_list,fontsize=15)
        ax.coastlines(linewidth=0.8,color='white')
        ax.add_feature(cfeature.BORDERS,linewidth=0.8)

        gl = ax.gridlines(crs=map_proj, draw_labels=True,
                          linestyle='--',color='white')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines       = False
        gl.ylines       = False
        gl.xlabel_style = {'size':11}
        gl.ylabel_style = {'size':11}

        if not clab:
            #cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax)
            cbar = plt.colorbar(cs,fraction=0.046,pad=0.07,orientation='horizontal',ax=ax)
        if clab==1:
            cbar = fig.colorbar(cs,aspect=15,orientation='vertical',ax=ax,ticks=range(0,6))
            cbar.set_ticks(range(0,11))
            cbar.set_ticklabels(['Clear','Liquid','SLW','Mixed','Ice','Unkown'])
        if clab==2:
            cbar = fig.colorbar(cs,fraction=0.046,pad=0.07,orientation='horizontal',ax=ax,ticks=range(0,11))
            cbar.set_ticks(range(0,11))
            cbar.set_ticklabels(['Clr','Ci','Cs','DC','Ac','As','Ns','Cu','Sc','St','Unk'])
        cbar.set_label('%s %s'%(var,units),fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        def resize_colobar(event):
            plt.draw()

            posn = ax.get_position()
            cbar_ax.set_position([posn.x0 + posn.width + 0.01,posn.y0,
                                  0.06, posn.height])

        fig.canvas.mpl_connect('resize_event', resize_colobar)

        return fig,ax


    def one_plot_cross(self,array,time,alt2,figsize,name_list,vmin,\
                            vmax,nround,units,color_map,var,save,norm=None,\
                            contour=False,xtype='time'):
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
        '''
        array2 = array.T.values
        time2  = time.T.values
        altt2  = alt2.values

        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.set_aspect('auto')

        if not norm:
            bounds =np.round( np.linspace(vmin,vmax,30, endpoint=True),nround)
        else:
            bounds = np.logspace(np.log10(vmin),np.log10(vmax),30)

        cs = ax.contourf(time2,altt2,array2,levels=bounds,cmap=color_map,norm=norm)

        ax.set_title(name_list,fontsize=15)

        if contour:
            styles = ['--','-']
            colors = ['yellow','k']
            for ct in range(len(contour)):
                plt.rcParams['contour.negative_linestyle'] = '--'
                cts = ax.contour(time2,altt2,contour[ct][0],levels=contour[ct][1],\
                           colors=colors[ct],ls=styles[ct],lw=0.3)
                # make labels only for black line
                if ct==1:
                    ax.clabel(cts,fontsize=9,inline=True,fmt='%1.0f',colors='k')
        else:
            pass

        if not norm:
            cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax)
        else:
            l_f = mticker.LogLocator(base=10)
            cbar = plt.colorbar(cs,ticks=l_f,aspect=15,orientation='vertical',ax=ax)

        cbar.set_label('%s %s'%(var,units),fontsize=14)
        cbar.ax.tick_params(labelsize=11)
        ax.set_ylabel('Altitude [km]',fontsize=14)
        ax.set_ylim(0,4)
        if xtype=='time':
            ax.set_xlabel('Time-UTC [seconds]',fontsize=14)
            locator = md.AutoDateLocator(minticks=3, maxticks=7)
            formatter = md.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.get_offset_text().set_size(13)
        else:
            ax.set_xlabel(xtype,fontsize=14)

        ax.tick_params('both',labelsize=13)

        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')

    def one_plot_radar_socra2(self,array,time,alt,figsize,name_list,vmin,\
                            vmax,nround,units,color_map,var,save,norm=None,\
                            flight=False,alpha=False,contour=False,ticks1=None,\
                            tickslb=None):
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
        alpha     : shade colors False, True
        contour   : False or tuple [values,levels]
        flight    : Flase of tuple [values, variable]
        '''

        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.set_aspect('auto')

        if not norm or norm!=mc.LogNorm():

            if norm=='Limit11':
                bounds = np.linspace(vmin,vmax,12)
                norm = mc.BoundaryNorm(bounds,color_map.N, extend='neither')
            else:
                bounds =np.round( np.linspace(vmin,vmax,30, endpoint=True),nround)
        else:
            bounds = np.logspace(np.log10(vmin),np.log10(vmax),30)

        if alpha:
            cs = ax.contourf(time,alt,array,levels=bounds,cmap=color_map,\
                             norm=norm,alpha=0.5)
        else:
            cs = ax.contourf(time,alt,array,levels=bounds,cmap=color_map,norm=norm)

        if flight:
            if flight[0] =='GGALT':
                falt = flight[1]/1000
                ax.plot(time[:,0],falt,'k',lw=1.5)
            if flight[0]=='Phase':
                ll = multicolored_lines(ax,flight[1][0],flight[1][1],flight[1][2],\
                                        slw_line)
                #ll = ax.scatter(flight[1][0],flight[1][1],c=flight[1][2],\
                #                cmap=plt.cm.get_cmap('Spectral',20))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("left", size="5%", pad=0.5)
                cbar2 = fig.colorbar(ll,cax=cax,\
                                    ticks=[0,1,2,3])
                cbar2.set_ticks([0,1,2,3])
                cbar2.set_ticklabels(['None','Liquid','Mixed','Ice'])
                cbar2.ax.yaxis.set_ticks_position('left')
                cbar2.ax.yaxis.set_label_position('left')
                #cbar2 = plt.colorbar(ll,aspect=15, orientation='vertical',ax=ax,\
                #                    ticks=[10,20,30,40,50,60,70,80,90,100,110])
                #cbar2.set_ticks([10,20,30,40,50,60,70,80,90,100,110])
                #cbar2.set_ticklabels(['Liq','Liq 2DC','Liq 2DS','SLW',
                #                     'Mixed','Mixed 2DC','Mixed 2DS',
                #                     'Ice','Ice 2DC','Ice 2DS','NaN'])

                cbar2.set_label('%s'%(flight[0]),fontsize=14)
                cbar2.ax.tick_params(labelsize=11)
            if flight[0]=='Scatter':
                yyy = np.array([2.5]*len(flight[1][0]))
                ax.scatter(flight[1][0][flight[1][2]==1],yyy[flight[1][2]==1],\
                           label='Liquid',color='g',s=2)
                ax.scatter(flight[1][0][flight[1][2]==2],yyy[flight[1][2]==2]+0.2,\
                           label='Mix',color='r',s=2)
                ax.scatter(flight[1][0][flight[1][2]==3],yyy[flight[1][2]==3]+0.4,\
                           label='Snow',color='purple',s=2)
                ax.legend(loc=0,ncol=3,fontsize=14)
                ax.plot(flight[1][0],flight[1][1],'k',lw=1.5)

        else:
            pass

        if contour:
            ax.contourf(time,alt,contour[0],levels=contour[1],cmap='twilight')
        else:
            pass

        ax.set_title(name_list,fontsize=15)

        if not norm or norm!=mc.LogNorm():
            #fmt = lambda x, pos: '{:.0f}'.format(x)
            cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax,\
            ticks=ticks1)
            if tickslb:
                cbar.ax.set_yticklabels(tickslb[0])

        else:
            l_f = mticker.LogLocator(base=10)
            cbar = plt.colorbar(cs,ticks=l_f,aspect=15,orientation='vertical',ax=ax)

        cbar.set_label('%s %s'%(var,units),fontsize=14)
        cbar.ax.tick_params(labelsize=11)

        ax.set_xlabel('Time-UTC [seconds]',fontsize=14)
        ax.set_ylabel('Altitude [km]',fontsize=14)
        ax.set_ylim(0,3)
        locator = md.AutoDateLocator(minticks=3, maxticks=7)
        formatter = md.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.tick_params('both',labelsize=13)
        ax.xaxis.get_offset_text().set_size(13)

        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')

    def one_plot_radar_capri(self,array,time,alt2,figsize,name_list,vmin,vmax,\
                             nround,units,color_map,var,save,norm=None,segment=False,\
                             contour=False,ticks1=None):
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
        '''

        if segment:
            array2 = array.loc[segment[0]:segment[1]].T.values
            time2  = time.loc[segment[0]:segment[1]].values.T
            altt2  = alt2.loc[segment[0]:segment[1]].values
        else:
            array2 = array.T.values
            time2  = time.T#.values
            altt2  = alt2.T#.values

        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.set_aspect('auto')

        if norm!=mc.LogNorm():
            bounds =np.round( np.linspace(vmin,vmax,30,endpoint=True),nround)
        else:
            bounds = np.logspace(np.log10(vmin),np.log10(vmax),30)

        cs = ax.contourf(time2,altt2,array2,levels=bounds,cmap=color_map,norm=norm)
        #cs = ax.pcolormesh(time2,altt2,array2,cmap=color_map,norm=norm)
        ax.set_title(name_list,fontsize=15)

        if contour:
            styles = ['-','--']
            colors = ['k','lightgray']
            for ct in range(len(contour)):
                plt.rcParams['contour.negative_linestyle'] = 'solid'
                cts = ax.contour(time2,altt2,contour[ct][0],levels=contour[ct][1],\
                           colors=colors[ct],ls=styles[ct],lw=0.3)
                ax.clabel(cts,fontsize=9,inline=True,fmt='%1.0f',colors='k')
        else:
            pass

        if norm!=mc.LogNorm():
            #cbar = plt.colorbar(cs,aspect=15,pad=0.15,orientation='horizontal',\
            #ax=ax,ticks=ticks1)
            cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax,\
            ticks=ticks1)
            #cbar.set_ticklabels(['Liq','Virga','SLW','Mixed','Ice/Mix','Ice'])
        else:
            l_f = mticker.LogLocator(base=10)
            cbar = plt.colorbar(cs,ticks=l_f,aspect=15,orientation='vertical',ax=ax)

        cbar.set_label('%s %s'%(var,units),fontsize=14)
        cbar.ax.tick_params(labelsize=13)
        ax.set_xlabel('Time [minutes]',fontsize=14)
        ax.set_ylabel('Altitude [km]',fontsize=14)
        ax.set_ylim(0,3)

        locator = md.AutoDateLocator(minticks=5, maxticks=10)
        formatter = md.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.tick_params('both',labelsize=13)
        ax.xaxis.get_offset_text().set_size(13)

        def resize_colobar(event):
            plt.draw()

            posn = ax.get_position()
            cbar_ax.set_position([posn.x0 + posn.width + 0.01,posn.y0,
                                  0.06, posn.height])

        fig.canvas.mpl_connect('resize_event', resize_colobar)


        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=300,bbox_inches='tight')
        plt.close('all')


    def one_psd_2d(self,array,time,alt,figsize,name_list,vmin,vmax,units,color_map,var,save):
        '''
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
        '''

        array2 = array
        time2  = time.T.values

        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.set_aspect('auto')

        cs = ax.pcolormesh(time2,alt,array2,vmin=vmin,vmax=vmax,cmap=color_map,norm=mc.LogNorm())

        ax.set_title(name_list,fontsize=15)

        cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax)

        cbar.set_label('%s %s'%(var,units),fontsize=14)
        cbar.ax.tick_params(labelsize=11)
        ax.set_xlabel('Time-UTC [seconds]',fontsize=14)
        ax.set_ylabel(r'Diameter [$mm$]',fontsize=14)
        ax.set_yscale('log')
        locator = md.AutoDateLocator(minticks=3, maxticks=7)
        formatter = md.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.tick_params('both',labelsize=13)
        ax.xaxis.get_offset_text().set_size(13)


        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')


    def one_boxplot(self,array,labels,positions,figsize,minmax,varunits,save):
        '''
        boxplot

        array     : array
        labels    : xlables
        var       : name of the variable and units
        save      : name save fig
        norm      : set center of colorbar at specific #
        '''

        fig,ax = plt.subplots(figsize=(figsize))
        # whis = Q1 - 1.5IQR
        ax.boxplot(array,showfliers=False,positions=positions)

        ax.set_xlabel(r'Temperature [$^o$C]',fontsize=14)
        ax.set_ylabel(r'%s'%(varunits),fontsize=14)
        plt.ylim(minmax[0],minmax[1])
        plt.xticks(range(len(labels)),labels)
        plt.xlim(-0.7,len(labels)+0.3)

        ax.tick_params('x',labelsize=10)
        ax.tick_params('y',labelsize=13)
        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')


    def one_cfad(self,array,xbin,ybin,figsize,var,xvar,cbarvar,save,type,vmin,\
                 vmax,xlim,ylim,ccolor,norm,arrayp=False,yrevert=False,alpha=False,\
                 xticks=False):
        '''
        Plot the CFAD and CFTD
        array   = cfad or cftd arrays
        xbin    = xvalues
        ybin    = y values
        type    = for name of fig : which type of normalisation
        yrevert = if invert yaxis
        vmin,vmax = min, max values of color
        xlim,ylim = min, max values for axes
        var      = ylabel
        xvar     = xlabel
        cbarvar  = cbar label
        arrayp   = list with the 25,50 and 75 percentiles
        '''

        fig,ax = plt.subplots(1,1,figsize=(figsize))
        ax.set_aspect('auto')
        if yrevert:
            ax.invert_yaxis()
            ax.yaxis.set_minor_locator(MultipleLocator(5))
        else:
            ax.yaxis.set_minor_locator(MultipleLocator(5))

        if alpha:
            cs = ax.pcolormesh(xbin,ybin,array,cmap=ccolor,\
                              norm=norm,alpha=0.5,vmin=vmin,vmax=vmax)
        else:
            cs = ax.pcolormesh(xbin,ybin,array,cmap=ccolor,\
                              norm=norm,vmin=vmin,vmax=vmax)

        if arrayp:
            ax.plot(arrayp[0][0],ybin,ls='--',color='k',lw=1.5)
            ax.plot(arrayp[0][1],ybin,color='k',lw=1.5)
            ax.plot(arrayp[0][2],ybin,ls='--',color='k',lw=1.5)

        cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax,extend='max')

        cbar.set_label('%s'%(cbarvar),fontsize=14)
        cbar.ax.tick_params(labelsize=11)

        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.set_xlabel('%s'%(xvar),fontsize=14)#.capitalize()),fontsize=14)
        ax.set_ylabel(r'%s'%(var),fontsize=14)#.capitalize()),fontsize=14)
        plt.xlim(xlim[0],xlim[1])
        #ax.set_xscale('log')
        plt.ylim(ylim[0],ylim[1])
        ax.tick_params('both',labelsize=13)
        if xticks:
            ax.set_xticks(np.arange(1,11,1))
            ax.set_xticklabels(['Dr','R','IC',\
                                 'Agg','WS','VI',\
                                 'LDGr','HDGr','Hail','BD'])

        #plt.show()
        plt.savefig(self.dir_save+'%s_capricorn_%s.png'%(save,type),dpi=500,bbox_inches='tight')
        plt.close('all')

    def one_cfad_profile(self,array,ybin,figsize,var,xvar,save,type,xlim,ylim,yrevert=False):
        '''
        Plot the CFAD and CFTD
        array   = cfad or cftd arrays
        xbin    = xvalues
        ybin    = y values
        type    = for name of fig : which type of normalisation
        yrevert = if invert yaxis
        vmin,vmax = min, max values of color
        xlim,ylim = min, max values for axes
        '''

        fig,ax = plt.subplots(1,1,figsize=(figsize))
        ax.set_aspect('auto')
        if yrevert:
            ax.invert_yaxis()
            ax.yaxis.set_minor_locator(MultipleLocator(5))
        else:
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        ax.plot(array[0],ybin,ls='--',color='k',lw=1.5)
        ax.plot(array[1],ybin,color='k',lw=1.5)
        ax.plot(array[2],ybin,ls='--',color='k',lw=1.5)

        ax.set_xlabel('%s'%(xvar.capitalize()),fontsize=14)
        ax.set_ylabel(r'%s'%(var.capitalize()),fontsize=14)
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
        ax.tick_params('both',labelsize=13)
        #plt.show()
        plt.savefig(self.dir_save+'%s_capricorn_%s.png'%(save,type),dpi=500,bbox_inches='tight')
        plt.close('all')

    def one_cradar(self,array,xbin,ybin,figsize,cbarvar,units,save,vmin,\
                 vmax,ccolor,norm,title,xlab,ylab,ylims,xlims,clab=False,contour=False):
        '''
        Plot information from OPOL radar (c type)
        array   = cfad or cftd arrays
        xbin    = xvalues
        ybin    = y values
        vmin,vmax = min, max values of color
        xlim,ylim = min, max values for axes
        cbarvar  = cbar label
        '''

        map_proj = ccrs.PlateCarree()
        fig,ax = plt.subplots(1,1,figsize=(7,5),
                   subplot_kw={'projection': map_proj})
        ax.set_aspect('auto')

        cs = ax.pcolormesh(xbin,ybin,array,vmin=vmin,vmax=vmax,cmap=ccolor,norm=norm)
        c1=plt.Circle((float(xbin[150,150]),float(ybin[150,150])),
                       radius=1.5,fill=False,color='k',lw=1)
        ax.add_artist(c1)

        if contour:
            plt.rcParams['contour.negative_linestyle'] = 'solid'
            cts = ax.contour(contour[0].lon,contour[0].lat,contour[0],\
                       colors='k',ls='-',lw=0.3)
            ax.clabel(cts,fontsize=9,inline=True,fmt='%1.0f',colors='k')

        ax.set_title(title,fontsize=15)

        ax.coastlines(linewidth=0.8,color='white')
        gl = ax.gridlines(crs=map_proj, draw_labels=True,
                              linestyle='--',color='white')
        ax.add_feature(cfeature.BORDERS,linewidth=0.8)

        gl.top_labels = False
        gl.right_labels = False
        gl.xlines       = False
        gl.ylines       = False
        gl.xlabel_style = {'size':12}
        gl.ylabel_style = {'size':12}

        if not clab:
            cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax)
        if clab==1:
            cbar = fig.colorbar(cs,aspect=15,orientation='vertical',ax=ax,ticks=range(1,11))
            cbar.set_ticks(np.arange(1,11,1))
            cbar.set_ticklabels(['Dr','R','IC',\
                                 'Agg','WS','VI',\
                                 'LDGr','HDGr','Hail','BD'])

        cbar.set_label('%s %s'%(cbarvar,units),fontsize=14)
        cbar.ax.tick_params(labelsize=11)
        ax.set_xlabel(xlab,fontsize=14)
        ax.set_ylabel(ylab,fontsize=14)
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])

        ax.tick_params('both',labelsize=13)

        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')

    def one_cradar2(self,array,xbin,ybin,figsize,cbarvar,units,save,vmin,\
                 vmax,ccolor,norm,title,xlab,ylab,ylims,xlims,clab=False,contour=False):
        '''
        Plot information from OPOL radar (c type)
        array   = cfad or cftd arrays
        xbin    = xvalues
        ybin    = y values
        vmin,vmax = min, max values of color
        xlim,ylim = min, max values for axes
        cbarvar  = cbar label
        '''

        fig,ax = plt.subplots(1,1,figsize=(7,5))
        ax.set_aspect('auto')

        cs = ax.pcolormesh(xbin,ybin,array,vmin=vmin,vmax=vmax,cmap=ccolor,norm=norm)

        if contour:
            plt.rcParams['contour.negative_linestyle'] = 'solid'
            cts = ax.contour(contour[0].lon,contour[0].lat,contour[0],\
                       colors='k',ls='-',lw=0.3)
            ax.clabel(cts,fontsize=9,inline=True,fmt='%1.0f',colors='k')

        ax.set_title(title,fontsize=15)

        if not clab:
            cbar = plt.colorbar(cs,aspect=15,orientation='vertical',ax=ax)
        if clab==1:
            cbar = fig.colorbar(cs,aspect=15,orientation='vertical',ax=ax,ticks=range(1,11))
            cbar.set_ticks(np.arange(1,11,1))
            cbar.set_ticklabels(['Dr','R','IC',\
                                 'Agg','WS','VI',\
                                 'LDGr','HDGr','Hail','BD'])

        cbar.set_label('%s %s'%(cbarvar,units),fontsize=14)
        cbar.ax.tick_params(labelsize=11)
        ax.set_xlabel(xlab,fontsize=14)
        ax.set_ylabel(ylab,fontsize=14)
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])

        ax.tick_params('both',labelsize=13)

        #plt.show()
        plt.savefig(self.dir_save+'%s.png'%(save),dpi=500,bbox_inches='tight')
        plt.close('all')
