# -*- coding: utf-8 -*-
"""
WRF visualization

@modified: Estefania Montoya Feb 2023
"""
%load_ext autoreload
%autoreload 2

# import the python libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import glob
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
from datetime import datetime
import xarray as xr
from netCDF4 import Dataset

from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair, GeoBounds,
                 get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim,
                 ALL_TIMES,ll_to_xy)


from Socrates_make_plots import *

# specify the location of data
dir  = dir
plot = plot_socrates(dir+'Figures/')

al_file = glob.glob(f"{dir}/*wrfout_d04*")
al_file.sort()

wrflist = []
for i in al_file[3:6]: #socra transect was between these hours
    wrflist.append(Dataset(i))

mixc           = getvar(wrflist,'QCLOUD',timeidx=ALL_TIMES, method='cat')
mixv           = getvar(wrflist,'QVAPOR',timeidx=ALL_TIMES, method='cat')
mixr           = getvar(wrflist,'QRAIN',timeidx=ALL_TIMES, method='cat')
mixi           = getvar(wrflist,'QICE',timeidx=ALL_TIMES, method='cat')
z              = getvar(wrflist,'height_agl',timeidx=ALL_TIMES, method='cat')

mixtotal = mixc + mixr + mixi # here is still in kg/kg then limit is 1e-5 instead of 0.01

mixc         = mixc.where((mixtotal>1*10**-5) & (mixc>0)) #paper 1 0.005
mixr         = mixr.where((mixtotal>1*10**-5) & (mixr>0))
mixi         = mixi.where((mixtotal>1*10**-5) & (mixi>0))


# find the section aroung the socrates in-cloud legs

start_point = CoordPair(lat=-59.9, lon=141.3)
end_point = CoordPair(lat=-55.9, lon=141.3)
height = z[:,:,:,:].mean(axis=0).mean(axis=1).mean(axis=1)


# Compute the vertical cross-section interpolation.  Also, include the
# lat/lon points along the cross-section in the metadata by setting latlon
# to True.
mc_1 = vertcross(mixc, z, levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)
mr_1 = vertcross(mixr, z,levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)
mi_1 = vertcross(mixi, z, levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)

start_point = CoordPair(lat=-59.9, lon=141.4)
end_point = CoordPair(lat=-55.9, lon=141.4)

# Compute the vertical cross-section interpolation.  Also, include the
# lat/lon points along the cross-section in the metadata by setting latlon
# to True.
mc_2 = vertcross(mixc, z, levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)
mr_2 = vertcross(mixr, z,levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)
mi_2 = vertcross(mixi, z, levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)

start_point = CoordPair(lat=-59.9, lon=141.5)
end_point = CoordPair(lat=-55.9, lon=141.5)

# Compute the vertical cross-section interpolation.  Also, include the
# lat/lon points along the cross-section in the metadata by setting latlon
# to True.
mc_3 = vertcross(mixc, z, levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)
mr_3 = vertcross(mixr, z,levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)
mi_3 = vertcross(mixi, z, levels=height,wrfin=wrflist, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True,timeidx=ALL_TIMES)

# check  ranges in the area
mc = (mc_1.mean(axis=0) + mc_2.mean(axis=0) + mc_3.mean(axis=0)) /3
mr = (mr_1.mean(axis=0) + mr_2.mean(axis=0) + mr_3.mean(axis=0)) /3
mi = (mi_1.mean(axis=0) + mi_2.mean(axis=0) + mi_3.mean(axis=0)) /3

lwc = mc + mr
