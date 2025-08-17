# -*- coding: utf-8 -*-
"""
WRF save files for profiles

@created: Estefania Montoya Nov 2023
"""


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

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords,ll_to_xy,ALL_TIMES)

import sys
import re

ext = sys.argv[1]#'trans4'
ty  = sys.argv[2]#'Sc'
parent = parent


def create_xr_stats(xarrayd,dim,name):
    """ 1d xarray
    one dimension is time
    """
    d1,d2,d3 = np.shape(xarrayd)
    #save mean, p10,p25,p50,p75,p90,std
    empt  = np.empty([7,d1])

    empt[2:,:] = xarrayd.quantile([0.10,0.25,0.5,0.75,0.90],dim=dim).values
    empt[0,:]  = xarrayd.mean(dim=dim).values
    empt[1,:]  = xarrayd.std(dim=dim).values

    to_nc = xr.Dataset({'stat':(['id','Time'],empt),
                        'id':np.arange(0,7),
                        'Time':xarrayd.Time.values})
    to_nc = to_nc.assign_attrs(description=\
    "id 0:mean, 1:std, 2:percentil 10, 3: 2:percentil 25, 2:percentil 50, 2:percentil 75, 2:percentil 90")

    #save the nc_file
    to_nc.to_netcdf(f'{parent}/codes_wrf/{name}.nc')
    return (f'file saved, check home directory for {name}')




# specify the location of data
dir  = f'{parent}WRF/WRF/case_{ext}/'

if ty =='Sc':
    area = [-48.8,-46,141.8,144] # Sc
if ty=='Cu':
    area = [-48.8,-46,145.8,148.5] # Cu

al_file = glob.glob(f"{dir}/*wrfout_d04*")
al_file.sort()

# Start and end dates (you can modify these)
start_date = dt.datetime(2018, 2, 17, 18, 00)
end_date   = dt.datetime(2018, 2, 17, 19, 20)
# Define a regular expression pattern to extract dates from file names
date_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}'

# we have in some folder the spin files
# also the profile is only for one period
wrflist = []
for file_path in al_file:
    # Extract the date from the file name using regular expression
    match = re.search(date_pattern, file_path)
    if match:
        file_date = dt.datetime.strptime(match.group(), '%Y-%m-%d_%H:%M:%S')
        # Check if the file date falls within the specified date range
        if start_date <= file_date <= end_date:
            # Open the NetCDF file and append to wrflist
            wrflist.append(Dataset(file_path))

if len(wrflist)==0:
    sys.exit("No datasets found! Check paths are correct")

frac           = getvar(wrflist,'low_cloudfrac',timeidx=ALL_TIMES,\
                        method='cat',mid_thresh=18000,high_thresh=20000)

# get point where ship is located
lats, lons = latlon_coords(frac)
xs,ys = ll_to_xy(wrflist,area[:2],area[2:])
x1,x2,y1,y2 = int(xs[0]),int(xs[1]),int(ys[0]),int(ys[1])

# 3D ---------
frac         = frac[:,y1:y2,x1:x2] # time, lat, lon

create_xr_stats(frac,['south_north','west_east'],f'cldfrac_trans4_{ext}_{ty}_areamean')
