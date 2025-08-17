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
import matplotlib.patches as patches

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords,ll_to_xy,ALL_TIMES)

import sys
import re

from scipy import ndimage
from skimage.measure import regionprops

ext = sys.argv[1]#'trans4'
ty  = sys.argv[2]#'Sc'
parent = parent
# specify the location of data
dir  = f'{parent}WRF_SO_post_frontal/final_sensitivity/case_{ext}'

if ty =='Sc':
    area = [-48.8,-46,141.8,144] # Sc
if ty=='Cu':
    area = [-48.8,-46,145.8,148.5] # Cu

al_file = glob.glob(f"{dir}/*wrfout_d04*")
al_file.sort()

# Start and end dates (you can modify these)
start_date = dt.datetime(2018, 2, 17, 18, 00)
end_date   = dt.datetime(2018, 2, 18, 00, 00)
# Define a regular expression pattern to extract dates from file names
date_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}'

# we have in some folder the spin files
# also the profile is only for one period
wrflist = []
for file_path in al_file:
    # Extract the date from the file name using regular expression
    match = re.search(date_pattern, file_path)
    if match:
        file_date = dt.datetime.strptime(match.group(), '%Y-%m-%d_%H_%M_%S')
        # Check if the file date falls within the specified date range
        if start_date <= file_date <= end_date:
            # Open the NetCDF file and append to wrflist
            wrflist.append(Dataset(file_path))

if len(wrflist)==0:
    sys.exit("No datasets found! Check paths are correct")



empt  = np.empty([11,len(wrflist)])

# set up the threshold exploration
#lay  = lats[y1:y2,x1:x2]
#lonx = lons[y1:y2,x1:x2]

# cell identification with polygons
from stardist.plot import render_label
import gc

for ta in np.arange(len(wrflist)):

    frac           = getvar(wrflist[ta],'ctt',units='degC',fill_nocloud=True)

    # get point where ship is located
    lats, lons = latlon_coords(frac)
    xs,ys = ll_to_xy(wrflist,area[:2],area[2:])
    x1,x2,y1,y2 = int(xs[0]),int(xs[1]),int(ys[0]),int(ys[1])

    # 3D ---------
    frac         = frac[y1:y2,x1:x2] # time, lat, lon

    thresh = -900 # for now -900 is no cloud, above 0 any extension of cloud
    frac2 = frac.fillna(thresh)

    # object id
    test = ndimage.label(frac2 > thresh)[0]

    # total number of different blops
    N = len(np.unique(test))
    # N counts the zeros which are clear sky
    #save mean, p10,p25,p50,p75,p90,std, min,max, frac
    areas = []; remove = []
    for i in range(0, N):
        object_i = test[test == i]
        #number of pixels
        pix = len(object_i)
        if pix <= 10:
           remove.append(i)

    test2 = test.copy()
    rem = np.array(remove)
    for ii in rem:
        test2[test2==ii] = 0 # set 0, clear sky value, to small objects


    '''
    plt.figure(figsize=(10,15))
    plt.subplot(1,3,1)
    plt.imshow(frac2[ta], cmap="gray")
    x, y = 130, 160  
    # Create a rectangle patch with width=10 and height=10
    rect = patches.Rectangle((x, y), 5, 2, linewidth=2, edgecolor='red', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.axis("off")
    plt.title("a)   Input image",fontsize=15)

    plt.subplot(1,3,2)
    plt.imshow(render_label(test),vmin=0,vmax=206)
    plt.axis("off")
    plt.title("b)   Object ID",fontsize=15)

    plt.subplot(1,3,3)
    plt.imshow(render_label(test2),vmin=0,vmax=206)
    plt.axis("off")
    plt.title("c)   Object ID filtered",fontsize=15)

    plt.savefig('example_ob_id3.png',dpi=500,bbox_inches='tight')
    '''


    # now that small areas are set to clear sky classify again
    N2 = len(np.unique(test2))
    for i in np.unique(test2): #not arange since there are numbers that wont exist
        object_i = test2[test2 == i]
        #number of pixels
        pix = len(object_i)
        areas.append(pix)

    # Clear variables that are no longer needed
    del test, test2, object_i, frac, frac2
    gc.collect()  # Force memory cleanup

    # now we have all areas append and statisitcs
    areas = np.array(areas)
    areas2 = pd.DataFrame(areas)

    #statistics id 0 will be clear sky
    empt[6:,ta] = areas2[1:].quantile([0.10,0.25,0.5,0.75,0.90]).values[:,0]
    empt[0,ta]  = areas2[1:].mean().values
    empt[1,ta]  = areas2[1:].std().values
    empt[2,ta]  = areas2[1:].min().values
    empt[3,ta]  = areas2[1:].max().values
    empt[4,ta]  = areas2[1:].sum()/areas2.sum()
    empt[5,ta]  = len(areas2)-1 # since zero is clear


# now there is a time series of cloud statictis characterizing each blop
to_nc = xr.Dataset({'stat':(['id','Time'],empt),
                    'id':np.arange(0,11),
                    'Time':np.arange(len(wrflist))})
to_nc = to_nc.assign_attrs(description=\
"id 0:mean, 1:std, 2:min, 3:max, 4: frac, 5:Number cores,6:percentil 10, 7: 2:percentil 25, 8:percentil 50, 9:percentil 75, 10:percentil 90")

#save the nc_file
to_nc.to_netcdf(f'{parent}/ObjectIDstats_case_{ext}_{ty}.nc')
