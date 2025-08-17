# -*- coding: utf-8 -*-
"""

@author: Estefania Montoya Aug 2023

This code is modified from the second paper plots
to fit the WRF comparison/validation purpose
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import cmocean.cm as cmo

# get cloud frac as seen from above (like satellite)
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords,ll_to_xy,ALL_TIMES)

# to track objects in "images"
from scipy import ndimage
from skimage.measure import regionprops

#------------------------------------------------------------------------------
# preliminary information for reading
# values were generated in gadi with other code
#------------------------------------------------------------------------------
parent = parent
ext    = 'trans4'

dir2     = f'{parent}WRF_SO_post_frontal/final_sensitivity/case_trans4/'

dates = pd.date_range('2018-02-17 18:00','2018-02-18 00:00',freq='600s')

# Himawari 8 information
for ti in dates:
    dti = str(ti)
    t = dti.replace('-','').replace(' ','').replace(':','')

    ##### model bt
    t2 = dti.replace(' ','_').replace(':','_')
    d = Dataset(f'{dir2}wrfout_d04_{t2}')
    #FOR NOW IT IS ONLY FOR ONE FILE!!!!
    wrf =  getvar(d,'ctt',units='degC',fill_nocloud=True)
    lats, lons = latlon_coords(wrf)

    areas_pd = {}
    for ind,area in enumerate([[-48.8,-46,141.8,144],[-48.8,-46,145.8,148.5]]):
        if ind==0:
            ty = 'Sc'
        if ind==1:
            ty = 'Cu'

        xs,ys = ll_to_xy(d,area[:2],area[2:])
        x1,x2,y1,y2 = int(xs[0]),int(xs[1]),int(ys[0]),int(ys[1])

        wrf2  =  wrf[y1:y2,x1:x2]

        #by area open/closed extimate the cloud objects and the area
        thresh = -900 # for now -900 is no cloud, above 0 any extension of cloud
        wrf2 = wrf2.fillna(thresh)
        # it will include 0: "empty areas" exclude them for counting objects
        # each number will be "a cloud" and it will be repeated for all pixels
        # it covers
        test = ndimage.label(wrf2 > thresh)

        N = len(np.unique(test[0]))

        areas = []
        for i in range(1, N):
            object_i = test[0][test[0] == i]
            #number of pixels
            pix = len(object_i)
            # keep only the pixels above 1 count
            if pix>1:
                areas.append(pix)
        areas = np.array(areas)
        areas2 = pd.DataFrame(areas,columns=[ti])
        areas_sts = areas2.describe()
        areas_pd[ty] = areas_sts
    d.close()
