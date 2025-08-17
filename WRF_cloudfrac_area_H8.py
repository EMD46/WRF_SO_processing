# -*- coding: utf-8 -*-
"""

@author: Estefania Montoya June 2022

This code is modified from the second paper plots
to fit the WRF comparison/validation purpose
"""

%load_ext autoreload
%autoreload 2

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

import shutil
import tarfile

from scipy import ndimage
from skimage.measure import regionprops


#------------------------------------------------------------------------------
# preliminary information for reading
# values were generated in gadi with other code
#------------------------------------------------------------------------------
parent = dir
ty  = 'Sc'#sys.argv[2]#'Sc'

if ty =='Sc':
    area = [-48.8,-46,141.8,144] # Sc
if ty=='Cu':
    area = [-48.8,-46,145.8,148.5] # Cu


dates = pd.date_range('2018-02-17 18','2018-02-18 00',freq='10min')

# Himawari 8 information
all_ch13 = []
for ti in dates:
    dti = str(ti).split(':')[0]
    #folder date pattern
    t = dti.split(' ')[0].replace('-','')
    #file date pattern
    date2 = dt.datetime.strptime(str(ti),'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')[:-1]

    fname = f'{parent}H8_BoM/{t}.tar'
    # Open the tar file
    TF = tarfile.open(fname,'r')

    member = [m for m in TF.getmembers() if t in m.name]
    # extract files with cloud information
    #member2 = [TF.extract(m,'./') for m in member if 'OBS' in m.name]
    member2 = [TF.extract(m,'./') for m in member if 'CLD' in m.name]


    extract_dir = dti.split(' ')[0].replace('-','/')
    h8_files = glob.glob(f'./{extract_dir}/*{date2}*')
    h8_files.sort()

    if len(h8_files)>0:

        # open  himawari fileopol.loc[date]
        # some H8 files were missing
        ctt = xr.open_dataset(h8_files[0],decode_times=False)
        ch13 = ctt.cloud_top_temperature.loc[:,-64:-34,120:167][0]
        ch = ctt.cloud_top_height.loc[:,-64:-34,120:167][0]

        ch13 = ch13.where(ch>500) # 400 m
        ch13 = ch13.assign_coords({'time':ti}).expand_dims(dim={'time':1})
        all_ch13.append(ch13)
        ctt.close()
        #-----------------
        # CAREFUL !!!!
        #----------------
        shutil.rmtree(f'./{extract_dir[:4]}/')
    else:
        exit("No file found")


all_ch13     = xr.concat(all_ch13,dim='time')
y1,y2,x1,x2  = area
frac         = all_ch13.loc[:,y1:y2,x1:x2] # time,lat, lon

thresh = -900 # for now -900 is no cloud, above 0 any extension of cloud
frac2 = frac.fillna(thresh)

d1,d2,d3 = np.shape(frac2)
empt  = np.empty([11,d1])

for ta in np.arange(d1):
    test = ndimage.label(frac2[ta] > thresh)[0]

    # total number of different blops
    N = len(np.unique(test))
    # N counts the zeros which are clear sky
    #save mean, p10,p25,p50,p75,p90,std, min,max, frac
    areas = []; remove = []
    for i in range(0, N):
        object_i = test[test == i]
        #number of pixels
        pix = len(object_i)
        if pix > 0:
           areas.append(pix)
        else:
           remove.append(i)
    areas = np.array(areas)
    areas2 = pd.DataFrame(areas)

    test2 = test.copy()
    rem = np.array(remove)
    for ii in rem:
        test2[test2==ii] = 0 # set 0, clear sky value

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
                    'Time':frac2.time.values})
to_nc = to_nc.assign_attrs(description=\
"id 0:mean, 1:std, 2:min, 3:max, 4: frac, 5:Number cores,6:percentil 10, 7: 2:percentil 25, 8:percentil 50, 9:percentil 75, 10:percentil 90")

#save the nc_file
to_nc.to_netcdf(f'{parent}/WRF_SO_post_frontal/pre_processed/ObjectIDstats_case_H8_{ty}.nc')
