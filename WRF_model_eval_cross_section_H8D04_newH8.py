"""
Compute general statistic for model
validation

Author: Estefania Montoya March 2024
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
from datetime import datetime
import xarray as xr
from netCDF4 import Dataset

import tarfile
import shutil


from Socrates_file_read import *
from Socrates_transects_dates import *
from Socrates_postprocessing import *

read = Read_files_socrates(dir_files)


# specify the location of data
parent = parent

dir  = f'{parent}SOCRATES_CAPRICORN/flight/RF12/EOLH8/'

dates = pd.date_range('2018-02-18 00:20','2018-02-18 01:00',freq='600s')
#pd.date_range('2018-02-18 00:20','2018-02-18 01:00',freq='600s')
#pd.date_range('2018-02-18 03:30','2018-02-18 05:20',freq='600s')

d04 = [-48.8,-46,141.8,144]
# south
#[-57.8,-55,136.6,144]
# north
#[-49,-45,141,149]
# north closed
# [-48.8,-46,141.8,144]
#north open
# [-48.8,-46,145.8,148.8]


# Himawari 8 information
ctt_a, cth_a, phase_a = [],[],[]
for ti in dates:
    dti = str(ti)
    t = dti.replace('-','').replace(' ','').replace(':','')[:-2]

    # in that period EOL has some missing files
    try:
        h8_file = f'satellite.SATCORPS_V2.2_HIMAWARI-8.2km.{t}.NC'

        # himawari date
        date2 = dt.datetime.strptime(t, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')

        h8f = xr.open_dataset(f'{parent}SOCRATES_CAPRICORN/flight/RF12/EOLH8/{h8_file}')

        min_lon, max_lon = d04[2], d04[3]  # Set the valid longitude range
        min_lat, max_lat = d04[0], d04[1]  # Set the valid latitude range

        ch8 = h8f.cloud_top_height
        chh8 = ch8.where((ch8['longitude'] >= min_lon) & (ch8['longitude'] <= max_lon) &
                         (ch8['latitude'] >= min_lat) & (ch8['latitude'] <= max_lat),
                         drop=True)
        bt8 = h8f.cloud_top_temperature
        bth8 = bt8.where((bt8['longitude'] >= min_lon) & (bt8['longitude'] <= max_lon) &
                         (bt8['latitude'] >= min_lat) & (bt8['latitude'] <= max_lat) & (bt8>250),
                         drop=True)-273.15
        ctt_a.append(bth8)
        cth_a.append(chh8)
    except:
        print(dti)
        continue


ctt_a2   = xr.concat(ctt_a,dim='time')
cth_a2   = xr.concat(cth_a,dim='time')


# in the south domain classification is pretty bad
cth_a2.median()
cth_a2.quantile(0.75) - cth_a2.quantile(0.25)

ctt_a2.median()
ctt_a2.quantile(0.75) - ctt_a2.quantile(0.25)

