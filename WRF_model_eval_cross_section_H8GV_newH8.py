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
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
from datetime import datetime
import xarray as xr
from netCDF4 import Dataset

from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair, GeoBounds,
                 get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim,
                 ALL_TIMES,ll_to_xy)

import tarfile
import shutil


from Socrates_file_read import *
from Socrates_transects_dates import *
from Socrates_postprocessing import *

read = Read_files_socrates(dir_files)


# specify the location of data
parent = parent

dir  = f'{parent}SOCRATES_CAPRICORN/flight/RF12/EOLH8/'


# flight
filesd = glob.glob(f'{parent}SOCRATES_CAPRICORN/*rf12_track*')
gv = pd.read_csv(filesd[0],sep=',')
gv.index = pd.to_datetime(gv['Unnamed: 0'])
gv.index.name = None
gv = gv.drop(columns=['Unnamed: 0'])
fd_m = gv.loc['2018-02-18 00:00':'2018-02-18 07:35']

# THE IDEA OF THIS ONE IS CONCATENATE THE SECTIONS BY THE LINE DONE BY
# THE AIRCRAFT EVERY 10 MINS (MODEL RESOLUTION)
def concat_cross(id_list):
    # this version is for 2d array
    pos = 0
    for i in np.arange(len(id_list)):
        print(pos)
        id_list[i] = id_list[i].assign_coords({'cross_line_idx':id_list[i].cross_line_idx + pos})
        # the last position on each is the first of the following
        id_list[i] = id_list[i][:-1]
        id_last = id_list[i].cross_line_idx[-1]+1
        pos = int(id_last)

    id_all = xr.concat(id_list,dim='cross_line_idx')
    return id_all

#############################################################################
#model
##############################################################################
dates = pd.date_range('2018-02-18 0:20','2018-02-18 01:00',freq='600s')
#pd.date_range('2018-02-18 03:30','2018-02-18 05:20',freq='600s')
start_date = dt.datetime(2018, 2, 18, 0, 20)
#dt.datetime(2018, 2, 18, 4, 30) 0, 20
end_date   = dt.datetime(2018, 2, 18, 1, 00)
#dt.datetime(2018, 2, 18, 5, 20) 1,00

d04 = [-48.8,-46,145.8,148.8]
# south
#[-57,-54,136.7,144.8]
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

        if ti == start_date:
            sec1 = ti+dt.timedelta(minutes=5)
            sec2 = ti+dt.timedelta(minutes=9,seconds=59)
        #elif ti == end_date:
        # this is for the norht domain
        #    sec1 = ti
        #    sec2 = ti+dt.timedelta(minutes=5
        else:
            sec1, sec2 = ti,ti+dt.timedelta(minutes=9,seconds=59)

        fd_sec    = fd_m.loc[sec1:sec2]
        lat1,lat2 = fd_sec.lat.min(),fd_sec.lat.max()
        lon1,lon2 = fd_sec.lon.min(),fd_sec.lon.max()

        min_lon, max_lon = lon1, lon2  # Set the valid longitude range
        min_lat, max_lat = lat1, lat2  # Set the valid latitude range

        ch8 = h8f.cloud_top_height
        cth = ch8.where((ch8['longitude'] >= min_lon) & (ch8['longitude'] <= max_lon) &
                         (ch8['latitude'] >= min_lat) & (ch8['latitude'] <= max_lat),
                         drop=True)
        bt8 = h8f.cloud_top_temperature
        ctt = bt8.where((bt8['longitude'] >= min_lon) & (bt8['longitude'] <= max_lon) &
                         (bt8['latitude'] >= min_lat) & (bt8['latitude'] <= max_lat) & (bt8>250),
                         drop=True)-273.15

        # this new version is tricky to collocate in the traj, then we take the square
        ctt_a.append(float(ctt.median()))
        cth_a.append(float(cth.median()))

    except:
        print(dti)
        continue



ctt_a2   = concat_cross(ctt_a)
cth_a2   = concat_cross(cth_a)
phase_a2 = concat_cross(phase_a)

ctt_a2.where((ctt_a2>-15) & (cth_a2>1) & (phase_a2>0))
