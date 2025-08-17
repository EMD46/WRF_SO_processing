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


from Socrates_file_read import *
from Socrates_transects_dates import *
from Socrates_postprocessing import *

read = Read_files_socrates(dir_files)


list_radar = [['reflectivity','doppler','cld_phase','crosspol_backscatter','T'],\
              ['HCR_DBZ','HCR_VEL','PID','elevation','TEMP',\
              'HSRL_Aerosol_Backscatter_Coefficient','HCR_LDR','HCR_ECHO_TYPE_2D']]


# CAPRICORN
radar_lidar_s   = read.merge_nc_flight_xr('data/ship/radar-lidar3/',
                                          list_radar[0],'hour',
                                          changedate=True,decode=True)

#SOCRATES
radar_lidar_f3  = read.merge_nc_flight_xr('data/flight/RF12/HCR_HSRL_q03_RF12',\
                                          list_radar[1],'time')
micro_1sps_f3  = read.one_nc_flight_xr(dict_files['RF12']['subdir'],inst[4])
s1,s2 = ['2018-02-18 03:30','2018-02-18 05:20']
# ['2018-02-18 00:25','2018-02-18 01:05']
#['2018-02-18 03:30','2018-02-18 05:20']
#RF12_ship['time_gv'][0]

######################################################################
# SOCRATES
# cloud top temperature
radar = radar_lidar_f3
cla   = 'PID'
cla2  = 'TEMP'
mtimes = pd.to_datetime(radar.time.loc[s1:s2][::2].values) #it is every second and not 0.5 seconds


# for socates we need to estimate the altitude
trange,rrange = np.meshgrid(mtimes,radar.range)
#mesh gps altitude: shape alt, time
rnew,raux     = np.meshgrid(micro_1sps_f3.GGALT.loc[mtimes],radar_lidar_f3.range)
# where does the sensor was looking up or down: shape time
elev          = radar_lidar_f3.elevation.loc[mtimes]
# from these meshgrids will work with rrange, rnew, elev
# now find positions where elev was positive --> sensor looking up
jup   = np.where(elev>0)[0]
jdown = np.where(elev<0)[0]

rrange[:,jup]   = rrange[:,jup] + rnew[:,jup]
rrange[:,jdown] = rnew[:,jdown] - rrange[:,jdown]
rrange          = rrange /1000.



ctt_a,cth_a = [],[]
surfp = []
for i in np.arange(len(mtimes)):
    layeri = pd.DataFrame(radar[cla].loc[mtimes][i],columns=['layer'])
    layeri.index.name = None

    layeri['position'] = np.arange(len(layeri))
    layeri['tag']      = np.isfinite(layeri.layer)
    layeri['altitude'] = rrange[:,i]*1000
    layeri['temp']     = radar[cla2].loc[mtimes][i].values

    # first row is a True preceded by a False
    fst = layeri.position[layeri['tag'] & ~ layeri['tag'].shift(1).fillna(False)]
    # last row is a True followed by a False
    lst = layeri.position[layeri['tag'] & ~ layeri['tag'].shift(-1).fillna(False)]

    # filter those which are adequately apart
    consecutive = [(i, j) for i, j in zip(fst.values, lst.values) if j > i+1]

    # IF SOCRATES ---> DEPENDS ON THE AIRCRAFT POSITION
    # time by time find first and last value identified as cloud
    if len(consecutive)>=1:
        con_t = []
        for pt in consecutive:
            aux = layeri[pt[0]:pt[1]]
            if abs(aux.altitude.values[0] - aux.altitude.values[-1])>=60:
                con_t.append({'Y_start':aux.altitude.values[0],'Y_end':aux.altitude.values[-1],
                              'T_start':aux.temp.values[0],'T_end':aux.temp.values[-1],
                              'PID_start':aux.layer.values[0],'PID_end':aux.layer.values[-1]})

                layer_times = pd.DataFrame(con_t)

                if layer_times.Y_start[0] > layer_times.Y_end[0]:
                    ctt_a.append(layer_times.T_start[0])
                    cth_a.append(layer_times.Y_start[0])
                    # now take the bottom not the top
                    surfp.append(layer_times.PID_end[0])
                else:
                    ctt_a.append(layer_times.T_end[0])
                    cth_a.append(layer_times.Y_end[0])
                    # not take the bottom not the top
                    surfp.append(layer_times.PID_start[0])


rads2 = np.array(surfp) # precip flags (1, 2, 3, 4), melting (7), Lfrozen (8), precip(10)
surfpp = len(rads2[(rads2==1) | (rads2==2) | (rads2==3) | \
             (rads2==4) | (rads2==7) | (rads2==8) | \
             (rads2==10)])/len(rads2)

# phase inside cloud no rain
# liquid = 5,6
# ice    = 8,9
# mixed = 7 (melting)
rads3 = radar[cla].loc[mtimes].values.ravel()
# percentages for values above 4
# melting layer....as liquid
len(rads3[(rads3==5) | (rads3==6) | (rads3==7)])/len(rads3[(rads3>4) & (rads3!=10)])


######################################################################
# CAPRICORN
radar = radar_lidar_s
cla   = 'cld_phase'
cla2  = 'T'
mtimes = pd.to_datetime(radar.hour.loc[s1:s2].values) #it is every second and not 0.5 seconds

ctt_a,cth_a = [],[]
for i in np.arange(len(mtimes)):
    fil = radar[cla].loc[:,mtimes][:,i].where(radar[cla].loc[:,mtimes][:,i].values>0)
    layeri = pd.DataFrame(fil.values,columns=['layer'])
    layeri.index.name = None

    layeri['position'] = np.arange(len(layeri))
    layeri['tag']      = np.isfinite(layeri.layer)
    layeri['altitude'] = radar.height*1000
    layeri['temp']     = radar[cla2].loc[:,mtimes][:,i].values

    # first row is a True preceded by a False
    fst = layeri.position[layeri['tag'] & ~ layeri['tag'].shift(1).fillna(False)]
    # last row is a True followed by a False
    lst = layeri.position[layeri['tag'] & ~ layeri['tag'].shift(-1).fillna(False)]

    # filter those which are adequately apart
    consecutive = [(i, j) for i, j in zip(fst.values, lst.values) if j > i+1]

    # IF CAPRICORN --> ALWAYS LOOKING UP
    # for this specific case is one layer cloud
    if len(consecutive)==1:
        for pt in consecutive[:1]:
            aux = layeri[pt[0]:pt[-1]]
            if abs(aux.altitude.values[0] - aux.altitude.values[-1])>=60:
                cth_a.append(aux.altitude.values[-1])
                ctt_a.append(aux.temp.values[-1])
    if len(consecutive)==2: # notices some tiny two layer super low cloud
        for pt in consecutive[1:]:
            aux = layeri[pt[0]:pt[-1]]
            if abs(aux.altitude.values[0] - aux.altitude.values[-1])>=60:
                cth_a.append(aux.altitude.values[-1])
                ctt_a.append(aux.temp.values[-1])

surfp = radar[cla].loc[:,mtimes][6] # this one is always looking up and about 10 m blind
# cloud phase -- inside cloud is values above 2 (warm p and ice virga) and below 7 (aerosol and virga)
rads3 = radar[cla].loc[:,mtimes].values.ravel()
len(rads3[(rads3==3)])/len(rads3[(rads3>2) & (rads3<7)])
