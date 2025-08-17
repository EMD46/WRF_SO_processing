# -*- coding: utf-8 -*-
"""
reading and plotting soundings from CAPRICORN project

"""

# COULDNT GET TO MAKE IT RUN IN PYTHON3.6....WORKED IN 2.7

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
                 cartopy_ylim, latlon_coords,ll_to_xy)

from Socrates_file_read import *

# specify the location of data
parent = parent

# specify the location of data
dir  = f'{parent}WRF_SO_post_frontal/final_sensitivity/case_trans4/'

al_file = glob.glob(f"{dir}/*2018-02-18_00_40*")
al_file.sort()

# FROM THE MODEL
i = al_file[-1]
nfile = Dataset(i)
pres_temp = getvar(nfile,'pressure')
lats, lons = latlon_coords(pres_temp)
# nearest id points to ship location
ti = i.split('/')[-1][11:]
time = dt.datetime.strptime(ti,'%Y-%m-%d_%H_%M_%S').strftime('%Y-%m-%d %H:%M:%S')
yship,xship = -46.99,146.36 #sd_m.loc[time]
x,y = ll_to_xy(nfile, yship, xship)
x,y = int(x),int(y)
pres_temp = pres_temp#[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values

tdry_temp = getvar(nfile,'tc')
tdry_temp = tdry_temp#[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values
dpnt_temp = getvar(nfile,'td')
dpnt_temp = dpnt_temp#[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values
u,v       = getvar(nfile,'uvmet')
uwin_temp = u#[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values
vwin_temp = v#[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values

df = pd.DataFrame(data=[pres_temp[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values,
                        tdry_temp[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values,
                        dpnt_temp[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values,
                        uwin_temp[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values,
                        vwin_temp[:,y-1:y+2,x-1:x+2].mean(axis=1).mean(axis=1).values],\
                index=['Pressure','Temperature','Dewpoint','u','v'])
df = df.T
# nan data should be deleted
df[df<-900] = np.nan
df = df.dropna().reset_index(drop=True)

# plot
# need to assing units to variables
p = df['Pressure'].values * units.hPa
T = df['Temperature'].values * units.degC
Td = df['Dewpoint'].values * units.degC
uwin = df['u'].values
vwin = df['v'].values



# for the shading
d04 = [-48.8,-46,145.8,148.8]
xs,ys = ll_to_xy(nfile,d04[:2],d04[2:])
x1,x2,y1,y2 = int(xs[0]),int(xs[1]),int(ys[0]),int(ys[1])
dfmin = pd.DataFrame(data=[pres_temp[:,y1:y2,x1:x2].quantile([0.10],dim='south_north')[0].quantile([0.10],dim='west_east')[0].values,
                        tdry_temp[:,y1:y2,x1:x2].quantile([0.10],dim='south_north')[0].quantile([0.10],dim='west_east')[0].values,
                        dpnt_temp[:,y1:y2,x1:x2].quantile([0.10],dim='south_north')[0].quantile([0.10],dim='west_east')[0].values],\
                index=['Pressure','Temperature','Dewpoint'])
dfmin = dfmin.T
# nan data should be deleted
dfmin[dfmin<-900] = np.nan
dfmin = dfmin.dropna().reset_index(drop=True)
pmin = dfmin['Pressure'].values * units.hPa
Tmin = dfmin['Temperature'].values * units.degC
Tdmin = dfmin['Dewpoint'].values * units.degC

dfmax = pd.DataFrame(data=[pres_temp[:,y1:y2,x1:x2].quantile([0.90],dim='south_north')[0].quantile([0.90],dim='west_east')[0].values,
                        tdry_temp[:,y1:y2,x1:x2].quantile([0.90],dim='south_north')[0].quantile([0.90],dim='west_east')[0].values,
                        dpnt_temp[:,y1:y2,x1:x2].quantile([0.90],dim='south_north')[0].quantile([0.90],dim='west_east')[0].values],\
                index=['Pressure','Temperature','Dewpoint'])
dfmax = dfmax.T
# nan data should be deleted
dfmax[dfmax<-900] = np.nan
dfmax = dfmax.dropna().reset_index(drop=True)
pmax = dfmax['Pressure'].values * units.hPa
Tmax = dfmax['Temperature'].values * units.degC
Tdmax = dfmax['Dewpoint'].values * units.degC

#plot data
fig = plt.figure(figsize=(5,6))
skew = SkewT(fig,aspect='auto')
skew.plot(p,T, 'r', label='T WRF',lw=1.7)
skew.plot(p,Td, 'b', label='Dp WRF',lw=1.7)
#plot shaded areas
skew.shade_area(p,Tmin,Tmax,color='orange')
skew.shade_area(p,Tdmin,Tdmax,color='skyblue')
#plot barbs
my_interval = np.arange(100, 1000, 50)
ix = mpcalc.resample_nn_1d(df['Pressure'].values, my_interval)
skew.plot_barbs(p[ix], uwin[ix], vwin[ix])
#skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-50, 50)
# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k')

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
# lables
plt.title(f'{time}',fontsize=10,y=-0.25)
plt.xlabel(r'Temperature [$^{\circ}C$]', fontsize=14)
plt.ylabel('Pressure [mb]', fontsize=14)
plt.tick_params('both',labelsize=13)


# FROM SOCRATES
f = glob.glob(f'{parent}SOCRATES_CAPRICORN/flight/RF12/dropsonde/Cu/*20180218_00*')
f.sort()
date  = f[0].split('_')
date1 = date[1].split('/')[-1][1:]
n1    = date1+date[2]
d1    = (dt.datetime.strptime(n1,'%Y%m%d%H%M%S') - dt.timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
read = Read_files_socrates('/media/emontoyaduqu/ESTEFANIA/dropbox/UniMelb/first_year/')
file  = read.plain_text_flight_dropsonde(f[0],d1,{'lat':float(date[4]),'lon':float(date[3])})
# end date format
d2      = file.index[-1].strftime('%Y-%m-%d %H:%M:%S')
n2      = file.index[-1].strftime('%Y%m%d%H%M%S')

# Drop any rows with all NaN values for T, Td, winds
file[file<-99] = np.nan
file = file.dropna().reset_index(drop=True)

# need to assing units to variables
p    = file['Press'].values[::-1] * units.hPa
T    = file['Temp'].values[::-1] * units.degC
Td   = file['Dewpt'].values[::-1] * units.degC
uwin = file['Uwind'].values[::-1]
vwin = file['Vwind'].values[::-1]

skew.plot(p,T, 'r', label='T RF12',alpha=0.5,lw=2.2)
skew.plot(p,Td, 'b', label='Dp RF12',alpha=0.5,lw=2.2)
#plot barbs
my_interval = np.arange(100, 1000, 50)
ix = mpcalc.resample_nn_1d(file['Press'].values[::-1], my_interval)
skew.plot_barbs(p[ix], uwin[ix], vwin[ix],alpha=0.5,lw=2.2)
skew.ax.set_ylim(1000, 600)
skew.ax.set_xlim(-30, 30)
plt.legend(loc=3, fontsize=14)
#skew.ax.set_ylim(1000, 100)
#skew.ax.set_xlim(-50, 50)
# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k',alpha=0.5)



plt.savefig(f'Sounding_WRF_vsSOCRATES_shaded.png',\
            dpi=500,bbox_inches='tight')
