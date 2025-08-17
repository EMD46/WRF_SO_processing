# -*- coding: utf-8 -*-
"""
reading and plotting soundings from CAPRICORN project

"""

# import the python libraries
from netCDF4 import Dataset,num2date
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import glob
import os
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
from datetime import datetime

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords,ll_to_xy)


# specify the location of data
parent = parent

# specify the location of data
dir  = f'{parent}WRF_SO_post_frontal/final_sensitivity/case_insitu/'

al_file = glob.glob(f"{dir}/*2018-02-18_04_30*")
al_file.sort()

# FROM THE MODEL
i = al_file[-1]
nfile = Dataset(i)
pres_temp = getvar(nfile,'pressure')
lats, lons = latlon_coords(pres_temp)
# nearest id points to ship location
ti = i.split('/')[-1][11:]
time = dt.datetime.strptime(ti,'%Y-%m-%d_%H_%M_%S').strftime('%Y-%m-%d %H:%M:%S')
yship,xship = -56.55,141.49 #sd_m.loc[time]
x,y = ll_to_xy(nfile, yship, xship)
x,y = int(x),int(y)
pres_temp = pres_temp#[:,y-1:y+2,x-1:x+2]#.mean(axis=1).mean(axis=1).values

tdry_temp = getvar(nfile,'tc')
tdry_temp = tdry_temp#[:,y-1:y+2,x-1:x+2]#.mean(axis=1).mean(axis=1).values
dpnt_temp = getvar(nfile,'td')
dpnt_temp = dpnt_temp#[:,y-1:y+2,x-1:x+2]#.mean(axis=1).mean(axis=1).values
u,v       = getvar(nfile,'uvmet')
uwin_temp = u#[:,y-1:y+2,x-1:x+2]#.mean(axis=1).mean(axis=1).values
vwin_temp = v#[:,y-1:y+2,x-1:x+2]#.mean(axis=1).mean(axis=1).values

cloud = getvar(nfile,'ctt',fill_nocloud=True)


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

# theta e from WRF
p_w = np.copy(p)
theta_e_wrf = mpcalc.equivalent_potential_temperature(p, T, Td).to('degC')

# for the shading

dfmin = pd.DataFrame(data=[pres_temp[:,:,x-5:x+6].quantile([0.10],dim='south_north')[0].quantile([0.10],dim='west_east')[0].values,
                        tdry_temp[:,:,x-5:x+6].quantile([0.10],dim='south_north')[0].quantile([0.10],dim='west_east')[0].values,
                        dpnt_temp[:,:,x-5:x+6].quantile([0.10],dim='south_north')[0].quantile([0.10],dim='west_east')[0].values],\
                index=['Pressure','Temperature','Dewpoint'])
dfmin = dfmin.T
# nan data should be deleted
dfmin[dfmin<-900] = np.nan
dfmin = dfmin.dropna().reset_index(drop=True)
pmin = dfmin['Pressure'].values * units.hPa
Tmin = dfmin['Temperature'].values * units.degC
Tdmin = dfmin['Dewpoint'].values * units.degC

dfmax = pd.DataFrame(data=[pres_temp[:,:,x-5:x+6].quantile([0.90],dim='south_north')[0].quantile([0.90],dim='west_east')[0].values,
                        tdry_temp[:,:,x-5:x+6].quantile([0.90],dim='south_north')[0].quantile([0.90],dim='west_east')[0].values,
                        dpnt_temp[:,:,x-5:x+6].quantile([0.90],dim='south_north')[0].quantile([0.90],dim='west_east')[0].values],\
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


# from Capricorn
al_file = glob.glob(f"{parent}/SOCRATES_CAPRICORN/ship/sounding/*20180218*")
al_file.sort()
al_file = al_file[1]

df = Dataset(al_file, mode='r')
pres_temp = df.variables['pres'][1:]
tdry_temp = df.variables['tdry'][1:]
dpnt_temp = df.variables['dp'][1:]
uwin_temp = df.variables['u_wind'][1:].data
vwin_temp = df.variables['v_wind'][1:].data

df = pd.DataFrame(data=[pres_temp,tdry_temp,dpnt_temp,uwin_temp,vwin_temp],\
                index=['Pressure','Temperature','Dewpoint','u','v'])
df = df.T
# nan data should be deleted
df[df<-900] = np.nan
df = df.dropna().reset_index(drop=True)

# need to assing units to variables
p = df['Pressure'].values * units.hPa
T = df['Temperature'].values * units.degC
Td = df['Dewpoint'].values * units.degC
uwin = df['u'].values
vwin = df['v'].values

# theta e from CAP
theta_e_cap = mpcalc.equivalent_potential_temperature(p, T, Td).to('degC')

skew.plot(p,T, 'r', label='T CAP',alpha=0.5,lw=2.2)
skew.plot(p,Td, 'b', label='Dp CAP',alpha=0.5,lw=2.2)
#plot barbs
my_interval = np.arange(100, 1000, 50)
ix = mpcalc.resample_nn_1d(pres_temp.data[::-1], my_interval)
skew.plot_barbs(p[ix], uwin[ix], vwin[ix],alpha=0.5,lw=2.2)
skew.ax.set_ylim(1000, 600)
skew.ax.set_xlim(-30, 30)
plt.legend(loc=3, fontsize=14)
#skew.ax.set_ylim(1000, 100)
#skew.ax.set_xlim(-50, 50)
# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k',alpha=0.5)

plt.savefig(f'Sounding_WRF_vsCAPRICORN_shaded.png',\
            dpi=500,bbox_inches='tight')


####################
# theta e plot

P0 = 1013.25 * units.hPa  # Sea level standard pressure
T0 = 288.15 * units.kelvin  # Sea level standard temperature
L = 0.0065 * units.kelvin / units.meter  # Temperature lapse rate
R = 287.05 * units.joule / (units.kilogram * units.kelvin)  # Specific gas constant
g = 9.80665 * units.meter / (units.second ** 2)  # Gravity

alt_w   = (T0 / L) * (1 - (p_w / P0) ** (R * L / g)) / 1000  # Altitude in km
alt_cap = (T0 / L) * (1 - (p / P0) ** (R * L / g)) / 1000  # Altitude in km

theta_e_gradient_wrf = np.gradient(theta_e_wrf.m, alt_w.m)
theta_e_gradient_cap = np.gradient(theta_e_cap.m, alt_cap.m)

plt.figure(figsize=(8, 6))
plt.plot(theta_e_wrf, alt_w, label="WRF", color="blue")
plt.plot(theta_e_cap, alt_cap, label="CAP", color="green")
plt.gca().invert_yaxis()  # Invert y-axis for pressure (high to low)
plt.xlabel("Equivalent Potential Temperature [K]")
plt.ylabel("Pressure [hPa]")
plt.ylim(0,2)
plt.xlim(10,15)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(theta_e_gradient_wrf, alt_w, label="WRF", color="blue")
plt.plot(theta_e_gradient_cap, alt_cap, label="CAP", color="green")
plt.gca().invert_yaxis()  # Invert y-axis for pressure (high to low)
plt.xlabel("Equivalent Potential Temperature [K]")
plt.ylabel("Pressure [hPa]")
plt.ylim(0,2)
#plt.xlim(10,15)
plt.legend()
plt.show()

