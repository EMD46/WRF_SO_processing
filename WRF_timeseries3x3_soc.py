# -*- coding: utf-8 -*-
"""
WRF visualization

@modified: Estefania Montoya Feb 2023
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

from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair, GeoBounds,
                 get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim,
                 ALL_TIMES,ll_to_xy)


from Socrates_file_read import *
from Socrates_transects_dates import *

read = Read_files_socrates(dir_files)


# specify the location of data
parent = parent

dir  = f'{parent}WRF_SO_post_frontal/final_sensitivity/case_insitu/'

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
    pos = 0
    for i in np.arange(len(id_list)):
        print(pos)
        id_list[i] = id_list[i].assign_coords({'cross_line_idx':id_list[i].cross_line_idx + pos})
        # the last position on each is the first of the following
        id_list[i] = id_list[i][:,:-1]
        # also not all height are the same, but almost set as one
        id_list[i] = id_list[i].assign_coords({'vertical':id_list[0].vertical})
        id_last = id_list[i].cross_line_idx[-1]+1
        pos = int(id_last)

    id_all = xr.concat(id_list,dim='cross_line_idx')
    return id_all

#############################################################################
#model
##############################################################################
start_date = dt.datetime(2018, 2, 18, 3, 30)
end_date   = dt.datetime(2018, 2, 18, 5, 20)
# Define a regular expression pattern to extract dates from file names
date_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}'

al_file = glob.glob(f"{dir}/*wrfout_d04*")
al_file.sort()


# convert Nc and others to volume they are in Kg^-1 to L^-1 with ideal gas law
Rv    = 461 #J/K kg

cloud_l, ice_l, liquid_l,Ni_l,Nc_l = [],[],[],[],[]
frac_l = []
for file_path in al_file:
    # Extract the date from the file name using regular expression
    match = re.search(date_pattern, file_path)
    if match:
        file_date = dt.datetime.strptime(match.group(), '%Y-%m-%d_%H_%M_%S')
        # Check if the file date falls within the specified date range
        if start_date <= file_date <= end_date:
            print(file_date)
            # Open the NetCDF file and append to wrflist
            d = Dataset(file_path)

            mixc = (getvar(d,'QCLOUD')*1000)
            mixr = (getvar(d,'QRAIN')*1000)
            mixi = (getvar(d,'QICE')*1000)
            z    = (getvar(d,'height_agl')/1000)
            p    = getvar(d,'p',units='Pa')
            temp = getvar(d,'temp')
            Nice = getvar(d,'QNICE')
            Nr   = getvar(d,'QNRAIN')
            frac = (getvar(d,'CLDFRA')*100)
            # fix Nice and Nr units
            Nicev = ((p*Nice)/(Rv*temp))* 0.001 # 1/L
            Nrv   = ((p*Nr)/(Rv*temp))*0.001



            # to id the location each 10 minutes
            # the flight is inside the area at 0:25 but we don have that file
            if file_date == start_date:
                sec1 = file_date+dt.timedelta(minutes=5)
                sec2 = file_date+dt.timedelta(minutes=9,seconds=59)
            elif file_date == end_date:
                sec1 = file_date
                sec2 = file_date+dt.timedelta(minutes=5)
            else:
                sec1, sec2 = file_date,file_date+dt.timedelta(minutes=9,seconds=59)

            fd_sec    = fd_m.loc[sec1:sec2]
            lat1,lat2 = fd_sec.lat.min(),fd_sec.lat.max()
            lon1,lon2 = fd_sec.lon.min(),fd_sec.lon.max()
            # the trajectory is from right to left then it goes
            # from max lat, max lon to min lat, min lon
            start_point = CoordPair(lat=lat2, lon=lon2)
            end_point = CoordPair(lat=lat1, lon=lon1)

            height = z[:,:].mean(axis=1).mean(axis=1)

            # now take the cross section between the two points from above
            lats, lons = latlon_coords(z)
            cld    = mixi + mixr + mixc

            # Compute the vertical cross-section interpolation.  Also, include the
            # lat/lon points along the cross-section in the metadata by setting latlon
            # to True.
            cloud = vertcross(mixc, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            ice = vertcross(mixi, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            liquid = vertcross(mixr, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            Ni = vertcross(Nicev, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            Nc = vertcross(Nrv, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            frac2 = vertcross(frac, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            # save them in a list
            cloud_l.append(cloud)
            ice_l.append(ice)
            liquid_l.append(liquid)
            Ni_l.append(Ni)
            Nc_l.append(Nc)
            frac_l.append(frac2)

            d.close()
# now we need to turn them into one 2d array but to concat we need to
#chnage the cross_line_id in a way they are consequtive (as if they were made
# in the same vert cross)

cloud_thresh = 1*10**-2#1*10**-14

cloud_all = concat_cross(cloud_l)
ice_all = concat_cross(ice_l)
liquid_all = concat_cross(liquid_l)
Ni_all = concat_cross(Ni_l)
Nc_all = concat_cross(Nc_l)
frac_all = concat_cross(frac_l)

cldm = xr.where((frac_all > 1) & ((ice_all > cloud_thresh) | \
                                  (liquid_all>cloud_thresh) | \
                                  (cloud_all>cloud_thresh)), 1, np.nan)

##############################################################################
# SOCRATES
#############################################################################
# # # read file o 1 sps
# Microphysics flight 3 1 sps
segment =  RF12_ship
micro_1sps_f3  = read.one_nc_flight_xr(dict_files['RF12']['subdir'],inst[4])
# 3 is 2dc and 2 is 2ds
psd_2ds        = read.one_nc_flight_xr(dict_files['RF12']['subdir'],inst[2])
psd_2dc        = read.one_nc_flight_xr(dict_files['RF12']['subdir'],inst[3])


phase          = xr.open_dataset(f'{parent}/SOCRATES_CAPRICORN/flight/SOCRATES_phase_product.nc')
phase          = phase.phase[phase.flight_number==12]
phase          = phase.assign_coords(time=micro_1sps_f3.Time.values)
phase          = phase.rename({'time':'Time'})


cloud1 = micro_1sps_f3.PLWCD_RWIO
cloud2 = micro_1sps_f3.PLWC2DSA_2V#.PLWC2DCA_RWOI
cloud3 = psd_2ds.IWC.assign_coords(Time=micro_1sps_f3.Time)
cloudt = cloud1 +  cloud2 +  cloud3

s1,s2 = segment['time_gv'][0]
seg   = micro_1sps_f3
yval  = seg.GGALT.loc[s1:s2]/1000
ph    = phase.loc[s1:s2]

condl = (cloudt.loc[s1:s2]>cloud_thresh) & (yval<=1.5)


# # Cloud droplet Concentration Nc

# # Liquid Water Content LWC
# CDP LWC vs Nc
condl2 = condl & ((ph==1))
dc  = seg.CONCD_RWIO.where(condl2)
dc  = dc.where(dc>0)/0.001
lwc = seg.PLWCD_RWIO.where(condl2)
lwc = lwc.where(lwc>0)
print('NC_cdp',dc[dc>0].min().values,dc.max().values,dc.mean().values)
print('lwc_cdp',lwc[lwc>0].min().values,lwc.max().values,lwc.mean().values)

# RAIN Water content (in the model >100, here they are >150)
ds2  = seg.CONC2DSA_2V.where(condl2)
ds2  = ds2.where(ds2>0)
lws2 = seg.PLWC2DSA_2V.where(condl2)
lws2 = lws2.where(lws2>0)
print('NC_2ds',ds2[ds2>0].min().values,ds2.max().values,ds2.mean().values)
print('lwc_2ds',lws2[lws2>0].min().values,lws2.max().values,lws2.mean().values)

# iCE QWATER CONTENT
condi = condl & ((ph==3) | (ph==2))
seg4 = psd_2ds.Nice[:len(micro_1sps_f3.Time)].assign_coords(Time=micro_1sps_f3.Time)
seg5 = psd_2ds.IWC[:len(micro_1sps_f3.Time)].assign_coords(Time=micro_1sps_f3.Time)

nis = seg4.loc[s1:s2].where(condi)/0.001
nis = nis.where(nis>0)
iws = seg5.loc[s1:s2].where(condi)
iws = iws.where(iws>0)

#############################################################################
# boxplot
#############################################################################
# remove nan values from model data (i.e outside cloud)

liquid_all = (liquid_all*cldm).values.ravel()
liquid_all = liquid_all[~np.isnan(liquid_all)]

cloud_all  = (cloud_all*cldm).values.ravel()
cloud_all  = cloud_all[~np.isnan(cloud_all)]

ice_all    = (ice_all*cldm).values.ravel()
ice_all    = ice_all[~np.isnan(ice_all)]

dataframe_m = pd.DataFrame({'Liquid':liquid_all + cloud_all,
                            'Ice':ice_all})

dataframe_s = pd.DataFrame({'Liquid':lws2 + lwc,
                            'Ice':iws})

dataframe_m['Type'] = 'WRF'
dataframe_s['Type'] = 'SOCRATES'
combined_df = pd.concat([dataframe_m, dataframe_s])

# Reshape the dataframe for seaborn boxplot
combined_df_melted = pd.melt(combined_df, id_vars='Type', var_name='Variable', value_name='Value')

import seaborn as sns
plt.figure(figsize=(8, 4))
sns.boxplot(x='Variable', y='Value', hue='Type', data=combined_df_melted,
            showfliers=False,
            palette={'WRF': '#BBCCEE', 'SOCRATES': '#EEEEBB'})
plt.legend(title='',fontsize=13)
plt.xlabel('')
plt.ylabel(r'Water Content [g/$m^3$]',fontsize=13)
plt.tick_params(axis='both',labelsize=13)
plt.savefig(f'Boxplot_watercontent_WRFSOC.png',\
            dpi=500,bbox_inches='tight')

###################################################3
# now with the number of particles
Ni_all = (Ni_all*cldm).values.ravel()
Ni_all = Ni_all[~np.isnan(Ni_all)]

Nc_all  = (Nc_all*cldm).values.ravel()
Nc_all  = Nc_all[~np.isnan(Nc_all)]

dataframe_m = pd.DataFrame({'Nrain':Nc_all,
            'Nice':Ni_all})

dataframe_s = pd.DataFrame({'Nrain':ds2,
              'Nice':nis})

dataframe_m['Type'] = 'WRF'
dataframe_s['Type'] = 'SOCRATES'
combined_df = pd.concat([dataframe_m, dataframe_s])

# Reshape the dataframe for seaborn boxplot
combined_df_melted = pd.melt(combined_df, id_vars='Type', var_name='Variable', value_name='Value')

dataframe_m2 = pd.DataFrame({
            'Nc':(Nc_all*0)})

dataframe_s2 = pd.DataFrame({
              'Nc':dc})

dataframe_m2['Type'] = 'WRF'
dataframe_s2['Type'] = 'SOCRATES'
combined_df2 = pd.concat([dataframe_m2, dataframe_s2])

# Reshape the dataframe for seaborn boxplot
combined_df_melted2 = pd.melt(combined_df2, id_vars='Type', var_name='Variable', value_name='Value')


import seaborn as sns
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(111)
sns.boxplot(ax=ax1,x='Variable', y='Value', hue='Type', data=combined_df_melted,
            showfliers=False,
            palette={'WRF': '#BBCCEE', 'SOCRATES': '#EEEEBB'})
ax2=ax1.twinx()
sns.boxplot(ax=ax2,x='Variable', y='Value', hue='Type', data=combined_df_melted2,
            showfliers=False,
            palette={'WRF': '#BBCCEE', 'SOCRATES': '#EEEEBB'},
            legend=False)

dataframe_m2 = pd.DataFrame({
            'Nc':(Nc_all[:4]*0)+200000})

dataframe_s2 = pd.DataFrame({
              'Nc':dc*0})

dataframe_m2['Type'] = 'WRF'
dataframe_s2['Type'] = 'SOCRATES'
combined_df2 = pd.concat([dataframe_m2, dataframe_s2])

# Reshape the dataframe for seaborn boxplot
combined_df_melted2 = pd.melt(combined_df2, id_vars='Type', var_name='Variable', value_name='Value')


sns.scatterplot(ax=ax2,data=combined_df_melted2, x="Variable", y="Value",
                hue="Type", s=500,marker='*',
                palette={'WRF': '#BBCCEE', 'SOCRATES': '#EEEEBB'},
                legend=False)

ax1.legend(title='',fontsize=13)
ax1.set_xlabel('')
ax1.set_ylabel(r'Particle Count [L$^{-1}$]',fontsize=13)
ax1.set_yscale('log')
ax1.set_ylim(10**-10)

ax2.set_ylabel(r'',fontsize=13)
ax2.set_yscale('log')
ax2.set_ylim(10**4,10**6)

plt.tick_params(axis='both',labelsize=13)
plt.savefig(f'Boxplot_particlecount_WRFSOC.png',\
            dpi=500,bbox_inches='tight')
