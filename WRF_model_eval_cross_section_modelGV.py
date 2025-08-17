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


# specify the location of data
parent = parent

dir  = f'{parent}WRF_SO_post_frontal/final_sensitivity/case_trans4/'

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
start_date = dt.datetime(2018, 2, 18, 0, 20)
#(dt.datetime(2018, 2, 18, 0, 20))
#dt.datetime(2018, 2, 18, 3, 30) 0,20
end_date   = dt.datetime(2018, 2, 18, 1, 00)
#dt.datetime(2018, 2, 18, 1, 00))
#dt.datetime(2018, 2, 18, 5, 20) 1,00
# Define a regular expression pattern to extract dates from file names
date_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}'

al_file = glob.glob(f"{dir}/*wrfout_d04*")
al_file.sort()


# convert Nc and others to volume they are in Kg^-1 to L^-1 with ideal gas law
Rv    = 461 #J/K kg

cloud_l, ice_l, liquid_l,Ni_l,Nc_l = [],[],[],[],[]
frac_l,temp_l,dewtemp_l = [],[],[]
z_l,ctt_l, cth_l  = [], [], []
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

            mixc = (getvar(d,'QCLOUD')*1000) #g/kg
            mixr = (getvar(d,'QRAIN')*1000)
            mixi = (getvar(d,'QICE')*1000)
            z    = (getvar(d,'height_agl')/1000)
            p    = getvar(d,'p',units='Pa')
            temp = getvar(d,'temp',units='K')
            dewtemp = getvar(d,'td',units='K')
            Nice = getvar(d,'QNICE')
            Nr   = getvar(d,'QNRAIN')
            frac = (getvar(d,'CLDFRA')*100)
            ctt  = getvar(d,'ctt',units='degC',fill_nocloud=True)

            mask = (frac>1) & ((mixr > 1*10**-14)  | (mixi > 1*10**-14))
            mixc = mixc.where(mask)
            mixr = mixr.where(mask)
            mixi = mixi.where(mask)

            # fix Nice and Nr units
            Nicev = ((p*Nice)/(Rv*temp))* 0.001 # 1/L
            Nrv   = ((p*Nr)/(Rv*temp))*0.001
            # fix water content units
            mixc  = ((p*mixc)/(Rv*temp)) # g/m3
            mixr  = ((p*mixr)/(Rv*temp)) # g/m3
            mixi  = ((p*mixi)/(Rv*temp)) # g/m3


            # Create a boolean mask where values are above the predefined number
            #furter limit by cloud detection with total mixing ratio
            ch = z.where(mask)
            #ch = cth_a.max(axis=0) # warning this is like as seen by satellite and we have one layer
            # then filter clouds higher than 2 km, visual inspection this would be a second layer
            #ch = ch.where(ch<1.8)

            # trun ctt as a 3d for a cross section
            ctt = ctt.expand_dims(bottom_top=np.arange(63))

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
            fra = vertcross(frac, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            tem = vertcross(temp, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            dewtem = vertcross(p, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            zz = vertcross(z, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            ctt = vertcross(ctt, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            cth = vertcross(ch, z, levels=height,wrfin=d, start_point=start_point,
                                end_point=end_point, latlon=True, meta=True)
            # save them in a list
            cloud_l.append(cloud)
            ice_l.append(ice)
            liquid_l.append(liquid)
            Ni_l.append(Ni)
            Nc_l.append(Nc)
            frac_l.append(fra)
            temp_l.append(tem)
            dewtemp_l.append(dewtem)
            z_l.append(zz)
            cth_l.append(cth)
            ctt_l.append(ctt)
            d.close()
# now we need to turn them into one 2d array but to concat we need to
#chnage the cross_line_id in a way they are consequtive (as if they were made
# in the same vert cross)


cloud_all = concat_cross(cloud_l)
ice_all = concat_cross(ice_l)
liquid_all = concat_cross(liquid_l)
Ni_all = concat_cross(Ni_l)
Nc_all = concat_cross(Nc_l)
frac_all = concat_cross(frac_l)
temp_all = concat_cross(temp_l)
dewtemp_all = concat_cross(dewtemp_l)
z_all = concat_cross(z_l)
cth_all = concat_cross(cth_l)
ctt_all = concat_cross(ctt_l)


cldm2 = xr.where((frac_all>1) & ((liquid_all > 1*10**-14)  | (ice_all > 1*10**-14)), 1, np.nan)
cldm3 = xr.where((liquid_all+cloud_all+ice_all)>0.01, 1, np.nan)



liquid_all2 = (liquid_all*cldm2).values.ravel()
liquid_all2 = liquid_all2[~np.isnan(liquid_all2)]

cloud_all  = (cloud_all*cldm2).values.ravel()
cloud_all  = cloud_all[~np.isnan(cloud_all)]

ice_all2    = (ice_all*cldm2).values.ravel()
ice_all2    = ice_all2[~np.isnan(ice_all2)]

# now with the number of particles
Ni_all = (Ni_all*cldm2).values.ravel()
Ni_all = Ni_all[~np.isnan(Ni_all)]

Nc_all  = (Nc_all*cldm2).values.ravel()
Nc_all  = Nc_all[~np.isnan(Nc_all)]


len(liquid_all2[(liquid_all2>=cloud_thresh) & (ice_all2<cloud_thresh)])/len(liquid_all2[np.isfinite(liquid_all2)])


# surface precip cloud thresh or 0.01?
surf = liquid_all[0].where((liquid_all[0]>cloud_thresh) | (ice_all[0]>cloud_thresh))


cth_all.median()
cth_all.quantile(0.75) - cth_all.quantile(0.25)

ctt_all.median()
ctt_all.quantile(0.75) - ctt_all.quantile(0.25)



np.median(total)
np.quantile(total,0.75) - np.quantile(total,0.25)

np.median(ice_all2[ice_all2>0])
np.quantile(ice_all2[ice_all2>0],0.75) - np.quantile(ice_all2[ice_all2>0],0.25)

np.median(Ni_all)
np.quantile(Ni_all[Ni_all>0],0.75) - np.quantile(Ni_all[Ni_all>0],0.25)

np.median(Nc_all)
np.quantile(Nc_all[Nc_all>0],0.75) - np.quantile(Nc_all[Nc_all>0],0.25)
