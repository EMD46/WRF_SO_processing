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

dir  = f'{parent}WRF_SO_post_frontal/final_sensitivity/case_insitu/'

d04 = [-58,-54,136,145]# 58
# south
#[-57.8,-55,136.5,144]
# north
#[-49,-45,141,149]
# north closed
# [-48.8,-46,141.8,144]
#north open
# [-48.8,-46,145.8,148.8]

#############################################################################
#model
##############################################################################
start_date = dt.datetime(2018, 2, 18, 3, 30)
#(dt.datetime(2018, 2, 18, 0, 20))
#dt.datetime(2018, 2, 18, 3, 30) 0,20
end_date   = dt.datetime(2018, 2, 18, 5, 20)
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

            mixc = (getvar(d,'QCLOUD')*1000)
            mixr = (getvar(d,'QRAIN')*1000)
            mixi = (getvar(d,'QICE')*1000)
            z    = (getvar(d,'height_agl')/1000)
            p    = getvar(d,'p',units='Pa')
            temp = getvar(d,'temp',units='K')
            Nice = getvar(d,'QNICE')
            Nr   = getvar(d,'QNRAIN')
            frac = (getvar(d,'CLDFRA')*100)
            ctt  = getvar(d,'ctt',units='degC',fill_nocloud=True)

            # fix Nice and Nr units
            Nicev = ((p*Nice)/(Rv*temp))* 0.001 # 1/L
            Nrv   = ((p*Nr)/(Rv*temp))*0.001
            # fix water content units
            mixc  = ((p*mixc)/(Rv*temp)) # g/m3
            mixr  = ((p*mixr)/(Rv*temp)) # g/m3
            mixi  = ((p*mixi)/(Rv*temp)) # g/m3

            mixtotal = mixc + mixr + mixi

            # Create a boolean mask where values are above the predefined number
            #furter limit by cloud detection with total mixing ratio
            mask = (frac>1) & (mixtotal> 1*10**-14)
            ch = z.where(mask)
            #ch = cth_a.max(axis=0) # warning this is like as seen by satellite and we have one layer
            # then filter clouds higher than 2 km, visual inspection this would be a second layer
            #ch = ch.where(ch<1.8)

            # trun ctt as a 3d for a cross section
            ctt = ctt.expand_dims(bottom_top=np.arange(63))

            lats, lons = latlon_coords(frac)
            xs,ys = ll_to_xy(d,d04[:2],d04[2:])
            x1,x2,y1,y2 = int(xs[0]),int(xs[1]),int(ys[0]),int(ys[1])

            # save them in a list
            cloud_l.append(mixc[:,y1:y2,x1:x2])
            ice_l.append(mixi[:,y1:y2,x1:x2])
            liquid_l.append(mixr[:,y1:y2,x1:x2])
            Ni_l.append(Nicev[:,y1:y2,x1:x2])
            Nc_l.append(Nrv[:,y1:y2,x1:x2])
            frac_l.append(frac[:,y1:y2,x1:x2])
            cth_l.append(ch[:,y1:y2,x1:x2])
            ctt_l.append(ctt[:,y1:y2,x1:x2])
            d.close()
# now we need to turn them into one 2d array but to concat we need to
#chnage the cross_line_id in a way they are consequtive (as if they were made
# in the same vert cross)


cloud_all = xr.concat(cloud_l,dim='time')
ice_all = xr.concat(ice_l,dim='time')
liquid_all = xr.concat(liquid_l,dim='time')
Ni_all = xr.concat(Ni_l,dim='time')
Nr_all = xr.concat(Nc_l,dim='time')
frac_all = xr.concat(frac_l,dim='time')
cth_all = xr.concat(cth_l,dim='time')
ctt_all = xr.concat(ctt_l,dim='time')

##
#This not valid since it is in g/m3, if run this remove conversion##############333
cloud_thresh = 1*10**-14
# this for clou AND THERE IS A SECOND LAYER OF CLOUDS NEED TO REMOVE to only capture low-level CLOUDS
# ctt and cth are visual inspections leaving a good variability to avoit overfiting
cldm2 = xr.where((frac_all > 1) & ((ice_all > cloud_thresh) | (liquid_all>cloud_thresh))\
                 & (ctt_all>-12) & (cth_all<=2), 1, np.nan)

# this valid
cldm2 = xr.where((liquid_all+cloud_all+ice_all)>0.01, 1, np.nan)

liquid_all2 = (liquid_all*cldm2).values.ravel()
liquid_all2 = liquid_all2[~np.isnan(liquid_all2)]

cloud_all  = (cloud_all*cldm2).values.ravel()
cloud_all  = cloud_all[~np.isnan(cloud_all)]

ice_all2    = (ice_all*cldm2).values.ravel()
ice_all2    = ice_all2[~np.isnan(ice_all2)]

# now with the number of particles
Ni_all = (Ni_all*cldm2)#.values.ravel()
#Ni_all = Ni_all[~np.isnan(Ni_all)]

Nr_all  = (Nr_all*cldm2)#.values.ravel()
#Nr_all  = Nr_all[~np.isnan(Nr_all)]

#cth
cth_all.max(axis=1).mean()
#LWC
total = liquid_all2 + cloud_all
#frac
ctt_all[:,0].values.ravel() #from top
#phase
len(liquid_all2[(liquid_all2>=cloud_thresh) & (ice_all2<cloud_thresh)])/len(liquid_all2[np.isfinite(liquid_all2)])


# surface precip cloud thresh or 0.01?
surf = liquid_all[:,0].where((liquid_all[:,0]>cloud_thresh) | (ice_all[:,0]>cloud_thresh))


cth_all.max(axis=1).median()
cth_all.max(axis=1).quantile(0.75) - cth_all.max(axis=1).quantile(0.25)

ctt_all.max(axis=1).median()
ctt_all.max(axis=1).quantile(0.75) - ctt_all.max(axis=1).quantile(0.25)

ctt_all[:,0].count()/(5*302*155)

np.median(total)
np.quantile(total,0.75) - np.quantile(total,0.25)

np.median(ice_all2[ice_all2>0])
np.quantile(ice_all2[ice_all2>0],0.75) - np.quantile(ice_all2[ice_all2>0],0.25)

Ni_all.median()
Ni_all.quantile(0.75) - Ni_all.quantile(0.25)

Nr_all.median()
Nr_all.quantile(0.75) - Nr_all.quantile(0.25)






