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


from Socrates_file_read import *
from Socrates_transects_dates import *
from Socrates_postprocessing import *

read = Read_files_socrates(dir_files)

# specify the location of data
parent = parent
s1,s2 = RF12_ship['time_gv'][0]

# flight
filesd = glob.glob(f'{parent}SOCRATES_CAPRICORN/*rf12_track*')
gv = pd.read_csv(filesd[0],sep=',')
gv.index = pd.to_datetime(gv['Unnamed: 0'])
gv.index.name = None
gv = gv.drop(columns=['Unnamed: 0'])
fd_m = gv.loc['2018-02-18 00:00':'2018-02-18 07:35']


##############################################################################
# SOCRATES
#############################################################################
# # # read file o 1 sps
# Microphysics flight 3 1 sps
segment =  RF12_ship
micro_1sps_f3  = read.one_nc_flight_xr(dict_files['RF12']['subdir'],inst[4])
# 3 is 2dc and 2 is 2ds
ds2            = read.one_nc_flight_xr(dict_files['RF12']['subdir'],inst[2])
#psd_2dc        = read.one_nc_flight_xr(dict_files['RF12']['subdir'],inst[3])

phase          = xr.open_dataset(f'{parent}/SOCRATES_CAPRICORN/flight/SOCRATES_phase_product.nc')
phase          = phase.phase[phase.flight_number==12]
phase          = phase.assign_coords(time=micro_1sps_f3.Time.values)
phase          = phase.rename({'time':'Time'})


cloud1 = micro_1sps_f3.PLWCD_RWIO
cloud2 = micro_1sps_f3.PLWC2DSA_2V#.PLWC2DCA_RWOI
cloud3 = ds2.IWC.assign_coords(Time=micro_1sps_f3.Time)
cloudt = cloud1 +  cloud2 +  cloud3
cloud_thresh = 1*10**-2 #1*10**-14


# thres model ice or liquid > 1*10**-14 g/kg
Rv = 461
cloud12 = (Rv*micro_1sps_f3.ATH1)/((micro_1sps_f3.PSFC*100)*cloud1**-1)
cloud22 = (Rv*micro_1sps_f3.ATH1)/((micro_1sps_f3.PSFC*100)*cloud2**-1)
cloud32 = (Rv*micro_1sps_f3.ATH1)/((micro_1sps_f3.PSFC*100)*cloud3**-1)
thres2  = 1*10**-14

condl12  = ((cloud12.loc[s1:s2] + cloud22.loc[s1:s2])>thres2) | (cloud32.loc[s1:s2]>thres2)

s1,s2 = segment['time_gv'][0]
yval  = micro_1sps_f3.GGALT.loc[s1:s2]/1000
ph    = phase.loc[s1:s2]

condl = (cloudt.loc[s1:s2]>cloud_thresh)#0.01 HEREEEEEEEEEEE
condl2 = condl & ((ph==1)) #liqu
condl3 = condl & ((ph==3) | (ph==2)) # ice and mixed

psd_cdp     = micro_1sps_f3.CCDP_RWIO[:,0,:]
psd_cdp_all = psd_cdp.assign_coords({'Vector31':psd_cdp.CellSizes})
psd_cdp_all = psd_cdp_all.rename({'Vector31':'new_bins'})
psd_cdp_all = psd_cdp_all.loc[s1:s2] # cm-3

dcd = psd_cdp_all/0.001 #to L-1
lcd = micro_1sps_f3.PLWCD_RWIO.loc[s1:s2].where(condl)

psd_2ds     = micro_1sps_f3.C2DSA_2V[:,0,:] # L
psd_2ds_all = psd_2ds.assign_coords({'Vector257':psd_2ds.CellSizes})
psd_2ds_all = psd_2ds_all.rename({'Vector257':'new_bins'})
psd_2ds_all = psd_2ds_all.loc[s1:s2]

psd_2ds_rain = assign_psd(psd_2ds_all,'C2DSA_2V',50).sum(axis=1)
psd_2ds_ice  = assign_psd(psd_2ds_all,'C2DSA_2V',200).sum(axis=1)


# particle count!!!
dc = dcd.sum(axis=1).where(condl) #from psd to one dim
dr = psd_2ds_rain.where(condl2)
di = psd_2ds_ice.where(condl3)

# water content
lwc = micro_1sps_f3.PLWC2DSA_2V.loc[s1:s2].where(condl2)
lwc = lwc + lcd # total water content
#lwc = lwc.where(lwc>0)

# ice water content
iwc = cloud3.loc[s1:s2].where(condl3)
#iwc = iwc.where(iwc>0)

# cloud top temperature
# firs id saw thooth that can be used
cloudt2 = ph[::2]#cloudt.loc[s1:s2][::2]
yval2   = yval[::2]
temp2   = micro_1sps_f3.ATH1.loc[s1:s2][::2]

con_t = []
for ind,(i,j) in enumerate(zip([650,871,2950,3061,3181],[870,1050,3060,3180,3280])):

    layeri = pd.DataFrame(cloudt2[i:j],columns=['layer'])
    layeri.index.name = None

    layeri['position'] = np.arange(len(layeri))
    layeri['tag']      = cloudt2[i:j]>0
    layeri['altitude'] = yval2[i:j]*1000
    layeri['temp']     = temp2[i:j]

    # first row is a True preceded by a False
    fst = layeri.position[layeri['tag'] & ~ layeri['tag'].shift(1).fillna(False)]
    # last row is a True followed by a False
    lst = layeri.position[layeri['tag'] & ~ layeri['tag'].shift(-1).fillna(False)]

    # filter those which are adequately apart
    consecutive = [(i, j) for i, j in zip(fst.values, lst.values) if j > i+1]
    # time by time find first and last value identified as cloud
    if len(consecutive)>=1:

        for pt in consecutive:
            aux = layeri[pt[0]:pt[1]]
            if abs(aux.altitude.values[0] - aux.altitude.values[-1])>=60:
                con_t.append({'Y_start':aux.altitude.values[0],'Y_end':aux.altitude.values[-1],
                              'T_start':aux.temp.values[0],'T_end':aux.temp.values[-1],
                              'PID_start':aux.layer.values[0],'PID_end':aux.layer.values[-1],
                              'Profile':ind})

layer_times = pd.DataFrame(con_t)

#then i hand picked the altitude and temperature from plotting each profile
#layer 1 1299.021362 -7.194171

cth = []

for i in [1,2,3,4,5,6]:
    if layer_times['Y_start'][i]>layer_times['Y_end'][i]:
       cth.append(layer_times['Y_start'][i])
    else:
       cth.append(layer_times['Y_end'][i])

cth_all = np.array(cth)
np.median(cth_all)
np.quantile(cth_all,0.75) - np.quantile(cth_all,0.25)

