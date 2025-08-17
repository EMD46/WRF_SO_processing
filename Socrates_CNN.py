# -*- coding: utf-8 -*-
"""
Visualise the Cloud Condensation Nuclei (CCN)
from socrates

@author: Estefania Montoya November 2020
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
#%matplotlib agg

# read classes from developed for file reading, plotting and some statistics
import sys
sys.path.append(PATH)

from Socrates_file_read import *
from Socrates_make_plots import *
from Socrates_transects_dates import *

#------------------------------------------------------------------------------
# preliminary information for reading
#------------------------------------------------------------------------------
dir     = dir
RF      = 'RF12'
#segment = RF12

# # # call reading class and plotting class
read = Read_files_socrates(dir)
plot = plot_socrates(dir+'figures/')


# # # read file o 1 sps
# Microphysics flight 3 1 sps
micro_1sps_f3  = read.one_nc_flight_xr(dict_files[RF]['subdir'],inst[4])
# 3 is 2dc and 2 is 2ds
psd_2ds        = read.one_nc_flight_xr(dict_files[RF]['subdir'],inst[2])
psd_2dc        = read.one_nc_flight_xr(dict_files[RF]['subdir'],inst[3])

cloud1 = micro_1sps_f3.PLWCD_RWIO
cloud2 = micro_1sps_f3.PLWC2DSA_2V#.PLWC2DCA_RWOI
cloud3 = psd_2ds.IWC.assign_coords(Time=micro_1sps_f3.Time)
cloud4 = micro_1sps_f3.PLWC2DCA_RWOI
cloud5 = psd_2dc.IWC.assign_coords(Time=micro_1sps_f3.Time)
cloudt = cloud1 +  cloud2 +  cloud3 #+ cloud4 + cloud5

condl = (cloudt>0.01)



plt.plot(micro_1sps_f3.Time.values,micro_1sps_f3.CONCN.where(condl).values)
plt.twinx()
plt.plot(micro_1sps_f3.Time.values,micro_1sps_f3.GGALT.values,color='black')





# with usahs

yy = micro_1sps_f3.GGALT.loc['2018-02-18T00:00:40':'2018-02-18T07:39:20'].resample(Time='10s').mean()
yy = yy.to_dataframe()
yy.index.name = None

# CCN data is every 10 seconds
ccn  = read.plain_text_cnn(dict_files[RF]['subdir'],inst[0],units=dict_files[RF]['units'])
ccn[ccn.CCN==ccn.CCN_error] = np.nan




aux  = ccn[yy.values>=2000]
aux2 = ccn[(yy<2000) & (yy>=500)]
aux3 = ccn[yy.values<500]

plt.figure(figsize=(6,4))
ax = plt.gca()
ax.errorbar(aux.Supersaturation.values,aux.CCN.values,yerr=aux.CCN_error,fmt='o',c='dimgray')
ax.errorbar(aux2.Supersaturation.values,aux2.CCN.values,yerr=aux2.CCN_error,fmt='o',c='red')
ax.errorbar(aux3.Supersaturation.values,aux3.CCN.values,yerr=aux3.CCN_error,fmt='o',c='black')
plt.yscale('log')
plt.ylim(10**-1,6*10**2)
plt.xlim(0,0.9)
plt.xlabel('Supersaturation [%]',fontsize=14)
plt.ylabel(r'CCN [$cm^{-3}$]',fontsize=14)
plt.tick_params('both',labelsize=13)
plt.savefig(dir+'/RF12_CCN_supersat.png',dpi=500,bbox_inches='tight')

plt.figure(figsize=(6,4))
ax = plt.gca()
ax.errorbar(yy[yy.values>=2000].values/1000,aux.CCN.values,yerr=aux.CCN_error,fmt='o',c='dimgray')
ax.errorbar(yy[(yy<2000) & (yy>=500)].values/1000,aux2.CCN.values,yerr=aux2.CCN_error,fmt='o',c='red')
ax.errorbar(yy[yy.values<500].values/1000,aux3.CCN.values,yerr=aux3.CCN_error,fmt='o',c='black')
plt.yscale('log')
plt.ylim(10**-1,6*10**2)
plt.xlabel('altitude [km]',fontsize=14)
plt.ylabel(r'CCN [$cm^{-3}$]',fontsize=14)
plt.tick_params('both',labelsize=13)
plt.savefig(dir+'/RF12_CCN_alt.png',dpi=500,bbox_inches='tight')
