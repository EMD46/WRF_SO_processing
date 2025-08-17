# -*- coding: utf-8 -*-
"""
Class for reading files from flights of
SOCRATES project. Different type formats
for each instrument or variables

@author: Estefania Montoya September 2020
"""

import numpy as np
import xarray as xr
import pandas as pd
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
from netCDF4 import num2date,Dataset


class Read_files_socrates():


    def __init__(self,main_dir):
        '''
        global variables: needed for each function
        that read a variable

        main_dir : parent directory of data
        '''

        self.dir = main_dir


    def one_nc_flight_xr(self,dir3,instrument):
        '''
        Read nc file of each flight mision

        #var3       : variable to read
        instrument : alias instrument will give the key of
                     which file to open to read variable var3
        dir3       : subdir where files are

        return     : array of variable var3
        '''
        file_dir    = glob.glob(self.dir+dir3+'/*'+instrument+'*')

        data_xr    = xr.open_dataset(file_dir[0])
        return data_xr

    def merge_nc_flight_xr(self,dir3,list_var,mergedim,changedate=False,subset=False,\
                           decode=False):
        '''
        Read nc file of each flight mision

        list_var   : list of variables to select
        dir3       : subdir where files are

        return     : array of variable var3
        '''
        # all files
        if subset:
            file_list = glob.glob(self.dir+dir3+f'/*{subset}*')
            file_list.sort()
        else:
            file_list = glob.glob(self.dir+dir3+'/*nc*')
            file_list.sort()

        # read files and merge
        dataset_all = []
        for i in file_list:
            if decode:
                data = xr.open_dataset(i,decode_times=False)
            else:
                data = xr.open_dataset(i)
            # get variables of interest
            data  = data[list_var]
            if changedate:
                datef = datetime.strptime(i.split('.')[-2],'%Y%m%d').strftime('%Y-%m-%d %H:%M:%S')
                date  = num2date(data[mergedim].values,units='hours since %s'%datef,\
                                 only_use_python_datetimes=True,only_use_cftime_datetimes=False)
                if mergedim=='hour':
                    data  = data.assign_coords(hour=date)
                if mergedim=='time':
                    data  = data.assign_coords(time=date)

            dataset_all.append(data)

        # concat list of files in the area wanted with resolution wanted
        dataset_all2 = xr.concat(dataset_all,dim=mergedim)

        return dataset_all2

    def plain_text_cnn(self,dir3,instrument,units):
        '''
        read plain text (txt,ascii,ict)

        instrument : alias instrument will give the key of
                     which file to open to read variable var3
        dir3       : subdir where files are

        return     : DataFrame
        '''
        file_dir           = glob.glob(self.dir+dir3+'/*'+instrument+'*')
        pd_dataset         = pd.read_csv(file_dir[0])
        pd_dataset.columns = pd_dataset.columns.str.replace(' ', '')

        # formating
        date_file        = num2date(pd_dataset['Start_UTC'],units=units,
                                    only_use_cftime_datetimes=False)
        pd_dataset.index = date_file
        pd_dataset       = pd_dataset.replace(-9999.0,np.nan)

        return pd_dataset


    def plain_text_flight_dropsonde(self,file,date,loc):
        '''
        read plain text (txt,ascii,ict)
        file       : file to read (full directory)
        date       : date to create fomr date_initial
        loc        : two constant columns where was drop lat,lon
                    format dic = {'lat':x,'lon':y}
        return     : DataFrame
        '''
        pd_dataset = pd.read_csv(file,sep=';')

        # formating
        date_file         = pd.date_range(date,freq='s',periods=len(pd_dataset))
        pd_dataset.index  = date_file
        pd_dataset        = pd_dataset.drop(['Unnamed: 0','Unnamed: 18','Time','hh','mm','ss'],axis=1)
        pd_dataset['lat'] = [loc['lat']]*len(pd_dataset)
        pd_dataset['lon'] = [loc['lon']]*len(pd_dataset)

        return pd_dataset


    def read_location_capricorn(self,dir3,date):
        '''
        So far I have no lat lon files for both capricorn projects
        I am using only radiometer data from 2016 to find the location
        date: string, format YMD (year, month day)
        '''

        A = Dataset(self.dir+dir3+'/Utah_MWR_LWP_PWV_%s_V1.cdf'%date)
        try:
            time = num2date(A['time_sec_since_1970'][:],units='seconds since 1970-01-01')
        except:
            time = num2date(A['sec_since_1970_time'][:],units='seconds since 1970-01-01')

        lat = A['latitude'][:]
        lon = A['longitude'][:]
        A.close()

        ds = xr.Dataset({'lat': xr.DataArray(data = lat, dims  = ['time'],
                        coords = {'time': time}),
                        'lon': xr.DataArray(data  = lon , dims  = ['time'],
                        coords = {'time': time})
                        })
        return ds

    def read_lwp_capricorn(self,dir3):
        '''
        I have so far two different sources for the LWP
        one in the same file as radar and lidar (2018)
        and the other in the radiometer file (2016)
        both have different units:
        2018: cm --> Total liquid water along LOS path
        2016: g/m2 --> liquid water path units
        from http://www.remss.com/measurements/
        the total water vapor 1 g/cm2 = 10 mm

        WARNING!!!!!!! POR AHORA NO LO HAGO
        '''
        return print('ja')
