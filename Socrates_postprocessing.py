# -*- coding: utf-8 -*-
"""
Functions I will use in more than one code
to do some statistical anaylises

important to know if function is not here
it is because it will not be used in more than one code

@author: Estefania Montoya December 2020
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from netCDF4 import num2date


# ------------------------- filter -------------------------------------------

# filter data from font
def filter_fronts(front,typef,datef,lonlim,latlim):
    '''
    since fronts are everywhere
    leave only the ones in the australian
    '''
    #take last hour since we take hour + 50 minutes
    front_t = front[typef].loc[:datef][-1,:,:,:]
    frontlon_t = front_t[:,:,1].values
    frontlat_t = front_t[:,:,0].values
    # nan to all values outside region
    frontlon_t[frontlon_t < min(lonlim)] = np.nan
    frontlon_t[frontlon_t > max(lonlim)] = np.nan
    frontlat_t[frontlat_t < min(latlim)] = np.nan
    frontlat_t[frontlat_t > max(latlim)] = np.nan
    return frontlon_t,frontlat_t

def closest(lst, K):
    '''
    Find the closest date in a given list of dates
    lst: list of dates
    K : target date to find closest in lst
    Taken from:
    https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
    '''
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]


def nearest(items, pivot):
    '''
    https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    '''

    return min(items, key=lambda x: abs(x - pivot))

def most_frequent(List):
    return max(set(List), key = List.count)

def find_nearest(array, value):
    '''
    https://www.codegrepper.com/code-examples/python/find+nearest+python
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_iqr(x):
    '''
    https://www.statology.org/interquartile-range-python/
    '''
    return np.subtract(*np.percentile(x, [75, 25]))

import math

def distance(origin, destination):
    '''
    calculate the distance between two points
    considering the sphere shape of earth
    using the haversine method
    taken from:
    https://gist.github.com/rochacbruno/2883505
    '''

    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    #d = radius * c
    d = np.rad2deg(c)
    return d


def hysplit_convert_txt_to_arrays(filename_, number_of_hours):
  '''
  Author: Nasim 2022
  convert txt files from the Hysplit model into arrays
  filename_ which file we want to read
  number_of_hours the backtrajectory hours
  output: array
  '''
  file_obj = open(filename_, 'r')

  line_list = file_obj.readlines()

  file_obj.close()

  file_traj_list = []
  traj_number = -1
  for line_inx, line_str in enumerate(line_list):
    if line_str == '     1 PRESSURE\n':
      traj_number += 1
      for r_ in range(number_of_hours + 1):
        new_line_list = line_list[line_inx + r_ + 1].split()
        new_line_list.append(traj_number)
        file_traj_list.append(new_line_list)

  arr_ = np.zeros((len(file_traj_list), 12), dtype=float)
  for r_ in range(len(file_traj_list)):
    for c_ in range(12):
      arr_[r_, c_] = file_traj_list[r_][c_ + 2]

  return arr_

def resize_psd(array,var,limit1,new_bins,micro_1sps_f3,limit2=False):
    '''
    takes a psd array and resize de bins
    of a given variables, considering the
    limit values to which we can trust data
    array : xarray
    var   : variable to resize
    limit : set zero all values below limi
    new_bins: array of new size bins of mid values
    '''
    array2 = array[var][:len(micro_1sps_f3.Time),:]
    array2 = array2.assign_coords(CIPcorrlen=array.bin_mid,Time=micro_1sps_f3.Time)
    array2[:,array.bin_mid<limit1] = 0.0
    if limit2:
        array2[:,array.bin_mid>limit2] = 0.0

    new_bins_loc = np.digitize(array.bin_mid,new_bins)
    resize_bin   = [np.nansum(array2.values[:,new_bins_loc==j],axis=1) for j in range(len(new_bins))]
    # set nan al zero values
    resize_bin = np.array(resize_bin)
    resize_bin[resize_bin==0] = np.nan
    # turn it again to xarray so que can search by dates
    foo = xr.DataArray(resize_bin.T, coords=[array2.Time.values, new_bins], dims=["Time", "new_bins"])
    return foo


def assign_psd(array,var,limit1,limit2=False):
    '''
    takes a psd array removes values Belowthe
    limit to which we can trust data
    array : xarray
    var   : variable to resize
    limit : set zero all values below limi
    '''
    array2 = array.copy()
    array2[:,array2.new_bins<limit1] = 0.0
    if limit2:
        array2[:,array2.new_bins>limit2] = 0.0
        new_bins = array2.new_bins[(array2.new_bins>=limit1) & (array2.new_bins<=limit2)]
    else:
        new_bins = array2.new_bins[array2.new_bins>=limit1]

    # turn it again to xarray so que can search by dates
    foo = xr.DataArray(array2.loc[:,new_bins].values,
                       coords=[array2.Time.values, new_bins],
                       dims=["Time", "new_bins"])
    return foo


def conditional_binning (wd,pollux,statistic,nbins = 16):
    '''
    seriewd: mean diameter series
    seriepollux: concentration series
    nbins: número de bins en los que se va a dividir la serie
    '''
    wd = wd.to_dataframe()
    pollux = pollux.to_dataframe()
    vbins = np.linspace(0,50,nbins)#[:,0]
    dgroups = wd.groupby(np.digitize(wd.values[:,0],vbins))

    pollution_matrix = []
    bn               = []
    for j in dgroups.groups.keys():
        daux = dgroups.get_group(j)
        if statistic == 'Mean' or statistic =='mean':
            pollution_matrix.append(float(pollux.loc[(daux.index)].mean().values))
        elif statistic == 'Count' or statistic == 'count':
            pollution_matrix.append(float(pollux.loc[(daux.index)].count().values))
        else:
            raise ValueError ('Invalid Statistic Method')
        bn.append(j-1)

    return np.array(pollution_matrix),np.round(vbins[np.array(bn)-1],0)*0.001




def apply_cld_mask(xxarray):
    '''
    depolarization_ratio and crosspol_backscatter
    might have info in clear sky sections that can
    create noise in the analyses
    Then: clear sky in cld_cover_type is 0
    make them as nan in the other arrays
    and replace it in the dataarray
    '''
    # work with the radar and lidar product
    new_var1 = np.zeros([len(xxarray.height.values),\
                         len(xxarray.cld_mask.hour.values)])
    for i in range(len(xxarray.height.values)):
        aux   = np.copy(xxarray.cld_mask[:,i,:].values)
        aux[aux==0] = np.nan
        aux[np.isfinite(aux)] = 1
        new_var1[i,:] = aux
    return new_var1


def cfad_cftd(y,cf,ybins,cfbins,ravel=False,filtery=False,filterx=False):
    '''
    contouf plots by frequency...similar to 2d histogram

    y         : either height of temperature array
    cf        : array which will estimate frequencies
    binspan   : len of y bins
    cfbinspan : len of cf bins
    ravel     : True when y is 2d
    '''

    # I need both variables to be 1d arrays with no nan
    # is there is nan in one array the same position in the other should
    # also be deleted
    if ravel: # both x and y are 2d
        yy  = y.ravel()
        cff = cf.ravel()
    else: # y is 1d and x is 2d
        yy  = np.meshgrid(range(np.shape(cf)[1]),y)[1].ravel()
        cff = cf.ravel()

    # depolarization_ratio has values larger than 1
    if filtery:
        y[y>1]  = np.nan
    if filterx:
        cf[cf>1] = np.nan

    # only finite values
    yt = yy[np.isfinite(yy) & np.isfinite(cff)]
    ct = cff[np.isfinite(yy) & np.isfinite(cff)]

    # identifiy which values or positions are from each y bin
    ybin_dig  = np.digitize(yt,ybins)

    cfd     = np.zeros([len(ybins),len(cfbins)-1])
    cfd_n   = np.zeros([len(ybins),len(cfbins)-1])
    percent = np.zeros([3,len(ybins)])
    for i in range(len(ybins)):
        x     = ct[ybin_dig==i+1]
        if len(x)>0:
            freq, bins = np.histogram(x,bins=cfbins,density=False)
            # normalization by row
            cfd[i,:] = freq/len(x)
            cfd_n[i,:] = freq/len(ct)
            test = freq/len(ct)
            test[test < 0.0006] = np.nan # consider from plot since light blue
            if len(test[np.isfinite(test)]) > 1:
                percent[:,i] = np.percentile(x,[25,50,75])
            else:
                percent[:,i] = [np.nan]*3
        else:
            cfd[i,:] = [0.0]*(len(cfbins)-1)
            cfd_n[i,:] = [0.0]*(len(cfbins)-1)
            percent[:,i] = [np.nan]*3
    return cfd, cfd_n, cfbins[:-1],ybins,percent


def hist2D(seriex,seriey,Nx,Ny):
    '''ambas series deben estar
    en el mismo periodo de tiempo
    y sin NAN
    Se aplica la regla de sturges
    para el numero de bins en cada variable'''

    #Nx=int(1 + 3.322 * np.log(len(seriex)))
    #Ny=int(1 + 3.322 * np.log(len(seriey)))

    #Nx, Ny = np.linspace(min(seriex),max(seriey), Nx), np.linspace(min(seriey),max(seriey), Ny)

    nbins = (Nx,Ny)
    #print len(seriex),len(seriey)
    H, xedges, yedges = np.histogram2d(seriex,seriey,bins=nbins) # OJO NORMED = TRUE NO NORMALIZA
    Hnorm = (H/len(seriex))*100
    #xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
    #ycenters = yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1])

    Hmasked = np.ma.masked_where(Hnorm==0.0,Hnorm) # Mask pixels with a value of zero
    return xedges,yedges,Hmasked





def conditional_contourf(x,y,z,ybins,xbins,y2d=False):
    '''
    contouf plots conditioned by x variable

    y         : y dimension
    x         : x new dimension
    z         : vaiable to find mean values given a x and y value
    '''

    if y2d:
        yy = y.ravel()
    else:
        yy = np.meshgrid(range(z.shape[1]),y)[1].ravel()
    xx = x.ravel()
    zz = z.ravel()

    # only finite values
    yt = yy[np.isfinite(yy) & np.isfinite(xx) & np.isfinite(zz)]
    xt = xx[np.isfinite(yy) & np.isfinite(xx) & np.isfinite(zz)]
    zt = zz[np.isfinite(yy) & np.isfinite(xx) & np.isfinite(zz)]

    # identifiy which values or positions are from each y bin
    ybin_dig  = np.digitize(yt,ybins)
    xbin_dig  = np.digitize(xt,xbins)

    cond = np.zeros([len(ybins),len(xbins)])

    for i in range(1,len(ybins)):
        for j in range(1,len(xbins)):
            zmean = zt[(ybin_dig==i) & (xbin_dig==j)]
            if len(zmean)> 20: #10
                cond[i-1,j-1] = np.percentile(zmean,50)
            else:
                cond[i-1,j-1] = np.nan

    return cond
