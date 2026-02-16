import numpy as np
from astropy.io import fits
import glob
import sys
import os
from astropy.time import Time

def stack(path):
    if not os.path.exists(path+'/stacked'):
        os.mkdir(path+'/stacked')

    file = sorted(glob.glob(path+'/sky_subed/*.fits'))
    cen_idx = [i for i in range(0,len(file),10)]
    for j in cen_idx:
        #print(j)
        list=[]
        date=[]
        if j != cen_idx[-1]:
            for i in range(10):
                #print(i+j)
                hdu = fits.open(file[i+j])[0].data 
                hdr = fits.open(file[i+j])[0].header
                time = hdr['DATE-OBS']
                t = Time(time, format='isot', scale='utc')
                list.append(hdu)
                date.append(t.mjd)
            array = np.array(list)
            date_array = np.array(date)
            median_time = Time(np.median(date_array), format='mjd', scale='utc').isot
            ran_hdu = fits.PrimaryHDU(hdu)
            ran_hdr = ran_hdu.header
            ran_hdr.append(('CENOBSDATE', str(median_time), 'Median OBSTIME'))
            
            max = np.max(array, axis=0)
            n = format(j//10,'04')
            fits.writeto(path+'/stacked/sate_max'+str(n)+'.fits', max,header=ran_hdr, overwrite=True)
        
        else:
            for i in range(len(file)-cen_idx[-1]):
                #print(j+i)
                hdu = fits.open(file[i+j])[0].data 
                hdr = fits.open(file[i+j])[0].header
                time = hdr['DATE-OBS']
                t = Time(time, format='isot', scale='utc')
                list.append(hdu)
                date.append(t.mjd)
            array = np.array(list)
            date_array = np.array(date)
            median_time = Time(np.median(date_array), format='mjd', scale='utc').isot
            ran_hdu = fits.PrimaryHDU(hdu)
            ran_hdr = ran_hdu.header
            ran_hdr.append(('CENOBSDATE', str(median_time), 'Median OBSTIME'))
            
            max = np.max(array, axis=0)
            n = format(j//10,'04')
            fits.writeto(path+'/stacked/sate_max'+str(n)+'.fits', max,header=ran_hdr, overwrite=True)
    """
            hdu =  fits.open(file[i+j])[0].data 
            list.append(hdu)

        array = np.array(list)
    
        max = np.max(array, axis=0)
        n = format(j,'04')
        fits.writeto(path+'/sate_max'+str(n)+'.fits', max, overwrite=True)
    """
    
import warnings

warnings.filterwarnings('ignore')
#stack('/volumes/ssd/satelite/BSHS/250604E')

from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
path = '/volumes/ssd/satelite/stack_plot'
p = sorted(glob.glob(path+'/*.fits'))
fig, ax = plt.subplots(1,2, figsize=(8,5))
title = ['BSHS data', 'UOU data']
for i in range(len(p)):
    hdu = fits.open(p[i])[0].data 
    mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3.)
    ax[i].imshow(hdu, origin='lower',cmap='grey', vmax=median+3*std, vmin=median-3*std)
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
    ax[i].set_title(title[i])
plt.show()
