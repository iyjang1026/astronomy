import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from photutils.aperture import RectangularAperture
import matplotlib.pyplot as plt
import sys
import os

def tbl(path):
    tbl = Table.read(path)
    ra = tbl['ra']*u.deg
    dec = tbl['dec']*u.deg
    return np.array([ra,dec])
    #print(SkyCoord(ra[0],dec[0]))

coord = tbl('/Users/jang-in-yeong/Documents/NGC6946_dw.csv')

def xy2wcs(path, coord):
    if not os.path.exists(path+'/thumbnail'):
        os.mkdir(path+'/thumbnail')
    hdul = fits.open(path+'/sky_subed/coadd.fits')
    hdu = hdul[0].data 
    hdr = hdul[0].header
    w = WCS(hdr)
    box_size = round(50/1.89)
    meam, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
    #plt.imshow(hdu, vmax=median+.09*std,vmin=median-.09*std,origin='lower')
    #fig, ax = plt.subplots(1,len(coord[0]))
    for i in range(len(coord[0])):
        x,y = w.world_to_pixel(SkyCoord(coord[0,i]*u.deg,coord[1,i]*u.deg)) #center of box
        arr = hdu[round(y-box_size):round(y+box_size),round(x-box_size):round(x+box_size)].astype(np.float32)
        #fits.writeto(path+'/thumbnail/'+str(format(i+1,'04'))+'.fits', arr, overwrite=True)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(arr, vmax=median+6*std, vmin=median-6*std, origin='lower', cmap='grey')
        #plt.imsave(path+'/thumbnail/'+str(format(i+1,'04')+'.png'),arr,vmax=median+3*std, vmin=median-3*std, origin='lower', cmap='grey')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.text(41,43,s=str(format(i+1,'02')),color='tomato', fontsize=30, fontweight='bold', bbox={'boxstyle':'square', 'fc':'white'})
        #plt.show()
        plt.savefig(path+'/thumbnail/'+str(format(i+1,'04')+'.png'))
        plt.close()
        #sys.exit()
        
        #box = RectangularAperture((x,y),round(100/1.89), round(100/1.89))
        #box.plot(color='C3')
        
    
        

xy2wcs('/volumes/ssd/intern/25_summer/NGC6946_L', coord)