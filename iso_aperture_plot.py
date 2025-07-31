import numpy as np
import astropy.io.fits as fits
from photutils.aperture import EllipticalAperture
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
import sys


hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/sky_subed/coadd.fits')[0].data 
tbl = Table.read('/volumes/ssd/intern/25_summer/M101_L/iso_tbl.csv', format='ascii.csv')

mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)

sma = tbl['sma'][21:]
x0, y0 = tbl['x0'][21:], tbl['y0'][21:]
eps = tbl['ellipticity'][21:]
pa = tbl['pa'][21:]

plt.imshow(hdu,vmax=10000, vmin=0,origin='lower')# median-9*std
for i in range(25):
    n = i
    aper = EllipticalAperture(positions=(x0[n],y0[n]), a=sma[n], b=sma[n]*(1-eps[n]), theta=pa[n])
    aper.plot(color='white')


plt.show()