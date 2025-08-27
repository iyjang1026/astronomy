import numpy as np
import astropy.io.fits as fits
from photutils.aperture import EllipticalAperture
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import sys

path = '/volumes/ssd/intern/25_summer/M51_L'

hdl = fits.open(path+'/sky_subed/coadd.fits')
hdu = hdl[0].data
hdr = hdl[0].header
wcs = WCS(hdr)
tbl = Table.read(path+'/tbl_median_test_1/tbl_-0.0.csv', format='ascii.csv')
mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)

sma = tbl['sma'][1:]
x0, y0 = tbl['x0'][1:], tbl['y0'][1:]
eps = tbl['ellipticity'][1:]
pa = tbl['pa'][1:]
#print(pa*np.pi/180)

#sys.exit()
median = np.median(hdu)
std = np.std(hdu)
fig, ax = plt.subplots(figsize=(6,5), subplot_kw=dict(projection=wcs))
ax.imshow(hdu,vmax=median+3*std, vmin=median-3*std,origin='lower')# median-9*std 
for i in range(len(sma)):
    if i%1 == 0:
        n = i
        aper = EllipticalAperture(positions=(x0[n],y0[n]), a=sma[n], b=sma[n]*(1-eps[n]), theta=pa[n]*np.pi/180)
        aper.plot(color='white', linewidth=0.7)

ax.set_xlabel('R.A.')
ax.set_ylabel('DEC')
ax.set_title('NGC 5195')
plt.show()