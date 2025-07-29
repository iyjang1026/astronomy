import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.table import Table

tbl = Table.read('/volumes/ssd/intern/25_summer/M101_L/iso_tbl1.csv', format='ascii.csv')
intens = tbl['intens']
sma = tbl['sma']
pa = tbl['pa']
eps = tbl['ellipticity']

d = 6730

kpc = d * np.tan((np.pi/180)*((sma*1.89)/3600))
fig, axes = plt.subplots()

axes.plot(kpc, np.log10(intens), '.-')
#axes.invert_yaxis()
axes.set_xlabel('sma')
axes.set_ylabel('log intens')
plt.show()
