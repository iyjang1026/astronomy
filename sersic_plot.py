import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Sersic1D
from astropy.table import Table

tbl = Table.read('/volumes/ssd/intern/25_summer/M101_L/iso_integ_median_tbl.csv', format='ascii.csv')
sma = tbl['sma'] #장반경
intens = tbl['intens']/(1.89**2)

d = 6730
kpc = d * np.tan((np.pi/180)*((sma*1.89)/3600))

bulge_sersic = Sersic1D(amplitude=2632,r_eff=0.65, n=4)(kpc)
disk = Sersic1D(amplitude=166.6, r_eff=10.7, n=1)(kpc)
def mag(x): #등급으로 변환
    mag = -2.5*np.log10(x)+28.39
    return mag


fig, ax = plt.subplots()
ax.plot(kpc, mag(bulge_sersic), label='Bulge Sersic')
ax.plot(kpc, mag(disk), label='Disk sersic')
ax.plot(kpc, mag(bulge_sersic + disk), label='bulge+disk')
ax.scatter(kpc, mag(intens), label='data', color='tomato', s=5)
ax.invert_yaxis()
ax.set_xlabel('sma(kpc)')
ax.set_ylabel('$Mag_r$')
plt.legend()
plt.show()