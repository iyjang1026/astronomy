import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Sersic1D
from astropy.table import Table, Column
import sys

tbl = Table.read('/volumes/ssd/intern/25_summer/M51_L/M51b_t_iso_tbl.csv', format='ascii.csv')
#upper_tbl = Table.read('/volumes/ssd/intern/25_summer/M101_L/tbl/3sigma/tbl_130.csv', format='ascii.csv')
#lower_tbl = Table.read('/volumes/ssd/intern/25_summer/M101_L/tbl/3sigma/tbl_-128.csv', format='ascii.csv')

#upper_intens = upper_tbl['intens']/(1.89**2)
#lower_intens = lower_tbl['intens']/(1.89**2)


sma = tbl['sma'] #장반경
intens = tbl['intens']/(1.89**2)
intens_err = tbl['intens_err']/(1.89**2)
#print(abs(np.log10(intens_err)), np.log10(intens_err))
#sys.exit()
d = 8580
kpc = d * np.tan((np.pi/180)*((sma*1.89)/3600))

pa = tbl['pa']
eps = tbl['ellipticity']
fig, ax = plt.subplots(2,1,figsize=(5,2), sharex=True)
plt.subplots_adjust(hspace=0)
ax[0].plot(kpc, pa, '.-')
ax[1].plot(kpc, eps, '.-')
ax[0].set_ylabel('PA(deg)')
ax[1].set_xlabel('sma(kpc)')
ax[1].set_ylabel('Ellipticity')
#fig.suptitle('')
plt.show()
sys.exit()


bulge_sersic = Sersic1D(amplitude=2736,r_eff=0.63, n=0.4)(kpc)
disk = Sersic1D(amplitude=209.6, r_eff=9.58, n=1)(kpc)
def mag(x): #등급으로 변환
    mag = -2.5*np.log10(x)+28.05#29.04#28.39
    return mag


fig, ax = plt.subplots()
#ax.fill_between(kpc, mag(upper_intens),mag(lower_intens), color='skyblue')
ax.plot(kpc, mag(bulge_sersic + disk), label='bulge+disk', c='C2')
ax.plot(kpc, mag(bulge_sersic), label='Bulge Profile', linestyle='dashed', c='C0')
ax.plot(kpc, mag(disk), label='Disk Profile', linestyle='dashed', c='tomato')
#ax.errorbar(kpc,intens,yerr=intens_err, c='tomato')#(abs(mag(upper_intens)-mag(lower_intens))), c='tomato')
ax.scatter(kpc,mag(intens), label='data', color='tomato', s=5)
#ax.set_yscale('log', base=10)
ax.invert_yaxis()
ax.set_xlabel('sma(kpc)')
ax.set_ylabel('$Mag_r$')
plt.legend()
plt.show()
